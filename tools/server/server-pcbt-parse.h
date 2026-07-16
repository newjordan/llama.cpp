// PCBT contract v1 — request parsing, validation, error mapping, and
// canonical digests (PCBT-2, schema slice). Self-contained over nlohmann
// json + the vendored sha256, so schema and error-mapping behavior is
// testable against tests/pcbt/fixtures/ without a server
// (tests/test-pcbt-parse.cpp locks digest parity with the Python reference
// in scripts/treebeard-pcbt-contract-lint.py).

#pragma once

#include "server-pcbt.h"

#include <nlohmann/json.hpp>

extern "C" {
#include "sha256.h"
}

#include <cctype>
#include <cstring>
#include <string>

// Error classes map 1:1 to HTTP statuses (contract §9).
enum class pcbt_error : uint8_t {
    NONE,             // -
    INVALID_REQUEST,  // 400
    NOT_FOUND,        // 404
    CONFLICT,         // 409
    EXPIRED,          // 410
    UNPROCESSABLE,    // 422
    CAPACITY,         // 503
};

static inline int pcbt_error_http(pcbt_error e) {
    switch (e) {
        case pcbt_error::NONE:            return 200;
        case pcbt_error::INVALID_REQUEST: return 400;
        case pcbt_error::NOT_FOUND:       return 404;
        case pcbt_error::CONFLICT:        return 409;
        case pcbt_error::EXPIRED:         return 410;
        case pcbt_error::UNPROCESSABLE:   return 422;
        case pcbt_error::CAPACITY:        return 503;
    }
    return 500;
}

struct pcbt_parse_result {
    pcbt_error  error = pcbt_error::NONE;
    std::string message;

    bool ok() const { return error == pcbt_error::NONE; }
    static pcbt_parse_result fail(pcbt_error e, std::string msg) {
        return { e, std::move(msg) };
    }
};

// Canonical JSON (contract §1): sorted keys, no insignificant whitespace,
// integers only. nlohmann::json objects already iterate key-sorted; dump()
// emits compact separators, so canonicalization reduces to rejecting floats
// and serializing. The cross-language golden digest test guards any
// serialization divergence from the Python reference.
static inline bool pcbt_json_integer_only(const nlohmann::json & j) {
    if (j.is_number_float()) {
        return false;
    }
    if (j.is_object() || j.is_array()) {
        for (const auto & v : j) {
            if (!pcbt_json_integer_only(v)) {
                return false;
            }
        }
    }
    return true;
}

static inline std::string pcbt_digest(const char * domain, const nlohmann::json & j) {
    const std::string canon = j.dump();
    sha256_t h;
    sha256_init(&h);
    sha256_update(&h, (const unsigned char *) domain, strlen(domain));
    const unsigned char nl = '\n';
    sha256_update(&h, &nl, 1);
    sha256_update(&h, (const unsigned char *) canon.data(), canon.size());
    unsigned char out[32];
    sha256_final(&h, out);
    static const char * hexd = "0123456789abcdef";
    std::string hex = "sha256:";
    for (unsigned char b : out) {
        hex += hexd[b >> 4];
        hex += hexd[b & 15];
    }
    return hex;
}

static inline bool pcbt_ident(const std::string & s, size_t lo, size_t hi) {
    if (s.size() < lo || s.size() > hi) {
        return false;
    }
    for (char c : s) {
        if (!isalnum((unsigned char) c) && c != '.' && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

static inline bool pcbt_int_in(const nlohmann::json & j, const char * key, int64_t lo, int64_t hi) {
    if (!j.contains(key) || !j[key].is_number_integer()) {
        return false;
    }
    const int64_t v = j[key].get<int64_t>();
    return lo <= v && v <= hi;
}

struct pcbt_create_request {
    std::string request_id;
    int32_t     source_node_id  = -1;
    int32_t     source_state_id = -1;
    int32_t     source_fork_id  = -1;
    std::vector<std::pair<std::string, nlohmann::json>> branches; // key -> completion request
    pcbt_budget budget;
    std::string acceptance_kind;
    std::string acceptance_name;
    std::string create_digest;   // pcbt.create.v1 over body minus request_id
};

// Contract §2. Unknown fields rejected; floats rejected anywhere in the body.
static inline pcbt_parse_result pcbt_parse_create(const nlohmann::json & body, pcbt_create_request & out) {
    using R = pcbt_parse_result;
    using E = pcbt_error;
    if (!body.is_object()) {
        return R::fail(E::INVALID_REQUEST, "body must be an object");
    }
    for (const auto & kv : body.items()) {
        const std::string & k = kv.key();
        if (k != "request_id" && k != "source" && k != "branches" &&
            k != "budget" && k != "acceptance_contract") {
            return R::fail(E::INVALID_REQUEST, "unknown field: " + k);
        }
    }
    for (const char * k : { "request_id", "source", "branches", "budget", "acceptance_contract" }) {
        if (!body.contains(k)) {
            return R::fail(E::INVALID_REQUEST, std::string("missing field: ") + k);
        }
    }
    if (!pcbt_json_integer_only(body)) {
        return R::fail(E::INVALID_REQUEST, "floats are forbidden in digested payloads");
    }
    if (!body["request_id"].is_string() || !pcbt_ident(body["request_id"], 1, 128)) {
        return R::fail(E::INVALID_REQUEST, "bad request_id");
    }
    out.request_id = body["request_id"];

    const auto & src = body["source"];
    if (!src.is_object() || !pcbt_int_in(src, "node_id", 0, INT32_MAX)) {
        return R::fail(E::INVALID_REQUEST, "bad source");
    }
    for (const auto & kv : src.items()) {
        if (kv.key() != "node_id" && kv.key() != "state_id" && kv.key() != "fork_id") {
            return R::fail(E::INVALID_REQUEST, "unknown source field: " + kv.key());
        }
    }
    out.source_node_id  = src["node_id"].get<int32_t>();
    out.source_state_id = src.contains("state_id") ? src["state_id"].get<int32_t>() : -1;
    out.source_fork_id  = src.contains("fork_id")  ? src["fork_id"].get<int32_t>()  : -1;

    const auto & br = body["branches"];
    if (!br.is_array() || br.size() < 2 || br.size() > 12) {
        return R::fail(E::INVALID_REQUEST, "branches must be 2..12");
    }
    for (const auto & b : br) {
        if (!b.is_object() || b.size() != 2 || !b.contains("key") || !b.contains("request")) {
            return R::fail(E::INVALID_REQUEST, "branch fields must be exactly key,request");
        }
        if (!b["key"].is_string() || !pcbt_ident(b["key"], 1, 64)) {
            return R::fail(E::INVALID_REQUEST, "bad branch key");
        }
        const std::string key = b["key"];
        for (const auto & seen : out.branches) {
            if (seen.first == key) {
                return R::fail(E::INVALID_REQUEST, "duplicate key " + key);
            }
        }
        const auto & req = b["request"];
        if (!req.is_object() || !pcbt_int_in(req, "max_tokens", 1, 262144)) {
            return R::fail(E::INVALID_REQUEST, "branch request needs int max_tokens");
        }
        if (req.contains("stream") && req["stream"].is_boolean() && req["stream"].get<bool>()) {
            return R::fail(E::INVALID_REQUEST, "streaming branches forbidden");
        }
        out.branches.emplace_back(key, req);
    }

    const auto & bud = body["budget"];
    if (!bud.is_object() || bud.size() != 4 ||
        !pcbt_int_in(bud, "max_slots", 2, 12) ||
        !pcbt_int_in(bud, "max_predicted_tokens", 1, 262144) ||
        !pcbt_int_in(bud, "deadline_ms", 1000, 600000) ||
        !pcbt_int_in(bud, "max_candidate_bytes", 1024, 8388608)) {
        return R::fail(E::INVALID_REQUEST, "bad budget");
    }
    if (bud["max_slots"].get<int64_t>() < (int64_t) out.branches.size()) {
        return R::fail(E::INVALID_REQUEST, "max_slots must cover branches");
    }
    out.budget.max_slots            = bud["max_slots"].get<uint32_t>();
    out.budget.max_predicted_tokens = bud["max_predicted_tokens"].get<uint32_t>();
    out.budget.deadline_ms          = bud["deadline_ms"].get<uint64_t>();
    out.budget.max_candidate_bytes  = bud["max_candidate_bytes"].get<uint64_t>();

    const auto & ac = body["acceptance_contract"];
    if (!ac.is_object() || ac.size() != 2 || !ac.contains("kind") || !ac.contains("name") ||
        !ac["kind"].is_string() || ac["kind"].get<std::string>() != "external" ||
        !ac["name"].is_string() || ac["name"].get<std::string>().empty() ||
        ac["name"].get<std::string>().size() > 64) {
        return R::fail(E::INVALID_REQUEST, "bad acceptance_contract");
    }
    out.acceptance_kind = ac["kind"];
    out.acceptance_name = ac["name"];

    nlohmann::json digest_body = body;
    digest_body.erase("request_id");
    out.create_digest = pcbt_digest("pcbt.create.v1", digest_body);
    return {};
}

struct pcbt_commit_request {
    int32_t     winner_node_id   = -1;
    int32_t     expected_fork_id = -1;
    std::string candidate_digest;
    std::string evidence_kind;
    std::string evidence_digest;
    nlohmann::json evidence;          // full bound evidence object
    std::string evidence_obj_digest;  // pcbt.evidence.v1
};

static inline pcbt_parse_result pcbt_parse_commit(const nlohmann::json & body, pcbt_commit_request & out) {
    using R = pcbt_parse_result;
    using E = pcbt_error;
    if (!body.is_object() || body.size() != 4 ||
        !pcbt_int_in(body, "winner_node_id", 0, INT32_MAX) ||
        !pcbt_int_in(body, "expected_fork_id", 0, INT32_MAX) ||
        !body.contains("candidate_digest") || !body["candidate_digest"].is_string() ||
        !body.contains("evidence")) {
        return R::fail(E::INVALID_REQUEST, "commit fields must be exactly winner_node_id, expected_fork_id, candidate_digest, evidence");
    }
    const std::string cd = body["candidate_digest"];
    if (cd.rfind("sha256:", 0) != 0) {
        return R::fail(E::INVALID_REQUEST, "candidate_digest must be sha256:<hex>");
    }
    const auto & ev = body["evidence"];
    if (!ev.is_object() || !ev.contains("kind") || !ev.contains("digest")) {
        return R::fail(E::INVALID_REQUEST, "evidence fields");
    }
    for (const auto & kv : ev.items()) {
        if (kv.key() != "kind" && kv.key() != "digest" && kv.key() != "summary") {
            return R::fail(E::INVALID_REQUEST, "unknown evidence field: " + kv.key());
        }
    }
    if (!ev["digest"].is_string() ||
        ev["digest"].get<std::string>().rfind("sha256:", 0) != 0) {
        return R::fail(E::INVALID_REQUEST, "evidence digest must be sha256:<hex>");
    }
    if (!pcbt_json_integer_only(body)) {
        return R::fail(E::INVALID_REQUEST, "floats are forbidden in digested payloads");
    }
    out.winner_node_id   = body["winner_node_id"].get<int32_t>();
    out.expected_fork_id = body["expected_fork_id"].get<int32_t>();
    out.candidate_digest = cd;
    out.evidence         = ev;
    out.evidence_kind    = ev["kind"].is_string() ? ev["kind"].get<std::string>() : "";
    out.evidence_digest  = ev["digest"];
    out.evidence_obj_digest = pcbt_digest("pcbt.evidence.v1", ev);
    return {};
}

struct pcbt_abort_request {
    int32_t     expected_fork_id = -1;
    std::string reason;
};

static inline pcbt_parse_result pcbt_parse_abort(const nlohmann::json & body, pcbt_abort_request & out) {
    using R = pcbt_parse_result;
    using E = pcbt_error;
    if (!body.is_object() || body.size() != 2 ||
        !pcbt_int_in(body, "expected_fork_id", 0, INT32_MAX) ||
        !body.contains("reason") || !body["reason"].is_string()) {
        return R::fail(E::INVALID_REQUEST, "abort fields");
    }
    const std::string reason = body["reason"];
    if (reason.empty() || reason.size() > 128) {
        return R::fail(E::INVALID_REQUEST, "reason length");
    }
    out.expected_fork_id = body["expected_fork_id"].get<int32_t>();
    out.reason = reason;
    return {};
}
