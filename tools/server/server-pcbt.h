// Proof-Carrying Branch Transactions — state-thread-owned records and the
// pure transition/accounting logic (PCBT-1).
//
// This header is deliberately self-contained (STL only) so the transaction
// state machine is testable in isolation, without llama or server linkage
// (tests/test-pcbt-state.cpp). Contract: docs/treebeard-pcbt-contract-v1.md.
// Ownership rule (invariant 1): only the server state thread mutates these
// records; HTTP threads parse and enqueue tasks.

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

enum class pcbt_status : uint8_t {
    CREATING,
    RUNNING,
    AWAITING_DECISION,
    COMMITTING,
    COMMITTED,
    ABORTING,
    ABORTED,
    EXPIRING,
    EXPIRED,
    FAILED,
};

static inline const char * pcbt_status_name(pcbt_status s) {
    switch (s) {
        case pcbt_status::CREATING:          return "creating";
        case pcbt_status::RUNNING:           return "running";
        case pcbt_status::AWAITING_DECISION: return "awaiting_decision";
        case pcbt_status::COMMITTING:        return "committing";
        case pcbt_status::COMMITTED:         return "committed";
        case pcbt_status::ABORTING:          return "aborting";
        case pcbt_status::ABORTED:           return "aborted";
        case pcbt_status::EXPIRING:          return "expiring";
        case pcbt_status::EXPIRED:           return "expired";
        case pcbt_status::FAILED:            return "failed";
    }
    return "?";
}

static inline bool pcbt_status_terminal(pcbt_status s) {
    return s == pcbt_status::COMMITTED || s == pcbt_status::ABORTED ||
           s == pcbt_status::EXPIRED   || s == pcbt_status::FAILED;
}

// The complete legal transition relation (contract §4). Everything not
// listed is illegal; FAILED is reachable from every pre-commit state.
static inline bool pcbt_transition_legal(pcbt_status from, pcbt_status to) {
    if (pcbt_status_terminal(from)) {
        return false;
    }
    const bool pre_decision = from == pcbt_status::CREATING ||
                              from == pcbt_status::RUNNING ||
                              from == pcbt_status::AWAITING_DECISION;
    switch (to) {
        case pcbt_status::RUNNING:           return from == pcbt_status::CREATING;
        case pcbt_status::AWAITING_DECISION: return from == pcbt_status::RUNNING;
        case pcbt_status::COMMITTING:        return from == pcbt_status::AWAITING_DECISION;
        case pcbt_status::COMMITTED:         return from == pcbt_status::COMMITTING;
        case pcbt_status::ABORTING:          return pre_decision;
        case pcbt_status::ABORTED:           return from == pcbt_status::ABORTING;
        case pcbt_status::EXPIRING:          return pre_decision;
        case pcbt_status::EXPIRED:           return from == pcbt_status::EXPIRING;
        case pcbt_status::FAILED:            return true;  // any pre-commit state (terminal excluded above)
        case pcbt_status::CREATING:          return false;
    }
    return false;
}

enum class pcbt_branch_phase : uint8_t {
    QUEUED,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELED,
};

static inline bool pcbt_branch_terminal(pcbt_branch_phase p) {
    return p == pcbt_branch_phase::COMPLETED || p == pcbt_branch_phase::FAILED ||
           p == pcbt_branch_phase::CANCELED;
}

struct pcbt_branch {
    std::string       key;
    int32_t           node_id       = -1;
    int32_t           slot_id       = -1;
    pcbt_branch_phase phase         = pcbt_branch_phase::QUEUED;
    uint32_t          prompt_tokens = 0;
    uint32_t          predicted_tokens = 0;
    uint64_t          wall_ms       = 0;
    std::string       finish_reason;
    std::string       output_digest;    // sha256:<hex> once terminal-completed
    std::string       candidate_digest; // pcbt.candidate.v1 domain digest
    std::vector<uint8_t> output_bytes;  // bounded by budget accounting below
    bool              body_truncated = false;
};

struct pcbt_event {
    uint64_t    seq     = 0;
    uint64_t    unix_ms = 0;
    std::string kind;
    std::string data;   // canonical JSON payload
};

struct pcbt_budget {
    uint32_t max_slots            = 0;
    uint32_t max_predicted_tokens = 0;
    uint64_t deadline_ms          = 0;   // duration from acceptance
    uint64_t max_candidate_bytes  = 0;   // aggregate across branches
};

struct pcbt_decision {
    int32_t     winner_node_id = -1;
    std::string candidate_digest;
    std::string evidence_kind;
    std::string evidence_digest;
    std::string decision_digest;         // pcbt.decision.v1
};

struct server_branch_transaction {
    uint64_t    id            = 0;
    std::string request_id;
    std::string create_digest;           // pcbt.create.v1
    // source assertions
    int32_t     source_node_id  = -1;
    int32_t     source_state_id = -1;    // -1 = unasserted
    int32_t     source_fork_id  = -1;    // -1 = unasserted
    int32_t     generation      = -1;    // family generation once forked
    pcbt_status status        = pcbt_status::CREATING;
    pcbt_budget budget;
    uint64_t    accepted_unix_ms = 0;
    uint64_t    deadline_unix_ms = 0;
    std::string terminal_reason;
    pcbt_decision decision;
    std::string receipt_digest;          // pcbt.receipt.v1 once terminal
    std::string receipt_json;            // canonical terminal receipt

    std::vector<pcbt_branch> branches;

    // bounded event ring (invariant 8); first_seq exposes truncation
    std::deque<pcbt_event> events;
    uint64_t next_seq       = 1;
    uint64_t first_seq      = 1;
    size_t   max_events     = 256;

    // exact aggregate candidate-byte accounting (invariant 8)
    uint64_t candidate_bytes_stored = 0;

    // --- state-thread-only mutations -------------------------------------

    bool transition(pcbt_status to) {
        if (!pcbt_transition_legal(status, to)) {
            return false;
        }
        status = to;
        return true;
    }

    void push_event(uint64_t unix_ms, std::string kind, std::string data) {
        events.push_back({ next_seq++, unix_ms, std::move(kind), std::move(data) });
        while (events.size() > max_events) {
            events.pop_front();
            ++first_seq;
        }
    }

    pcbt_branch * find_branch(const std::string & key) {
        for (auto & b : branches) {
            if (b.key == key) {
                return &b;
            }
        }
        return nullptr;
    }

    // Store candidate bytes under the aggregate cap; overflow keeps digest
    // and length only (contract §5). Returns false when truncated.
    bool store_candidate_bytes(pcbt_branch & b, const uint8_t * data, size_t len) {
        const uint64_t room = budget.max_candidate_bytes > candidate_bytes_stored
                            ? budget.max_candidate_bytes - candidate_bytes_stored : 0;
        const size_t keep = (size_t) std::min<uint64_t>(room, len);
        b.output_bytes.assign(data, data + keep);
        b.body_truncated = keep < len;
        candidate_bytes_stored += keep;
        return !b.body_truncated;
    }

    // Every branch is terminal and at least one candidate completed.
    bool ready_for_decision() const {
        bool any_completed = false;
        for (const auto & b : branches) {
            if (!pcbt_branch_terminal(b.phase)) {
                return false;
            }
            any_completed |= b.phase == pcbt_branch_phase::COMPLETED;
        }
        return any_completed;
    }

    bool all_branches_failed_or_canceled() const {
        for (const auto & b : branches) {
            if (!pcbt_branch_terminal(b.phase) || b.phase == pcbt_branch_phase::COMPLETED) {
                return false;
            }
        }
        return !branches.empty();
    }

    // Idempotent terminal cleanup of member bookkeeping (invariant 9): safe
    // to call repeatedly, after partial branch completion, and after slot
    // release. Slot/KV release itself is the StateTree layer's job.
    void cleanup_members() {
        for (auto & b : branches) {
            if (!pcbt_branch_terminal(b.phase)) {
                b.phase = pcbt_branch_phase::CANCELED;
            }
            b.slot_id = -1;
            b.output_bytes.clear();
            b.output_bytes.shrink_to_fit();
        }
        candidate_bytes_stored = 0;
    }
};

// Process-local registry: monotonic ids, request_id idempotency index, and
// non-mutating lookups. All access is state-thread-owned.
struct pcbt_registry {
    uint64_t next_id = 1;
    std::map<uint64_t, server_branch_transaction> by_id;
    std::map<std::string, uint64_t> by_request_id;
    size_t max_transactions = 64;   // bounded registry (invariant 8)

    // nullptr on capacity; existing transaction on exact idempotent retry;
    // conflict reported via `conflict` when request_id reuses a different body.
    server_branch_transaction * create(const std::string & request_id,
                                       const std::string & create_digest,
                                       bool & conflict) {
        conflict = false;
        auto it = by_request_id.find(request_id);
        if (it != by_request_id.end()) {
            auto & existing = by_id.at(it->second);
            if (existing.create_digest != create_digest) {
                conflict = true;
                return nullptr;
            }
            return &existing;
        }
        if (by_id.size() >= max_transactions) {
            return nullptr;
        }
        const uint64_t id = next_id++;
        auto & tx = by_id[id];
        tx.id = id;
        tx.request_id = request_id;
        tx.create_digest = create_digest;
        by_request_id.emplace(request_id, id);
        return &tx;
    }

    server_branch_transaction * find(uint64_t id) {
        auto it = by_id.find(id);
        return it == by_id.end() ? nullptr : &it->second;
    }
};
