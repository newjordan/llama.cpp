// PCBT-2 schema slice exit gate: the C++ parser accepts/rejects exactly the
// committed contract fixtures and reproduces the Python reference's golden
// create digest (cross-language canonicalization lock).
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-pcbt-parse.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static nlohmann::json load(const fs::path & p) {
    std::ifstream f(p);
    assert(f.good());
    return nlohmann::json::parse(f);
}

int main(int argc, char ** argv) {
    assert(argc > 1 && "usage: test-pcbt-parse <fixtures-dir>");
    const fs::path dir = argv[1];
    assert(fs::is_directory(dir));

    std::string golden;
    {
        std::ifstream g(dir / "create-valid-minimal.golden-digest");
        assert(g.good());
        std::getline(g, golden);
    }

    int checked = 0;
    std::string create_min_digest;
    for (const auto & ent : fs::directory_iterator(dir)) {
        const std::string name = ent.path().filename().string();
        if (ent.path().extension() != ".json") {
            continue;
        }
        const bool expect_valid = name.find("-valid") != std::string::npos;
        const auto doc = load(ent.path());

        pcbt_parse_result res;
        if (name.rfind("create", 0) == 0) {
            pcbt_create_request out;
            res = pcbt_parse_create(doc, out);
            if (name == "create-valid-minimal.json" && res.ok()) {
                create_min_digest = out.create_digest;
            }
        } else if (name.rfind("commit", 0) == 0) {
            pcbt_commit_request out;
            res = pcbt_parse_commit(doc, out);
        } else if (name.rfind("abort", 0) == 0) {
            pcbt_abort_request out;
            res = pcbt_parse_abort(doc, out);
        } else {
            continue;
        }

        if (res.ok() != expect_valid) {
            std::cerr << name << ": expected " << (expect_valid ? "VALID" : "INVALID")
                      << " got " << (res.ok() ? "valid" : ("invalid: " + res.message)) << "\n";
            assert(false);
        }
        if (!res.ok()) {
            assert(pcbt_error_http(res.error) == 400);
        }
        ++checked;
    }
    assert(checked >= 11);
    assert(!create_min_digest.empty());
    if (create_min_digest != golden) {
        std::cerr << "CANONICALIZATION DRIFT vs Python reference:\n  golden " << golden
                  << "\n  c++    " << create_min_digest << "\n";
        assert(false);
    }
    std::cout << "PCBT parse tests passed (" << checked
              << " fixtures; golden digest reproduced)\n";
    return 0;
}
