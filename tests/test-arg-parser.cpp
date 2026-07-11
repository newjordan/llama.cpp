#include "arg.h"
#include "common.h"
#include "download.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    // negated arg
    argv = {"binary_name", "--no-mmap"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // negative StateTree lease
    argv = {"binary_name", "--statetree-lease-ms", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // partially parsed StateTree lease
    argv = {"binary_name", "--statetree-lease-ms", "250ms"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // negative StateTree state byte budget
    argv = {"binary_name", "--statetree-max-state-bytes", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // partially parsed StateTree state byte budget
    argv = {"binary_name", "--statetree-max-state-bytes", "4294967296bytes"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // negative or partially parsed StateTree snapshot budget
    argv = {"binary_name", "--statetree-max-snapshot-bytes", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-bytes", "536870912bytes"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--statetree-max-snapshot-disk-bytes", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-disk-bytes", "1073741824bytes"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-load-bytes", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-load-bytes", "536870912bytes"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-manifest-bytes", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-max-snapshot-manifest-bytes", "67108864bytes"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    argv = {"binary_name", "--statetree-snapshot-compat-id", std::string(257, 'x')};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    argv = {
        "binary_name",
        "--statetree-lease-ms", "250",
        "--statetree-max-state-bytes", "4294967296",
        "--statetree-max-snapshot-bytes", "536870912",
        "--statetree-snapshot-store", ".",
        "--statetree-snapshot-compat-id", "model-runtime-test",
        "--statetree-max-snapshot-disk-bytes", "1073741824",
        "--statetree-max-snapshot-load-bytes", "536870912",
        "--statetree-max-snapshot-manifest-bytes", "67108864",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.statetree_lease_ms == 250);
    assert(params.statetree_max_state_bytes == 4294967296ULL);
    assert(params.statetree_max_snapshot_bytes == 536870912ULL);
    assert(params.statetree_snapshot_store == ".");
    assert(params.statetree_snapshot_compat_id == "model-runtime-test");
    assert(params.statetree_max_snapshot_disk_bytes == 1073741824ULL);
    assert(params.statetree_max_snapshot_load_bytes == 536870912ULL);
    assert(params.statetree_max_snapshot_manifest_bytes == 67108864ULL);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_STATETREE_LEASE_MS", "250ms", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    unsetenv("LLAMA_ARG_STATETREE_LEASE_MS");

    setenv("LLAMA_ARG_STATETREE_LEASE_MS", "300", true);
    setenv("LLAMA_ARG_STATETREE_MAX_STATE_BYTES", "4294967296", true);
    setenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_BYTES", "536870912", true);
    setenv("LLAMA_ARG_STATETREE_SNAPSHOT_STORE", ".", true);
    setenv("LLAMA_ARG_STATETREE_SNAPSHOT_COMPAT_ID", "model-runtime-env", true);
    setenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_DISK_BYTES", "1073741824", true);
    setenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_LOAD_BYTES", "536870912", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.statetree_lease_ms == 300);
    assert(params.statetree_max_state_bytes == 4294967296ULL);
    assert(params.statetree_max_snapshot_bytes == 536870912ULL);
    assert(params.statetree_snapshot_store == ".");
    assert(params.statetree_snapshot_compat_id == "model-runtime-env");
    assert(params.statetree_max_snapshot_disk_bytes == 1073741824ULL);
    assert(params.statetree_max_snapshot_load_bytes == 536870912ULL);
    unsetenv("LLAMA_ARG_STATETREE_LEASE_MS");
    unsetenv("LLAMA_ARG_STATETREE_MAX_STATE_BYTES");
    unsetenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_BYTES");
    unsetenv("LLAMA_ARG_STATETREE_SNAPSHOT_STORE");
    unsetenv("LLAMA_ARG_STATETREE_SNAPSHOT_COMPAT_ID");
    unsetenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_DISK_BYTES");
    unsetenv("LLAMA_ARG_STATETREE_MAX_SNAPSHOT_LOAD_BYTES");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_MMAP", "0", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.use_mmap == false);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}
