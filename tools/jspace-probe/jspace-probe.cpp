#include "arg.h"
#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct probe_args {
    struct named_vector {
        std::string name;
        std::string path;
    };

    std::vector<llama_token>  token_ids;
    std::vector<named_vector> probe_vectors;
    std::vector<float>        probe_strengths;
    std::vector<char *>       common_argv;
    bool                      self_test = false;
};

static std::string trim(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }

    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::vector<llama_token> parse_token_id_list(const std::string & value) {
    std::vector<llama_token> result;
    size_t begin = 0;

    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = trim(value.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.empty()) {
            throw std::invalid_argument("--token-ids contains an empty item");
        }

        errno = 0;
        char * parse_end = nullptr;
        const long long parsed = std::strtoll(item.c_str(), &parse_end, 10);
        if (errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' ||
                parsed < 0 || parsed > std::numeric_limits<llama_token>::max()) {
            throw std::invalid_argument("invalid token ID in --token-ids: '" + item + "'");
        }
        result.push_back(static_cast<llama_token>(parsed));

        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }

    return result;
}

static std::vector<float> parse_strength_list(const std::string & value) {
    std::vector<float> result;
    size_t begin = 0;

    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = trim(value.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.empty()) {
            throw std::invalid_argument("--probe-strengths contains an empty item");
        }

        errno = 0;
        char * parse_end = nullptr;
        const float parsed = std::strtof(item.c_str(), &parse_end);
        if (errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' || !std::isfinite(parsed)) {
            throw std::invalid_argument("invalid strength in --probe-strengths: '" + item + "'");
        }
        result.push_back(parsed);

        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }

    return result;
}

static probe_args::named_vector parse_named_vector(const std::string & value) {
    const size_t separator = value.find('=');
    if (separator == std::string::npos) {
        throw std::invalid_argument("--probe-vector requires NAME=PATH");
    }

    probe_args::named_vector result {
        trim(value.substr(0, separator)),
        trim(value.substr(separator + 1)),
    };
    if (result.name.empty() || result.path.empty()) {
        throw std::invalid_argument("--probe-vector requires non-empty NAME and PATH values");
    }
    return result;
}

static probe_args preprocess_args(int argc, char ** argv) {
    probe_args result;
    result.common_argv.reserve(argc);
    result.common_argv.push_back(argv[0]);

    constexpr const char * token_prefix = "--token-ids=";
    constexpr const char * vector_prefix = "--probe-vector=";
    constexpr const char * strength_prefix = "--probe-strengths=";

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--self-test") {
            result.self_test = true;
            continue;
        }

        if (arg == "--token-ids") {
            if (++i >= argc) {
                throw std::invalid_argument("--token-ids requires a comma-separated value");
            }
            auto parsed = parse_token_id_list(argv[i]);
            result.token_ids.insert(result.token_ids.end(), parsed.begin(), parsed.end());
            continue;
        }
        if (arg.compare(0, std::strlen(token_prefix), token_prefix) == 0) {
            auto parsed = parse_token_id_list(arg.substr(std::strlen(token_prefix)));
            result.token_ids.insert(result.token_ids.end(), parsed.begin(), parsed.end());
            continue;
        }

        if (arg == "--probe-vector") {
            if (++i >= argc) {
                throw std::invalid_argument("--probe-vector requires NAME=PATH");
            }
            result.probe_vectors.push_back(parse_named_vector(argv[i]));
            continue;
        }
        if (arg.compare(0, std::strlen(vector_prefix), vector_prefix) == 0) {
            result.probe_vectors.push_back(parse_named_vector(arg.substr(std::strlen(vector_prefix))));
            continue;
        }

        if (arg == "--probe-strengths") {
            if (++i >= argc) {
                throw std::invalid_argument("--probe-strengths requires a comma-separated value");
            }
            auto parsed = parse_strength_list(argv[i]);
            result.probe_strengths.insert(result.probe_strengths.end(), parsed.begin(), parsed.end());
            continue;
        }
        if (arg.compare(0, std::strlen(strength_prefix), strength_prefix) == 0) {
            auto parsed = parse_strength_list(arg.substr(std::strlen(strength_prefix)));
            result.probe_strengths.insert(result.probe_strengths.end(), parsed.begin(), parsed.end());
            continue;
        }

        result.common_argv.push_back(argv[i]);
    }

    return result;
}

static double logits_logsumexp(const float * logits, size_t count) {
    double max_logit = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity()) {
            throw std::runtime_error("model produced a non-finite positive or NaN logit");
        }
        if (std::isfinite(logits[i])) {
            max_logit = std::max(max_logit, static_cast<double>(logits[i]));
        }
    }

    if (!std::isfinite(max_logit)) {
        throw std::runtime_error("model did not produce any finite logits");
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(logits[i])) {
            sum += std::exp(static_cast<double>(logits[i]) - max_logit);
        }
    }

    return max_logit + std::log(sum);
}

static void print_usage(int, char ** argv) {
    std::printf("\nJ-Space causal logit probe:\n");
    std::printf("\n  %s -m model.gguf -p PROMPT --token-ids ID,ID,... [common options]\n", argv[0]);
    std::printf("\nProbe options:\n");
    std::printf("  --token-ids ID,ID,...  vocabulary IDs to report; may be repeated\n");
    std::printf("  --probe-vector NAME=PATH\n");
    std::printf("                           named control vector to sweep; may be repeated\n");
    std::printf("  --probe-strengths S,S,...\n");
    std::printf("                           shared finite strengths for every probe vector\n");
    std::printf("  --self-test            run the model-independent deterministic smoke test\n\n");
}

static int run_self_test() {
    const auto ids = parse_token_id_list("1, 2,248319");
    if (ids != std::vector<llama_token>({ 1, 2, 248319 })) {
        throw std::runtime_error("token ID parser self-test failed");
    }

    const float logits[] = { 0.0f, static_cast<float>(std::log(2.0)) };
    const double actual = logits_logsumexp(logits, 2);
    const double expected = std::log(3.0);
    if (std::abs(actual - expected) > 1e-6) {
        throw std::runtime_error("log-sum-exp self-test failed");
    }

    const auto strengths = parse_strength_list("-2,-1,0,1,2");
    if (strengths != std::vector<float>({ -2.0f, -1.0f, 0.0f, 1.0f, 2.0f })) {
        throw std::runtime_error("strength parser self-test failed");
    }
    const auto vector = parse_named_vector("joy=/tmp/joy.gguf");
    if (vector.name != "joy" || vector.path != "/tmp/joy.gguf") {
        throw std::runtime_error("named vector parser self-test failed");
    }

    json output = {
        { "schema", "treebeard.jspace.logit_probe.self_test.v1" },
        { "ok", true },
    };
    std::cout << output.dump(2) << '\n';
    return 0;
}

static std::string model_meta(const llama_model * model, const char * key) {
    std::vector<char> value(256);
    int32_t needed = llama_model_meta_val_str(model, key, value.data(), value.size());
    if (needed < 0) {
        return {};
    }
    if (static_cast<size_t>(needed) >= value.size()) {
        value.resize(static_cast<size_t>(needed) + 1);
        needed = llama_model_meta_val_str(model, key, value.data(), value.size());
        if (needed < 0) {
            return {};
        }
    }
    return std::string(value.data(), static_cast<size_t>(needed));
}

struct loaded_probe_vector {
    probe_args::named_vector      spec;
    common_control_vector_data    data;
};

static void apply_control_vector(
        llama_context *                         ctx,
        const common_control_vector_data *      cvec,
        int32_t                                 layer_start,
        int32_t                                 layer_end,
        size_t                                  full_size) {
    if (cvec == nullptr) {
        if (llama_set_adapter_cvec(ctx, nullptr, 0, 0, 0, 0) != 0) {
            throw std::runtime_error("failed to clear control vector");
        }
        return;
    }

    // Fill every model layer so switching from a wider vector cannot leave stale
    // device tensors behind in llama_adapter_cvec.
    std::vector<float> full_data(full_size, 0.0f);
    std::copy_n(cvec->data.begin(), std::min(cvec->data.size(), full_data.size()), full_data.begin());
    if (llama_set_adapter_cvec(
                ctx,
                full_data.data(),
                full_data.size(),
                cvec->n_embd,
                layer_start,
                layer_end) != 0) {
        throw std::runtime_error("control vector is incompatible with the model or layer range");
    }
}

static common_control_vector_data compose_control_vectors(
        const common_control_vector_data * base,
        const common_control_vector_data & probe,
        float                              strength,
        size_t                             full_size) {
    common_control_vector_data result {
        probe.n_embd,
        std::vector<float>(full_size, 0.0f),
    };

    if (base != nullptr) {
        if (base->n_embd != probe.n_embd) {
            throw std::runtime_error("probe vector embedding size does not match base control vector");
        }
        std::copy_n(base->data.begin(), std::min(base->data.size(), result.data.size()), result.data.begin());
    }
    for (size_t i = 0; i < std::min(probe.data.size(), result.data.size()); ++i) {
        result.data[i] += probe.data[i] * strength;
    }

    return result;
}

static json evaluate_prompt(
        llama_context *                    ctx,
        const llama_vocab *                vocab,
        const std::vector<llama_token> &   prompt_tokens,
        const std::vector<llama_token> &   requested_ids) {
    // Qwen3.6 is hybrid recurrent/attention. Clearing data, not only metadata,
    // resets both its KV cache and recurrent/GDN state between paired doses.
    llama_synchronize(ctx);
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        throw std::runtime_error("decoder context has no model memory to reset");
    }
    llama_memory_clear(memory, true);
    llama_perf_context_reset(ctx);

    const size_t n_batch = llama_n_batch(ctx);
    for (size_t offset = 0; offset < prompt_tokens.size(); offset += n_batch) {
        const size_t count = std::min(n_batch, prompt_tokens.size() - offset);
        llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(prompt_tokens.data() + offset),
                static_cast<int32_t>(count));
        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            throw std::runtime_error("prompt evaluation failed at token offset " +
                    std::to_string(offset) + " with llama_decode code " + std::to_string(rc));
        }
    }

    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("last-position logits are unavailable");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const double log_z = logits_logsumexp(logits, static_cast<size_t>(n_vocab));

    json result;
    result["logsumexp"] = log_z;
    result["tokens"] = json::array();
    for (llama_token id : requested_ids) {
        const double logit = logits[id];
        json item = {
            { "id", id },
            { "piece", common_token_to_piece(vocab, id, true) },
        };
        if (std::isfinite(logit)) {
            item["logit"] = logit;
            item["logprob"] = logit - log_z;
        } else {
            item["logit"] = nullptr;
            item["logprob"] = nullptr;
        }
        result["tokens"].push_back(std::move(item));
    }
    return result;
}

static int run_probe(common_params & params, const probe_args & probe) {
    const auto & requested_ids = probe.token_ids;

    // Apply control vectors ourselves after common initialization. This follows the
    // common loader path but lets the probe fail closed if loading or application fails.
    const auto control_vectors = params.control_vectors;
    params.control_vectors.clear();
    auto llama_init = common_init_from_params(params);
    params.control_vectors = control_vectors;

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        throw std::runtime_error("failed to initialize model and context");
    }

    if (llama_model_has_encoder(model)) {
        throw std::runtime_error("J-Space causal probing currently requires a decoder-only model");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id : requested_ids) {
        if (id < 0 || id >= n_vocab) {
            throw std::invalid_argument("requested token ID " + std::to_string(id) +
                    " is outside model vocabulary [0, " + std::to_string(n_vocab) + ")");
        }
    }

    const bool add_special = llama_vocab_get_add_bos(vocab);
    const auto prompt_tokens = common_tokenize(ctx, params.prompt, add_special, params.parse_special);
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("prompt tokenized to zero tokens; provide a non-empty prompt with -p or -f");
    }
    if (prompt_tokens.size() > llama_n_ctx(ctx)) {
        throw std::invalid_argument("prompt has " + std::to_string(prompt_tokens.size()) +
                " tokens but context is only " + std::to_string(llama_n_ctx(ctx)) +
                "; increase it with -c");
    }

    const size_t n_batch = llama_n_batch(ctx);
    if (n_batch == 0) {
        throw std::runtime_error("context reports a zero logical batch size");
    }

    const bool has_any_control = !control_vectors.empty() || !probe.probe_vectors.empty();
    int32_t layer_start = params.control_vector_layer_start;
    int32_t layer_end = params.control_vector_layer_end;
    if (has_any_control) {
        if (layer_start <= 0) {
            layer_start = 1;
        }
        if (layer_end <= 0) {
            layer_end = llama_model_n_layer(model);
        }
        if (layer_start < 1 || layer_end < layer_start || layer_end > llama_model_n_layer(model)) {
            throw std::invalid_argument("invalid control-vector layer range [" +
                    std::to_string(layer_start) + ", " + std::to_string(layer_end) + "]");
        }
    }

    common_control_vector_data base_cvec;
    const common_control_vector_data * base_cvec_ptr = nullptr;
    if (!control_vectors.empty()) {
        base_cvec = common_control_vector_load(control_vectors);
        if (base_cvec.n_embd < 0) {
            throw std::runtime_error("failed to load base control vector");
        }
        base_cvec_ptr = &base_cvec;
    }

    std::vector<loaded_probe_vector> loaded_vectors;
    loaded_vectors.reserve(probe.probe_vectors.size());
    for (const auto & spec : probe.probe_vectors) {
        auto data = common_control_vector_load({ { 1.0f, spec.path } });
        if (data.n_embd < 0) {
            throw std::runtime_error("failed to load probe vector '" + spec.name + "' from " + spec.path);
        }
        if (data.n_embd != llama_model_n_embd(model)) {
            throw std::runtime_error("probe vector '" + spec.name + "' embedding size does not match model");
        }
        if (base_cvec_ptr != nullptr && data.n_embd != base_cvec_ptr->n_embd) {
            throw std::runtime_error("probe vector '" + spec.name + "' embedding size does not match base vector");
        }
        loaded_vectors.push_back({ spec, std::move(data) });
    }

    const size_t cvec_full_size = static_cast<size_t>(llama_model_n_embd(model)) *
            static_cast<size_t>(std::max(0, llama_model_n_layer(model) - 1));
    if (base_cvec_ptr != nullptr && base_cvec_ptr->data.size() > cvec_full_size) {
        throw std::runtime_error("base control vector contains a layer beyond the model's controllable range");
    }
    for (const auto & loaded : loaded_vectors) {
        if (loaded.data.data.size() > cvec_full_size) {
            throw std::runtime_error("probe vector '" + loaded.spec.name +
                    "' contains a layer beyond the model's controllable range");
        }
    }
    apply_control_vector(ctx, base_cvec_ptr, layer_start, layer_end, cvec_full_size);
    const json baseline = evaluate_prompt(ctx, vocab, prompt_tokens, requested_ids);

    char description[512] = {};
    llama_model_desc(model, description, sizeof(description));

    json output;
    output["schema"] = "treebeard.jspace.logit_probe.v1";
    output["model"] = {
        { "path", params.model.path },
        { "architecture", model_meta(model, "general.architecture") },
        { "name", model_meta(model, "general.name") },
        { "description", description },
        { "parameter_count", llama_model_n_params(model) },
        { "vocabulary_size", n_vocab },
        { "layer_count", llama_model_n_layer(model) },
        { "embedding_size", llama_model_n_embd(model) },
    };
    output["probe"] = {
        { "prompt_token_count", prompt_tokens.size() },
        { "position", prompt_tokens.size() - 1 },
        { "position_basis", "zero_based_prompt_token" },
        { "last_prompt_token_id", prompt_tokens.back() },
        { "last_prompt_token_piece", common_token_to_piece(vocab, prompt_tokens.back(), true) },
        { "distribution", "raw_full_vocabulary_softmax" },
        { "temperature", 1.0 },
        { "state_reset", "llama_memory_clear(data=true) before every evaluation" },
    };

    json vectors = json::array();
    for (const auto & info : control_vectors) {
        vectors.push_back({
            { "path", info.fname },
            { "scale", info.strength },
        });
    }
    output["control"] = {
        { "base_vectors", std::move(vectors) },
        { "layer_start", has_any_control ? json(layer_start) : json(nullptr) },
        { "layer_end", has_any_control ? json(layer_end) : json(nullptr) },
    };
    output["baseline"] = baseline;
    output["sweeps"] = json::array();

    for (const auto & loaded : loaded_vectors) {
        json sweep = {
            { "name", loaded.spec.name },
            { "path", loaded.spec.path },
            { "runs", json::array() },
        };

        for (float strength : probe.probe_strengths) {
            json result;
            bool reused_baseline = strength == 0.0f;
            if (reused_baseline) {
                result = baseline;
            } else {
                const auto composed = compose_control_vectors(
                        base_cvec_ptr, loaded.data, strength, cvec_full_size);
                apply_control_vector(ctx, &composed, layer_start, layer_end, cvec_full_size);
                result = evaluate_prompt(ctx, vocab, prompt_tokens, requested_ids);
            }

            json run = {
                { "strength", strength },
                { "reused_baseline", reused_baseline },
                { "logsumexp", result.at("logsumexp") },
                { "tokens", result.at("tokens") },
            };
            sweep["runs"].push_back(std::move(run));
        }
        output["sweeps"].push_back(std::move(sweep));
    }

    std::cout << output.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    bool backend_initialized = false;

    try {
        std::setlocale(LC_NUMERIC, "C");

        probe_args probe = preprocess_args(argc, argv);
        if (probe.self_test) {
            return run_self_test();
        }

        common_params params;
        params.n_ctx = 4096;
        params.n_predict = 0;
        params.warmup = false;

        common_init();
        if (!common_params_parse(
                    static_cast<int>(probe.common_argv.size()),
                    probe.common_argv.data(),
                    params,
                    LLAMA_EXAMPLE_COMMON,
                    print_usage)) {
            return 1;
        }
        if (probe.token_ids.empty()) {
            print_usage(argc, argv);
            throw std::invalid_argument("at least one token ID is required via --token-ids");
        }
        if (probe.probe_vectors.empty() != probe.probe_strengths.empty()) {
            throw std::invalid_argument("--probe-vector and --probe-strengths must be used together");
        }
        for (size_t i = 0; i < probe.probe_vectors.size(); ++i) {
            for (size_t j = i + 1; j < probe.probe_vectors.size(); ++j) {
                if (probe.probe_vectors[i].name == probe.probe_vectors[j].name) {
                    throw std::invalid_argument("duplicate --probe-vector name: '" +
                            probe.probe_vectors[i].name + "'");
                }
            }
        }
        if (params.embedding) {
            throw std::invalid_argument("embedding mode is incompatible with a causal logit probe");
        }

        // This tool does no generation and should not spend a second evaluation on warmup.
        params.n_predict = 0;
        params.warmup = false;

        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);

        const int result = run_probe(params, probe);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        if (backend_initialized) {
            llama_backend_free();
        }
        std::cerr << "llama-jspace-probe: error: " << error.what() << '\n';
        return 1;
    }
}
