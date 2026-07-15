#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct extractor_args {
    std::string manifest_path;
    std::string out_prefix;
    std::string verified_model_sha256;
    std::string verified_manifest_sha256;
    std::string pooling = "last";
    std::vector<int32_t> layers;
    std::vector<char *> common_argv;
    bool self_test = false;
};

static std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string parse_sha256(const std::string & value, const char * name) {
    const std::string parsed = trim(value);
    if (parsed.size() != 64 || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
        throw std::invalid_argument(std::string(name) + " must be 64 lowercase hexadecimal characters");
    }
    return parsed;
}

static std::vector<int32_t> parse_layers(const std::string & value) {
    std::vector<int32_t> result;
    std::set<int32_t> seen;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = trim(value.substr(
                begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.empty()) {
            throw std::invalid_argument("--layers contains an empty item");
        }
        errno = 0;
        char * parse_end = nullptr;
        const long parsed = std::strtol(item.c_str(), &parse_end, 10);
        if (errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' ||
                parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("--layers contains an invalid layer: " + item);
        }
        const auto layer = static_cast<int32_t>(parsed);
        if (!seen.insert(layer).second) {
            throw std::invalid_argument("--layers contains a duplicate layer: " + item);
        }
        result.push_back(layer);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (result.empty()) {
        throw std::invalid_argument("--layers must contain at least one layer");
    }
    return result;
}

static extractor_args preprocess_args(int argc, char ** argv) {
    extractor_args result;
    result.common_argv.push_back(argv[0]);
    constexpr const char * manifest_prefix = "--manifest=";
    constexpr const char * out_prefix = "--out-prefix=";
    constexpr const char * layers_prefix = "--layers=";
    constexpr const char * pooling_prefix = "--pooling=";
    constexpr const char * model_sha_prefix = "--verified-model-sha256=";
    constexpr const char * manifest_sha_prefix = "--verified-manifest-sha256=";

    auto require_value = [&](int & i, const char * name) -> std::string {
        if (++i >= argc) {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        return argv[i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--self-test") {
            result.self_test = true;
        } else if (arg == "--manifest") {
            result.manifest_path = require_value(i, "--manifest");
        } else if (arg.compare(0, std::strlen(manifest_prefix), manifest_prefix) == 0) {
            result.manifest_path = arg.substr(std::strlen(manifest_prefix));
        } else if (arg == "--out-prefix") {
            result.out_prefix = require_value(i, "--out-prefix");
        } else if (arg.compare(0, std::strlen(out_prefix), out_prefix) == 0) {
            result.out_prefix = arg.substr(std::strlen(out_prefix));
        } else if (arg == "--layers") {
            result.layers = parse_layers(require_value(i, "--layers"));
        } else if (arg.compare(0, std::strlen(layers_prefix), layers_prefix) == 0) {
            result.layers = parse_layers(arg.substr(std::strlen(layers_prefix)));
        } else if (arg == "--pooling") {
            result.pooling = require_value(i, "--pooling");
        } else if (arg.compare(0, std::strlen(pooling_prefix), pooling_prefix) == 0) {
            result.pooling = arg.substr(std::strlen(pooling_prefix));
        } else if (arg == "--verified-model-sha256") {
            result.verified_model_sha256 = parse_sha256(
                    require_value(i, "--verified-model-sha256"), "--verified-model-sha256");
        } else if (arg.compare(0, std::strlen(model_sha_prefix), model_sha_prefix) == 0) {
            result.verified_model_sha256 = parse_sha256(
                    arg.substr(std::strlen(model_sha_prefix)), "--verified-model-sha256");
        } else if (arg == "--verified-manifest-sha256") {
            result.verified_manifest_sha256 = parse_sha256(
                    require_value(i, "--verified-manifest-sha256"), "--verified-manifest-sha256");
        } else if (arg.compare(0, std::strlen(manifest_sha_prefix), manifest_sha_prefix) == 0) {
            result.verified_manifest_sha256 = parse_sha256(
                    arg.substr(std::strlen(manifest_sha_prefix)), "--verified-manifest-sha256");
        } else {
            result.common_argv.push_back(argv[i]);
        }
    }
    if (result.pooling != "last" && result.pooling != "last-mean") {
        throw std::invalid_argument("--pooling must be last or last-mean");
    }
    return result;
}

static void print_usage(int, char ** argv) {
    std::printf("Usage: %s [common model options] --manifest FILE --out-prefix PATH --layers LIST\\n", argv[0]);
    std::printf("\\nJ-Space G1 options:\\n");
    std::printf("  --manifest FILE                   frozen G1 dataset/control JSON\\n");
    std::printf("  --out-prefix PATH                 write PATH.json and PATH.f32 atomically\\n");
    std::printf("  --layers 2,3,...                  ordered zero-based l_out layers to retain\\n");
    std::printf("  --pooling last|last-mean          token summaries per retained layer\\n");
    std::printf("  --verified-model-sha256 SHA       runner-verified exact model identity\\n");
    std::printf("  --verified-manifest-sha256 SHA    runner-verified frozen dataset identity\\n");
    std::printf("  --self-test                       model-independent parser smoke test\\n\\n");
}

struct activation_capture {
    std::unordered_map<std::string, size_t> tensor_indices;
    std::vector<std::vector<float>> values;
    std::vector<bool> captured;
    std::string error;
    int64_t dimension = 0;
    bool include_mean = false;

    void reset(size_t n_layers) {
        values.assign(n_layers, {});
        captured.assign(n_layers, false);
        error.clear();
        dimension = 0;
    }
};

static bool capture_activations(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<activation_capture *>(user_data);
    const auto it = capture->tensor_indices.find(tensor->name);
    if (ask) {
        return it != capture->tensor_indices.end();
    }
    if (it == capture->tensor_indices.end()) {
        return true;
    }
    if (capture->captured[it->second]) {
        capture->error = "activation tensor was evaluated more than once: " + std::string(tensor->name);
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != sizeof(float) ||
            tensor->nb[1] < static_cast<size_t>(tensor->ne[0]) * sizeof(float)) {
        capture->error = "activation tensor has unsupported type or layout: " + std::string(tensor->name);
        return true;
    }
    if (capture->dimension != 0 && capture->dimension != tensor->ne[0]) {
        capture->error = "activation layers have inconsistent dimensions";
        return true;
    }
    capture->dimension = tensor->ne[0];
    const size_t dimension = static_cast<size_t>(tensor->ne[0]);
    const size_t nbytes = dimension * sizeof(float);
    const size_t offset = static_cast<size_t>(tensor->ne[1] - 1) * tensor->nb[1];
    if (offset + nbytes > ggml_nbytes(tensor)) {
        capture->error = "final activation column is out of bounds: " + std::string(tensor->name);
        return true;
    }
    auto & values = capture->values[it->second];
    values.resize(dimension * (capture->include_mean ? 2 : 1));
    ggml_backend_tensor_get(tensor, values.data(), offset, nbytes);
    if (capture->include_mean) {
        std::vector<float> column(dimension);
        std::vector<double> sums(dimension, 0.0);
        for (int64_t token = 0; token < tensor->ne[1]; ++token) {
            ggml_backend_tensor_get(
                tensor, column.data(), static_cast<size_t>(token) * tensor->nb[1], nbytes);
            for (size_t feature = 0; feature < dimension; ++feature) {
                sums[feature] += column[feature];
            }
        }
        const double inverse_tokens = 1.0 / static_cast<double>(tensor->ne[1]);
        for (size_t feature = 0; feature < dimension; ++feature) {
            values[dimension + feature] = static_cast<float>(sums[feature] * inverse_tokens);
        }
    }
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
        capture->error = "activation tensor contains a non-finite value: " + std::string(tensor->name);
        return true;
    }
    capture->captured[it->second] = true;
    return true;
}

static json read_manifest(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + path);
    }
    json document = json::parse(input);
    const std::string schema = document.value("schema", "");
    if ((schema != "treebeard.jspace.g1.dataset.v1" &&
            schema != "treebeard.jspace.g1.dataset.v2" &&
            schema != "treebeard.jspace.g1.dataset.v3" &&
            schema != "treebeard.jspace.g1.controls.v1" &&
            schema != "treebeard.jspace.g1.controls.v2") ||
            !document.contains("rows") || !document["rows"].is_array() || document["rows"].empty()) {
        throw std::runtime_error("unsupported or empty J-Space G1 manifest");
    }
    std::set<std::string> sample_ids;
    for (const auto & row : document["rows"]) {
        for (const char * field : { "sample_id", "split", "label", "text_sha256", "text" }) {
            if (!row.contains(field) || !row[field].is_string() || row[field].get<std::string>().empty()) {
                throw std::runtime_error(std::string("manifest row has invalid field: ") + field);
            }
        }
        if ((schema == "treebeard.jspace.g1.dataset.v1" ||
                schema == "treebeard.jspace.g1.dataset.v2") && row.value("anchor_echo", true)) {
            throw std::runtime_error("manifest row is not certified anchor-free");
        }
        if (!sample_ids.insert(row["sample_id"].get<std::string>()).second) {
            throw std::runtime_error("manifest contains a duplicate sample_id");
        }
    }
    return document;
}

static int run_self_test() {
    const auto layers = parse_layers("2,3,11,39");
    if (layers != std::vector<int32_t>({ 2, 3, 11, 39 })) {
        throw std::runtime_error("layer parser self-test failed");
    }
    bool rejected_duplicate = false;
    try {
        (void) parse_layers("2,2");
    } catch (const std::invalid_argument &) {
        rejected_duplicate = true;
    }
    if (!rejected_duplicate || parse_sha256(std::string(64, 'a'), "digest").size() != 64) {
        throw std::runtime_error("argument validation self-test failed");
    }
    std::cout << json({
        { "schema", "treebeard.jspace.g1.extractor-self-test.v1" },
        { "ok", true },
    }).dump() << '\n';
    return 0;
}

static int run_extractor(common_params & params, const extractor_args & args) {
    json manifest = read_manifest(args.manifest_path);
    activation_capture capture;
    capture.include_mean = args.pooling == "last-mean";
    for (size_t i = 0; i < args.layers.size(); ++i) {
        capture.tensor_indices.emplace("l_out-" + std::to_string(args.layers[i]), i);
    }
    params.cb_eval = capture_activations;
    params.cb_eval_user_data = &capture;
    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr || llama_model_has_encoder(model)) {
        throw std::runtime_error("J-Space sensor extraction requires a decoder-only model and context");
    }
    for (int32_t layer : args.layers) {
        if (layer < 0 || layer >= llama_model_n_layer(model)) {
            throw std::invalid_argument("requested layer is outside the model: " + std::to_string(layer));
        }
    }
    if (llama_model_n_embd(model) <= 0) {
        throw std::runtime_error("model reports an invalid embedding dimension");
    }
    const uint32_t one_decode_limit = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    if (one_decode_limit == 0) {
        throw std::runtime_error("context reports a zero batch or microbatch size");
    }

    const std::filesystem::path raw_path = args.out_prefix + ".f32";
    const std::filesystem::path meta_path = args.out_prefix + ".json";
    const std::filesystem::path raw_tmp = args.out_prefix + ".f32.tmp";
    const std::filesystem::path meta_tmp = args.out_prefix + ".json.tmp";
    if (!raw_path.parent_path().empty()) {
        std::filesystem::create_directories(raw_path.parent_path());
    }
    std::error_code ignored;
    std::filesystem::remove(raw_tmp, ignored);
    std::filesystem::remove(meta_tmp, ignored);
    std::ofstream raw(raw_tmp, std::ios::binary | std::ios::trunc);
    if (!raw) {
        throw std::runtime_error("failed to open temporary activation output: " + raw_tmp.string());
    }

    json output_rows = json::array();
    size_t min_tokens = std::numeric_limits<size_t>::max();
    size_t max_tokens = 0;
    uint64_t total_tokens = 0;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_special = llama_vocab_get_add_bos(vocab);
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        throw std::runtime_error("context has no model memory");
    }

    const auto & rows = manifest["rows"];
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const auto & row = rows[row_index];
        const std::string text = row["text"].get<std::string>();
        const auto tokens = common_tokenize(ctx, text, add_special, false);
        if (tokens.empty() || tokens.size() > llama_n_ctx(ctx) || tokens.size() > one_decode_limit) {
            throw std::runtime_error(
                    "sample " + row["sample_id"].get<std::string>() + " has " +
                    std::to_string(tokens.size()) + " tokens; the one-decode limit is " +
                    std::to_string(one_decode_limit));
        }
        capture.reset(args.layers.size());
        llama_synchronize(ctx);
        llama_memory_clear(memory, true);
        const int32_t rc = llama_decode(
                ctx, llama_batch_get_one(
                    const_cast<llama_token *>(tokens.data()), static_cast<int32_t>(tokens.size())));
        if (rc != 0) {
            throw std::runtime_error(
                    "decode failed for " + row["sample_id"].get<std::string>() +
                    " with code " + std::to_string(rc));
        }
        llama_synchronize(ctx);
        if (!capture.error.empty()) {
            throw std::runtime_error(capture.error);
        }
        if (capture.dimension != llama_model_n_embd(model) ||
                !std::all_of(capture.captured.begin(), capture.captured.end(), [](bool value) { return value; })) {
            throw std::runtime_error("one or more requested activation layers were not captured");
        }
        for (const auto & layer_values : capture.values) {
            raw.write(reinterpret_cast<const char *>(layer_values.data()),
                    static_cast<std::streamsize>(layer_values.size() * sizeof(float)));
        }
        if (!raw) {
            throw std::runtime_error("failed while writing activation output");
        }

        min_tokens = std::min(min_tokens, tokens.size());
        max_tokens = std::max(max_tokens, tokens.size());
        total_tokens += tokens.size();
        output_rows.push_back({
            { "row", row_index },
            { "sample_id", row["sample_id"] },
            { "split", row["split"] },
            { "label", row["label"] },
            { "text_sha256", row["text_sha256"] },
            { "token_count", tokens.size() },
        });
        if ((row_index + 1) % 64 == 0 || row_index + 1 == rows.size()) {
            std::cerr << "llama-jspace-sensor-extract: " << (row_index + 1) << "/" << rows.size()
                      << " samples, " << total_tokens << " tokens\n";
        }
    }
    raw.close();
    if (!raw) {
        throw std::runtime_error("failed to finalize activation output");
    }

    char description[512] = {};
    llama_model_desc(model, description, sizeof(description));
    const uint64_t pooling_width = capture.include_mean ? 2 : 1;
    const uint64_t raw_bytes = static_cast<uint64_t>(rows.size()) * args.layers.size() * pooling_width *
            static_cast<uint64_t>(llama_model_n_embd(model)) * sizeof(float);
    if (std::filesystem::file_size(raw_tmp) != raw_bytes) {
        throw std::runtime_error("activation output size does not match its declared shape");
    }
    json metadata = {
        { "schema", "treebeard.jspace.g1.activations.v1" },
        { "status", capture.include_mean ?
            "exact_runtime_last_mean_token_residuals" : "exact_runtime_last_token_residuals" },
        { "model", {
            { "description", description },
            { "decoder_only", true },
            { "runner_verified_sha256", args.verified_model_sha256 },
            { "n_layer", llama_model_n_layer(model) },
            { "n_embd", llama_model_n_embd(model) },
        } },
        { "dataset", {
            { "manifest", args.manifest_path },
            { "runner_verified_sha256", args.verified_manifest_sha256 },
            { "schema", manifest["schema"] },
        } },
        { "capture", {
            { "tensor_pattern", "l_out-{zero_based_layer}" },
            { "layers", args.layers },
            { "layer_types", "Qwen3.6 default: full attention iff (layer+1)%4==0; otherwise DeltaNet" },
            { "position", capture.include_mean ? "last token, token mean" : "last token" },
            { "pooling", capture.include_mean ?
                json::array({ "last", "mean" }) : json::array({ "last" }) },
            { "chat_template", false },
            { "parse_special", false },
            { "add_bos_from_model", add_special },
            { "memory_reset", "llama_memory_clear(data=true) before every sample" },
            { "one_decode_per_sample", true },
        } },
        { "raw", {
            { "path", raw_path.filename().string() },
            { "dtype", "little_endian_float32" },
            { "shape", capture.include_mean ?
                json::array({ rows.size(), args.layers.size(), 2, llama_model_n_embd(model) }) :
                json::array({ rows.size(), args.layers.size(), llama_model_n_embd(model) }) },
            { "order", capture.include_mean ?
                "C: sample, layer, pooling(last,mean), embedding" :
                "C: sample, layer, embedding" },
            { "bytes", raw_bytes },
        } },
        { "token_counts", {
            { "minimum", min_tokens },
            { "maximum", max_tokens },
            { "total", total_tokens },
        } },
        { "rows", std::move(output_rows) },
    };
    {
        std::ofstream meta(meta_tmp, std::ios::trunc);
        if (!meta) {
            throw std::runtime_error("failed to open temporary metadata output");
        }
        meta << metadata.dump(2) << '\n';
        if (!meta) {
            throw std::runtime_error("failed to finalize metadata output");
        }
    }
    std::filesystem::rename(raw_tmp, raw_path);
    std::filesystem::rename(meta_tmp, meta_path);
    std::cout << json({
        { "status", "pass" },
        { "metadata", meta_path.string() },
        { "activations", raw_path.string() },
        { "rows", rows.size() },
        { "tokens", total_tokens },
        { "bytes", raw_bytes },
    }).dump() << '\n';
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    bool backend_initialized = false;
    try {
        std::setlocale(LC_NUMERIC, "C");
        extractor_args args = preprocess_args(argc, argv);
        if (args.self_test) {
            return run_self_test();
        }
        if (args.manifest_path.empty() || args.out_prefix.empty() || args.layers.empty() ||
                args.verified_model_sha256.empty() || args.verified_manifest_sha256.empty()) {
            print_usage(argc, argv);
            throw std::invalid_argument(
                    "manifest, output prefix, layers, model SHA, and manifest SHA are required");
        }

        common_params params;
        params.n_ctx = 128;
        params.n_batch = 128;
        params.n_ubatch = 128;
        params.n_predict = 0;
        params.warmup = false;
        params.parse_special = false;
        common_init();
        if (!common_params_parse(
                    static_cast<int>(args.common_argv.size()),
                    args.common_argv.data(),
                    params,
                    LLAMA_EXAMPLE_COMMON,
                    print_usage)) {
            return 1;
        }
        if (params.embedding || !params.control_vectors.empty()) {
            throw std::invalid_argument("embedding mode and control vectors are forbidden during G1 extraction");
        }
        params.n_predict = 0;
        params.warmup = false;
        params.parse_special = false;

        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);
        const int result = run_extractor(params, args);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        if (backend_initialized) {
            llama_backend_free();
        }
        std::cerr << "llama-jspace-sensor-extract: error: " << error.what() << '\n';
        return 1;
    }
}
