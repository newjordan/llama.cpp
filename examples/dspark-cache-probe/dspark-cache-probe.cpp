#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct probe_params {
    std::string model;
    std::string prompt = "Leanstral DSpark cache probe.";
    std::string prompt_file;
    std::string out_dir;
    std::vector<int> layers = { 1, 9, 17, 25, 33 };
    int n_gpu_layers = 99;
    int n_ctx = 0;
    int n_batch = 0;
    int n_ubatch = 0;
    int n_threads = 8;
    int n_threads_batch = 8;
    bool write_binary = true;
    bool write_last_hidden = true;
    bool help = false;
};

struct layer_stats {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double mean = 0.0;
    double l2 = 0.0;
    uint64_t finite = 0;
    uint64_t nonfinite = 0;
    std::vector<float> first8;
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "\nusage:\n"
        "  %s -m MODEL.gguf --out-dir DIR [options]\n\n"
        "options:\n"
        "  --prompt TEXT             prompt to tokenize and prefill\n"
        "  --prompt-file PATH        read prompt from file\n"
        "  --layers CSV              layer ids, default 1,9,17,25,33\n"
        "  -ngl N                    GPU layers, default 99\n"
        "  -c N                      context size, default prompt tokens\n"
        "  -b N                      batch size, default prompt tokens\n"
        "  -ub N                     ubatch size, default batch size\n"
        "  -t N                      decode threads, default 8\n"
        "  -tb N                     batch threads, default decode threads\n"
        "  --no-binary               write JSON only\n"
        "  --no-last-hidden          skip final hidden-state extraction\n\n",
        argv0);
}

static std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b";  break;
            case '\f': out << "\\f";  break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << (char) c;
                }
        }
    }
    return out.str();
}

static std::string now_utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm {};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::vector<std::string> argv_vector(int argc, char ** argv) {
    std::vector<std::string> out;
    out.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        out.emplace_back(argv[i]);
    }
    return out;
}

static std::string join_argv(const std::vector<std::string> & args) {
    std::ostringstream out;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) {
            out << ' ';
        }
        const std::string & arg = args[i];
        const bool needs_quotes = arg.find_first_of(" \t\n\"'\\") != std::string::npos;
        if (!needs_quotes) {
            out << arg;
            continue;
        }
        out << '"';
        for (char c : arg) {
            if (c == '"' || c == '\\') {
                out << '\\';
            }
            out << c;
        }
        out << '"';
    }
    return out.str();
}

static std::vector<int> parse_layers(const std::string & raw) {
    std::vector<int> out;
    std::stringstream ss(raw);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) {
            continue;
        }
        size_t dash = part.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::stoi(part));
            continue;
        }
        int first = std::stoi(part.substr(0, dash));
        int last  = std::stoi(part.substr(dash + 1));
        if (last < first) {
            throw std::runtime_error("decreasing layer range: " + part);
        }
        for (int v = first; v <= last; ++v) {
            out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::string read_file_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool parse_args(int argc, char ** argv, probe_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            params.model = need_value(arg.c_str());
        } else if (arg == "--prompt") {
            params.prompt = need_value(arg.c_str());
        } else if (arg == "--prompt-file") {
            params.prompt_file = need_value(arg.c_str());
        } else if (arg == "--out-dir") {
            params.out_dir = need_value(arg.c_str());
        } else if (arg == "--layers") {
            params.layers = parse_layers(need_value(arg.c_str()));
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::stoi(need_value(arg.c_str()));
        } else if (arg == "-c" || arg == "--ctx-size") {
            params.n_ctx = std::stoi(need_value(arg.c_str()));
        } else if (arg == "-b" || arg == "--batch-size") {
            params.n_batch = std::stoi(need_value(arg.c_str()));
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            params.n_ubatch = std::stoi(need_value(arg.c_str()));
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(need_value(arg.c_str()));
            params.n_threads_batch = params.n_threads;
        } else if (arg == "-tb" || arg == "--threads-batch") {
            params.n_threads_batch = std::stoi(need_value(arg.c_str()));
        } else if (arg == "--no-binary") {
            params.write_binary = false;
        } else if (arg == "--no-last-hidden") {
            params.write_last_hidden = false;
        } else if (arg == "-h" || arg == "--help") {
            params.help = true;
            return false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (!params.prompt_file.empty()) {
        params.prompt = read_file_text(params.prompt_file);
    }
    if (params.model.empty() || params.out_dir.empty()) {
        return false;
    }
    if (params.layers.empty()) {
        throw std::runtime_error("at least one layer must be selected");
    }
    return true;
}

static layer_stats summarize(const float * data, size_t n) {
    layer_stats stats;
    stats.first8.reserve(std::min<size_t>(8, n));

    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (stats.first8.size() < 8) {
            stats.first8.push_back(v);
        }
        if (!std::isfinite(v)) {
            stats.nonfinite++;
            continue;
        }
        const double dv = (double) v;
        stats.min = std::min(stats.min, dv);
        stats.max = std::max(stats.max, dv);
        sum += dv;
        sum_sq += dv * dv;
        stats.finite++;
    }
    if (stats.finite > 0) {
        stats.mean = sum / (double) stats.finite;
        stats.l2 = std::sqrt(sum_sq);
    } else {
        stats.min = 0.0;
        stats.max = 0.0;
    }
    return stats;
}

static bool write_binary_file(const std::filesystem::path & path, const float * data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), (std::streamsize) (n * sizeof(float)));
    return (bool) out;
}

static std::string meta_string(const llama_model * model, const char * key) {
    char buf[512];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) {
        return "";
    }
    return std::string(buf);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    probe_params params;
    const std::string started_at_utc = now_utc();
    const std::vector<std::string> cli_argv = argv_vector(argc, argv);
    try {
        if (!parse_args(argc, argv, params)) {
            usage(argv[0]);
            return params.help ? 0 : 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 1;
    }

    std::filesystem::create_directories(params.out_dir);

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.c_str());
        llama_backend_free();
        return 2;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);

    for (int layer : params.layers) {
        if (layer < 0 || layer >= n_layer) {
            fprintf(stderr, "error: selected layer %d outside [0, %d)\n", layer, n_layer);
            llama_model_free(model);
            llama_backend_free();
            return 3;
        }
    }

    int n_prompt = -llama_tokenize(vocab, params.prompt.c_str(), (int32_t) params.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "error: tokenization produced no tokens\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, params.prompt.c_str(), (int32_t) params.prompt.size(), tokens.data(), n_prompt, true, true) < 0) {
        fprintf(stderr, "error: failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params.n_ctx > 0 ? params.n_ctx : n_prompt;
    cparams.n_batch = params.n_batch > 0 ? params.n_batch : n_prompt;
    cparams.n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : cparams.n_batch;
    cparams.no_perf = false;
    cparams.embeddings = params.write_last_hidden;

    if ((int32_t) cparams.n_ctx < n_prompt || (int32_t) cparams.n_batch < n_prompt) {
        fprintf(stderr, "error: prompt tokens exceed n_ctx or n_batch\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    llama_set_n_threads(ctx, params.n_threads, params.n_threads_batch);
    for (int layer : params.layers) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) layer, true);
    }

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int32_t i = 0; i < n_prompt; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
        batch.n_tokens++;
    }

    const int64_t t0 = ggml_time_us();
    const int decode_rc = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    const int64_t t1 = ggml_time_us();

    if (decode_rc != 0) {
        fprintf(stderr, "error: llama_decode failed with code %d\n", decode_rc);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 7;
    }

    struct layer_result {
        int layer;
        std::filesystem::path path;
        layer_stats stats;
    };
    std::vector<layer_result> results;
    const size_t row_floats = (size_t) n_embd;
    const size_t total_floats = (size_t) n_prompt * row_floats;

    std::filesystem::path last_hidden_path;
    layer_stats last_hidden_stats;
    if (params.write_last_hidden) {
        const float * last_hidden = llama_get_embeddings(ctx);
        if (last_hidden == nullptr) {
            fprintf(stderr, "error: missing final hidden-state embeddings\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 8;
        }
        if (params.write_binary) {
            last_hidden_path = std::filesystem::path(params.out_dir) / "target-last-hidden.f32";
            if (!write_binary_file(last_hidden_path, last_hidden, total_floats)) {
                fprintf(stderr, "error: failed to write %s\n", last_hidden_path.string().c_str());
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 9;
            }
        }
        last_hidden_stats = summarize(last_hidden, total_floats);
    }

    for (int layer : params.layers) {
        const float * data = llama_get_embeddings_layer_inp(ctx, (uint32_t) layer);
        if (data == nullptr) {
            fprintf(stderr, "error: missing layer input data for layer %d\n", layer);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 8;
        }

        std::filesystem::path layer_path;
        if (params.write_binary) {
            char name[64];
            snprintf(name, sizeof(name), "layer-%03d-input.f32", layer);
            layer_path = std::filesystem::path(params.out_dir) / name;
            if (!write_binary_file(layer_path, data, total_floats)) {
                fprintf(stderr, "error: failed to write %s\n", layer_path.string().c_str());
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 9;
            }
        }

        results.push_back({ layer, layer_path, summarize(data, total_floats) });
    }

    char desc[512];
    llama_model_desc(model, desc, sizeof(desc));

    std::filesystem::path json_path = std::filesystem::path(params.out_dir) / "probe_result.json";
    std::ofstream json(json_path);
    if (!json) {
        fprintf(stderr, "error: failed to write %s\n", json_path.string().c_str());
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 10;
    }

    json << "{\n";
    json << "  \"condition\": {\n";
    json << "    \"name\": \"llama_dspark_layer_input_probe\",\n";
    json << "    \"label\": \"controlled_nvfp4_layer_input_extraction\",\n";
    json << "    \"started_at_utc\": \"" << started_at_utc << "\",\n";
    json << "    \"finished_at_utc\": \"" << now_utc() << "\",\n";
    json << "    \"command\": \"" << json_escape(join_argv(cli_argv)) << "\",\n";
    json << "    \"argv\": [";
    for (size_t i = 0; i < cli_argv.size(); ++i) {
        json << (i ? ", " : "") << "\"" << json_escape(cli_argv[i]) << "\"";
    }
    json << "],\n";
    json << "    \"model_path\": \"" << json_escape(params.model) << "\",\n";
    json << "    \"out_dir\": \"" << json_escape(params.out_dir) << "\",\n";
    json << "    \"prompt\": \"" << json_escape(params.prompt) << "\",\n";
    json << "    \"prompt_file\": \"" << json_escape(params.prompt_file) << "\",\n";
    json << "    \"n_gpu_layers\": " << params.n_gpu_layers << ",\n";
    json << "    \"n_ctx_requested\": " << cparams.n_ctx << ",\n";
    json << "    \"n_batch_requested\": " << cparams.n_batch << ",\n";
    json << "    \"n_ubatch_requested\": " << cparams.n_ubatch << ",\n";
    json << "    \"n_threads\": " << params.n_threads << ",\n";
    json << "    \"n_threads_batch\": " << params.n_threads_batch << ",\n";
    json << "    \"write_last_hidden\": " << (params.write_last_hidden ? "true" : "false") << "\n";
    json << "  },\n";
    json << "  \"model\": {\n";
    json << "    \"description\": \"" << json_escape(desc) << "\",\n";
    json << "    \"architecture\": \"" << json_escape(meta_string(model, "general.architecture")) << "\",\n";
    json << "    \"file_type\": \"" << json_escape(meta_string(model, "general.file_type")) << "\",\n";
    json << "    \"n_layer\": " << n_layer << ",\n";
    json << "    \"n_embd\": " << n_embd << ",\n";
    json << "    \"n_ctx_train\": " << llama_model_n_ctx_train(model) << ",\n";
    json << "    \"n_vocab\": " << llama_vocab_n_tokens(vocab) << "\n";
    json << "  },\n";
    json << "  \"context\": {\n";
    json << "    \"n_ctx_actual\": " << llama_n_ctx(ctx) << ",\n";
    json << "    \"n_batch_actual\": " << llama_n_batch(ctx) << ",\n";
    json << "    \"n_ubatch_actual\": " << llama_n_ubatch(ctx) << "\n";
    json << "  },\n";
    json << "  \"tokenization\": {\n";
    json << "    \"n_tokens\": " << n_prompt << ",\n";
    json << "    \"tokens\": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        json << (i ? ", " : "") << tokens[i];
    }
    json << "]\n";
    json << "  },\n";
    json << "  \"decode\": {\n";
    json << "    \"return_code\": " << decode_rc << ",\n";
    json << "    \"elapsed_ms\": " << ((t1 - t0) / 1000.0) << "\n";
    json << "  },\n";
    json << "  \"outputs\": {\n";
    json << "    \"dtype\": \"float32\",\n";
    json << "    \"shape_per_layer\": [" << n_prompt << ", " << n_embd << "],\n";
    json << "    \"row_major\": true,\n";
    json << "    \"binary_files_written\": " << (params.write_binary ? "true" : "false") << ",\n";
    json << "    \"target_last_hidden_state\": ";
    if (params.write_last_hidden) {
        json << "{\n";
        json << "      \"path\": \"" << json_escape(last_hidden_path.string()) << "\",\n";
        json << "      \"bytes\": " << (params.write_binary ? total_floats * sizeof(float) : 0) << ",\n";
        json << "      \"finite\": " << last_hidden_stats.finite << ",\n";
        json << "      \"nonfinite\": " << last_hidden_stats.nonfinite << ",\n";
        json << "      \"min\": " << last_hidden_stats.min << ",\n";
        json << "      \"max\": " << last_hidden_stats.max << ",\n";
        json << "      \"mean\": " << last_hidden_stats.mean << ",\n";
        json << "      \"l2\": " << last_hidden_stats.l2 << ",\n";
        json << "      \"first8\": [";
        for (size_t j = 0; j < last_hidden_stats.first8.size(); ++j) {
            json << (j ? ", " : "") << last_hidden_stats.first8[j];
        }
        json << "]\n";
        json << "    },\n";
    } else {
        json << "null,\n";
    }
    json << "    \"layers\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & r = results[i];
        json << "      {\n";
        json << "        \"layer\": " << r.layer << ",\n";
        json << "        \"path\": \"" << json_escape(r.path.string()) << "\",\n";
        json << "        \"bytes\": " << (params.write_binary ? total_floats * sizeof(float) : 0) << ",\n";
        json << "        \"finite\": " << r.stats.finite << ",\n";
        json << "        \"nonfinite\": " << r.stats.nonfinite << ",\n";
        json << "        \"min\": " << r.stats.min << ",\n";
        json << "        \"max\": " << r.stats.max << ",\n";
        json << "        \"mean\": " << r.stats.mean << ",\n";
        json << "        \"l2\": " << r.stats.l2 << ",\n";
        json << "        \"first8\": [";
        for (size_t j = 0; j < r.stats.first8.size(); ++j) {
            json << (j ? ", " : "") << r.stats.first8[j];
        }
        json << "]\n";
        json << "      }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    json << "    ]\n";
    json << "  }\n";
    json << "}\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", json_path.string().c_str());
    return 0;
}
