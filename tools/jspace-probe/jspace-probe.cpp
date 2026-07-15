#include "arg.h"
#include "common.h"
#include "fibonacci-pool.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
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
    std::string               verified_model_sha256;
    size_t                    disabled_invariance_tokens = 0;
    size_t                    sequence_lifecycle_tokens = 0;
    bool                      include_residual_vector = false;
    size_t                    fibonacci_pool_max_tokens = 0;
    bool                      self_test = false;
};

struct residual_norm_capture {
    std::string                     tensor_name;
    std::vector<float>              values;
    std::vector<std::vector<float>> recent_columns;
    std::string                     error;
    size_t                          max_columns = 0;
    size_t                          expected_columns = 0;
    size_t                          columns_seen = 0;
    double                          l2        = 0.0;
    double                          rms       = 0.0;
    int64_t                         dimension = 0;
    bool                            captured  = false;

    void reset() {
        values.clear();
        recent_columns.clear();
        error.clear();
        columns_seen = 0;
        l2 = 0.0;
        rms = 0.0;
        dimension = 0;
        captured = false;
    }
};

static bool capture_residual_norm(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<residual_norm_capture *>(user_data);
    const bool matches = !capture->tensor_name.empty() &&
            std::strcmp(tensor->name, capture->tensor_name.c_str()) == 0;

    if (ask) {
        return matches;
    }
    if (!matches) {
        return true;
    }

    if (tensor->type != GGML_TYPE_F32) {
        capture->error = "residual tensor " + capture->tensor_name +
                " has unsupported type " + ggml_type_name(tensor->type) + " (expected f32)";
        return true;
    }
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            tensor->nb[0] != sizeof(float) ||
            tensor->nb[1] < static_cast<size_t>(tensor->ne[0]) * sizeof(float)) {
        capture->error = "residual tensor " + capture->tensor_name + " has an unsupported layout";
        return true;
    }

    const size_t width = static_cast<size_t>(tensor->ne[0]);
    const size_t nbytes = width * sizeof(float);
    const size_t column_count = static_cast<size_t>(tensor->ne[1]);
    const bool pool_enabled = capture->max_columns > 0;
    for (size_t column = 0; column < column_count; ++column) {
        const size_t global_column = capture->columns_seen + column;
        const bool retained_pool_column = pool_enabled &&
                global_column < capture->expected_columns &&
                global_column + capture->max_columns >= capture->expected_columns;
        const bool final_column = pool_enabled ?
                global_column + 1 == capture->expected_columns :
                column + 1 == column_count;
        if (!retained_pool_column && !final_column) {
            continue;
        }
        const size_t offset = column * tensor->nb[1];
        if (offset + nbytes > ggml_nbytes(tensor)) {
            capture->error = "column of residual tensor " + capture->tensor_name + " is out of bounds";
            return true;
        }

        std::vector<float> values(width);
        ggml_backend_tensor_get(tensor, values.data(), offset, nbytes);
        for (float value : values) {
            if (!std::isfinite(value)) {
                capture->error = "residual tensor " + capture->tensor_name + " contains a non-finite value";
                return true;
            }
        }
        if (final_column) {
            capture->values = values;
        }
        if (retained_pool_column) {
            capture->recent_columns.push_back(std::move(values));
        }
    }

    capture->columns_seen += column_count;
    if (capture->values.empty()) {
        return true;
    }
    double sum_squares = 0.0;
    for (float value : capture->values) {
        sum_squares += static_cast<double>(value) * value;
    }

    capture->dimension = tensor->ne[0];
    capture->l2 = std::sqrt(sum_squares);
    capture->rms = std::sqrt(sum_squares / static_cast<double>(capture->dimension));
    capture->captured = true;
    return true;
}

static json residual_capture_to_json(
        const residual_norm_capture & capture,
        size_t                        position,
        bool                          include_values,
        size_t                        fibonacci_pool_max_tokens) {
    json result = {
        { "tensor", capture.tensor_name },
        { "position", position },
        { "dimension", capture.dimension },
        { "l2", capture.l2 },
        { "rms", capture.rms },
    };
    if (include_values) {
        result["values"] = capture.values;
    }
    if (fibonacci_pool_max_tokens > 0) {
        if (capture.expected_columns < capture.recent_columns.size()) {
            throw std::logic_error(
                    "Fibonacci retained suffix exceeds the prompt-token domain");
        }
        const size_t retained_start =
                capture.expected_columns - capture.recent_columns.size();
        const auto horizons = jspace::fibonacci_horizons_up_to(
                fibonacci_pool_max_tokens);
        const auto rows = jspace::suffix_boxcar_pool(
                capture.recent_columns, horizons);
        json row_records = json::array();
        for (const auto & row : rows) {
            json record = {
                { "horizon_tokens", row.horizon },
                { "start_position", position + 1 - row.horizon },
                { "end_position_exclusive", position + 1 },
                { "dimension", capture.dimension },
                { "matrix_weight", 1.0 / static_cast<double>(row.horizon) },
                { "l2", row.l2 },
                { "rms", row.rms },
            };
            if (include_values) {
                record["values"] = row.values;
            }
            row_records.push_back(std::move(record));
        }
        result["fibonacci_pool"] = {
            { "status", "diagnostic_observer_candidate_only" },
            { "algorithm", "nested_uniform_suffix_mean_v1" },
            { "basis", "causal row-stochastic boxcars at Fibonacci horizons" },
            { "seed_lengths", { 1, 2 } },
            { "requested_max_tokens", fibonacci_pool_max_tokens },
            { "available_prompt_tokens", capture.expected_columns },
            { "largest_horizon_tokens", rows.empty() ? 0 : rows.back().horizon },
            { "captured_columns", capture.recent_columns.size() },
            { "retained_bytes", capture.recent_columns.size() *
                    static_cast<size_t>(capture.dimension) * sizeof(float) },
            { "columns_seen", capture.columns_seen },
            { "token_domain", "tokenized prompt including added special tokens" },
            { "matrix", {
                { "shape", { rows.size(), capture.expected_columns } },
                { "storage", "sparse_suffix_boxcar" },
                { "column_basis", "zero_based_prompt_token" },
                { "row_sum", 1.0 },
            } },
            { "retained_suffix", {
                { "shape", { capture.recent_columns.size(), capture.dimension } },
                { "start_position", retained_start },
                { "end_position_exclusive", capture.expected_columns },
            } },
            { "rows", std::move(row_records) },
        };
    }
    return result;
}

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

static size_t parse_fibonacci_pool_max_tokens(const std::string & value) {
    const std::string item = trim(value);
    errno = 0;
    char * parse_end = nullptr;
    const unsigned long long parsed = std::strtoull(item.c_str(), &parse_end, 10);
    if (item.empty() || errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' ||
            parsed == 0 || parsed > 144) {
        throw std::invalid_argument(
                "--fibonacci-pool-max must be an integer from 1 through 144");
    }
    return static_cast<size_t>(parsed);
}

static size_t parse_disabled_invariance_tokens(const std::string & value) {
    const std::string item = trim(value);
    errno = 0;
    char * parse_end = nullptr;
    const unsigned long long parsed = std::strtoull(item.c_str(), &parse_end, 10);
    if (item.empty() || errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' ||
            parsed == 0 || parsed > 32) {
        throw std::invalid_argument(
                "--verify-disabled-invariance must be an integer from 1 through 32");
    }
    return static_cast<size_t>(parsed);
}

static size_t parse_sequence_lifecycle_tokens(const std::string & value) {
    const std::string item = trim(value);
    errno = 0;
    char * parse_end = nullptr;
    const unsigned long long parsed = std::strtoull(item.c_str(), &parse_end, 10);
    if (item.empty() || errno == ERANGE || parse_end == item.c_str() || *parse_end != '\0' ||
            parsed == 0 || parsed > 16) {
        throw std::invalid_argument(
                "--verify-sequence-lifecycle must be an integer from 1 through 16");
    }
    return static_cast<size_t>(parsed);
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

static std::string parse_sha256(const std::string & value, const char * option) {
    const std::string item = trim(value);
    if (item.size() != 64 || !std::all_of(item.begin(), item.end(), [](unsigned char ch) {
            return std::isxdigit(ch) != 0;
        })) {
        throw std::invalid_argument(std::string(option) + " requires exactly 64 hexadecimal characters");
    }

    std::string result = item;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return result;
}

static probe_args preprocess_args(int argc, char ** argv) {
    probe_args result;
    result.common_argv.reserve(argc);
    result.common_argv.push_back(argv[0]);

    constexpr const char * token_prefix = "--token-ids=";
    constexpr const char * vector_prefix = "--probe-vector=";
    constexpr const char * strength_prefix = "--probe-strengths=";
    constexpr const char * fibonacci_prefix = "--fibonacci-pool-max=";
    constexpr const char * model_sha_prefix = "--verified-model-sha256=";
    constexpr const char * disabled_invariance_prefix = "--verify-disabled-invariance=";
    constexpr const char * sequence_lifecycle_prefix = "--verify-sequence-lifecycle=";

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--self-test") {
            result.self_test = true;
            continue;
        }
        if (arg == "--include-residual-vector") {
            result.include_residual_vector = true;
            continue;
        }
        if (arg == "--verify-disabled-invariance") {
            if (++i >= argc) {
                throw std::invalid_argument("--verify-disabled-invariance requires a token count");
            }
            result.disabled_invariance_tokens = parse_disabled_invariance_tokens(argv[i]);
            continue;
        }
        if (arg.compare(0, std::strlen(disabled_invariance_prefix), disabled_invariance_prefix) == 0) {
            result.disabled_invariance_tokens = parse_disabled_invariance_tokens(
                    arg.substr(std::strlen(disabled_invariance_prefix)));
            continue;
        }
        if (arg == "--verify-sequence-lifecycle") {
            if (++i >= argc) {
                throw std::invalid_argument("--verify-sequence-lifecycle requires a token count");
            }
            result.sequence_lifecycle_tokens = parse_sequence_lifecycle_tokens(argv[i]);
            continue;
        }
        if (arg.compare(0, std::strlen(sequence_lifecycle_prefix), sequence_lifecycle_prefix) == 0) {
            result.sequence_lifecycle_tokens = parse_sequence_lifecycle_tokens(
                    arg.substr(std::strlen(sequence_lifecycle_prefix)));
            continue;
        }
        if (arg == "--verified-model-sha256") {
            if (++i >= argc) {
                throw std::invalid_argument("--verified-model-sha256 requires a SHA-256 value");
            }
            if (!result.verified_model_sha256.empty()) {
                throw std::invalid_argument("--verified-model-sha256 may be specified only once");
            }
            result.verified_model_sha256 = parse_sha256(argv[i], "--verified-model-sha256");
            continue;
        }
        if (arg.compare(0, std::strlen(model_sha_prefix), model_sha_prefix) == 0) {
            if (!result.verified_model_sha256.empty()) {
                throw std::invalid_argument("--verified-model-sha256 may be specified only once");
            }
            result.verified_model_sha256 = parse_sha256(
                    arg.substr(std::strlen(model_sha_prefix)), "--verified-model-sha256");
            continue;
        }
        if (arg == "--fibonacci-pool-max") {
            if (++i >= argc) {
                throw std::invalid_argument("--fibonacci-pool-max requires an integer value");
            }
            result.fibonacci_pool_max_tokens = parse_fibonacci_pool_max_tokens(argv[i]);
            continue;
        }
        if (arg.compare(0, std::strlen(fibonacci_prefix), fibonacci_prefix) == 0) {
            result.fibonacci_pool_max_tokens = parse_fibonacci_pool_max_tokens(
                    arg.substr(std::strlen(fibonacci_prefix)));
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

struct divergence_metrics {
    double kl_base_to_run = 0.0;
    double kl_run_to_base = 0.0;
    double jensen_shannon = 0.0;
};

static double logaddexp(double lhs, double rhs) {
    if (!std::isfinite(lhs)) {
        return rhs;
    }
    if (!std::isfinite(rhs)) {
        return lhs;
    }
    const double maximum = std::max(lhs, rhs);
    return maximum + std::log(std::exp(lhs - maximum) + std::exp(rhs - maximum));
}

static divergence_metrics distribution_divergence(
        const std::vector<float> & baseline_logits,
        double                     baseline_log_z,
        const std::vector<float> & run_logits,
        double                     run_log_z) {
    if (baseline_logits.size() != run_logits.size()) {
        throw std::runtime_error("cannot compare logit distributions with different vocabulary sizes");
    }

    divergence_metrics result;
    constexpr double log_two = 0.693147180559945309417232121458176568;

    for (size_t i = 0; i < baseline_logits.size(); ++i) {
        const double log_p = std::isfinite(baseline_logits[i]) ? baseline_logits[i] - baseline_log_z :
                -std::numeric_limits<double>::infinity();
        const double log_q = std::isfinite(run_logits[i]) ? run_logits[i] - run_log_z :
                -std::numeric_limits<double>::infinity();
        const double p = std::isfinite(log_p) ? std::exp(log_p) : 0.0;
        const double q = std::isfinite(log_q) ? std::exp(log_q) : 0.0;

        if (p > 0.0) {
            result.kl_base_to_run += std::isfinite(log_q) ? p * (log_p - log_q) :
                    std::numeric_limits<double>::infinity();
        }
        if (q > 0.0) {
            result.kl_run_to_base += std::isfinite(log_p) ? q * (log_q - log_p) :
                    std::numeric_limits<double>::infinity();
        }
        if (p > 0.0 || q > 0.0) {
            const double log_m = logaddexp(log_p, log_q) - log_two;
            if (p > 0.0) {
                result.jensen_shannon += 0.5 * p * (log_p - log_m);
            }
            if (q > 0.0) {
                result.jensen_shannon += 0.5 * q * (log_q - log_m);
            }
        }
    }

    if (std::isfinite(result.kl_base_to_run)) {
        result.kl_base_to_run = std::max(0.0, result.kl_base_to_run);
    }
    if (std::isfinite(result.kl_run_to_base)) {
        result.kl_run_to_base = std::max(0.0, result.kl_run_to_base);
    }
    result.jensen_shannon = std::max(0.0, result.jensen_shannon);
    return result;
}

static json divergence_to_json(const divergence_metrics & metrics) {
    auto finite_number = [](double value) -> json {
        return std::isfinite(value) ? json(value) : json(nullptr);
    };
    return {
        { "units", "nats" },
        { "kl_base_to_run", finite_number(metrics.kl_base_to_run) },
        { "kl_run_to_base", finite_number(metrics.kl_run_to_base) },
        { "jensen_shannon", finite_number(metrics.jensen_shannon) },
    };
}

static void print_usage(int, char ** argv) {
    std::printf("\nJ-Space causal logit probe:\n");
    std::printf("\n  %s -m model.gguf -p PROMPT --token-ids ID,ID,... [common options]\n", argv[0]);
    std::printf("\nProbe options:\n");
    std::printf("  --token-ids ID,ID,...  vocabulary IDs to report; may be repeated\n");
    std::printf("  --probe-vector NAME=PATH\n");
    std::printf("                           named control vector to sweep; may be repeated\n");
    std::printf("  --verified-model-sha256 HEX\n");
    std::printf("                           runner-verified model digest; required for vectors\n");
    std::printf("  --probe-strengths S,S,...\n");
    std::printf("                           shared finite strengths for every probe vector\n");
    std::printf("  --include-residual-vector\n");
    std::printf("                           include the final post-block residual values in JSON\n");
    std::printf("  --fibonacci-pool-max N\n");
    std::printf("                           pool residual columns over 1,2,3,5,... token horizons\n");
    std::printf("  --verify-disabled-invariance N\n");
    std::printf("                           require exact disabled-path parity for N greedy tokens\n");
    std::printf("  --verify-sequence-lifecycle N\n");
    std::printf("                           require exact two-sequence isolation for N greedy tokens\n");
    std::printf("  --self-test            run the model-independent deterministic smoke test\n\n");
}

static void run_identity_self_test();

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
    if (parse_fibonacci_pool_max_tokens("144") != 144) {
        throw std::runtime_error("Fibonacci pool-max parser self-test failed");
    }
    if (parse_disabled_invariance_tokens("32") != 32) {
        throw std::runtime_error("disabled-invariance token parser self-test failed");
    }
    if (parse_sequence_lifecycle_tokens("16") != 16) {
        throw std::runtime_error("sequence-lifecycle token parser self-test failed");
    }
    bool rejected_bad_pool_max = false;
    try {
        (void) parse_fibonacci_pool_max_tokens("145");
    } catch (const std::invalid_argument &) {
        rejected_bad_pool_max = true;
    }
    if (!rejected_bad_pool_max) {
        throw std::runtime_error("Fibonacci pool-max bound self-test failed");
    }
    const auto vector = parse_named_vector("joy=/tmp/joy.gguf");
    if (vector.name != "joy" || vector.path != "/tmp/joy.gguf") {
        throw std::runtime_error("named vector parser self-test failed");
    }
    if (parse_sha256(std::string(64, 'A'), "self-test") != std::string(64, 'a')) {
        throw std::runtime_error("SHA-256 parser normalization self-test failed");
    }
    run_identity_self_test();

    residual_norm_capture residual;
    residual.tensor_name = "l_out-39";
    residual.values = { 1.0f, -2.0f };
    residual.dimension = 2;
    residual.l2 = std::sqrt(5.0);
    residual.rms = std::sqrt(2.5);
    residual.captured = true;
    const json norm_only = residual_capture_to_json(residual, 7, false, 0);
    const json with_values = residual_capture_to_json(residual, 7, true, 0);
    if (norm_only.contains("values") || !with_values.contains("values") ||
            with_values.at("values") != json({ 1.0f, -2.0f })) {
        throw std::runtime_error("residual-vector JSON self-test failed");
    }
    residual.recent_columns = { { 3.0f, 2.0f }, residual.values };
    residual.max_columns = 2;
    residual.expected_columns = 5;
    residual.columns_seen = 5;
    const json pool_norm_only = residual_capture_to_json(residual, 4, false, 2);
    const json pool_with_values = residual_capture_to_json(residual, 4, true, 2);
    const auto & pool_rows = pool_norm_only.at("fibonacci_pool").at("rows");
    const auto & pool_value_rows = pool_with_values.at("fibonacci_pool").at("rows");
    if (pool_rows.size() != 2 || pool_rows[0].contains("values") ||
            !pool_value_rows[0].contains("values") ||
            pool_value_rows[0].at("values") != with_values.at("values") ||
            pool_norm_only.at("fibonacci_pool").at("matrix").at("shape") !=
                    json({ 2, 5 }) ||
            pool_norm_only.at("fibonacci_pool").at("retained_suffix").at("shape") !=
                    json({ 2, 2 }) ||
            pool_norm_only.at("fibonacci_pool").at("retained_suffix").at("start_position") != 3 ||
            pool_rows[0].at("start_position") != 4 ||
            pool_rows[1].at("start_position") != 3) {
        throw std::runtime_error("Fibonacci pooling JSON/vector-gating self-test failed");
    }

    const auto horizons = jspace::fibonacci_horizons(6);
    if (horizons != std::vector<size_t>({ 1, 2, 3, 5, 8, 13 })) {
        throw std::runtime_error("Fibonacci horizon self-test failed");
    }
    if (jspace::fibonacci_horizons_up_to(10) !=
            std::vector<size_t>({ 1, 2, 3, 5, 8 })) {
        throw std::runtime_error("bounded Fibonacci horizon self-test failed");
    }
    std::vector<std::vector<float>> pool_columns;
    for (int value = 1; value <= 8; ++value) {
        pool_columns.push_back({ static_cast<float>(value), 2.0f });
    }
    const auto pooled = jspace::fibonacci_suffix_pool(pool_columns, 5);
    const std::vector<double> expected_means = { 8.0, 7.5, 7.0, 6.0, 4.5 };
    if (pooled.size() != expected_means.size()) {
        throw std::runtime_error("Fibonacci pooling row-count self-test failed");
    }
    for (size_t i = 0; i < pooled.size(); ++i) {
        if (std::abs(pooled[i].values[0] - expected_means[i]) > 1e-6 ||
                std::abs(pooled[i].values[1] - 2.0f) > 1e-6) {
            throw std::runtime_error("Fibonacci pooling mean/constant-preservation self-test failed");
        }
    }
    // At t=7, the five-sample suffix sum [4,5,6,7,8] factors into
    // the newest three [6,7,8] and the delayed older two [4,5].
    const double recursive_five =
            (pooled[2].values[0] * 3.0) + ((4.0 + 5.0));
    if (std::abs(recursive_five - pooled[3].values[0] * 5.0) > 1e-12) {
        throw std::runtime_error("delayed Fibonacci block-recursion self-test failed");
    }
    const auto comparator_pool = jspace::suffix_boxcar_pool(
            pool_columns, { 1, 4, 8 });
    if (comparator_pool.size() != 3 ||
            std::abs(comparator_pool[0].values[0] - 8.0f) > 1e-6 ||
            std::abs(comparator_pool[1].values[0] - 6.5f) > 1e-6 ||
            std::abs(comparator_pool[2].values[0] - 4.5f) > 1e-6) {
        throw std::runtime_error("generic suffix-boxcar self-test failed");
    }
    bool rejected_bad_horizons = false;
    try {
        (void) jspace::suffix_boxcar_pool(pool_columns, { 1, 3, 3 });
    } catch (const std::invalid_argument &) {
        rejected_bad_horizons = true;
    }
    if (!rejected_bad_horizons) {
        throw std::runtime_error("suffix-boxcar horizon-validation self-test failed");
    }
    auto nonfinite_columns = pool_columns;
    nonfinite_columns.back()[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_nonfinite_pool = false;
    try {
        (void) jspace::fibonacci_suffix_pool(nonfinite_columns, 5);
    } catch (const std::invalid_argument &) {
        rejected_nonfinite_pool = true;
    }
    if (!rejected_nonfinite_pool) {
        throw std::runtime_error("non-finite Fibonacci input self-test failed");
    }

    const std::vector<float> distribution_a = {
        static_cast<float>(std::log(0.8)),
        static_cast<float>(std::log(0.2)),
    };
    const std::vector<float> distribution_b = {
        static_cast<float>(std::log(0.5)),
        static_cast<float>(std::log(0.5)),
    };
    const auto same_divergence = distribution_divergence(distribution_a, 0.0, distribution_a, 0.0);
    const auto forward_divergence = distribution_divergence(distribution_a, 0.0, distribution_b, 0.0);
    const auto reverse_divergence = distribution_divergence(distribution_b, 0.0, distribution_a, 0.0);
    if (same_divergence.kl_base_to_run > 1e-12 || same_divergence.kl_run_to_base > 1e-12 ||
            same_divergence.jensen_shannon > 1e-12 ||
            std::abs(forward_divergence.kl_base_to_run - reverse_divergence.kl_run_to_base) > 1e-12 ||
            std::abs(forward_divergence.kl_run_to_base - reverse_divergence.kl_base_to_run) > 1e-12 ||
            std::abs(forward_divergence.kl_base_to_run - 0.1927447570217575) > 1e-7 ||
            std::abs(forward_divergence.kl_run_to_base - 0.2231435513142097) > 1e-7 ||
            std::abs(forward_divergence.jensen_shannon - 0.0506718369855659) > 1e-7) {
        throw std::runtime_error("distribution-divergence self-test failed");
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

struct probe_vector_identity {
    std::string schema;
    std::string axis;
    std::string base_model_sha256;
    std::string direction_normalization;
};

static std::string require_gguf_string(
        const struct gguf_context * ctx,
        const std::string &         path,
        const char *                key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("probe vector '" + path + "' is missing string metadata '" + key + "'");
    }
    return gguf_get_val_str(ctx, key_id);
}

static probe_vector_identity load_probe_vector_identity(const std::string & path) {
    const struct gguf_init_params params {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    std::unique_ptr<struct gguf_context, decltype(&gguf_free)> ctx(
            gguf_init_from_file(path.c_str(), params), &gguf_free);
    if (!ctx) {
        throw std::runtime_error("failed to read probe-vector metadata from " + path);
    }

    return {
        require_gguf_string(ctx.get(), path, "jspace.schema"),
        require_gguf_string(ctx.get(), path, "jspace.axis"),
        require_gguf_string(ctx.get(), path, "jspace.base_model_sha256"),
        require_gguf_string(ctx.get(), path, "jspace.direction_normalization"),
    };
}

static void validate_probe_vector_identity(
        const probe_vector_identity & identity,
        const std::string &           requested_axis,
        const std::string &           verified_model_sha256) {
    if (identity.schema != "treebeard.jspace.phase0-cvec.v0") {
        throw std::runtime_error("unsupported J-Space probe-vector schema '" + identity.schema + "'");
    }
    if (parse_sha256(identity.base_model_sha256, "jspace.base_model_sha256") != verified_model_sha256) {
        throw std::runtime_error("probe-vector base-model SHA-256 does not match the runner-verified model");
    }
    if (identity.axis != requested_axis) {
        throw std::runtime_error("probe-vector axis metadata '" + identity.axis +
                "' does not match requested name '" + requested_axis + "'");
    }
    static const std::vector<std::string> supported_normalizations {
        "unit_l2_regularized_dual",
        "raw_regularized_dual",
        "direct_pre_rms_contrast",
    };
    if (std::find(supported_normalizations.begin(), supported_normalizations.end(),
                  identity.direction_normalization) == supported_normalizations.end()) {
        throw std::runtime_error("unsupported probe-vector direction normalization '" +
                identity.direction_normalization + "'");
    }
}

static void run_identity_self_test() {
    const std::string model_sha256(64, 'a');
    const probe_vector_identity valid {
        "treebeard.jspace.phase0-cvec.v0",
        "joy",
        model_sha256,
        "raw_regularized_dual",
    };
    validate_probe_vector_identity(valid, "joy", model_sha256);

    auto must_reject = [&](probe_vector_identity identity, const std::string & axis,
                           const std::string & digest) {
        try {
            validate_probe_vector_identity(identity, axis, digest);
        } catch (const std::exception &) {
            return;
        }
        throw std::runtime_error("probe-vector identity self-test accepted an invalid artifact");
    };

    auto invalid = valid;
    invalid.schema = "treebeard.jspace.unknown";
    must_reject(invalid, "joy", model_sha256);
    invalid = valid;
    invalid.base_model_sha256 = std::string(64, 'b');
    must_reject(invalid, "joy", model_sha256);
    invalid = valid;
    invalid.axis = "sadness";
    must_reject(invalid, "joy", model_sha256);
    invalid = valid;
    invalid.direction_normalization.clear();
    must_reject(invalid, "joy", model_sha256);
}

struct loaded_probe_vector {
    probe_args::named_vector      spec;
    common_control_vector_data    data;
    probe_vector_identity         identity;
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

struct evaluation_result {
    json               output;
    std::vector<float> logits;
    double             log_z = 0.0;
};

static llama_token greedy_top_1(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("cannot sample from empty logits");
    }
    return static_cast<llama_token>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

static std::vector<float> decode_one_token_logits(
        llama_context *         ctx,
        llama_token             token,
        residual_norm_capture & residual_capture) {
    residual_capture.reset();
    llama_batch batch = llama_batch_get_one(&token, 1);
    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        throw std::runtime_error("disabled-invariance decode failed with code " + std::to_string(rc));
    }
    llama_synchronize(ctx);

    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("disabled-invariance logits are unavailable");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    return std::vector<float>(logits, logits + n_vocab);
}

static std::vector<uint8_t> capture_sequence_state(llama_context * ctx, llama_seq_id seq_id = 0) {
    llama_synchronize(ctx);
    const size_t size = llama_state_seq_get_size(ctx, seq_id);
    if (size == 0) {
        throw std::runtime_error("disabled-invariance sequence state is empty");
    }
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data(ctx, data.data(), data.size(), seq_id);
    if (written != data.size()) {
        throw std::runtime_error("disabled-invariance sequence state serialization failed");
    }
    return data;
}

struct disabled_invariance_trace {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token>        sampled_tokens;
    std::vector<uint8_t>            sequence_state;
};

static std::array<std::vector<float>, 2> decode_sequence_pair(
        llama_context * ctx,
        llama_token     token_0,
        llama_token     token_1,
        llama_pos       pos) {
    llama_batch batch = llama_batch_init(2, 0, 1);
    batch.n_tokens = 2;
    batch.token[0] = token_0;
    batch.token[1] = token_1;
    batch.pos[0] = pos;
    batch.pos[1] = pos;
    batch.n_seq_id[0] = 1;
    batch.n_seq_id[1] = 1;
    batch.seq_id[0][0] = 0;
    batch.seq_id[1][0] = 1;
    batch.logits[0] = true;
    batch.logits[1] = true;

    const int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("sequence-lifecycle paired decode failed with code " + std::to_string(rc));
    }
    llama_synchronize(ctx);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::array<std::vector<float>, 2> result;
    for (int32_t i = 0; i < 2; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        if (logits == nullptr) {
            throw std::runtime_error("sequence-lifecycle paired logits are unavailable");
        }
        result[i].assign(logits, logits + n_vocab);
    }
    return result;
}

static std::array<disabled_invariance_trace, 2> run_sequence_pair_trace(
        llama_context *                  ctx,
        const std::vector<llama_token> & prompt_tokens,
        size_t                           token_count,
        float                            scale_0,
        float                            scale_1) {
    if (llama_adapter_cvec_seq_set(ctx, 0, scale_0) != 0 ||
            llama_adapter_cvec_seq_set(ctx, 1, scale_1) != 0) {
        throw std::runtime_error("failed to set request-scoped control-vector scales");
    }

    llama_synchronize(ctx);
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        throw std::runtime_error("sequence-lifecycle context has no model memory");
    }
    llama_memory_clear(memory, true);

    std::array<std::vector<float>, 2> logits;
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        logits = decode_sequence_pair(ctx, prompt_tokens[i], prompt_tokens[i], static_cast<llama_pos>(i));
    }

    std::array<disabled_invariance_trace, 2> result;
    for (auto & trace : result) {
        trace.logits.reserve(token_count);
        trace.sampled_tokens.reserve(token_count);
    }

    for (size_t step = 0; step < token_count; ++step) {
        std::array<llama_token, 2> sampled;
        for (size_t seq = 0; seq < result.size(); ++seq) {
            sampled[seq] = greedy_top_1(logits[seq]);
            result[seq].logits.push_back(std::move(logits[seq]));
            result[seq].sampled_tokens.push_back(sampled[seq]);
        }
        if (step + 1 < token_count) {
            logits = decode_sequence_pair(
                    ctx,
                    sampled[0],
                    sampled[1],
                    static_cast<llama_pos>(prompt_tokens.size() + step));
        }
    }

    result[0].sequence_state = capture_sequence_state(ctx, 0);
    result[1].sequence_state = capture_sequence_state(ctx, 1);
    return result;
}

static disabled_invariance_trace finish_disabled_invariance_trace(
        llama_context *            ctx,
        std::vector<float>          initial_logits,
        size_t                      token_count,
        residual_norm_capture &    residual_capture) {
    disabled_invariance_trace result;
    result.logits.reserve(token_count);
    result.sampled_tokens.reserve(token_count);

    auto logits = std::move(initial_logits);
    for (size_t i = 0; i < token_count; ++i) {
        const llama_token token = greedy_top_1(logits);
        result.logits.push_back(std::move(logits));
        result.sampled_tokens.push_back(token);
        if (i + 1 < token_count) {
            logits = decode_one_token_logits(ctx, token, residual_capture);
        }
    }
    result.sequence_state = capture_sequence_state(ctx);
    return result;
}

static bool bit_identical(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    return lhs.size() == rhs.size() &&
            (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

static void require_trace_identity(
        const disabled_invariance_trace & baseline,
        const disabled_invariance_trace & candidate,
        const char *                      label) {
    if (baseline.sampled_tokens != candidate.sampled_tokens) {
        throw std::runtime_error(std::string(label) + " changed the greedy token sequence");
    }
    if (baseline.logits.size() != candidate.logits.size()) {
        throw std::runtime_error(std::string(label) + " changed the logit-step count");
    }
    for (size_t i = 0; i < baseline.logits.size(); ++i) {
        if (!bit_identical(baseline.logits[i], candidate.logits[i])) {
            size_t mismatch_count = 0;
            size_t max_index = 0;
            double max_abs = 0.0;
            for (size_t j = 0; j < baseline.logits[i].size(); ++j) {
                uint32_t baseline_bits = 0;
                uint32_t candidate_bits = 0;
                std::memcpy(&baseline_bits, &baseline.logits[i][j], sizeof(baseline_bits));
                std::memcpy(&candidate_bits, &candidate.logits[i][j], sizeof(candidate_bits));
                if (baseline_bits == candidate_bits) {
                    continue;
                }
                ++mismatch_count;
                const double abs_diff = std::abs(
                        static_cast<double>(baseline.logits[i][j]) - candidate.logits[i][j]);
                if (abs_diff > max_abs) {
                    max_abs = abs_diff;
                    max_index = j;
                }
            }
            throw std::runtime_error(
                    std::string(label) + " changed logits at greedy step " + std::to_string(i) +
                    ": mismatches=" + std::to_string(mismatch_count) +
                    " max_abs=" + std::to_string(max_abs) +
                    " max_token=" + std::to_string(max_index) +
                    " baseline=" + std::to_string(baseline.logits[i][max_index]) +
                    " candidate=" + std::to_string(candidate.logits[i][max_index]));
        }
    }
    if (baseline.sequence_state != candidate.sequence_state) {
        throw std::runtime_error(std::string(label) + " changed serialized logical sequence state");
    }
}

static json verify_sequence_lifecycle(
        llama_context *                  ctx,
        const std::vector<llama_token> & prompt_tokens,
        size_t                           token_count) {
    const auto all_off = run_sequence_pair_trace(ctx, prompt_tokens, token_count, 0.0f, 0.0f);
    const auto all_on  = run_sequence_pair_trace(ctx, prompt_tokens, token_count, 1.0f, 1.0f);
    const auto mixed   = run_sequence_pair_trace(ctx, prompt_tokens, token_count, 1.0f, 0.0f);

    require_trace_identity(all_on[0], mixed[0], "mixed active sequence");
    require_trace_identity(all_off[1], mixed[1], "mixed protected sequence");

    bool actuator_observed = false;
    for (size_t step = 0; step < token_count && !actuator_observed; ++step) {
        actuator_observed = !bit_identical(all_on[0].logits[step], all_off[0].logits[step]);
    }
    if (!actuator_observed) {
        throw std::runtime_error("sequence-lifecycle vector produced no observable logit change");
    }

    if (!llama_adapter_cvec_seq_mode(ctx) ||
            llama_adapter_cvec_seq_get(ctx, 0) != 1.0f ||
            llama_adapter_cvec_seq_get(ctx, 1) != 0.0f) {
        throw std::runtime_error("sequence-lifecycle mixed scale state is inconsistent");
    }

    // Snapshot the active head, then exercise the exact whole-sequence hooks
    // used by server reset/cancel/fork/commit/slot-reuse paths.
    const auto snapshot_state = capture_sequence_state(ctx, 0);
    const float snapshot_scale = llama_adapter_cvec_seq_get(ctx, 0);

    common_context_seq_rm(ctx, 1, -1, -1);
    common_context_seq_cp(ctx, 0, 1, -1, -1);
    if (llama_adapter_cvec_seq_get(ctx, 1) != snapshot_scale) {
        throw std::runtime_error("sequence-lifecycle fork did not copy controller state");
    }

    common_context_seq_rm(ctx, 1, -1, -1);
    if (llama_adapter_cvec_seq_get(ctx, 1) != 0.0f) {
        throw std::runtime_error("sequence-lifecycle cancellation did not clear controller state");
    }

    if (llama_adapter_cvec_seq_set(ctx, 1, -0.5f) != 0) {
        throw std::runtime_error("sequence-lifecycle slot-reuse setup failed");
    }
    common_context_seq_rm(ctx, 1, -1, -1);
    if (llama_adapter_cvec_seq_get(ctx, 1) != 0.0f) {
        throw std::runtime_error("sequence-lifecycle slot reuse retained controller state");
    }

    common_context_seq_rm(ctx, 0, -1, -1);
    const size_t restored = llama_state_seq_set_data(
            ctx, snapshot_state.data(), snapshot_state.size(), 0);
    if (restored != snapshot_state.size() ||
            llama_adapter_cvec_seq_set(ctx, 0, snapshot_scale) != 0 ||
            capture_sequence_state(ctx, 0) != snapshot_state) {
        throw std::runtime_error("sequence-lifecycle snapshot restore did not reproduce branch state");
    }

    if (llama_adapter_cvec_seq_set(ctx, 1, -1.0f) != 0) {
        throw std::runtime_error("sequence-lifecycle commit setup failed");
    }
    llama_memory_seq_keep(llama_get_memory(ctx), 0);
    llama_adapter_cvec_seq_keep(ctx, 0);
    if (llama_adapter_cvec_seq_get(ctx, 0) != snapshot_scale ||
            llama_adapter_cvec_seq_get(ctx, 1) != 0.0f) {
        throw std::runtime_error("sequence-lifecycle commit retained losing controller state");
    }

    return {
        { "status", "pass" },
        { "mode", "per_token_scale_by_llama_seq_id" },
        { "sequences", 2 },
        { "steps", token_count },
        { "same_shape_controls", { "all_off", "all_on", "mixed" } },
        { "active_matches_all_on_bit_exact", true },
        { "protected_matches_all_off_bit_exact", true },
        { "serialized_state_matches_controls", true },
        { "actuator_change_observed", true },
        { "lifecycle", {
            { "reset", "pass" },
            { "cancel", "pass" },
            { "fork", "pass" },
            { "commit", "pass" },
            { "slot_reuse", "pass" },
            { "snapshot_restore", "pass" },
        } },
        { "logit_bytes_compared", 2 * token_count * all_on[0].logits[0].size() * sizeof(float) },
        { "sequence_state_bytes_compared", mixed[0].sequence_state.size() + mixed[1].sequence_state.size() },
    };
}

static evaluation_result evaluate_prompt(
        llama_context *                    ctx,
        const llama_vocab *                vocab,
        const std::vector<llama_token> &   prompt_tokens,
        const std::vector<llama_token> &   requested_ids,
        residual_norm_capture &            residual_capture,
        bool                               include_residual_vector,
        size_t                             fibonacci_pool_max_tokens) {
    // Qwen3.6 is hybrid recurrent/attention. Clearing data, not only metadata,
    // resets both its KV cache and recurrent/GDN state between paired doses.
    llama_synchronize(ctx);
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        throw std::runtime_error("decoder context has no model memory to reset");
    }
    llama_memory_clear(memory, true);
    llama_perf_context_reset(ctx);
    residual_capture.reset();

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
    if (!residual_capture.error.empty()) {
        throw std::runtime_error(residual_capture.error);
    }
    if (!residual_capture.captured) {
        throw std::runtime_error("evaluation did not expose residual tensor " + residual_capture.tensor_name);
    }
    if (fibonacci_pool_max_tokens > 0) {
        if (residual_capture.columns_seen != prompt_tokens.size()) {
            throw std::runtime_error(
                    "Fibonacci pooling saw " + std::to_string(residual_capture.columns_seen) +
                    " residual columns for a " + std::to_string(prompt_tokens.size()) +
                    "-token prompt");
        }
        if (residual_capture.recent_columns.size() != residual_capture.max_columns) {
            throw std::runtime_error(
                    "Fibonacci pooling retained an unexpected number of residual columns");
        }
        const auto pooled = jspace::suffix_boxcar_pool(
                residual_capture.recent_columns,
                jspace::fibonacci_horizons_up_to(fibonacci_pool_max_tokens));
        if (pooled.empty() || pooled.front().horizon != 1 ||
                pooled.front().values != residual_capture.values ||
                pooled.front().l2 != residual_capture.l2 ||
                pooled.front().rms != residual_capture.rms) {
            throw std::runtime_error(
                    "Fibonacci horizon-one pool does not exactly match the final residual");
        }
    }

    evaluation_result result;
    result.log_z = log_z;
    result.logits.assign(logits, logits + n_vocab);
    result.output["logsumexp"] = log_z;
    result.output["residual"] = residual_capture_to_json(
            residual_capture,
            prompt_tokens.size() - 1,
            include_residual_vector,
            fibonacci_pool_max_tokens);
    result.output["tokens"] = json::array();
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
        result.output["tokens"].push_back(std::move(item));
    }
    return result;
}

static int run_probe(common_params & params, const probe_args & probe) {
    const auto & requested_ids = probe.token_ids;

    // Identity is cheap to validate and must fail before allocating the model or
    // loading any vector tensor data.
    std::vector<probe_vector_identity> vector_identities;
    vector_identities.reserve(probe.probe_vectors.size());
    for (const auto & spec : probe.probe_vectors) {
        auto identity = load_probe_vector_identity(spec.path);
        validate_probe_vector_identity(identity, spec.name, probe.verified_model_sha256);
        vector_identities.push_back(std::move(identity));
    }

    residual_norm_capture residual_capture;
    params.cb_eval = capture_residual_norm;
    params.cb_eval_user_data = &residual_capture;

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
    residual_capture.tensor_name = "l_out-" + std::to_string(llama_model_n_layer(model) - 1);

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
    if (probe.fibonacci_pool_max_tokens > 0) {
        const auto horizons = jspace::fibonacci_horizons_up_to(
                std::min(prompt_tokens.size(), probe.fibonacci_pool_max_tokens));
        residual_capture.expected_columns = prompt_tokens.size();
        residual_capture.max_columns = horizons.back();
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
    for (size_t i = 0; i < probe.probe_vectors.size(); ++i) {
        const auto & spec = probe.probe_vectors[i];
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
        loaded_vectors.push_back({ spec, std::move(data), std::move(vector_identities[i]) });
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
    const evaluation_result baseline = evaluate_prompt(
            ctx,
            vocab,
            prompt_tokens,
            requested_ids,
            residual_capture,
            probe.include_residual_vector,
            probe.fibonacci_pool_max_tokens);

    json disabled_invariance;
    if (probe.disabled_invariance_tokens > 0) {
        auto baseline_trace = finish_disabled_invariance_trace(
                ctx, baseline.logits, probe.disabled_invariance_tokens, residual_capture);

        auto replay_initial = evaluate_prompt(
                ctx,
                vocab,
                prompt_tokens,
                requested_ids,
                residual_capture,
                probe.include_residual_vector,
                probe.fibonacci_pool_max_tokens);
        auto replay_trace = finish_disabled_invariance_trace(
                ctx,
                std::move(replay_initial.logits),
                probe.disabled_invariance_tokens,
                residual_capture);
        require_trace_identity(baseline_trace, replay_trace, "no-API replay control");

        apply_control_vector(ctx, nullptr, layer_start, layer_end, cvec_full_size);
        auto rebuild_initial = evaluate_prompt(
                ctx,
                vocab,
                prompt_tokens,
                requested_ids,
                residual_capture,
                probe.include_residual_vector,
                probe.fibonacci_pool_max_tokens);
        auto rebuild_trace = finish_disabled_invariance_trace(
                ctx,
                std::move(rebuild_initial.logits),
                probe.disabled_invariance_tokens,
                residual_capture);
        require_trace_identity(baseline_trace, rebuild_trace, "no-artifact graph-rebuild control");

        const auto enabled = compose_control_vectors(
                nullptr, loaded_vectors.front().data, 1.0f, cvec_full_size);
        apply_control_vector(ctx, &enabled, layer_start, layer_end, cvec_full_size);
        apply_control_vector(ctx, nullptr, layer_start, layer_end, cvec_full_size);

        auto disabled_initial = evaluate_prompt(
                ctx,
                vocab,
                prompt_tokens,
                requested_ids,
                residual_capture,
                probe.include_residual_vector,
                probe.fibonacci_pool_max_tokens);
        auto disabled_trace = finish_disabled_invariance_trace(
                ctx,
                std::move(disabled_initial.logits),
                probe.disabled_invariance_tokens,
                residual_capture);
        require_trace_identity(baseline_trace, disabled_trace, "disabled controller");

        json tokens = json::array();
        for (llama_token token : baseline_trace.sampled_tokens) {
            tokens.push_back({
                { "id", token },
                { "piece", common_token_to_piece(vocab, token, true) },
            });
        }
        disabled_invariance = {
            { "status", "pass" },
            { "artifact", loaded_vectors.front().spec.path },
            { "disable_api", "llama_set_adapter_cvec(data=null)" },
            { "sampler", "greedy_top_1_lowest_token_id_tie_break" },
            { "steps", probe.disabled_invariance_tokens },
            { "sampled_tokens", std::move(tokens) },
            { "controls", {
                { "no_api_replay", "pass" },
                { "no_artifact_graph_rebuild", "pass" },
            } },
            { "logit_values_per_step", n_vocab },
            { "logit_bytes_compared", probe.disabled_invariance_tokens *
                    static_cast<size_t>(n_vocab) * sizeof(float) },
            { "sequence_state_bytes_compared", baseline_trace.sequence_state.size() },
            { "logits_bit_identical", true },
            { "sampled_tokens_identical", true },
            { "sequence_state_bit_identical", true },
        };
    }

    json sequence_lifecycle;
    if (probe.sequence_lifecycle_tokens > 0) {
        const auto enabled = compose_control_vectors(
                nullptr, loaded_vectors.front().data, 1.0f, cvec_full_size);
        apply_control_vector(ctx, &enabled, layer_start, layer_end, cvec_full_size);
        sequence_lifecycle = verify_sequence_lifecycle(
                ctx, prompt_tokens, probe.sequence_lifecycle_tokens);
        sequence_lifecycle["artifact"] = loaded_vectors.front().spec.path;
    }

    char description[512] = {};
    llama_model_desc(model, description, sizeof(description));

    json output;
    output["schema"] = "treebeard.jspace.logit_probe.v1";
    output["model"] = {
        { "path", params.model.path },
        { "runner_verified_sha256", probe.verified_model_sha256.empty()
                ? json(nullptr) : json(probe.verified_model_sha256) },
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
        { "residual_tensor", residual_capture.tensor_name },
        { "state_reset", "llama_memory_clear(data=true) before every evaluation" },
        { "warmup", (probe.disabled_invariance_tokens > 0 || probe.sequence_lifecycle_tokens > 0) ?
                "common empty run before invariant baseline" : "disabled" },
    };
    if (probe.fibonacci_pool_max_tokens > 0) {
        output["probe"]["fibonacci_pool"] = {
            { "requested_max_tokens", probe.fibonacci_pool_max_tokens },
            { "requested_horizons", jspace::fibonacci_horizons_up_to(
                    probe.fibonacci_pool_max_tokens) },
            { "emitted_horizons", jspace::fibonacci_horizons_up_to(
                    std::min(prompt_tokens.size(), probe.fibonacci_pool_max_tokens)) },
            { "domain", "post-final-block residual columns" },
        };
    }

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
    output["baseline"] = baseline.output;
    if (!disabled_invariance.is_null()) {
        output["disabled_invariance"] = std::move(disabled_invariance);
    }
    if (!sequence_lifecycle.is_null()) {
        output["sequence_lifecycle"] = std::move(sequence_lifecycle);
    }
    output["sweeps"] = json::array();

    for (const auto & loaded : loaded_vectors) {
        json sweep = {
            { "name", loaded.spec.name },
            { "path", loaded.spec.path },
            { "artifact_identity", {
                { "schema", loaded.identity.schema },
                { "axis", loaded.identity.axis },
                { "base_model_sha256", loaded.identity.base_model_sha256 },
                { "direction_normalization", loaded.identity.direction_normalization },
            } },
            { "runs", json::array() },
        };

        for (float strength : probe.probe_strengths) {
            json result_json;
            divergence_metrics divergence;
            bool reused_baseline = strength == 0.0f;
            if (reused_baseline) {
                result_json = baseline.output;
            } else {
                const auto composed = compose_control_vectors(
                        base_cvec_ptr, loaded.data, strength, cvec_full_size);
                apply_control_vector(ctx, &composed, layer_start, layer_end, cvec_full_size);
                auto result = evaluate_prompt(
                        ctx,
                        vocab,
                        prompt_tokens,
                        requested_ids,
                        residual_capture,
                        probe.include_residual_vector,
                        probe.fibonacci_pool_max_tokens);
                divergence = distribution_divergence(
                        baseline.logits, baseline.log_z, result.logits, result.log_z);
                result_json = std::move(result.output);
            }

            json run = {
                { "strength", strength },
                { "reused_baseline", reused_baseline },
                { "logsumexp", result_json.at("logsumexp") },
                { "collateral", divergence_to_json(divergence) },
                { "residual", result_json.at("residual") },
                { "tokens", result_json.at("tokens") },
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
        if (!probe.probe_vectors.empty() && probe.verified_model_sha256.empty()) {
            throw std::invalid_argument(
                    "--verified-model-sha256 is required whenever --probe-vector is used");
        }
        if (probe.disabled_invariance_tokens > 0) {
            if (probe.probe_vectors.size() != 1) {
                throw std::invalid_argument(
                        "--verify-disabled-invariance requires exactly one --probe-vector");
            }
            if (!params.control_vectors.empty()) {
                throw std::invalid_argument(
                        "--verify-disabled-invariance cannot be combined with base control vectors");
            }
        }
        if (probe.sequence_lifecycle_tokens > 0) {
            if (probe.probe_vectors.size() != 1) {
                throw std::invalid_argument(
                        "--verify-sequence-lifecycle requires exactly one --probe-vector");
            }
            if (!params.control_vectors.empty()) {
                throw std::invalid_argument(
                        "--verify-sequence-lifecycle cannot be combined with base control vectors");
            }
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

        params.n_predict = 0;
        params.warmup = probe.disabled_invariance_tokens > 0 || probe.sequence_lifecycle_tokens > 0;
        if (probe.sequence_lifecycle_tokens > 0) {
            params.n_parallel = std::max(params.n_parallel, 2);
        }

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
