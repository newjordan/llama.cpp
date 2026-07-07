#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

struct common_speculative_mtp_policy_choice {
    int index = 0;

    float score = 0.0f;
    float expected_accept = 0.0f;
    float route_cost = 0.0f;
};

struct common_speculative_mtp_route_head {
    int32_t hidden_size = 0;
    int32_t vocab_size  = 0;
    int32_t rank        = 0;

    float scale = 1.0f;
    float route_cost_target_scale = 1.0f;
    std::string route_cost_output_transform = "identity";
    std::string route_cost_target = "unknown";

    std::vector<float> context_weight; // [rank, hidden_size], row-major
    std::vector<float> context_bias;   // [rank]
    std::vector<float> token_embed;    // [vocab_size, rank], row-major
    std::vector<float> token_bias;     // [vocab_size]

    bool loaded() const {
        return hidden_size > 0 && vocab_size > 0 && rank > 0 &&
            context_weight.size() == (size_t) rank * hidden_size &&
            context_bias.size() == (size_t) rank &&
            token_embed.size() == (size_t) vocab_size * rank &&
            token_bias.size() == (size_t) vocab_size;
    }

    void compute_context(const float * hidden, int32_t n_embd, std::vector<float> & out) const {
        GGML_ASSERT(loaded());
        GGML_ASSERT(hidden != nullptr);
        GGML_ASSERT(n_embd == hidden_size);

        out.resize(rank);
        for (int32_t r = 0; r < rank; ++r) {
            const float * w = context_weight.data() + (size_t) r * hidden_size;
            float sum = context_bias[r];
            for (int32_t i = 0; i < hidden_size; ++i) {
                sum += w[i] * hidden[i];
            }
            out[r] = sum;
        }
    }

    float decode_route_cost(float raw) const {
        if (route_cost_output_transform == "identity") {
            return raw;
        }
        if (route_cost_output_transform == "log1p_scaled") {
            const float target_scale = std::max(1.0f, route_cost_target_scale);
            const float log_scale = std::log1p(target_scale);
            const float x = std::min(std::max(0.0f, raw) * log_scale, 80.0f);
            return std::expm1(x);
        }
        GGML_ABORT("unsupported MTP route-cost output transform");
    }

    float estimate_from_context(const std::vector<float> & context, llama_token id) const {
        if (!loaded() || id < 0 || id >= vocab_size || (int32_t) context.size() != rank) {
            return 0.0f;
        }

        const float * e = token_embed.data() + (size_t) id * rank;
        float sum = token_bias[id];
        for (int32_t r = 0; r < rank; ++r) {
            sum += scale * context[r] * e[r];
        }
        return decode_route_cost(sum);
    }
};

static std::string common_speculative_dirname(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

static std::string common_speculative_join_path(const std::string & dir, const std::string & file) {
    if (file.empty()) {
        return dir;
    }
    if (file[0] == '/' || (file.size() > 2 && file[1] == ':')) {
        return file;
    }
    if (dir.empty() || dir == ".") {
        return file;
    }
    return dir + "/" + file;
}

static std::vector<float> common_speculative_read_f32_file(const std::string & path, size_t count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open route-head tensor: " + path);
    }

    const std::streamsize size = in.tellg();
    const std::streamsize expected = (std::streamsize) (count * sizeof(float));
    if (size != expected) {
        throw std::runtime_error(string_format(
            "route-head tensor %s has %lld bytes, expected %lld",
            path.c_str(), (long long) size, (long long) expected));
    }

    std::vector<float> data(count);
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char *>(data.data()), expected)) {
        throw std::runtime_error("failed to read route-head tensor: " + path);
    }
    return data;
}

static std::vector<int64_t> common_speculative_json_shape(const nlohmann::json & tensor) {
    std::vector<int64_t> shape;
    for (const auto & value : tensor.at("shape")) {
        shape.push_back(value.get<int64_t>());
    }
    return shape;
}

static void common_speculative_expect_shape(
        const nlohmann::json & tensor,
        const std::vector<int64_t> & expected,
        const char * name) {
    const auto shape = common_speculative_json_shape(tensor);
    if (shape != expected) {
        throw std::runtime_error(string_format("route-head tensor %s has unexpected shape", name));
    }
}

static std::vector<float> common_speculative_load_tensor(
        const nlohmann::json & tensors,
        const std::string & base_dir,
        const char * name,
        const std::vector<int64_t> & expected_shape) {
    const auto & tensor = tensors.at(name);
    common_speculative_expect_shape(tensor, expected_shape, name);

    if (tensor.at("dtype").get<std::string>() != "float32") {
        throw std::runtime_error(string_format("route-head tensor %s must be float32", name));
    }

    size_t count = 1;
    for (int64_t dim : expected_shape) {
        count *= (size_t) dim;
    }
    return common_speculative_read_f32_file(
        common_speculative_join_path(base_dir, tensor.at("path").get<std::string>()),
        count);
}

static common_speculative_mtp_route_head common_speculative_load_mtp_route_head(
        const std::string & manifest_path,
        int32_t expected_hidden_size,
        int32_t expected_vocab_size) {
    std::ifstream in(manifest_path);
    if (!in) {
        throw std::runtime_error("failed to open route-head manifest: " + manifest_path);
    }

    nlohmann::json manifest;
    in >> manifest;

    if (manifest.at("schema_version").get<int>() != 1 ||
            manifest.at("format").get<std::string>() != "ornith_mtp_route_head_low_rank_v1") {
        throw std::runtime_error("unsupported route-head manifest format: " + manifest_path);
    }

    common_speculative_mtp_route_head head;
    head.hidden_size = manifest.at("hidden_size").get<int32_t>();
    head.vocab_size  = manifest.at("vocab_size").get<int32_t>();
    head.rank        = manifest.at("rank").get<int32_t>();
    head.scale       = manifest.value("scale", std::pow((float) head.rank, -0.5f));
    head.route_cost_target_scale = manifest.value("route_cost_target_scale", 1.0f);
    head.route_cost_output_transform = manifest.value(
        "route_cost_output_transform",
        manifest.contains("route_cost_target_scale") ? "log1p_scaled" : "identity");
    head.route_cost_target = manifest.value("route_cost_target", "unknown");

    if (head.hidden_size != expected_hidden_size) {
        throw std::runtime_error(string_format(
            "route-head hidden_size=%d does not match MTP hidden size=%d",
            head.hidden_size, expected_hidden_size));
    }
    if (head.vocab_size != expected_vocab_size) {
        throw std::runtime_error(string_format(
            "route-head vocab_size=%d does not match draft vocab size=%d",
            head.vocab_size, expected_vocab_size));
    }
    if (head.rank <= 0) {
        throw std::runtime_error("route-head rank must be positive");
    }
    if (head.route_cost_target_scale < 1.0f) {
        throw std::runtime_error("route-head route_cost_target_scale must be >= 1");
    }
    if (head.route_cost_output_transform != "identity" &&
            head.route_cost_output_transform != "log1p_scaled") {
        throw std::runtime_error("unsupported route-head route_cost_output_transform: " +
                head.route_cost_output_transform);
    }

    const auto & tensors = manifest.at("tensors");
    const std::string base_dir = common_speculative_dirname(manifest_path);
    head.context_weight = common_speculative_load_tensor(
        tensors, base_dir, "route_context_proj_weight", { head.rank, head.hidden_size });
    head.context_bias = common_speculative_load_tensor(
        tensors, base_dir, "route_context_proj_bias", { head.rank });
    head.token_embed = common_speculative_load_tensor(
        tensors, base_dir, "route_token_embed_weight", { head.vocab_size, head.rank });
    auto token_bias_2d = common_speculative_load_tensor(
        tensors, base_dir, "route_token_bias_weight", { head.vocab_size, 1 });
    head.token_bias = std::move(token_bias_2d);

    GGML_ASSERT(head.loaded());
    return head;
}

static common_speculative_mtp_policy_choice common_speculative_mtp_select_candidate(
        const llama_token_data_array * cur_p,
        const common_speculative_mtp_route_head * route_head,
        const float * h_row,
        int32_t n_embd,
        std::vector<float> & route_context,
        int32_t branch_k,
        float p_min,
        float route_alpha,
        float token_alpha,
        int step) {
    common_speculative_mtp_policy_choice best;

    if (cur_p == nullptr || cur_p->size == 0) {
        return best;
    }

    const int n_candidates = std::max(1, std::min<int>((int) cur_p->size, std::max(1, branch_k)));
    float best_score = -std::numeric_limits<float>::infinity();
    const bool use_route_head = route_head != nullptr && route_head->loaded() && h_row != nullptr && route_alpha > 0.0f;
    if (use_route_head) {
        route_head->compute_context(h_row, n_embd, route_context);
    }

    for (int k = 0; k < n_candidates; ++k) {
        const auto & cand = cur_p->data[k];
        const float p = std::max(0.0f, cand.p);

        if (p < p_min) {
            continue;
        }

        GGML_UNUSED(step);
        const float route_cost = use_route_head
            ? std::max(0.0f, route_head->estimate_from_context(route_context, cand.id))
            : 0.0f;
        const float denom = 1.0f + std::max(0.0f, token_alpha) + std::max(0.0f, route_alpha) * route_cost;
        const float score = p / denom;

        if (score > best_score) {
            best_score = score;
            best.index = k;
            best.score = score;
            best.expected_accept = p;
            best.route_cost = route_cost;
        }
    }

    if (!std::isfinite(best_score)) {
        best.index = 0;
        best.score = 0.0f;
        best.expected_accept = std::max(0.0f, cur_p->data[0].p);
        best.route_cost = use_route_head
            ? route_head->estimate_from_context(route_context, cur_p->data[0].id)
            : 0.0f;
    }

    return best;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    common_speculative_mtp_route_head route_head;
    std::vector<float> route_context;

    bool is_mem_shared = false;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    struct mtp_tree_branch {
        llama_seq_id seq_id = -1;
        int32_t branch = 0;
        int32_t i_batch = -1;

        llama_tokens tokens;
        common_sampler_ptr smpl;

        float prefix_p = 1.0f;
        float expected_accept = 0.0f;
        float route_cost = 0.0f;
        float score = 0.0f;

        bool active = true;
    };

    struct mtp_tree_candidate {
        int parent = -1;
        int parent_i_batch = -1;
        int index = 0;
        llama_token id = LLAMA_TOKEN_NULL;

        float p = 0.0f;
        float route_cost = 0.0f;
        float score = 0.0f;
    };

    bool has_tree_scratch() const {
        return params.mtp_tree_width > 1;
    }

    llama_seq_id scratch_seq_id(llama_seq_id seq_id, int32_t branch) const {
        GGML_ASSERT(branch > 0 && branch < params.mtp_tree_width);
        return (llama_seq_id) n_seq + seq_id*(params.mtp_tree_width - 1) + (branch - 1);
    }

    llama_seq_id tree_seq_id(llama_seq_id seq_id, int32_t branch) const {
        GGML_ASSERT(branch >= 0 && branch < params.mtp_tree_width);
        return branch == 0 ? seq_id : scratch_seq_id(seq_id, branch);
    }

    float score_branch(float expected_accept, float route_cost, int32_t n_tokens) const {
        const float denom =
            1.0f +
            std::max(0.0f, params.mtp_token_alpha) * (float) n_tokens +
            std::max(0.0f, params.mtp_route_alpha) * std::max(0.0f, route_cost) +
            1.0e-6f;

        return expected_accept / denom;
    }

    float estimate_route_cost(const float * h_row, llama_token id) {
        if (!route_head.loaded() || h_row == nullptr || params.mtp_route_alpha <= 0.0f) {
            return 0.0f;
        }

        route_head.compute_context(h_row, n_embd, route_context);
        return std::max(0.0f, route_head.estimate_from_context(route_context, id));
    }

    void append_branch_token(mtp_tree_branch & branch, llama_token id, float p, float route_cost) const {
        branch.tokens.push_back(id);
        branch.prefix_p *= std::max(0.0f, p);
        branch.expected_accept += branch.prefix_p;
        branch.route_cost += std::max(0.0f, route_cost);
        branch.score = score_branch(branch.expected_accept, branch.route_cost, (int32_t) branch.tokens.size());
    }

    bool better_branch(const mtp_tree_branch & lhs, const mtp_tree_branch & rhs) const {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        if (lhs.expected_accept != rhs.expected_accept) {
            return lhs.expected_accept > rhs.expected_accept;
        }
        return lhs.tokens.size() > rhs.tokens.size();
    }

    mtp_tree_branch make_child_branch(
            const mtp_tree_branch & parent,
            llama_seq_id seq_id,
            int32_t branch_slot,
            const mtp_tree_candidate & cand) const {
        mtp_tree_branch child;
        child.seq_id = seq_id;
        child.branch = branch_slot;
        child.i_batch = -1;
        child.tokens = parent.tokens;
        child.smpl.reset(common_sampler_clone(parent.smpl.get()));
        child.prefix_p = parent.prefix_p;
        child.expected_accept = parent.expected_accept;
        child.route_cost = parent.route_cost;
        child.score = parent.score;
        child.active = true;

        common_sampler_accept(child.smpl.get(), cand.id, true);
        append_branch_token(child, cand.id, cand.p, cand.route_cost);

        return child;
    }

    void clear_tree_scratch(llama_seq_id seq_id) {
        if (!has_tree_scratch()) {
            return;
        }

        auto * mem_dft = llama_get_memory(params.ctx_dft);
        for (int32_t branch = 1; branch < params.mtp_tree_width; ++branch) {
            llama_memory_seq_rm(mem_dft, scratch_seq_id(seq_id, branch), -1, -1);
        }
    }

    void clear_tree_scratch() {
        if (!has_tree_scratch()) {
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            clear_tree_scratch(seq_id);
        }
    }

    void draft_tree_for_seq(
            llama_seq_id seq_id,
            common_speculative_draft_params & dp,
            int32_t root_i_batch) {
        auto * ctx_dft = params.ctx_dft;
        auto * mem_dft = llama_get_memory(ctx_dft);
        auto * root_smpl = smpls[seq_id].get();

        auto & result = *dp.result;
        result.clear();

        const int32_t max_len = std::max<int32_t>(0, std::min<int32_t>(
            std::max(0, params.n_max),
            dp.n_max > 0 ? dp.n_max : params.n_max));
        const int32_t max_depth = std::max<int32_t>(0, std::min<int32_t>(params.mtp_tree_depth, max_len));
        if (max_depth <= 0 || root_i_batch < 0) {
            return;
        }

        common_sampler_sample(root_smpl, ctx_dft, root_i_batch, true);
        const float * h_root = llama_get_embeddings_nextn_ith(ctx_dft, root_i_batch);
        const auto * cur_p = common_sampler_get_candidates(root_smpl, true);

        if (cur_p == nullptr || cur_p->size == 0) {
            return;
        }

        std::vector<mtp_tree_candidate> candidates;
        const int32_t n_candidates = std::min<int32_t>(
            (int32_t) cur_p->size,
            std::max<int32_t>(1, params.mtp_branch_k));
        candidates.reserve((size_t) n_candidates);

        for (int32_t k = 0; k < n_candidates; ++k) {
            const auto & cand = cur_p->data[k];
            const float p = std::max(0.0f, cand.p);
            if (p < params.p_min) {
                continue;
            }

            const float route_cost = estimate_route_cost(h_root, cand.id);
            candidates.push_back({
                /* .parent         = */ -1,
                /* .parent_i_batch = */ root_i_batch,
                /* .index          = */ k,
                /* .id             = */ cand.id,
                /* .p              = */ p,
                /* .route_cost     = */ route_cost,
                /* .score          = */ score_branch(p, route_cost, 1),
            });
        }

        if (candidates.empty()) {
            return;
        }

        std::sort(candidates.begin(), candidates.end(), [](const mtp_tree_candidate & lhs, const mtp_tree_candidate & rhs) {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.p > rhs.p;
        });

        const int32_t n_branches = std::min<int32_t>((int32_t) candidates.size(), params.mtp_tree_width);
        std::vector<mtp_tree_branch> branches;
        branches.reserve((size_t) n_branches);

        clear_tree_scratch(seq_id);

        common_batch_clear(batch);
        std::vector<int32_t> decode_branch_indices;
        decode_branch_indices.reserve((size_t) n_branches);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (int32_t b = 0; b < n_branches; ++b) {
            const auto & cand = candidates[b];

            mtp_tree_branch branch;
            branch.branch = b;
            branch.seq_id = tree_seq_id(seq_id, b);
            branch.i_batch = root_i_batch;
            branch.smpl.reset(common_sampler_clone(root_smpl));
            common_sampler_accept(branch.smpl.get(), cand.id, true);
            append_branch_token(branch, cand.id, cand.p, cand.route_cost);

            if (b > 0) {
                llama_memory_seq_rm(mem_dft, branch.seq_id, -1, -1);
                llama_memory_seq_cp(mem_dft, seq_id, branch.seq_id, -1, -1);
            }

            LOG_DBG(" - seq_id %d, MTP tree root branch %d selected candidate %d, score %.6f, p %.6f, route_cost %.3f\n",
                    seq_id, b, cand.index, branch.score, cand.p, cand.route_cost);

            branches.push_back(std::move(branch));

            if (max_depth > 1) {
                const auto & state = branches.back();
                const llama_pos pos = is_mem_shared ? dp.n_past : dp.n_past + (llama_pos) state.tokens.size();
                common_batch_add(batch, state.tokens.back(), pos, { state.seq_id }, true);
                std::memcpy(batch.embd + (size_t) n_embd*(batch.n_tokens - 1), h_root, row_bytes);
                decode_branch_indices.push_back((int32_t) branches.size() - 1);
            }
        }

        if (batch.n_tokens > 0) {
            const int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode initial tree branches returned %d\n", __func__, ret);
                batch.n_tokens = 0;
            } else {
                for (int32_t i = 0; i < (int32_t) decode_branch_indices.size(); ++i) {
                    branches[decode_branch_indices[i]].i_batch = i;
                }
            }
        }

        for (int32_t depth = 1; depth < max_depth; ++depth) {
            std::vector<mtp_tree_candidate> expansions;
            expansions.reserve((size_t) branches.size() * std::max<int32_t>(1, params.mtp_branch_k));

            for (int32_t b = 0; b < (int32_t) branches.size(); ++b) {
                auto & branch = branches[b];
                if (!branch.active || (int32_t) branch.tokens.size() >= max_depth || branch.i_batch < 0) {
                    branch.active = false;
                    continue;
                }

                common_sampler_sample(branch.smpl.get(), ctx_dft, branch.i_batch, true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, branch.i_batch);
                const auto * branch_p = common_sampler_get_candidates(branch.smpl.get(), true);
                if (branch_p == nullptr || branch_p->size == 0) {
                    branch.active = false;
                    continue;
                }

                const int32_t n_candidates = std::min<int32_t>(
                    (int32_t) branch_p->size,
                    std::max<int32_t>(1, params.mtp_branch_k));

                for (int32_t k = 0; k < n_candidates; ++k) {
                    const auto & cand = branch_p->data[k];
                    const float p = std::max(0.0f, cand.p);
                    if (p < params.p_min) {
                        continue;
                    }

                    const float route_cost = estimate_route_cost(h_row, cand.id);
                    const float child_prefix = branch.prefix_p * p;
                    const float child_expected = branch.expected_accept + child_prefix;
                    const float child_route_cost = branch.route_cost + route_cost;
                    expansions.push_back({
                        /* .parent         = */ b,
                        /* .parent_i_batch = */ branch.i_batch,
                        /* .index          = */ k,
                        /* .id             = */ cand.id,
                        /* .p              = */ p,
                        /* .route_cost     = */ route_cost,
                        /* .score          = */ score_branch(child_expected, child_route_cost, (int32_t) branch.tokens.size() + 1),
                    });
                }
            }

            if (expansions.empty()) {
                break;
            }

            std::sort(expansions.begin(), expansions.end(), [](const mtp_tree_candidate & lhs, const mtp_tree_candidate & rhs) {
                if (lhs.score != rhs.score) {
                    return lhs.score > rhs.score;
                }
                return lhs.p > rhs.p;
            });
            if ((int32_t) expansions.size() > params.mtp_tree_width) {
                expansions.resize(params.mtp_tree_width);
            }

            std::vector<int32_t> assigned_slot(expansions.size(), -1);
            std::vector<bool> slot_used(params.mtp_tree_width, false);

            for (int32_t i = 0; i < (int32_t) expansions.size(); ++i) {
                const int32_t parent = expansions[i].parent;
                GGML_ASSERT(parent >= 0 && parent < (int32_t) branches.size());
                const int32_t parent_slot = branches[parent].branch;
                if (parent_slot >= 0 && parent_slot < params.mtp_tree_width && !slot_used[parent_slot]) {
                    assigned_slot[i] = parent_slot;
                    slot_used[parent_slot] = true;
                }
            }

            for (int32_t i = 0; i < (int32_t) expansions.size(); ++i) {
                if (assigned_slot[i] >= 0) {
                    continue;
                }
                for (int32_t slot = 0; slot < params.mtp_tree_width; ++slot) {
                    if (!slot_used[slot]) {
                        assigned_slot[i] = slot;
                        slot_used[slot] = true;
                        break;
                    }
                }
                GGML_ASSERT(assigned_slot[i] >= 0);
            }

            std::vector<mtp_tree_branch> next_branches;
            next_branches.reserve(expansions.size());

            common_batch_clear(batch);
            decode_branch_indices.clear();

            for (int32_t i = 0; i < (int32_t) expansions.size(); ++i) {
                const auto & exp = expansions[i];
                const auto & parent = branches[exp.parent];
                const int32_t branch_slot = assigned_slot[i];
                const llama_seq_id dst_seq = tree_seq_id(seq_id, branch_slot);

                if (dst_seq != parent.seq_id) {
                    llama_memory_seq_rm(mem_dft, dst_seq, -1, -1);
                    llama_memory_seq_cp(mem_dft, parent.seq_id, dst_seq, -1, -1);
                }

                auto child = make_child_branch(parent, dst_seq, branch_slot, exp);

                if ((int32_t) child.tokens.size() < max_depth) {
                    const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, exp.parent_i_batch);
                    const llama_pos pos = is_mem_shared ? dp.n_past : dp.n_past + (llama_pos) child.tokens.size();
                    common_batch_add(batch, child.tokens.back(), pos, { child.seq_id }, true);
                    std::memcpy(batch.embd + (size_t) n_embd*(batch.n_tokens - 1), h_row, row_bytes);
                    decode_branch_indices.push_back((int32_t) next_branches.size());
                } else {
                    child.active = false;
                }

                next_branches.push_back(std::move(child));
            }

            if (batch.n_tokens == 0) {
                branches = std::move(next_branches);
                break;
            }

            const int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode tree depth %d returned %d\n", __func__, depth, ret);
                break;
            }

            for (int32_t i = 0; i < (int32_t) decode_branch_indices.size(); ++i) {
                next_branches[decode_branch_indices[i]].i_batch = i;
            }

            branches = std::move(next_branches);
        }

        if (branches.empty()) {
            return;
        }

        int32_t best = 0;
        for (int32_t i = 1; i < (int32_t) branches.size(); ++i) {
            if (better_branch(branches[i], branches[best])) {
                best = i;
            }
        }

        result = branches[best].tokens;
        LOG_DBG(" - seq_id %d, MTP tree selected branch %d, tokens=%zu, score %.6f, expected_accept %.6f, route_cost %.3f\n",
                seq_id, branches[best].branch, result.size(), branches[best].score,
                branches[best].expected_accept, branches[best].route_cost);
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");

        this->params.mtp_branch_k = std::max(1, this->params.mtp_branch_k);
        this->params.mtp_tree_width = std::max(1, this->params.mtp_tree_width);
        const int32_t n_max = std::max(0, this->params.n_max);
        if (this->params.mtp_tree_depth <= 0) {
            this->params.mtp_tree_depth = n_max;
        } else {
            this->params.mtp_tree_depth = std::min(this->params.mtp_tree_depth, n_max);
        }

        const uint32_t ctx_dft_n_seq = llama_n_seq_max(ctx_dft);
        const uint64_t required_n_seq = (uint64_t) n_seq * (uint64_t) this->params.mtp_tree_width;
        if (ctx_dft_n_seq < required_n_seq) {
            throw std::runtime_error(string_format(
                "MTP tree width requires ctx_dft n_seq_max >= %u*%d = %" PRIu64 ", got %u",
                n_seq, this->params.mtp_tree_width, required_n_seq, ctx_dft_n_seq));
        }

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - mtp_branch_k=%d, mtp_tree_width=%d, mtp_tree_depth=%d, mtp_route_alpha=%.3f, mtp_token_alpha=%.3f\n", __func__,
                this->params.mtp_branch_k, this->params.mtp_tree_width, this->params.mtp_tree_depth,
                this->params.mtp_route_alpha, this->params.mtp_token_alpha);
        LOG_INF("%s: - ctx_dft n_seq_max=%u, active n_seq=%u, reserved MTP branch seqs=%" PRIu64 "\n", __func__,
                ctx_dft_n_seq, n_seq, required_n_seq);
        if (!this->params.mtp_route_head.empty()) {
            const llama_vocab * vocab_dft = llama_model_get_vocab(llama_get_model(ctx_dft));
            route_head = common_speculative_load_mtp_route_head(
                this->params.mtp_route_head,
                n_embd,
                llama_vocab_n_tokens(vocab_dft));
            LOG_INF("%s: - loaded MTP route head rank=%d target=%s transform=%s from %s\n", __func__,
                    route_head.rank,
                    route_head.route_cost_target.c_str(),
                    route_head.route_cost_output_transform.c_str(),
                    this->params.mtp_route_head.c_str());
        } else if (this->params.mtp_route_alpha > 0.0f) {
            LOG_WRN("%s: mtp_route_alpha > 0 but no route-head manifest was provided; route cost stays zero\n", __func__);
        }
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);

        clear_tree_scratch();
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (has_tree_scratch()) {
            for (int32_t branch = 1; branch < params.mtp_tree_width; ++branch) {
                llama_memory_seq_rm(llama_get_memory(params.ctx_dft), scratch_seq_id(seq_id, branch), -1, -1);
            }
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);
        std::vector<int32_t> root_i_batch(n_seq, -1);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            root_i_batch[seq_id] = batch.n_tokens;
            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        if (has_tree_scratch()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                draft_tree_for_seq(seq_id, dparams.at(seq_id), root_i_batch[seq_id]);
            }

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting) {
                    continue;
                }

                if (dp.result->size() < (size_t) params.n_min) {
                    dp.result->clear();
                }

                last_n_drafted[seq_id] = (uint16_t) dp.result->size();
            }

            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const auto choice = common_speculative_mtp_select_candidate(
                    cur_p,
                    route_head.loaded() ? &route_head : nullptr,
                    h_row,
                    n_embd,
                    route_context,
                    params.mtp_branch_k,
                    params.p_min,
                    params.mtp_route_alpha,
                    params.mtp_token_alpha,
                    i
                );
                const auto & selected = cur_p->data[choice.index];

                if (choice.index != 0) {
                    LOG_DBG(" - seq_id %d, MTP branch policy selected candidate %d, score %.6f, p %.6f, route_cost %.3f\n",
                            seq_id, choice.index, choice.score, selected.p, choice.route_cost);
                }

                // add drafted token for each sequence
                const llama_token id = selected.id;

                // only collect very high-confidence draft tokens
                if (selected.p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                }
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);

        if (has_tree_scratch()) {
            for (int32_t branch = 1; branch < params.mtp_tree_width; ++branch) {
                llama_memory_seq_rm(llama_get_memory(params.ctx_dft), scratch_seq_id(seq_id, branch), -1, -1);
            }
        }
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 9);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
