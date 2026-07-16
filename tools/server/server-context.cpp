#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-pcbt-parse.h"
#include "server-queue.h"
#include "server-snapshot-store.h"
#include "server-snapshot-manifest.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "ggml-cpp.h"

extern "C" {
#include "sha256.h"
}

// TODO: tmp until the mtmd draft processing is refactored [TAG_MTMD_DRAFT_PROCESSING]
#include "../../src/llama-ext.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <filesystem>
#include <utility>
#include <fstream>
#include <limits>
#include <numeric>
#include <map>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <thread>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_MILLISECONDS = 1000;
constexpr int DURABLE_IO_POLLING_MILLISECONDS = 10;
constexpr size_t SERVER_STATETREE_JOURNAL_CAPACITY = 1024;

struct server_sha256 {
    server_sha256() {
#if !defined(_WIN32)
        const auto & api = openssl_api();
        if (api.available()) {
            openssl_ctx = api.ctx_new();
            if (openssl_ctx != nullptr && api.digest_init(openssl_ctx, api.sha256(), nullptr) == 1) {
                use_openssl = true;
                return;
            }
            if (openssl_ctx != nullptr) {
                api.ctx_free(openssl_ctx);
                openssl_ctx = nullptr;
            }
        }
#endif
        sha256_init(&fallback);
    }

    ~server_sha256() {
#if !defined(_WIN32)
        if (openssl_ctx != nullptr) {
            openssl_api().ctx_free(openssl_ctx);
        }
#endif
    }

    void update(const void * data, size_t size) {
        if (size == 0) {
            return;
        }
#if !defined(_WIN32)
        if (use_openssl) {
            GGML_ASSERT(openssl_api().digest_update(openssl_ctx, data, size) == 1);
            return;
        }
#endif
        sha256_update(&fallback, static_cast<const unsigned char *>(data), size);
    }

    void final(unsigned char digest[SHA256_DIGEST_SIZE]) {
#if !defined(_WIN32)
        if (use_openssl) {
            unsigned int size = 0;
            GGML_ASSERT(openssl_api().digest_final(openssl_ctx, digest, &size) == 1);
            GGML_ASSERT(size == SHA256_DIGEST_SIZE);
            return;
        }
#endif
        sha256_final(&fallback, digest);
    }

private:
#if !defined(_WIN32)
    struct dynamic_openssl {
        using ctx_new_t = void * (*)();
        using ctx_free_t = void (*)(void *);
        using sha256_t = const void * (*)();
        using digest_init_t = int (*)(void *, const void *, void *);
        using digest_update_t = int (*)(void *, const void *, size_t);
        using digest_final_t = int (*)(void *, unsigned char *, unsigned int *);

        void * handle = nullptr;
        ctx_new_t ctx_new = nullptr;
        ctx_free_t ctx_free = nullptr;
        sha256_t sha256 = nullptr;
        digest_init_t digest_init = nullptr;
        digest_update_t digest_update = nullptr;
        digest_final_t digest_final = nullptr;

        dynamic_openssl() {
            static constexpr const char * names[] = {
                "libcrypto.so.3",
                "libcrypto.so.1.1",
                "libcrypto.so",
            };
            for (const char * name : names) {
                handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
                if (handle != nullptr) {
                    break;
                }
            }
            if (handle == nullptr) {
                return;
            }
            ctx_new = reinterpret_cast<ctx_new_t>(dlsym(handle, "EVP_MD_CTX_new"));
            ctx_free = reinterpret_cast<ctx_free_t>(dlsym(handle, "EVP_MD_CTX_free"));
            sha256 = reinterpret_cast<sha256_t>(dlsym(handle, "EVP_sha256"));
            digest_init = reinterpret_cast<digest_init_t>(dlsym(handle, "EVP_DigestInit_ex"));
            digest_update = reinterpret_cast<digest_update_t>(dlsym(handle, "EVP_DigestUpdate"));
            digest_final = reinterpret_cast<digest_final_t>(dlsym(handle, "EVP_DigestFinal_ex"));
            if (!available()) {
                dlclose(handle);
                handle = nullptr;
            }
        }

        ~dynamic_openssl() {
            if (handle != nullptr) {
                dlclose(handle);
            }
        }

        bool available() const {
            return handle != nullptr && ctx_new != nullptr && ctx_free != nullptr && sha256 != nullptr &&
                digest_init != nullptr && digest_update != nullptr && digest_final != nullptr;
        }
    };

    static const dynamic_openssl & openssl_api() {
        static const dynamic_openssl api;
        return api;
    }

    void * openssl_ctx = nullptr;
    bool use_openssl = false;
#endif
    sha256_t fallback {};
};

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

static uint32_t server_mtp_tree_width(const common_params & params) {
    return std::max<int32_t>(1, params.speculative.draft.mtp_tree_width);
}

static uint32_t server_mtp_n_seq_max(const common_params & params) {
    const uint64_t n_seq = (uint64_t) std::max<int32_t>(1, params.n_parallel) * server_mtp_tree_width(params);
    return (uint32_t) std::min<uint64_t>(n_seq, (uint64_t) std::numeric_limits<uint32_t>::max());
}

static int64_t server_optional_fork_id(const json & data) {
    if (!data.is_object()) {
        throw std::invalid_argument("request body must be an object");
    }
    if (!data.contains("fork_id")) {
        return -1;
    }
    if (!data.at("fork_id").is_number_integer()) {
        throw std::invalid_argument("fork_id must be a non-negative integer");
    }
    const int64_t fork_id = data.at("fork_id").get<int64_t>();
    if (fork_id < 0) {
        throw std::invalid_argument("fork_id must be a non-negative integer");
    }
    return fork_id;
}

static int64_t server_optional_state_id(const json & data) {
    if (!data.is_object()) {
        throw std::invalid_argument("request body must be an object");
    }
    if (!data.contains("state_id")) {
        return -1;
    }
    if (!data.at("state_id").is_number_integer()) {
        throw std::invalid_argument("state_id must be a non-negative integer");
    }
    const int64_t state_id = data.at("state_id").get<int64_t>();
    if (state_id < 0) {
        throw std::invalid_argument("state_id must be a non-negative integer");
    }
    return state_id;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    llama_tokens spec_serial;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    // A forked prompt remains idle but is unavailable to automatic scheduling.
    int fork_source_id = -1;
    int64_t fork_id    = -1;
    int64_t state_id   = -1;
    int64_t node_id    = -1;
    int64_t parent_node_id = -1;
    int64_t materialized_snapshot_id = -1;
    std::string materialized_content_digest;
    uint64_t retention_touch = 0;
    int64_t lease_deadline_us = -1;

    server_prompt prompt;

    void prompt_metadata_clear() {
        prompt.tokens.clear();
        prompt.data.main.clear();
        prompt.data.drft.clear();
        prompt.checkpoints.clear();
        fork_source_id = -1;
        fork_id = -1;
        state_id = -1;
        node_id = -1;
        parent_node_id = -1;
        materialized_snapshot_id = -1;
        materialized_content_digest.clear();
        retention_touch = 0;
        lease_deadline_us = -1;
    }

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    bool prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        const bool was_fork_reserved = is_fork_reserved();
        prompt_metadata_clear();
        return was_fork_reserved;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;
    std::function<void(int /* id_slot */)> callback_on_deferred;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    std::vector<int32_t> n_draft_per_round;
    std::vector<int32_t> n_draft_accepted_per_round;
    int32_t n_draft_anchor = 0;
    int32_t n_draft_anchor_match = 0;
    int32_t n_draft_anchor_fallback = 0;
    std::vector<llama_token> draft_anchor_serial_tokens;
    std::vector<llama_token> draft_anchor_batched_tokens;
    int32_t n_draft_audit_tokens = 0;
    int32_t n_draft_audit_tokens_matched = 0;
    int32_t n_draft_audit_fallback = 0;
    std::vector<int32_t> draft_audit_first_mismatch;
    std::vector<llama_token> draft_audit_serial_tokens;
    std::vector<llama_token> draft_audit_batched_tokens;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
            spec_serial.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        n_draft_per_round.clear();
        n_draft_accepted_per_round.clear();
        n_draft_anchor = 0;
        n_draft_anchor_match = 0;
        n_draft_anchor_fallback = 0;
        draft_anchor_serial_tokens.clear();
        draft_anchor_batched_tokens.clear();
        n_draft_audit_tokens = 0;
        n_draft_audit_tokens_matched = 0;
        n_draft_audit_fallback = 0;
        draft_audit_first_mismatch.clear();
        draft_audit_serial_tokens.clear();
        draft_audit_batched_tokens.clear();

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_nextn(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool is_fork_reserved() const {
        return fork_source_id >= 0;
    }

    bool is_fork_root() const {
        return fork_source_id == id;
    }

    size_t prompt_state_bytes() const {
        return prompt.size();
    }

    bool is_available() const {
        return !is_processing() && !is_fork_reserved();
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        if (task->params.speculative_n_max >= 0) {
            n_draft_max = std::min(n_draft_max, task->params.speculative_n_max);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
            timings.draft_n_per_round = n_draft_per_round;
            timings.draft_n_accepted_per_round = n_draft_accepted_per_round;
        }
        if (n_draft_anchor > 0) {
            timings.draft_anchor_n = n_draft_anchor;
            timings.draft_anchor_match_n = n_draft_anchor_match;
            timings.draft_anchor_fallback_n = n_draft_anchor_fallback;
            timings.draft_anchor_serial_tokens = draft_anchor_serial_tokens;
            timings.draft_anchor_batched_tokens = draft_anchor_batched_tokens;
            timings.draft_audit_tokens = n_draft_audit_tokens;
            timings.draft_audit_tokens_matched = n_draft_audit_tokens_matched;
            timings.draft_audit_fallback_n = n_draft_audit_fallback;
            timings.draft_audit_first_mismatch = draft_audit_first_mismatch;
            timings.draft_audit_serial_tokens = draft_audit_serial_tokens;
            timings.draft_audit_batched_tokens = draft_audit_batched_tokens;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total);
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false, bool lease_pinned = false) const {
        json res;
        const size_t prompt_data_bytes       = prompt.data.size();
        const size_t prompt_checkpoint_bytes = prompt.checkpoint_size();
        const int64_t now_us = ggml_time_us();
        lease_pinned = is_fork_reserved() && (lease_pinned || is_processing());
        const int64_t lease_remaining_ms = lease_pinned || lease_deadline_us < 0
            ? -1
            : std::max<int64_t>(0, (lease_deadline_us - now_us + 999) / 1000);

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
            {"is_reserved",   is_fork_reserved()},
            {"fork_source_id", fork_source_id},
            {"fork_id",        fork_id},
            {"state_id",       state_id},
            {"node_id",        node_id},
            {"parent_node_id", parent_node_id},
            {"retention_touch", retention_touch},
            {"lease_pinned", lease_pinned},
            {"lease_remaining_ms", lease_remaining_ms},
            {"lease_expired", !lease_pinned && lease_deadline_us >= 0 && lease_deadline_us <= now_us},
            {"jspace_control_scale", llama_adapter_cvec_seq_mode(ctx_tgt)
                ? json(llama_adapter_cvec_seq_get(ctx_tgt, id))
                : json(nullptr)},
            {"n_prompt_checkpoints", prompt.checkpoints.size()},
            {"n_prompt_data_bytes", prompt_data_bytes},
            {"n_prompt_checkpoint_bytes", prompt_checkpoint_bytes},
            {"n_prompt_state_bytes", prompt_data_bytes + prompt_checkpoint_bytes},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other, bool copy_prompt_state) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        if (copy_prompt_state) {
            other.prompt = prompt.clone();
        } else {
            server_prompt prompt_without_state;
            prompt_without_state.tokens = prompt.tokens.clone();
            other.prompt = std::move(prompt_without_state);
        }
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // Zero-copy StateTree fork of `source`'s evaluated prefix into the
    // validated `destinations` (idle, unique, source excluded). On success
    // every family member (source included) carries a fresh node id under
    // one new fork generation. On failure `error` is set and nothing was
    // mutated. Extracted from SERVER_TASK_TYPE_SLOT_FORK so PCBT create
    // reuses the exact fork semantics (PCBT-3).
    struct statetree_fork_family_result {
        int64_t state_id       = -1;
        int64_t fork_id        = -1;
        int64_t parent_fork_id = -1;
        int64_t parent_node_id = -1;
    };

    bool statetree_fork_family(server_slot * source,
                               const std::vector<server_slot *> & destinations,
                               statetree_fork_family_result & out,
                               std::string & error) {
        const int id_slot = source->id;

        std::vector<server_tokens> prompt_copies;
        try {
            prompt_copies.reserve(destinations.size());
            for (size_t i = 0; i < destinations.size(); ++i) {
                prompt_copies.push_back(source->prompt.tokens.clone());
            }
        } catch (const std::exception & e) {
            error = std::string("Failed to clone slot prompt: ") + e.what();
            return false;
        }

        if (statetree_next_fork_id > (uint64_t) std::numeric_limits<int64_t>::max()) {
            error = "StateTree fork generation space is exhausted";
            return false;
        }
        if (source->state_id < 0 &&
                statetree_next_state_id > (uint64_t) std::numeric_limits<int64_t>::max()) {
            error = "StateTree logical state space is exhausted";
            return false;
        }
        const size_t n_nodes = destinations.size() + 1;
        const uint64_t node_limit = (uint64_t) std::numeric_limits<int64_t>::max();
        if (statetree_next_node_id > node_limit ||
                n_nodes - 1 > node_limit - statetree_next_node_id) {
            error = "StateTree branch node space is exhausted";
            return false;
        }

        out.parent_fork_id = source->fork_id;
        out.parent_node_id = source->node_id;
        out.state_id = source->state_id >= 0
            ? source->state_id
            : (int64_t) statetree_next_state_id++;
        out.fork_id = (int64_t) statetree_next_fork_id++;

        for (size_t i = 0; i < destinations.size(); ++i) {
            server_slot * destination = destinations[i];

            destination->prompt_clear(false);
            common_context_seq_cp(ctx_tgt, id_slot, destination->id, -1, -1);

            server_prompt prompt;
            prompt.tokens = std::move(prompt_copies[i]);
            destination->prompt = std::move(prompt);
            destination->task_prev.reset();
        }

        source->prompt.checkpoints.clear();
        std::vector<server_slot *> family_members = destinations;
        family_members.push_back(source);
        std::sort(family_members.begin(), family_members.end(), [](const auto * left, const auto * right) {
            return left->id < right->id;
        });
        for (server_slot * member : family_members) {
            member->fork_source_id = id_slot;
            member->state_id = out.state_id;
            member->node_id = (int64_t) statetree_next_node_id++;
            member->parent_node_id = out.parent_node_id;
            member->materialized_snapshot_id = -1;
            member->materialized_content_digest.clear();
            member->fork_id = out.fork_id;
        }
        touch_family(id_slot, out.fork_id, false);
        return true;
    }

    // Proof-carrying branch transactions: state-thread-owned registry and
    // the task handler (invariant 1; docs/treebeard-pcbt-contract-v1.md).
    pcbt_registry pcbt_txs;

    json pcbt_transaction_view(const server_branch_transaction & tx) const {
        json branches = json::array();
        for (const auto & b : tx.branches) {
            branches.push_back({
                {"key", b.key}, {"node_id", b.node_id}, {"slot_id", b.slot_id},
                {"phase", b.phase == pcbt_branch_phase::QUEUED ? "queued"
                        : b.phase == pcbt_branch_phase::RUNNING ? "running"
                        : b.phase == pcbt_branch_phase::COMPLETED ? "completed"
                        : b.phase == pcbt_branch_phase::FAILED ? "failed" : "canceled"},
            });
        }
        return json {
            {"transaction_id", tx.id},
            {"status", pcbt_status_name(tx.status)},
            {"generation", tx.generation},
            {"deadline_unix_ms", tx.deadline_unix_ms},
            {"create_digest", tx.create_digest},
            {"branches", std::move(branches)},
        };
    }

    std::unique_ptr<server_task_result_pcbt> handle_pcbt(const server_task & task) {
        auto res = std::make_unique<server_task_result_pcbt>();
        auto fail = [&](pcbt_error e, const char * cls, std::string msg) {
            res->http_status = pcbt_error_http(e);
            res->error_class = cls;
            res->message     = std::move(msg);
        };
        using OP = server_task::pcbt_action;
        switch (task.pcbt.op) {
            case OP::CREATE: {
                const auto body = json::parse(task.pcbt.body_json, nullptr, false);
                pcbt_create_request req;
                const auto pr = pcbt_parse_create(body, req);
                if (!pr.ok()) {
                    fail(pr.error, "invalid_request", pr.message);
                    break;
                }
                // Idempotency probe is read-only; registry mutation may only
                // happen after a successful fork (invariant 2).
                const auto it = pcbt_txs.by_request_id.find(req.request_id);
                if (it != pcbt_txs.by_request_id.end()) {
                    auto & existing = pcbt_txs.by_id.at(it->second);
                    if (existing.create_digest != req.create_digest) {
                        fail(pcbt_error::CONFLICT, "conflict",
                             "request_id reused with a different body");
                        break;
                    }
                    res->payload = pcbt_transaction_view(existing);
                    break;
                }
                // Slot lifecycle (PCBT-4 scheduling, PCBT-7 expiry/abort) is
                // not wired yet, so the create surface stays opt-in for test
                // servers only.
                static const bool pcbt_enabled = []() {
                    const char * env = getenv("TREEBEARD_PCBT_ENABLE");
                    return env != nullptr && atoi(env) != 0;
                }();
                if (!pcbt_enabled) {
                    fail(pcbt_error::CAPACITY, "capacity",
                         "pcbt: transactions not yet enabled (set TREEBEARD_PCBT_ENABLE=1 on test servers)");
                    break;
                }
                if (!params_base.kv_unified || ctx_dft || spec ||
                        !params_base.lora_adapters.empty()) {
                    fail(pcbt_error::UNPROCESSABLE, "unprocessable",
                         "pcbt requires a unified KV cache without speculative decoding or LoRA");
                    break;
                }
                // Resolve the committed singleton source by immutable node id.
                server_slot * source = nullptr;
                for (server_slot & slot : slots) {
                    if (slot.node_id >= 0 && slot.node_id == (int64_t) req.source_node_id) {
                        source = &slot;
                        break;
                    }
                }
                if (source == nullptr) {
                    fail(pcbt_error::NOT_FOUND, "not_found", "unknown source node");
                    break;
                }
                if ((req.source_state_id >= 0 && source->state_id != (int64_t) req.source_state_id) ||
                        (req.source_fork_id >= 0 && source->fork_id != (int64_t) req.source_fork_id)) {
                    fail(pcbt_error::CONFLICT, "conflict", "stale source assertion");
                    break;
                }
                if (source->is_processing() || source->prompt.tokens.empty() || !source->lora.empty()) {
                    fail(pcbt_error::UNPROCESSABLE, "unprocessable",
                         "source must be idle with a cached prompt");
                    break;
                }
                // Branch 0 decodes on the source; branches 1..N-1 need idle
                // destination slots. All-or-nothing: pick every destination
                // before mutating anything.
                std::vector<server_slot *> destinations;
                for (server_slot & slot : slots) {
                    if (destinations.size() + 1 >= req.branches.size()) {
                        break;
                    }
                    if (slot.id != source->id && slot.is_available() &&
                            slot.fork_id < 0 && !slot.is_processing()) {
                        destinations.push_back(&slot);
                    }
                }
                if (destinations.size() + 1 < req.branches.size()) {
                    fail(pcbt_error::CAPACITY, "capacity",
                         "insufficient idle slots for the declared branches");
                    break;
                }
                statetree_fork_family_result fork_res;
                std::string fork_error;
                if (!statetree_fork_family(source, destinations, fork_res, fork_error)) {
                    fail(pcbt_error::UNPROCESSABLE, "unprocessable", fork_error);
                    break;
                }
                // Fork succeeded: registry mutation is now legal.
                bool conflict = false;
                auto * tx = pcbt_txs.create(req.request_id, req.create_digest, conflict);
                GGML_ASSERT(tx != nullptr && !conflict);
                tx->source_node_id  = req.source_node_id;
                tx->source_state_id = req.source_state_id;
                tx->source_fork_id  = req.source_fork_id;
                tx->generation      = (int32_t) fork_res.fork_id;
                tx->budget          = req.budget;
                tx->accepted_unix_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                tx->deadline_unix_ms = tx->accepted_unix_ms + req.budget.deadline_ms;
                tx->branches.reserve(req.branches.size());
                for (size_t i = 0; i < req.branches.size(); ++i) {
                    pcbt_branch b;
                    b.key     = req.branches[i].first;
                    server_slot * host = i == 0 ? source : destinations[i - 1];
                    b.slot_id = host->id;
                    b.node_id = (int32_t) host->node_id;
                    tx->branches.push_back(std::move(b));
                }
                json assignments = json::object();
                for (const auto & b : tx->branches) {
                    assignments[b.key] = {{"node_id", b.node_id}, {"slot_id", b.slot_id}};
                }
                tx->push_event(tx->accepted_unix_ms, "create", assignments.dump());
                res->payload = pcbt_transaction_view(*tx);
                break;
            }
            case OP::OBSERVE:
            case OP::EVENTS: {
                auto * tx = pcbt_txs.find(task.pcbt.transaction_id);
                if (tx == nullptr) {
                    fail(pcbt_error::NOT_FOUND, "not_found", "unknown transaction");
                    break;
                }
                json view = pcbt_transaction_view(*tx);
                if (task.pcbt.op == OP::EVENTS) {
                    json events = json::array();
                    for (const auto & e : tx->events) {
                        if (e.seq > task.pcbt.after_seq) {
                            events.push_back({{"seq", e.seq}, {"unix_ms", e.unix_ms},
                                              {"kind", e.kind}, {"data", e.data}});
                        }
                    }
                    view["events"]    = std::move(events);
                    view["first_seq"] = tx->first_seq;
                }
                res->payload = std::move(view);
                break;
            }
            case OP::COMMIT:
            case OP::ABORT: {
                auto * tx = pcbt_txs.find(task.pcbt.transaction_id);
                if (tx == nullptr) {
                    fail(pcbt_error::NOT_FOUND, "not_found", "unknown transaction");
                    break;
                }
                fail(pcbt_error::UNPROCESSABLE, "unprocessable",
                     "pcbt: decision handling lands with PCBT-6/7");
                break;
            }
            default:
                fail(pcbt_error::INVALID_REQUEST, "invalid_request", "unknown pcbt op");
                break;
        }
        return res;
    }

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        stop_durable_io_worker();
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;
    std::unique_ptr<server_snapshot_store> snapshot_store;
    std::unique_ptr<server_snapshot_manifest> snapshot_manifest;
    std::map<std::string, server_snapshot_store_entry> durable_store_entries;
    uint64_t durable_store_disk_bytes = 0;
    uint64_t durable_store_disk_budget_bytes = 0;
    uint64_t durable_store_disk_high_water_bytes = 0;
    uint64_t durable_store_recovered_temp_files = 0;
    uint64_t durable_store_ignored_corrupt_files = 0;
    uint64_t durable_store_runtime_integrity_failures = 0;
    uint64_t durable_store_orphaned_disk_bytes = 0;
    std::map<std::string, server_snapshot_manifest_ref> durable_manifest_refs;
    std::set<std::string> durable_managed_digests;
    std::map<std::string, server_snapshot_logical_head> durable_logical_heads;
    uint64_t durable_manifest_revision = 0;
    uint64_t durable_manifest_file_bytes = 0;
    uint64_t durable_manifest_budget_bytes = 0;
    uint64_t durable_manifest_high_water_bytes = 0;
    uint64_t durable_manifest_record_count = 0;
    uint64_t durable_manifest_recovered_temp_files = 0;
    uint64_t durable_manifest_recovered_tail_bytes = 0;
    uint64_t durable_manifest_compactions = 0;
    uint64_t durable_manifest_recovered_publish_commits = 0;
    uint64_t durable_manifest_recovered_publish_aborts = 0;
    uint64_t durable_managed_recovered_erases = 0;
    uint64_t durable_managed_recovered_bytes = 0;

    server_metrics metrics;

    struct statetree_family {
        int source_id = -1;
        int64_t state_id = -1;
        int64_t fork_id = -1;
        uint64_t touch = 0;
        int64_t lease_deadline_us = -1;
        size_t state_bytes = 0;
        bool active = false;
        std::vector<server_slot *> members;
    };

    struct statetree_node_ref {
        int id_slot = -1;
        int64_t node_id = -1;
        int64_t parent_node_id = -1;
        int64_t materialized_snapshot_id = -1;
        std::string materialized_content_digest;

        json to_json() const {
            return json {
                { "id_slot", id_slot },
                { "node_id", node_id },
                { "parent_node_id", parent_node_id },
                { "materialized_snapshot_id", materialized_snapshot_id >= 0
                    ? json(materialized_snapshot_id)
                    : json(nullptr) },
                { "materialized_content_digest", materialized_content_digest.empty()
                    ? json(nullptr)
                    : json(materialized_content_digest) },
            };
        }
    };

    struct statetree_snapshot_payload {
        llama_tokens tokens;
        std::vector<uint8_t> state;
        std::string digest;

        size_t payload_bytes() const {
            return state.size() + tokens.size() * sizeof(llama_token);
        }
    };

    struct statetree_snapshot {
        int64_t snapshot_id = -1;
        int64_t state_id = -1;
        int64_t source_node_id = -1;
        int64_t source_fork_id = -1;
        int source_slot = -1;
        int64_t captured_at_us = 0;
        std::shared_ptr<const statetree_snapshot_payload> payload;

        size_t payload_bytes() const {
            return payload->payload_bytes();
        }

        json to_json() const {
            return json {
                { "snapshot_id", snapshot_id },
                { "state_id", state_id },
                { "source_node_id", source_node_id },
                { "source_fork_id", source_fork_id },
                { "source_slot", source_slot },
                { "captured_at_us", captured_at_us },
                { "n_tokens", payload->tokens.size() },
                { "state_bytes", payload->state.size() },
                { "token_bytes", payload->tokens.size() * sizeof(llama_token) },
                { "payload_bytes", payload_bytes() },
                { "digest", payload->digest },
            };
        }
    };

    enum class durable_io_operation {
        spill,
        publish,
        publish_advance,
        load,
        erase,
        retain,
        release,
        compact,
        prune,
        head_create,
        head_advance,
        head_delete,
        head_load,
    };

    struct durable_io_job {
        uint64_t id = 0;
        durable_io_operation operation = durable_io_operation::load;
        server_task request;
        std::shared_ptr<const statetree_snapshot_payload> hot_payload;
        std::string digest;
        std::string owner;
        std::string retention_class;
        uint64_t max_load_bytes = 0;
        uint64_t disk_reservation_bytes = 0;
        uint64_t load_reservation_bytes = 0;
        uint64_t target_disk_bytes = 0;
        int64_t enqueued_us = 0;
    };

    struct durable_io_completion {
        struct cache_eviction {
            std::string digest;
            uint64_t released_refs = 0;
            uint64_t file_bytes = 0;
            bool object_erased = false;
            bool forgotten = false;
        };
        durable_io_job job;
        bool success = false;
        bool object_published = false;
        bool object_erased = false;
        bool managed_forgotten = false;
        bool unavailable = false;
        std::string error;
        server_snapshot_store_spill_result spill;
        server_snapshot_store_payload load;
        server_snapshot_store_entry erase;
        server_snapshot_manifest_result manifest;
        server_snapshot_publish_begin_result publish_begin;
        server_snapshot_publish_advance_begin_result publish_advance_begin;
        server_snapshot_publish_advance_result publish_advance;
        server_snapshot_manifest_compact_result compact;
        server_snapshot_logical_head_result head;
        std::vector<cache_eviction> cache_evictions;
        uint64_t prune_disk_bytes_before = 0;
        uint64_t prune_disk_bytes_after = 0;
        uint64_t disk_bytes = 0;
        uint64_t disk_budget_bytes = 0;
        uint64_t disk_high_water_bytes = 0;
        uint64_t recovered_temp_files = 0;
        uint64_t ignored_corrupt_files = 0;
        uint64_t runtime_integrity_failures = 0;
        uint64_t orphaned_disk_bytes = 0;
        uint64_t manifest_revision = 0;
        uint64_t manifest_file_bytes = 0;
        uint64_t manifest_budget_bytes = 0;
        uint64_t manifest_high_water_bytes = 0;
        uint64_t manifest_record_count = 0;
        uint64_t manifest_recovered_temp_files = 0;
        uint64_t manifest_recovered_tail_bytes = 0;
        uint64_t manifest_compactions = 0;
        int64_t started_us = 0;
        int64_t finished_us = 0;
    };

    struct statetree_journal_entry {
        uint64_t sequence = 0;
        int64_t timestamp_us = 0;
        std::string event;
        int64_t state_id = -1;
        int64_t fork_id = -1;
        int64_t parent_fork_id = -1;
        int source_slot = -1;
        size_t state_bytes = 0;
        std::vector<int> slots;
        std::vector<statetree_node_ref> nodes;

        json to_json() const {
            json node_values = json::array();
            for (const auto & node : nodes) {
                node_values.push_back(node.to_json());
            }
            return json {
                { "sequence", sequence },
                { "timestamp_us", timestamp_us },
                { "event", event },
                { "state_id", state_id },
                { "fork_id", fork_id },
                { "parent_fork_id", parent_fork_id },
                { "source_slot", source_slot },
                { "state_bytes", state_bytes },
                { "slots", slots },
                { "nodes", node_values },
            };
        }
    };

    struct retention_stats {
        size_t state_bytes = 0;
        size_t retained_bytes = 0;
        size_t active_bytes = 0;
        int n_families = 0;
        int n_active_families = 0;
    };

    uint64_t retention_touch_next = 0;
    uint64_t statetree_next_state_id = 0;
    uint64_t statetree_next_node_id = 0;
    uint64_t statetree_next_fork_id = 0;
    uint64_t statetree_next_snapshot_id = 0;
    uint64_t statetree_journal_next_sequence = 1;
    static constexpr size_t statetree_journal_capacity = SERVER_STATETREE_JOURNAL_CAPACITY;
    std::deque<statetree_journal_entry> statetree_journal;
    uint64_t state_high_water_bytes = 0;
    uint64_t retained_high_water_bytes = 0;
    uint64_t statetree_expired_total = 0;
    uint64_t statetree_evicted_total = 0;
    uint64_t statetree_reclaimed_bytes_total = 0;
    uint64_t statetree_renewed_total = 0;
    uint64_t statetree_pressure_rejected_total = 0;
    uint64_t snapshot_bytes = 0;
    uint64_t snapshot_high_water_bytes = 0;
    uint64_t snapshots_captured_total = 0;
    uint64_t snapshots_materialized_total = 0;
    uint64_t snapshots_erased_total = 0;
    uint64_t snapshot_rejected_total = 0;
    uint64_t durable_spilled_total = 0;
    uint64_t durable_materialized_total = 0;
    uint64_t durable_erased_total = 0;
    uint64_t durable_retained_total = 0;
    uint64_t durable_released_total = 0;
    uint64_t durable_compacted_total = 0;
    uint64_t durable_cache_evicted_total = 0;
    uint64_t durable_cache_reclaimed_bytes_total = 0;
    uint64_t durable_rejected_total = 0;
    uint64_t durable_io_completed_total = 0;
    uint64_t durable_io_cancelled_loads_total = 0;
    uint64_t durable_io_queue_high_water = 0;
    uint64_t durable_io_reserved_disk_bytes = 0;
    uint64_t durable_io_reserved_disk_high_water = 0;
    uint64_t durable_io_reserved_load_bytes = 0;
    uint64_t durable_io_reserved_load_high_water = 0;
    uint64_t durable_io_next_id = 1;
    std::unordered_set<std::string> durable_io_pending_spill_digests;
    std::atomic<uint64_t> durable_io_pending {0};
    std::thread durable_io_thread;
    std::mutex durable_io_mutex;
    std::condition_variable durable_io_condition;
    std::deque<durable_io_job> durable_io_jobs;
    std::map<uint64_t, durable_io_completion> durable_io_completions;
    bool durable_io_stopping = false;
    std::map<int64_t, statetree_snapshot> statetree_snapshots;
    std::map<std::string, std::shared_ptr<const statetree_snapshot_payload>> statetree_snapshot_contents;

    json json_ui_settings = json::object();    // Primary: new name
    json json_webui_settings = json::object();    // Deprecated: use json_ui_settings instead (kept for compat)

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    bool retention_enabled() const {
        return params_base.statetree_lease_ms > 0 || params_base.statetree_max_state_bytes > 0;
    }

    void record_statetree_event(
            const char * event,
            int64_t state_id,
            int64_t fork_id,
            int source_slot,
            std::vector<int> event_slots,
            size_t state_bytes,
            int64_t parent_fork_id = -1,
            std::vector<statetree_node_ref> event_nodes = {}) {
        if (state_id < 0) {
            return;
        }
        if (statetree_journal.size() == statetree_journal_capacity) {
            statetree_journal.pop_front();
        }
        statetree_journal.push_back(statetree_journal_entry {
            statetree_journal_next_sequence++,
            ggml_time_us(),
            event,
            state_id,
            fork_id,
            parent_fork_id,
            source_slot,
            state_bytes,
            std::move(event_slots),
            std::move(event_nodes),
        });
    }

    std::vector<statetree_node_ref> get_statetree_node_refs(
            const std::vector<server_slot *> & members) const {
        std::vector<statetree_node_ref> result;
        result.reserve(members.size());
        for (const server_slot * member : members) {
            if (member->node_id >= 0) {
                result.push_back({ member->id, member->node_id, member->parent_node_id,
                        member->materialized_snapshot_id, member->materialized_content_digest });
            }
        }
        std::sort(result.begin(), result.end(), [](const auto & left, const auto & right) {
            return left.id_slot < right.id_slot;
        });
        return result;
    }

    json get_statetree_nodes_json(const std::vector<server_slot *> & members) const {
        json result = json::array();
        for (const auto & node : get_statetree_node_refs(members)) {
            result.push_back(node.to_json());
        }
        return result;
    }

    std::vector<statetree_family> collect_statetree_families() {
        std::vector<statetree_family> families;
        for (server_slot & slot : slots) {
            if (!slot.is_fork_reserved()) {
                continue;
            }

            auto family = std::find_if(families.begin(), families.end(), [&](const statetree_family & value) {
                return value.source_id == slot.fork_source_id && value.fork_id == slot.fork_id;
            });
            if (family == families.end()) {
                statetree_family value;
                value.source_id = slot.fork_source_id;
                value.state_id = slot.state_id;
                value.fork_id = slot.fork_id;
                value.touch = slot.retention_touch;
                value.lease_deadline_us = slot.lease_deadline_us;
                families.push_back(std::move(value));
                family = std::prev(families.end());
            }
            GGML_ASSERT(family->state_id == slot.state_id);
            family->state_bytes += slot.prompt_state_bytes();
            family->active = family->active || slot.is_processing();
            family->members.push_back(&slot);
        }

        for (auto & family : families) {
            std::sort(family.members.begin(), family.members.end(), [](const server_slot * left, const server_slot * right) {
                return left->id < right->id;
            });
        }
        return families;
    }

    json get_statetree_states_json() {
        json result = json::array();
        auto families = collect_statetree_families();
        std::sort(families.begin(), families.end(), [](const statetree_family & left, const statetree_family & right) {
            return left.state_id < right.state_id;
        });
        const int64_t now_us = ggml_time_us();
        for (const auto & family : families) {
            std::vector<int> members;
            json heads = json::array();
            members.reserve(family.members.size());
            size_t n_tokens = 0;
            for (const server_slot * slot : family.members) {
                members.push_back(slot->id);
                heads.push_back(statetree_node_ref {
                    slot->id,
                    slot->node_id,
                    slot->parent_node_id,
                    slot->materialized_snapshot_id,
                    slot->materialized_content_digest,
                }.to_json());
                n_tokens += slot->prompt.tokens.size();
            }
            const bool singleton = family.members.size() == 1;
            const bool committed = singleton && family.members.front()->is_fork_root();
            result.push_back({
                { "state_id", family.state_id },
                { "fork_id", family.fork_id },
                { "source_slot", family.source_id },
                { "canonical_slot", singleton ? json(family.members.front()->id) : json(nullptr) },
                { "canonical_node_id", singleton ? json(family.members.front()->node_id) : json(nullptr) },
                { "status", committed ? "committed" : (singleton ? "single_head" : "forked") },
                { "members", members },
                { "heads", heads },
                { "n_members", members.size() },
                { "n_tokens", n_tokens },
                { "state_bytes", family.state_bytes },
                { "active", family.active },
                { "retention_touch", family.touch },
                { "lease_remaining_ms", family.active || family.lease_deadline_us < 0
                    ? -1
                    : std::max<int64_t>(0, (family.lease_deadline_us - now_us + 999) / 1000) },
            });
        }
        return result;
    }

    json get_statetree_journal_json() const {
        json result = json::array();
        for (const auto & entry : statetree_journal) {
            result.push_back(entry.to_json());
        }
        return result;
    }

    static void snapshot_hash_u64(server_sha256 & hash, uint64_t value) {
        unsigned char bytes[8];
        for (size_t i = 0; i < sizeof(bytes); ++i) {
            bytes[i] = (unsigned char) (value >> (i * 8));
        }
        hash.update(bytes, sizeof(bytes));
    }

    static constexpr size_t snapshot_cvec_trailer_size = 12;

    static void snapshot_append_cvec_scale(std::vector<uint8_t> & state, float scale) {
        static constexpr unsigned char magic[8] = { 'J', 'S', 'P', 'C', 'V', 'E', 'C', '1' };
        const size_t offset = state.size();
        state.resize(offset + snapshot_cvec_trailer_size);
        std::memcpy(state.data() + offset, magic, sizeof(magic));
        std::memcpy(state.data() + offset + sizeof(magic), &scale, sizeof(scale));
    }

    static bool snapshot_get_cvec_scale(
            const std::vector<uint8_t> & state,
            size_t &                     base_state_size,
            float &                      scale) {
        static constexpr unsigned char magic[8] = { 'J', 'S', 'P', 'C', 'V', 'E', 'C', '1' };
        base_state_size = state.size();
        if (state.size() < snapshot_cvec_trailer_size) {
            return false;
        }
        const size_t offset = state.size() - snapshot_cvec_trailer_size;
        if (std::memcmp(state.data() + offset, magic, sizeof(magic)) != 0) {
            return false;
        }
        std::memcpy(&scale, state.data() + offset + sizeof(magic), sizeof(scale));
        if (!std::isfinite(scale)) {
            throw std::runtime_error("StateTree snapshot has a non-finite J-Space scale");
        }
        base_state_size = offset;
        return true;
    }

    static std::string snapshot_content_digest(
            const llama_tokens & tokens,
            const std::vector<uint8_t> & state) {
        static constexpr unsigned char domain[] = "turbo-statetree-snapshot-v1\0";
        server_sha256 hash;
        hash.update(domain, sizeof(domain));
        snapshot_hash_u64(hash, tokens.size());
        for (const llama_token token : tokens) {
            const uint32_t value = (uint32_t) token;
            unsigned char bytes[4];
            for (size_t i = 0; i < sizeof(bytes); ++i) {
                bytes[i] = (unsigned char) (value >> (i * 8));
            }
            hash.update(bytes, sizeof(bytes));
        }
        snapshot_hash_u64(hash, state.size());
        if (!state.empty()) {
            hash.update(state.data(), state.size());
        }
        unsigned char digest[SHA256_DIGEST_SIZE];
        hash.final(digest);
        std::ostringstream encoded;
        encoded << "sha256:" << std::hex << std::setfill('0');
        for (const unsigned char byte : digest) {
            encoded << std::setw(2) << (unsigned int) byte;
        }
        return encoded.str();
    }

    json get_statetree_snapshots_json() const {
        json result = json::array();
        for (const auto & entry : statetree_snapshots) {
            result.push_back(entry.second.to_json());
        }
        return result;
    }

    json get_snapshot_store_contents_json() const {
        json result = json::array();
        if (!snapshot_store) {
            return result;
        }
        for (const auto & item : durable_store_entries) {
            const auto & entry = item.second;
            result.push_back({
                {"digest", entry.digest},
                {"n_tokens", entry.n_tokens},
                {"state_bytes", entry.state_bytes},
                {"payload_bytes", entry.payload_bytes},
                {"file_bytes", entry.file_bytes},
                {"ref_count", durable_manifest_ref_count(entry.digest)},
                {"owner_ref_count", (uint64_t) std::count_if(
                    durable_manifest_refs.begin(), durable_manifest_refs.end(), [&](const auto & item) {
                        return item.second.digest == entry.digest;
                    })},
                {"head_ref_count", (uint64_t) std::count_if(
                    durable_logical_heads.begin(), durable_logical_heads.end(), [&](const auto & item) {
                        return item.second.digest == entry.digest;
                    })},
                {"managed", durable_managed_digests.find(entry.digest) != durable_managed_digests.end()},
            });
        }
        return result;
    }

    void refresh_durable_store_cache() {
        if (!snapshot_store) {
            durable_store_entries.clear();
            durable_store_disk_bytes = 0;
            durable_store_disk_budget_bytes = 0;
            durable_store_disk_high_water_bytes = 0;
            durable_store_recovered_temp_files = 0;
            durable_store_ignored_corrupt_files = 0;
            durable_store_runtime_integrity_failures = 0;
            durable_store_orphaned_disk_bytes = 0;
            return;
        }
        durable_store_entries = snapshot_store->entries();
        durable_store_disk_bytes = snapshot_store->disk_bytes();
        durable_store_disk_budget_bytes = snapshot_store->disk_budget_bytes();
        durable_store_disk_high_water_bytes = snapshot_store->disk_high_water_bytes();
        durable_store_recovered_temp_files = snapshot_store->recovered_temp_files();
        durable_store_ignored_corrupt_files = snapshot_store->ignored_corrupt_files();
        durable_store_runtime_integrity_failures = snapshot_store->runtime_integrity_failures();
        durable_store_orphaned_disk_bytes = snapshot_store->orphaned_disk_bytes();
    }

    void refresh_durable_manifest_cache() {
        if (!snapshot_manifest) {
            durable_manifest_refs.clear();
            durable_managed_digests.clear();
            durable_logical_heads.clear();
            durable_manifest_revision = 0;
            durable_manifest_file_bytes = 0;
            durable_manifest_budget_bytes = 0;
            durable_manifest_high_water_bytes = 0;
            durable_manifest_record_count = 0;
            durable_manifest_recovered_temp_files = 0;
            durable_manifest_recovered_tail_bytes = 0;
            durable_manifest_compactions = 0;
            return;
        }
        durable_manifest_refs = snapshot_manifest->refs();
        durable_managed_digests = snapshot_manifest->managed_digests();
        durable_logical_heads = snapshot_manifest->logical_heads();
        durable_manifest_revision = snapshot_manifest->revision();
        durable_manifest_file_bytes = snapshot_manifest->file_bytes();
        durable_manifest_budget_bytes = snapshot_manifest->byte_budget();
        durable_manifest_high_water_bytes = snapshot_manifest->high_water_bytes();
        durable_manifest_record_count = snapshot_manifest->record_count();
        durable_manifest_recovered_temp_files = snapshot_manifest->recovered_temp_files();
        durable_manifest_recovered_tail_bytes = snapshot_manifest->recovered_tail_bytes();
        durable_manifest_compactions = snapshot_manifest->compactions();
    }

    uint64_t durable_manifest_ref_count(const std::string & digest) const {
        const uint64_t owner_refs = (uint64_t) std::count_if(
                durable_manifest_refs.begin(), durable_manifest_refs.end(), [&](const auto & item) {
                    return item.second.digest == digest;
                });
        const uint64_t head_refs = (uint64_t) std::count_if(
                durable_logical_heads.begin(), durable_logical_heads.end(), [&](const auto & item) {
                    return item.second.digest == digest;
                });
        return owner_refs + head_refs;
    }

    json get_snapshot_manifest_refs_json() const {
        json result = json::array();
        for (const auto & item : durable_manifest_refs) {
            const auto & ref = item.second;
            result.push_back({
                {"owner", ref.owner},
                {"digest", ref.digest},
                {"retention_class", ref.retention_class},
                {"revision", ref.revision},
            });
        }
        return result;
    }

    json get_snapshot_managed_digests_json() const {
        return json(durable_managed_digests);
    }

    json get_snapshot_logical_heads_json() const {
        json result = json::array();
        for (const auto & item : durable_logical_heads) {
            const auto & head = item.second;
            result.push_back({
                {"name", head.name},
                {"digest", head.digest},
                {"parent_digest", head.parent_digest.empty() ? json(nullptr) : json(head.parent_digest)},
                {"generation", head.generation},
                {"revision", head.revision},
            });
        }
        return result;
    }

    void apply_durable_store_cache(durable_io_completion & completion) {
        if (completion.success || completion.object_published || completion.object_erased) {
            switch (completion.job.operation) {
                case durable_io_operation::spill:
                case durable_io_operation::publish:
                case durable_io_operation::publish_advance:
                    durable_store_entries[completion.spill.entry.digest] = completion.spill.entry;
                    break;
                case durable_io_operation::load:
                    break;
                case durable_io_operation::erase:
                    durable_store_entries.erase(completion.erase.digest);
                    break;
                case durable_io_operation::retain:
                case durable_io_operation::release:
                case durable_io_operation::compact:
                case durable_io_operation::prune:
                case durable_io_operation::head_create:
                case durable_io_operation::head_advance:
                case durable_io_operation::head_delete:
                case durable_io_operation::head_load:
                    break;
            }
        }
        for (const auto & eviction : completion.cache_evictions) {
            if (eviction.object_erased) {
                durable_store_entries.erase(eviction.digest);
            }
        }
        durable_store_disk_bytes = completion.disk_bytes;
        durable_store_disk_budget_bytes = completion.disk_budget_bytes;
        durable_store_disk_high_water_bytes = completion.disk_high_water_bytes;
        durable_store_recovered_temp_files = completion.recovered_temp_files;
        durable_store_ignored_corrupt_files = completion.ignored_corrupt_files;
        durable_store_runtime_integrity_failures = completion.runtime_integrity_failures;
        durable_store_orphaned_disk_bytes = completion.orphaned_disk_bytes;
    }

    void apply_durable_manifest_cache(const durable_io_completion & completion) {
        if (completion.success) {
            switch (completion.job.operation) {
                case durable_io_operation::retain:
                case durable_io_operation::publish:
                    durable_manifest_refs[completion.manifest.ref.owner + '\0' + completion.manifest.ref.digest] =
                        completion.manifest.ref;
                    if (completion.job.operation == durable_io_operation::publish) {
                        durable_managed_digests.insert(completion.manifest.ref.digest);
                    }
                    break;
                case durable_io_operation::publish_advance:
                    durable_manifest_refs[
                        completion.publish_advance.ref.owner + '\0' +
                        completion.publish_advance.ref.digest] = completion.publish_advance.ref;
                    durable_managed_digests.insert(completion.publish_advance.ref.digest);
                    durable_logical_heads[completion.publish_advance.head.name] =
                        completion.publish_advance.head;
                    break;
                case durable_io_operation::release:
                    durable_manifest_refs.erase(
                            completion.manifest.ref.owner + '\0' + completion.manifest.ref.digest);
                    break;
                case durable_io_operation::head_create:
                case durable_io_operation::head_advance:
                    durable_logical_heads[completion.head.head.name] = completion.head.head;
                    break;
                case durable_io_operation::head_delete:
                    durable_logical_heads.erase(completion.head.head.name);
                    break;
                case durable_io_operation::spill:
                case durable_io_operation::load:
                case durable_io_operation::erase:
                case durable_io_operation::compact:
                case durable_io_operation::prune:
                case durable_io_operation::head_load:
                    break;
            }
        }
        for (const auto & eviction : completion.cache_evictions) {
            for (auto it = durable_manifest_refs.begin(); it != durable_manifest_refs.end();) {
                if (it->second.digest == eviction.digest) {
                    it = durable_manifest_refs.erase(it);
                } else {
                    ++it;
                }
            }
            if (eviction.forgotten) {
                durable_managed_digests.erase(eviction.digest);
            }
        }
        if (completion.managed_forgotten) {
            durable_managed_digests.erase(completion.job.digest);
        }
        durable_manifest_revision = completion.manifest_revision;
        durable_manifest_file_bytes = completion.manifest_file_bytes;
        durable_manifest_budget_bytes = completion.manifest_budget_bytes;
        durable_manifest_high_water_bytes = completion.manifest_high_water_bytes;
        durable_manifest_record_count = completion.manifest_record_count;
        durable_manifest_recovered_temp_files = completion.manifest_recovered_temp_files;
        durable_manifest_recovered_tail_bytes = completion.manifest_recovered_tail_bytes;
        durable_manifest_compactions = completion.manifest_compactions;
    }

    void prune_managed_cache(
            durable_io_completion & completion,
            uint64_t target_disk_bytes,
            const std::string & protected_digest = {}) {
        completion.prune_disk_bytes_before = snapshot_store->disk_bytes();
        if (completion.prune_disk_bytes_before <= target_disk_bytes) {
            completion.prune_disk_bytes_after = completion.prune_disk_bytes_before;
            return;
        }

        std::vector<server_snapshot_cache_eviction_candidate> planned;
        uint64_t reclaimable_bytes = 0;
        for (const auto & candidate : snapshot_manifest->cache_eviction_candidates()) {
            if (candidate.digest == protected_digest) {
                continue;
            }
            planned.push_back(candidate);
            server_snapshot_store_entry entry;
            if (snapshot_store->find_entry(candidate.digest, entry)) {
                reclaimable_bytes += entry.file_bytes;
            }
            if (reclaimable_bytes >= completion.prune_disk_bytes_before - target_disk_bytes) {
                break;
            }
        }
        if (reclaimable_bytes < completion.prune_disk_bytes_before - target_disk_bytes) {
            throw server_snapshot_store_error(
                    "durable disk pressure cannot be reclaimed without crossing a pinned or unmanaged fence",
                    true);
        }

        for (const auto & selected : planned) {
            if (snapshot_store->disk_bytes() <= target_disk_bytes) {
                break;
            }

            durable_io_completion::cache_eviction eviction;
            eviction.digest = selected.digest;
            server_snapshot_store_entry entry;
            const bool has_object = snapshot_store->find_entry(selected.digest, entry);
            if (has_object) {
                eviction.file_bytes = entry.file_bytes;
            }
            if (selected.refs > 0) {
                const auto evicted = snapshot_manifest->evict_cache(selected.digest);
                eviction.released_refs = evicted.released_refs;
            }
            completion.cache_evictions.push_back(eviction);
            if (has_object) {
                snapshot_store->erase(selected.digest);
                completion.cache_evictions.back().object_erased = true;
            }
            snapshot_manifest->forget_managed(selected.digest);
            completion.cache_evictions.back().forgotten = true;
        }
        completion.prune_disk_bytes_after = snapshot_store->disk_bytes();
    }

    void start_durable_io_worker() {
        if (!snapshot_store || durable_io_thread.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(durable_io_mutex);
            durable_io_stopping = false;
        }
        durable_io_thread = std::thread([this]() {
            while (true) {
                durable_io_job job;
                {
                    std::unique_lock<std::mutex> lock(durable_io_mutex);
                    durable_io_condition.wait(lock, [this]() {
                        return durable_io_stopping || !durable_io_jobs.empty();
                    });
                    if (durable_io_stopping) {
                        break;
                    }
                    job = std::move(durable_io_jobs.front());
                    durable_io_jobs.pop_front();
                }

                durable_io_completion completion;
                completion.job = std::move(job);
                completion.started_us = ggml_time_us();
                try {
                    switch (completion.job.operation) {
                        case durable_io_operation::spill:
                            GGML_ASSERT(completion.job.hot_payload);
                            completion.spill = snapshot_store->spill(
                                    completion.job.hot_payload->digest,
                                    completion.job.hot_payload->tokens,
                                    completion.job.hot_payload->state);
                            break;
                        case durable_io_operation::publish:
                            {
                                GGML_ASSERT(completion.job.hot_payload);
                                completion.publish_begin = snapshot_manifest->begin_publish(
                                        completion.job.owner,
                                        completion.job.digest,
                                        completion.job.retention_class);
                                try {
                                    server_snapshot_store_entry existing_entry;
                                    if (!snapshot_store->find_entry(completion.job.digest, existing_entry)) {
                                        const uint64_t projected = snapshot_store->projected_file_bytes(
                                                completion.job.hot_payload->tokens.size(),
                                                completion.job.hot_payload->state.size());
                                        const uint64_t budget = snapshot_store->disk_budget_bytes();
                                        if (projected > budget) {
                                            throw server_snapshot_store_error(
                                                    "managed snapshot object exceeds the durable disk budget", true);
                                        }
                                        prune_managed_cache(
                                                completion, budget - projected, completion.job.digest);
                                    }
                                    completion.spill = snapshot_store->spill(
                                            completion.job.hot_payload->digest,
                                            completion.job.hot_payload->tokens,
                                            completion.job.hot_payload->state);
                                    completion.object_published = true;
                                } catch (...) {
                                    if (!completion.publish_begin.already_retained) {
                                        try {
                                            snapshot_manifest->abort_publish(
                                                    completion.job.owner, completion.job.digest);
                                        } catch (const std::exception & abort_error) {
                                            throw server_snapshot_manifest_error(
                                                    std::string("snapshot publish failed and its durable intent could not be aborted: ") +
                                                    abort_error.what());
                                        }
                                    }
                                    throw;
                                }
                                if (completion.publish_begin.already_retained) {
                                    if (completion.publish_begin.existing_ref.retention_class ==
                                            completion.job.retention_class) {
                                        completion.manifest = {
                                            completion.publish_begin.existing_ref,
                                            true,
                                            snapshot_manifest->revision(),
                                            snapshot_manifest->file_bytes(),
                                        };
                                    } else {
                                        completion.manifest = snapshot_manifest->retain(
                                                completion.job.owner,
                                                completion.job.digest,
                                                completion.job.retention_class);
                                    }
                                } else {
                                    completion.manifest = snapshot_manifest->commit_publish(
                                            completion.job.owner, completion.job.digest);
                                }
                            } break;
                        case durable_io_operation::publish_advance:
                            {
                                GGML_ASSERT(completion.job.hot_payload);
                                completion.publish_advance_begin =
                                    snapshot_manifest->begin_publish_advance(
                                        completion.job.owner,
                                        completion.job.digest,
                                        completion.job.retention_class,
                                        completion.job.request.slot_action.head_name,
                                        completion.job.request.slot_action.expected_generation,
                                        completion.job.request.slot_action.expected_digest);
                                try {
                                    server_snapshot_store_entry existing_entry;
                                    if (!snapshot_store->find_entry(completion.job.digest, existing_entry)) {
                                        const uint64_t projected = snapshot_store->projected_file_bytes(
                                                completion.job.hot_payload->tokens.size(),
                                                completion.job.hot_payload->state.size());
                                        const uint64_t budget = snapshot_store->disk_budget_bytes();
                                        if (projected > budget) {
                                            throw server_snapshot_store_error(
                                                    "atomic publish-and-advance object exceeds the durable disk budget",
                                                    true);
                                        }
                                        prune_managed_cache(
                                                completion, budget - projected, completion.job.digest);
                                    }
                                    completion.spill = snapshot_store->spill(
                                            completion.job.hot_payload->digest,
                                            completion.job.hot_payload->tokens,
                                            completion.job.hot_payload->state);
                                    completion.object_published = true;
                                } catch (...) {
                                    if (!completion.publish_advance_begin.already_committed) {
                                        try {
                                            snapshot_manifest->abort_publish_advance(
                                                    completion.job.owner, completion.job.digest);
                                        } catch (const std::exception & abort_error) {
                                            throw server_snapshot_manifest_error(
                                                    std::string("atomic publish-and-advance failed and its durable intent could not be aborted: ") +
                                                    abort_error.what());
                                        }
                                    }
                                    throw;
                                }
                                if (completion.publish_advance_begin.already_committed) {
                                    completion.publish_advance = {
                                        completion.publish_advance_begin.ref,
                                        completion.publish_advance_begin.head,
                                        true,
                                        snapshot_manifest->revision(),
                                        snapshot_manifest->file_bytes(),
                                    };
                                } else {
                                    completion.publish_advance =
                                        snapshot_manifest->commit_publish_advance(
                                            completion.job.owner, completion.job.digest);
                                }
                            } break;
                        case durable_io_operation::load:
                            completion.load = snapshot_store->load(
                                    completion.job.digest,
                                    completion.job.max_load_bytes);
                            break;
                        case durable_io_operation::erase:
                            if (snapshot_manifest->ref_count(completion.job.digest) > 0 ||
                                    snapshot_manifest->publish_intent_count(completion.job.digest) > 0) {
                                throw server_snapshot_store_error(
                                        "durable snapshot content is retained by manifest owners", true);
                            }
                            completion.erase = snapshot_store->erase(completion.job.digest);
                            completion.object_erased = true;
                            if (snapshot_manifest->managed_digests().find(completion.job.digest) !=
                                    snapshot_manifest->managed_digests().end()) {
                                snapshot_manifest->forget_managed(completion.job.digest);
                                completion.managed_forgotten = true;
                            }
                            break;
                        case durable_io_operation::retain:
                            {
                                server_snapshot_store_entry entry;
                                if (!snapshot_store->find_entry(completion.job.digest, entry)) {
                                    throw server_snapshot_manifest_error(
                                            "durable snapshot content is unavailable", true);
                                }
                                completion.manifest = snapshot_manifest->retain(
                                        completion.job.owner,
                                        completion.job.digest,
                                        completion.job.retention_class);
                            } break;
                        case durable_io_operation::release:
                            completion.manifest = snapshot_manifest->release(
                                    completion.job.owner, completion.job.digest);
                            break;
                        case durable_io_operation::compact:
                            completion.compact = snapshot_manifest->compact();
                            break;
                        case durable_io_operation::prune:
                            prune_managed_cache(completion, completion.job.target_disk_bytes);
                            break;
                        case durable_io_operation::head_create:
                            {
                                server_snapshot_store_entry entry;
                                if (!snapshot_store->find_entry(completion.job.digest, entry)) {
                                    throw server_snapshot_manifest_error(
                                            "logical head content is unavailable", true);
                                }
                                completion.head = snapshot_manifest->create_head(
                                        completion.job.request.slot_action.head_name,
                                        completion.job.digest);
                            } break;
                        case durable_io_operation::head_advance:
                            {
                                server_snapshot_store_entry entry;
                                if (!snapshot_store->find_entry(completion.job.digest, entry)) {
                                    throw server_snapshot_manifest_error(
                                            "logical head target content is unavailable", true);
                                }
                                completion.head = snapshot_manifest->advance_head(
                                        completion.job.request.slot_action.head_name,
                                        completion.job.request.slot_action.expected_generation,
                                        completion.job.request.slot_action.expected_digest,
                                        completion.job.digest);
                            } break;
                        case durable_io_operation::head_delete:
                            completion.head = snapshot_manifest->delete_head(
                                    completion.job.request.slot_action.head_name,
                                    completion.job.request.slot_action.expected_generation,
                                    completion.job.request.slot_action.expected_digest);
                            break;
                        case durable_io_operation::head_load:
                            {
                                const auto & heads = snapshot_manifest->logical_heads();
                                const auto found = heads.find(completion.job.request.slot_action.head_name);
                                if (found == heads.end() ||
                                        found->second.generation !=
                                            completion.job.request.slot_action.expected_generation ||
                                        found->second.digest !=
                                            completion.job.request.slot_action.expected_digest) {
                                    throw server_snapshot_manifest_error(
                                            "logical head compare-and-swap conflict", true);
                                }
                                completion.head = {found->second, true,
                                    snapshot_manifest->revision(), snapshot_manifest->file_bytes()};
                                completion.load = snapshot_store->load(
                                        found->second.digest, completion.job.max_load_bytes);
                            } break;
                    }
                    completion.success = true;
                } catch (const server_snapshot_store_error & error) {
                    completion.unavailable = error.unavailable;
                    completion.error = error.what();
                } catch (const server_snapshot_manifest_error & error) {
                    completion.unavailable = error.unavailable;
                    completion.error = error.what();
                } catch (const std::exception & error) {
                    completion.error = std::string("durable snapshot I/O failed: ") + error.what();
                }
                completion.finished_us = ggml_time_us();
                completion.disk_bytes = snapshot_store->disk_bytes();
                completion.disk_budget_bytes = snapshot_store->disk_budget_bytes();
                completion.disk_high_water_bytes = snapshot_store->disk_high_water_bytes();
                completion.recovered_temp_files = snapshot_store->recovered_temp_files();
                completion.ignored_corrupt_files = snapshot_store->ignored_corrupt_files();
                completion.runtime_integrity_failures = snapshot_store->runtime_integrity_failures();
                completion.orphaned_disk_bytes = snapshot_store->orphaned_disk_bytes();
                completion.manifest_revision = snapshot_manifest->revision();
                completion.manifest_file_bytes = snapshot_manifest->file_bytes();
                completion.manifest_budget_bytes = snapshot_manifest->byte_budget();
                completion.manifest_high_water_bytes = snapshot_manifest->high_water_bytes();
                completion.manifest_record_count = snapshot_manifest->record_count();
                completion.manifest_recovered_temp_files = snapshot_manifest->recovered_temp_files();
                completion.manifest_recovered_tail_bytes = snapshot_manifest->recovered_tail_bytes();
                completion.manifest_compactions = snapshot_manifest->compactions();

                const uint64_t completion_id = completion.job.id;
                bool publish = false;
                {
                    std::lock_guard<std::mutex> lock(durable_io_mutex);
                    if (!durable_io_stopping) {
                        const auto inserted = durable_io_completions.emplace(
                                completion_id, std::move(completion));
                        GGML_ASSERT(inserted.second);
                        publish = true;
                    }
                }
                if (publish) {
                    server_task completed(SERVER_TASK_TYPE_DURABLE_IO_COMPLETE);
                    completed.id = queue_tasks.get_new_id();
                    completed.slot_action.durable_io_id = completion_id;
                    queue_tasks.post(std::move(completed));
                } else {
                    send_error(completion.job.request,
                            "durable snapshot I/O was canceled during server shutdown",
                            ERROR_TYPE_UNAVAILABLE);
                }
            }
        });
    }

    void stop_durable_io_worker() {
        std::vector<int> cancelled_task_ids;
        {
            std::lock_guard<std::mutex> lock(durable_io_mutex);
            durable_io_stopping = true;
            for (const auto & job : durable_io_jobs) {
                cancelled_task_ids.push_back(job.request.id);
            }
            for (const auto & item : durable_io_completions) {
                cancelled_task_ids.push_back(item.second.job.request.id);
            }
            durable_io_jobs.clear();
            durable_io_completions.clear();
        }
        for (const int id_task : cancelled_task_ids) {
            send_error(id_task,
                    "durable snapshot I/O was canceled during server shutdown",
                    ERROR_TYPE_UNAVAILABLE);
        }
        durable_io_condition.notify_all();
        if (durable_io_thread.joinable()) {
            durable_io_thread.join();
        }
        {
            std::lock_guard<std::mutex> lock(durable_io_mutex);
            GGML_ASSERT(durable_io_jobs.empty());
            GGML_ASSERT(durable_io_completions.empty());
        }
    }

    bool enqueue_durable_io(durable_io_job & job) {
        {
            std::lock_guard<std::mutex> lock(durable_io_mutex);
            if (durable_io_stopping || !durable_io_thread.joinable()) {
                return false;
            }
            durable_io_jobs.push_back(std::move(job));
        }
        const uint64_t pending = durable_io_pending.fetch_add(1) + 1;
        durable_io_queue_high_water = std::max(durable_io_queue_high_water, pending);
        durable_io_condition.notify_one();
        return true;
    }

    bool take_durable_io_completion(uint64_t id, durable_io_completion & completion) {
        std::lock_guard<std::mutex> lock(durable_io_mutex);
        const auto found = durable_io_completions.find(id);
        if (found == durable_io_completions.end()) {
            return false;
        }
        completion = std::move(found->second);
        durable_io_completions.erase(found);
        return true;
    }

    void release_durable_io_reservations(const durable_io_job & job) {
        if (job.disk_reservation_bytes > 0) {
            GGML_ASSERT(durable_io_reserved_disk_bytes >= job.disk_reservation_bytes);
            durable_io_reserved_disk_bytes -= job.disk_reservation_bytes;
            durable_io_pending_spill_digests.erase(job.digest);
        }
        if (job.load_reservation_bytes > 0) {
            GGML_ASSERT(durable_io_reserved_load_bytes >= job.load_reservation_bytes);
            durable_io_reserved_load_bytes -= job.load_reservation_bytes;
        }
        const uint64_t previous = durable_io_pending.fetch_sub(1);
        GGML_ASSERT(previous > 0);
        durable_io_completed_total++;
    }

    statetree_snapshot * resolve_snapshot(server_task & task) {
        auto found = statetree_snapshots.find(task.slot_action.snapshot_id);
        if (found == statetree_snapshots.end()) {
            send_error(task, "StateTree snapshot is unavailable", ERROR_TYPE_UNAVAILABLE);
            return nullptr;
        }
        if (!task.slot_action.digest.empty() && task.slot_action.digest != found->second.payload->digest) {
            send_error(task, "StateTree snapshot does not match the requested digest", ERROR_TYPE_UNAVAILABLE);
            return nullptr;
        }
        return &found->second;
    }

    retention_stats get_retention_stats() {
        retention_stats result;
        auto families = collect_statetree_families();
        result.n_families = families.size();
        result.n_active_families = std::count_if(families.begin(), families.end(), [](const statetree_family & family) {
            return family.active;
        });

        for (const server_slot & slot : slots) {
            const size_t bytes = slot.prompt_state_bytes();
            result.state_bytes += bytes;
            if (slot.is_processing()) {
                result.active_bytes += bytes;
            } else {
                result.retained_bytes += bytes;
            }
        }
        return result;
    }

    int64_t lease_remaining_ms(const server_slot & slot, int64_t now_us) const {
        if (slot.lease_deadline_us < 0) {
            return -1;
        }
        return std::max<int64_t>(0, (slot.lease_deadline_us - now_us + 999) / 1000);
    }

    bool touch_family(int source_id, int64_t fork_id, bool renewal) {
        const int64_t now_us = ggml_time_us();
        const int64_t deadline_us = params_base.statetree_lease_ms > 0
            ? now_us + (int64_t) params_base.statetree_lease_ms * 1000
            : -1;
        const uint64_t touch = ++retention_touch_next;
        bool found = false;
        for (server_slot & slot : slots) {
            if (slot.fork_source_id != source_id || slot.fork_id != fork_id) {
                continue;
            }
            slot.retention_touch = touch;
            slot.lease_deadline_us = deadline_us;
            found = true;
        }
        if (found && renewal) {
            statetree_renewed_total++;
        }
        return found;
    }

    bool family_is_active(int source_id, int64_t fork_id) const {
        return std::any_of(slots.begin(), slots.end(), [&](const server_slot & slot) {
            return slot.fork_source_id == source_id && slot.fork_id == fork_id && slot.is_processing();
        });
    }

    bool family_is_expired(const server_slot & slot, int64_t now_us) const {
        return slot.lease_deadline_us >= 0 && slot.lease_deadline_us <= now_us &&
            !family_is_active(slot.fork_source_id, slot.fork_id);
    }

    bool validate_slot_fence(const server_task & task, server_slot & slot) {
        if (task.slot_action.fork_id >= 0) {
            if (!slot.is_fork_reserved() || slot.fork_id != task.slot_action.fork_id ||
                    family_is_expired(slot, ggml_time_us())) {
                send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                if (slot.is_available()) {
                    slot.callback_on_deferred(slot.id);
                }
                return false;
            }
        } else if (slot.is_fork_reserved() && retention_enabled()) {
            send_error(task, "A generation-fenced fork_id is required for this reserved slot",
                    ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    server_slot * resolve_slot_action_target(server_task & task, const char * invalid_slot_error) {
        server_slot * slot = nullptr;
        if (task.slot_action.node_id >= 0) {
            for (server_slot & candidate : slots) {
                if (candidate.node_id == task.slot_action.node_id) {
                    GGML_ASSERT(slot == nullptr);
                    slot = &candidate;
                }
            }
            if (slot == nullptr || !slot->is_fork_reserved()) {
                send_error(task, "StateTree branch node is unavailable", ERROR_TYPE_UNAVAILABLE);
                return nullptr;
            }
            if ((task.slot_action.id_slot >= 0 && task.slot_action.id_slot != slot->id) ||
                    (task.slot_action.state_id >= 0 && task.slot_action.state_id != slot->state_id) ||
                    (task.slot_action.fork_id >= 0 && task.slot_action.fork_id != slot->fork_id)) {
                send_error(task, "StateTree branch node does not match the requested identity",
                        ERROR_TYPE_UNAVAILABLE);
                return nullptr;
            }
            task.slot_action.id_slot = slot->id;
            task.slot_action.fork_id = slot->fork_id;
            return slot;
        }

        slot = get_slot_by_id(task.slot_action.id_slot);
        if (slot == nullptr) {
            send_error(task, invalid_slot_error, ERROR_TYPE_INVALID_REQUEST);
            return nullptr;
        }
        if (task.slot_action.state_id >= 0 && task.slot_action.state_id != slot->state_id) {
            send_error(task, "StateTree state does not match the requested slot", ERROR_TYPE_UNAVAILABLE);
            return nullptr;
        }
        return slot;
    }

    void release_family(const statetree_family & family, bool expired) {
        GGML_ASSERT(!family.active);
        const size_t reclaimed = family.state_bytes;
        SRV_INF("StateTree %s family source=%d fork=%" PRId64 " members=%zu state=%.3f MiB\n",
                expired ? "expiring" : "evicting", family.source_id, family.fork_id,
                family.members.size(), reclaimed / (1024.0 * 1024.0));

        std::vector<int> released;
        released.reserve(family.members.size());
        for (server_slot * slot : family.members) {
            released.push_back(slot->id);
        }
        const auto released_nodes = get_statetree_node_refs(family.members);
        record_statetree_event(
                expired ? "expire" : "evict",
                family.state_id,
                family.fork_id,
                family.source_id,
                released,
                family.state_bytes,
                -1,
                released_nodes);
        for (server_slot * slot : family.members) {
            slot->prompt_clear(false);
        }

        if (expired) {
            statetree_expired_total++;
        } else {
            statetree_evicted_total++;
        }
        statetree_reclaimed_bytes_total += reclaimed;

        for (const int id_slot : released) {
            slots[id_slot].callback_on_deferred(id_slot);
        }
    }

    void release_idle_slot(server_slot & slot) {
        GGML_ASSERT(!slot.is_processing());
        GGML_ASSERT(!slot.is_fork_reserved());
        const size_t reclaimed = slot.prompt_state_bytes();
        SRV_INF("evicting idle slot %d state=%.3f MiB for retained-state budget\n",
                slot.id, reclaimed / (1024.0 * 1024.0));
        slot.prompt_clear(false);
        statetree_evicted_total++;
        statetree_reclaimed_bytes_total += reclaimed;
        slot.callback_on_deferred(slot.id);
    }

    bool reserve_state_bytes(size_t added_bytes, size_t removed_bytes, const server_slot & owner) {
        const uint64_t budget = params_base.statetree_max_state_bytes;
        if (budget == 0) {
            return true;
        }

        size_t non_evictable_bytes = 0;
        for (const server_slot & slot : slots) {
            const bool owns_family = owner.is_fork_reserved() && slot.is_fork_reserved() &&
                slot.fork_source_id == owner.fork_source_id && slot.fork_id == owner.fork_id;
            if (slot.is_processing() || slot.id == owner.id || owns_family) {
                non_evictable_bytes += slot.prompt_state_bytes();
            }
        }
        const size_t non_evictable_base = non_evictable_bytes > removed_bytes
            ? non_evictable_bytes - removed_bytes
            : 0;
        if (non_evictable_base > budget || added_bytes > budget - non_evictable_base) {
            statetree_pressure_rejected_total++;
            return false;
        }

        auto exceeds_budget = [&]() {
            const size_t current = get_retention_stats().state_bytes;
            const size_t projected_base = current > removed_bytes ? current - removed_bytes : 0;
            return projected_base > budget || added_bytes > budget - projected_base;
        };

        while (exceeds_budget()) {
            auto families = collect_statetree_families();
            std::sort(families.begin(), families.end(), [](const statetree_family & left, const statetree_family & right) {
                if (left.touch != right.touch) {
                    return left.touch < right.touch;
                }
                if (left.source_id != right.source_id) {
                    return left.source_id < right.source_id;
                }
                return left.fork_id < right.fork_id;
            });

            statetree_family * family_victim = nullptr;
            for (auto & family : families) {
                const bool owns_family = owner.is_fork_reserved() &&
                    family.source_id == owner.fork_source_id && family.fork_id == owner.fork_id;
                if (!family.active && !owns_family && family.state_bytes > 0) {
                    family_victim = &family;
                    break;
                }
            }

            server_slot * slot_victim = nullptr;
            for (server_slot & slot : slots) {
                if (slot.id == owner.id || slot.is_processing() || slot.is_fork_reserved() || slot.prompt_state_bytes() == 0) {
                    continue;
                }
                if (slot_victim == nullptr ||
                        slot.retention_touch < slot_victim->retention_touch ||
                        (slot.retention_touch == slot_victim->retention_touch && slot.id < slot_victim->id)) {
                    slot_victim = &slot;
                }
            }

            const bool use_family = family_victim != nullptr &&
                (slot_victim == nullptr ||
                 family_victim->touch < slot_victim->retention_touch ||
                 (family_victim->touch == slot_victim->retention_touch && family_victim->source_id <= slot_victim->id));
            if (use_family) {
                release_family(*family_victim, false);
            } else if (slot_victim != nullptr) {
                release_idle_slot(*slot_victim);
            } else {
                statetree_pressure_rejected_total++;
                return false;
            }
        }
        return true;
    }

    void maintain_retention() {
        const int64_t now_us = ggml_time_us();

        for (const auto & family : collect_statetree_families()) {
            if (!family.active && family.lease_deadline_us >= 0 && family.lease_deadline_us <= now_us) {
                release_family(family, true);
            }
        }

        const uint64_t budget = params_base.statetree_max_state_bytes;
        if (budget > 0) {
            while (get_retention_stats().state_bytes > budget) {
                auto families = collect_statetree_families();
                std::sort(families.begin(), families.end(), [](const statetree_family & left, const statetree_family & right) {
                    if (left.touch != right.touch) {
                        return left.touch < right.touch;
                    }
                    if (left.source_id != right.source_id) {
                        return left.source_id < right.source_id;
                    }
                    return left.fork_id < right.fork_id;
                });

                statetree_family * family_victim = nullptr;
                for (auto & family : families) {
                    if (!family.active && family.state_bytes > 0) {
                        family_victim = &family;
                        break;
                    }
                }

                server_slot * slot_victim = nullptr;
                for (server_slot & slot : slots) {
                    if (slot.is_processing() || slot.is_fork_reserved() || slot.prompt_state_bytes() == 0) {
                        continue;
                    }
                    if (slot_victim == nullptr ||
                            slot.retention_touch < slot_victim->retention_touch ||
                            (slot.retention_touch == slot_victim->retention_touch && slot.id < slot_victim->id)) {
                        slot_victim = &slot;
                    }
                }

                const bool use_family = family_victim != nullptr &&
                    (slot_victim == nullptr ||
                     family_victim->touch < slot_victim->retention_touch ||
                     (family_victim->touch == slot_victim->retention_touch && family_victim->source_id <= slot_victim->id));
                if (use_family) {
                    release_family(*family_victim, false);
                } else if (slot_victim != nullptr) {
                    release_idle_slot(*slot_victim);
                } else {
                    statetree_pressure_rejected_total++;
                    break;
                }
            }
        }

        const auto stats = get_retention_stats();
        state_high_water_bytes = std::max<uint64_t>(state_high_water_bytes, stats.state_bytes);
        retained_high_water_bytes = std::max<uint64_t>(retained_high_water_bytes, stats.retained_bytes);
    }

    int64_t retention_idle_wait_ms() {
        maintain_retention();
        const int64_t now_us = ggml_time_us();
        int64_t wait_ms = 1000;
        for (const auto & family : collect_statetree_families()) {
            if (family.active || family.lease_deadline_us < 0) {
                continue;
            }
            wait_ms = std::min(wait_ms,
                    std::max<int64_t>(1, (family.lease_deadline_us - now_us + 999) / 1000));
        }
        return wait_ms;
    }

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;
        params_base.n_outputs_max = server_n_outputs_max(params_base);

        const bool durable_store_requested = !params_base.statetree_snapshot_store.empty() ||
            !params_base.statetree_snapshot_compat_id.empty() ||
            params_base.statetree_max_snapshot_disk_bytes > 0 ||
            params_base.statetree_max_snapshot_load_bytes > 0 ||
            params_base.statetree_max_snapshot_manifest_bytes > 0;
        if (durable_store_requested && (params_base.statetree_snapshot_store.empty() ||
                params_base.statetree_snapshot_compat_id.empty() ||
                params_base.statetree_max_snapshot_disk_bytes == 0 ||
                params_base.statetree_max_snapshot_load_bytes == 0 ||
                params_base.statetree_max_snapshot_manifest_bytes == 0)) {
            SRV_ERR("%s", "durable StateTree content requires store path, compatibility ID, and positive disk/load/manifest budgets\n");
            return false;
        }
        if (!is_resume && durable_store_requested) {
            try {
                snapshot_store = std::make_unique<server_snapshot_store>(
                        params_base.statetree_snapshot_store,
                        params_base.statetree_snapshot_compat_id,
                        params_base.statetree_max_snapshot_disk_bytes,
                        [](const llama_tokens & tokens, const std::vector<uint8_t> & state) {
                            return snapshot_content_digest(tokens, state);
                        });
                snapshot_manifest = std::make_unique<server_snapshot_manifest>(
                        snapshot_store->namespace_path(),
                        params_base.statetree_snapshot_compat_id,
                        params_base.statetree_max_snapshot_manifest_bytes);
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize durable StateTree content: %s\n", e.what());
                return false;
            }
        } else if (!durable_store_requested) {
            snapshot_store.reset();
            snapshot_manifest.reset();
        }
        refresh_durable_store_cache();
        refresh_durable_manifest_cache();
        if (snapshot_manifest) {
            const auto pending_publishes = snapshot_manifest->publish_intents();
            try {
                for (const auto & item : pending_publishes) {
                    const auto & intent = item.second;
                    server_snapshot_store_entry entry;
                    if (!snapshot_store->find_entry(intent.digest, entry)) {
                        snapshot_manifest->abort_publish(intent.owner, intent.digest);
                        durable_manifest_recovered_publish_aborts++;
                        continue;
                    }
                    if (entry.payload_bytes > params_base.statetree_max_snapshot_load_bytes) {
                        throw server_snapshot_manifest_error(
                                "pending snapshot publish exceeds the configured recovery load budget", true);
                    }
                    try {
                        snapshot_store->load(
                                intent.digest, params_base.statetree_max_snapshot_load_bytes);
                        snapshot_manifest->commit_publish(intent.owner, intent.digest);
                        durable_manifest_recovered_publish_commits++;
                    } catch (const server_snapshot_store_error &) {
                        snapshot_manifest->abort_publish(intent.owner, intent.digest);
                        snapshot_store->erase(intent.digest);
                        durable_manifest_recovered_publish_aborts++;
                    }
                }
                const auto pending_publish_advances = snapshot_manifest->publish_advance_intents();
                for (const auto & item : pending_publish_advances) {
                    const auto & intent = item.second;
                    server_snapshot_store_entry entry;
                    if (!snapshot_store->find_entry(intent.digest, entry)) {
                        snapshot_manifest->abort_publish_advance(intent.owner, intent.digest);
                        durable_manifest_recovered_publish_aborts++;
                        continue;
                    }
                    if (entry.payload_bytes > params_base.statetree_max_snapshot_load_bytes) {
                        throw server_snapshot_manifest_error(
                                "pending atomic publish-and-advance exceeds the configured recovery load budget",
                                true);
                    }
                    try {
                        snapshot_store->load(
                                intent.digest, params_base.statetree_max_snapshot_load_bytes);
                        snapshot_manifest->commit_publish_advance(intent.owner, intent.digest);
                        durable_manifest_recovered_publish_commits++;
                    } catch (const server_snapshot_store_error &) {
                        snapshot_manifest->abort_publish_advance(intent.owner, intent.digest);
                        snapshot_store->erase(intent.digest);
                        durable_manifest_recovered_publish_aborts++;
                    }
                }
                const auto managed_digests = snapshot_manifest->managed_digests();
                for (const auto & digest : managed_digests) {
                    if (snapshot_manifest->ref_count(digest) != 0 ||
                            snapshot_manifest->publish_intent_count(digest) != 0) {
                        continue;
                    }
                    server_snapshot_store_entry entry;
                    if (snapshot_store->find_entry(digest, entry)) {
                        snapshot_store->erase(digest);
                        durable_managed_recovered_erases++;
                        durable_managed_recovered_bytes += entry.file_bytes;
                    }
                    snapshot_manifest->forget_managed(digest);
                }
            } catch (const std::exception & e) {
                SRV_ERR("failed to reconcile durable snapshot publish intent: %s\n", e.what());
                return false;
            }
            refresh_durable_store_cache();
            refresh_durable_manifest_cache();
        }
        for (const auto & item : durable_manifest_refs) {
            if (durable_store_entries.find(item.second.digest) == durable_store_entries.end()) {
                SRV_ERR("durable snapshot manifest owner '%s' references missing content '%s'\n",
                        item.second.owner.c_str(), item.second.digest.c_str());
                return false;
            }
        }
        for (const auto & item : durable_logical_heads) {
            if (durable_store_entries.find(item.second.digest) == durable_store_entries.end()) {
                SRV_ERR("durable logical head '%s' references missing content '%s'\n",
                        item.second.name.c_str(), item.second.digest.c_str());
                return false;
            }
        }

        std::string & mmproj_path = params_base.mmproj.path;
        bool has_mmproj = !mmproj_path.empty();
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_INF("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB\n", total / (1024.0 * 1024.0));
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            const bool has_draft = params_base.speculative.has_dft();

            if (has_draft || spec_mtp) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = spec_mtp
                    ? (int32_t) server_mtp_n_seq_max(params_base)
                    : params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                    cparams_dft.n_seq_max = server_mtp_n_seq_max(params_base);
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes =
                            (measure_model_bytes ? dmd[j].mb.model : 0) +
                            dmd[j].mb.context +
                            dmd[j].mb.compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();

            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cparams.n_seq_max = server_mtp_n_seq_max(params_base);
                cparams.n_outputs_max = server_mtp_n_seq_max(params_base);
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq  = 0;
            cparams.ctx_other = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end()) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            auto cparams_mtp = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.type_k        = params_base.speculative.draft.cache_type_k;
            cparams_mtp.type_v        = params_base.speculative.draft.cache_type_v;
            cparams_mtp.n_seq_max     = server_mtp_n_seq_max(params_base);
            cparams_mtp.n_rs_seq      = 0;
            cparams_mtp.n_outputs_max = server_mtp_n_seq_max(params_base);
            cparams_mtp.ctx_other     = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        }

        if (has_mmproj) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                server_slot & released = slots[id_slot];
                if (released.is_fork_reserved()) {
                    if (!family_is_active(released.fork_source_id, released.fork_id)) {
                        touch_family(released.fork_source_id, released.fork_id, false);
                    }
                } else if (released.prompt_state_bytes() > 0) {
                    released.retention_touch = ++retention_touch_next;
                }
                released.callback_on_deferred(id_slot);
            };

            slot.callback_on_deferred = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (params_base.statetree_lease_ms > 0) {
            SRV_INF("StateTree family lease = %d ms\n", params_base.statetree_lease_ms);
        } else {
            SRV_INF("%s", "StateTree family lease disabled\n");
        }
        if (params_base.statetree_max_state_bytes > 0) {
            SRV_INF("StateTree live state budget = %.3f MiB\n",
                    params_base.statetree_max_state_bytes / (1024.0 * 1024.0));
        } else {
            SRV_INF("%s", "StateTree live state budget disabled\n");
        }
        if (params_base.statetree_max_snapshot_bytes > 0) {
            SRV_INF("StateTree immutable snapshot budget = %.3f MiB\n",
                    params_base.statetree_max_snapshot_bytes / (1024.0 * 1024.0));
        } else {
            SRV_INF("%s", "StateTree immutable snapshots disabled\n");
        }
        if (snapshot_store) {
            SRV_INF("StateTree durable snapshot namespace = %s, objects=%zu, disk=%.3f/%.3f MiB\n",
                    snapshot_store->namespace_id().c_str(), snapshot_store->entries().size(),
                    snapshot_store->disk_bytes() / (1024.0 * 1024.0),
                    snapshot_store->disk_budget_bytes() / (1024.0 * 1024.0));
            if (snapshot_store->ignored_corrupt_files() > 0) {
                SRV_WRN("StateTree durable snapshot namespace ignored %" PRIu64 " corrupt objects\n",
                        snapshot_store->ignored_corrupt_files());
            }
        } else {
            SRV_INF("%s", "StateTree durable snapshot content disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_idle([this]() {
            return retention_idle_wait_ms();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });
        queue_tasks.on_idle_sleep_inhibited([this]() {
            return durable_io_pending.load() > 0 || std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.is_fork_reserved();
            });
        });
        queue_tasks.on_terminated_task([this](server_task && task) {
            if (task.type != SERVER_TASK_TYPE_CANCEL && task.type != SERVER_TASK_TYPE_DURABLE_IO_COMPLETE) {
                send_error(task, "server is shutting down", ERROR_TYPE_UNAVAILABLE);
            }
        });

        try {
            start_durable_io_worker();
        } catch (const std::exception & error) {
            SRV_ERR("failed to start durable snapshot I/O worker: %s\n", error.what());
            return false;
        }

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_INF("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate UI settings (from either new ui_config_json or deprecated webui_config_json)
        {
            const std::string & cfg = !params_base.ui_config_json.empty()
                ? params_base.ui_config_json
                : params_base.webui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                    json_webui_settings = json_settings; // deprecated: keep in sync
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (!slot.is_available()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (!slot.is_available()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;
            update_cache = update_cache && !llama_adapter_cvec_seq_mode(ctx_tgt);

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                maintain_retention();

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (!slot.is_available()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        if (task.params.jspace_control_scale && ctx_dft) {
            send_error(task,
                    "Request-scoped J-Space is unavailable with a separate draft context",
                    ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        if (task.params.jspace_control_scale && !llama_adapter_cvec_seq_mode(ctx_tgt)) {
            for (const auto & candidate : slots) {
                if (candidate.is_processing() || candidate.is_fork_reserved()) {
                    send_error(task,
                            "Cannot enter request-scoped J-Space mode while another slot is active or reserved",
                            ERROR_TYPE_UNAVAILABLE);
                    return false;
                }
            }
            // Cached KV/recurrent state was computed under the legacy global
            // vector and cannot be relabeled as request-scoped state.
            for (auto & candidate : slots) {
                if (!candidate.prompt.tokens.empty()) {
                    candidate.prompt_clear(false);
                }
            }
        }

        if (task.params.jspace_control_scale || llama_adapter_cvec_seq_mode(ctx_tgt)) {
            const float requested_scale = task.params.jspace_control_scale.value_or(0.0f);
            const float current_scale = llama_adapter_cvec_seq_get(ctx_tgt, slot.id);
            if (!slot.prompt.tokens.empty() && current_scale != requested_scale) {
                if (slot.is_fork_reserved()) {
                    send_error(task,
                            "A reserved StateTree branch cannot change J-Space scale",
                            ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                slot.prompt_clear(false);
            }
            if (llama_adapter_cvec_seq_set(ctx_tgt, slot.id, requested_scale) != 0) {
                send_error(task,
                        "jspace_control_scale requires an active server control vector",
                        ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (slot.is_available() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const size_t checkpoint_bytes =
            llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) +
            (ctx_dft ? llama_state_seq_get_size_ext(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0);
        size_t replaced_bytes = 0;
        if (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            size_t n_replace = slot.prompt.checkpoints.size() - params_base.n_ctx_checkpoints + 1;
            for (const auto & checkpoint : slot.prompt.checkpoints) {
                if (n_replace-- == 0) {
                    break;
                }
                replaced_bytes += checkpoint.size();
            }
        }
        if (!reserve_state_bytes(checkpoint_bytes, replaced_bytes, slot)) {
            const size_t additional_bytes = checkpoint_bytes > replaced_bytes ? checkpoint_bytes - replaced_bytes : 0;
            SLT_WRN(slot, "skipping context checkpoint: retained-state budget needs %zu additional bytes\n",
                    additional_bytes);
            return;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

        const auto retention = get_retention_stats();
        state_high_water_bytes = std::max<uint64_t>(state_high_water_bytes, retention.state_bytes);
        retained_high_water_bytes = std::max<uint64_t>(retained_high_water_bytes, retention.retained_bytes);
    }

    bool materialize_snapshot_content(
            server_task & task,
            const llama_tokens & tokens,
            const std::vector<uint8_t> & state,
            int64_t state_id,
            int64_t parent_node_id,
            int64_t snapshot_id,
            const std::string & content_digest,
            json & result) {
        if (!check_no_mtmd(task.id)) {
            return false;
        }
        if (ctx_dft || spec || !params_base.lora_adapters.empty()) {
            send_error(task, "StateTree snapshot materialization is unavailable with the active model configuration",
                    ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }

        server_slot * destination = nullptr;
        if (task.slot_action.id_slot >= 0) {
            destination = get_slot_by_id(task.slot_action.id_slot);
            if (destination == nullptr) {
                send_error(task, "Invalid destination slot ID", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            if (!destination->is_available()) {
                send_error(task, "Destination slot is unavailable", ERROR_TYPE_UNAVAILABLE);
                return false;
            }
        } else {
            const auto free_slots = get_free_slots(1, -1);
            if (free_slots.empty()) {
                send_error(task, "No destination slot is available", ERROR_TYPE_UNAVAILABLE);
                return false;
            }
            destination = free_slots.front();
        }

        const bool allocate_state_id = state_id < 0;
        if (statetree_next_fork_id > (uint64_t) std::numeric_limits<int64_t>::max() ||
                statetree_next_node_id > (uint64_t) std::numeric_limits<int64_t>::max() ||
                (allocate_state_id && statetree_next_state_id > (uint64_t) std::numeric_limits<int64_t>::max())) {
            send_error(task, "StateTree identity space is exhausted", ERROR_TYPE_SERVER);
            return false;
        }

        server_tokens restored_tokens;
        try {
            restored_tokens = server_tokens(tokens, false);
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to clone StateTree snapshot tokens: ") + e.what(),
                    ERROR_TYPE_SERVER);
            return false;
        }

        const int64_t t_start = ggml_time_us();
        destination->prompt_clear(false);
        size_t base_state_size = 0;
        float cvec_scale = 0.0f;
        bool has_cvec_scale = false;
        try {
            has_cvec_scale = snapshot_get_cvec_scale(state, base_state_size, cvec_scale);
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to parse StateTree snapshot controller state: ") + e.what(),
                    ERROR_TYPE_SERVER);
            destination->callback_on_deferred(destination->id);
            return false;
        }
        const size_t read = llama_state_seq_set_data_ext(
                ctx_tgt, state.data(), base_state_size, destination->id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (read != base_state_size || (has_cvec_scale &&
                llama_adapter_cvec_seq_set(ctx_tgt, destination->id, cvec_scale) != 0)) {
            destination->prompt_clear(false);
            send_error(task, "Failed to restore the complete StateTree snapshot", ERROR_TYPE_SERVER);
            destination->callback_on_deferred(destination->id);
            return false;
        }

        const int64_t final_state_id = allocate_state_id
            ? (int64_t) statetree_next_state_id++
            : state_id;
        const int64_t fork_id = (int64_t) statetree_next_fork_id++;
        const int64_t node_id = (int64_t) statetree_next_node_id++;
        server_prompt restored_prompt;
        restored_prompt.tokens = std::move(restored_tokens);
        destination->prompt = std::move(restored_prompt);
        destination->task_prev.reset();
        destination->fork_source_id = destination->id;
        destination->state_id = final_state_id;
        destination->node_id = node_id;
        destination->parent_node_id = parent_node_id;
        destination->materialized_snapshot_id = snapshot_id;
        destination->materialized_content_digest = content_digest;
        destination->fork_id = fork_id;
        touch_family(destination->id, fork_id, false);

        const int64_t t_end = ggml_time_us();
        result = {
            {"id_slot", destination->id},
            {"state_id", final_state_id},
            {"node_id", node_id},
            {"parent_node_id", parent_node_id},
            {"fork_id", fork_id},
            {"n_tokens", tokens.size()},
            {"digest", content_digest},
            {"timings", {{"materialize_ms", (t_end - t_start) / 1000.0}}},
        };
        return true;
    }

    void process_single_task(server_task && task) {
        maintain_retention();

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    int id_slot = task.id_slot;
                    const int id_task = task.id;

                    if (task.node_id >= 0) {
                        server_slot * logical_node = nullptr;
                        for (server_slot & candidate : slots) {
                            if (candidate.node_id == task.node_id) {
                                GGML_ASSERT(logical_node == nullptr);
                                logical_node = &candidate;
                            }
                        }
                        if (logical_node == nullptr) {
                            send_error(task, "StateTree branch node is unavailable", ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                        if ((id_slot >= 0 && id_slot != logical_node->id) ||
                                (task.state_id >= 0 && task.state_id != logical_node->state_id) ||
                                (task.fork_id >= 0 && task.fork_id != logical_node->fork_id)) {
                            send_error(task, "StateTree branch node does not match the requested identity",
                                    ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                        id_slot = logical_node->id;
                    }

                    if (task.state_id >= 0 && task.node_id < 0) {
                        if (task.fork_id < 0) {
                            send_error(task, "A logical state_id requires a generation-fenced fork_id",
                                    ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                        std::vector<server_slot *> logical_members;
                        for (server_slot & candidate : slots) {
                            if (candidate.state_id == task.state_id && candidate.fork_id == task.fork_id) {
                                logical_members.push_back(&candidate);
                            }
                        }
                        if (logical_members.empty()) {
                            send_error(task, "Logical StateTree state is unavailable", ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                        if (id_slot < 0) {
                            if (logical_members.size() != 1) {
                                send_error(task,
                                        "Logical StateTree state has multiple heads; provide an exact id_slot",
                                        ERROR_TYPE_INVALID_REQUEST);
                                break;
                            }
                            id_slot = logical_members.front()->id;
                        } else if (std::none_of(
                                logical_members.begin(), logical_members.end(),
                                [id_slot](const server_slot * member) { return member->id == id_slot; })) {
                            send_error(task, "Logical StateTree state does not own the requested slot",
                                    ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                    }

                    if (task.fork_id >= 0 &&
                            (id_slot < 0 || (size_t) id_slot >= slots.size())) {
                        send_error(task, "A generation-fenced fork_id requires an exact valid id_slot",
                                ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (id_slot != -1) {
                        if (task.node_id >= 0) {
                            if (!slot->is_fork_reserved() || slot->node_id != task.node_id ||
                                    (task.fork_id >= 0 && slot->fork_id != task.fork_id) ||
                                    (task.state_id >= 0 && slot->state_id != task.state_id) ||
                                    family_is_expired(*slot, ggml_time_us())) {
                                send_error(task, "StateTree branch node is unavailable", ERROR_TYPE_UNAVAILABLE);
                                if (slot->is_available()) {
                                    slot->callback_on_deferred(slot->id);
                                }
                                break;
                            }
                        } else if (task.fork_id >= 0) {
                            if (!slot->is_fork_reserved() || slot->fork_id != task.fork_id ||
                                    (task.state_id >= 0 && slot->state_id != task.state_id) ||
                                    family_is_expired(*slot, ggml_time_us())) {
                                send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                                if (slot->is_available()) {
                                    slot->callback_on_deferred(slot->id);
                                }
                                break;
                            }
                        } else if (slot->is_fork_reserved() && retention_enabled()) {
                            send_error(task, "A generation-fenced fork_id or node_id is required for this reserved slot",
                                    ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                        if (slot->is_fork_reserved()) {
                            touch_family(slot->fork_source_id, slot->fork_id, true);
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    } else if (slot->is_fork_reserved()) {
                        touch_family(slot->fork_source_id, slot->fork_id, true);
                    }

                    if (params_base.cache_idle_slots && !llama_adapter_cvec_seq_mode(ctx_tgt)) {
                        for (auto & slot : slots) {
                            if (slot.is_available()) {
                                SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear(false);
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            const bool was_fork_reserved = slot.is_fork_reserved();
                            slot.release();
                            if (llama_adapter_cvec_seq_mode(ctx_tgt) && !was_fork_reserved) {
                                slot.prompt_clear(false);
                            }
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;
                    int n_reserved_slots   = 0;

                    for (server_slot & slot : slots) {
                        const bool lease_pinned = slot.is_fork_reserved() &&
                            family_is_active(slot.fork_source_id, slot.fork_id);
                        json slot_data = slot.to_json(slots_debug == 0, lease_pinned);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else if (slot.is_fork_reserved()) {
                            n_reserved_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_reserved_slots    = n_reserved_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    const auto retention = get_retention_stats();
                    res->statetree_state_bytes = retention.state_bytes;
                    res->statetree_retained_bytes = retention.retained_bytes;
                    res->statetree_active_bytes = retention.active_bytes;
                    res->statetree_state_budget_bytes = params_base.statetree_max_state_bytes;
                    res->statetree_state_high_water_bytes = state_high_water_bytes;
                    res->statetree_retained_high_water_bytes = retained_high_water_bytes;
                    res->statetree_expired_total = statetree_expired_total;
                    res->statetree_evicted_total = statetree_evicted_total;
                    res->statetree_reclaimed_bytes_total = statetree_reclaimed_bytes_total;
                    res->statetree_renewed_total = statetree_renewed_total;
                    res->statetree_pressure_rejected_total = statetree_pressure_rejected_total;
                    res->statetree_snapshot_bytes = snapshot_bytes;
                    res->statetree_snapshot_budget_bytes = params_base.statetree_max_snapshot_bytes;
                    res->statetree_snapshot_high_water_bytes = snapshot_high_water_bytes;
                    res->statetree_snapshot_count = statetree_snapshots.size();
                    res->statetree_snapshot_content_count = statetree_snapshot_contents.size();
                    res->statetree_snapshots_captured_total = snapshots_captured_total;
                    res->statetree_snapshots_materialized_total = snapshots_materialized_total;
                    res->statetree_snapshots_erased_total = snapshots_erased_total;
                    res->statetree_snapshot_rejected_total = snapshot_rejected_total;
                    if (snapshot_store) {
                        res->statetree_durable_disk_bytes = durable_store_disk_bytes;
                        res->statetree_durable_disk_budget_bytes = durable_store_disk_budget_bytes;
                        res->statetree_durable_disk_high_water_bytes = durable_store_disk_high_water_bytes;
                        res->statetree_durable_content_count = durable_store_entries.size();
                        res->statetree_durable_recovered_temp_files = durable_store_recovered_temp_files;
                        res->statetree_durable_ignored_corrupt_files = durable_store_ignored_corrupt_files;
                        res->statetree_durable_runtime_integrity_failures = durable_store_runtime_integrity_failures;
                        res->statetree_durable_orphaned_disk_bytes = durable_store_orphaned_disk_bytes;
                    }
                    res->statetree_durable_spilled_total = durable_spilled_total;
                    res->statetree_durable_materialized_total = durable_materialized_total;
                    res->statetree_durable_erased_total = durable_erased_total;
                    res->statetree_durable_rejected_total = durable_rejected_total;
                    res->statetree_durable_io_pending = durable_io_pending.load();
                    res->statetree_durable_io_queue_high_water = durable_io_queue_high_water;
                    res->statetree_durable_io_completed_total = durable_io_completed_total;
                    res->statetree_durable_io_cancelled_loads_total = durable_io_cancelled_loads_total;
                    res->statetree_durable_io_reserved_disk_bytes = durable_io_reserved_disk_bytes;
                    res->statetree_durable_io_reserved_disk_high_water = durable_io_reserved_disk_high_water;
                    res->statetree_durable_io_reserved_load_bytes = durable_io_reserved_load_bytes;
                    res->statetree_durable_io_reserved_load_high_water = durable_io_reserved_load_high_water;
                    res->statetree_durable_manifest_refs = durable_manifest_refs.size();
                    res->statetree_durable_manifest_revision = durable_manifest_revision;
                    res->statetree_durable_manifest_file_bytes = durable_manifest_file_bytes;
                    res->statetree_durable_manifest_budget_bytes = durable_manifest_budget_bytes;
                    res->statetree_durable_manifest_high_water_bytes = durable_manifest_high_water_bytes;
                    res->statetree_durable_manifest_record_count = durable_manifest_record_count;
                    res->statetree_durable_manifest_recovered_temp_files =
                        durable_manifest_recovered_temp_files;
                    res->statetree_durable_manifest_recovered_tail_bytes =
                        durable_manifest_recovered_tail_bytes;
                    res->statetree_durable_manifest_compactions = durable_manifest_compactions;
                    res->statetree_durable_manifest_recovered_publish_commits =
                        durable_manifest_recovered_publish_commits;
                    res->statetree_durable_manifest_recovered_publish_aborts =
                        durable_manifest_recovered_publish_aborts;
                    res->statetree_durable_managed_count = durable_managed_digests.size();
                    res->statetree_durable_cache_evicted_total = durable_cache_evicted_total;
                    res->statetree_durable_cache_reclaimed_bytes_total =
                        durable_cache_reclaimed_bytes_total;
                    res->statetree_durable_managed_recovered_erases = durable_managed_recovered_erases;
                    res->statetree_durable_managed_recovered_bytes = durable_managed_recovered_bytes;
                    res->statetree_durable_retained_total = durable_retained_total;
                    res->statetree_durable_released_total = durable_released_total;
                    res->statetree_durable_compacted_total = durable_compacted_total;
                    res->n_statetree_families = retention.n_families;
                    res->n_statetree_active_families = retention.n_active_families;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!validate_slot_fence(task, *slot)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                    slot->callback_on_deferred(slot->id);
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!validate_slot_fence(task, *slot)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        const auto restored_nodes = get_statetree_node_refs({ slot });
                        record_statetree_event(
                                "restore_failed",
                                slot->state_id,
                                slot->fork_id,
                                slot->fork_source_id,
                                { slot->id },
                                slot->prompt_state_bytes(),
                                -1,
                                restored_nodes);
                        slot->prompt_clear(false);
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        slot->callback_on_deferred(slot->id);
                        break;
                    }
                    tokens.resize(token_count);
                    const auto restored_nodes = get_statetree_node_refs({ slot });
                    record_statetree_event(
                            "restore",
                            slot->state_id,
                            slot->fork_id,
                            slot->fork_source_id,
                            { slot->id },
                            slot->prompt_state_bytes(),
                            -1,
                            restored_nodes);
                    slot->prompt_metadata_clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                    slot->callback_on_deferred(slot->id);
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    server_slot * slot = resolve_slot_action_target(task, "Invalid slot ID");
                    if (slot == nullptr) {
                        break;
                    }
                    const int id_slot = slot->id;
                    if (!validate_slot_fence(task, *slot)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();
                    const int64_t erased_state_id = slot->state_id;
                    const int64_t erased_fork_id = slot->fork_id;
                    const int erased_source_id = slot->fork_source_id;
                    const size_t erased_state_bytes = slot->prompt_state_bytes();
                    const auto erased_nodes = get_statetree_node_refs({ slot });

                    record_statetree_event(
                            "erase",
                            erased_state_id,
                            erased_fork_id,
                            erased_source_id,
                            { id_slot },
                            erased_state_bytes,
                            -1,
                            erased_nodes);

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->state_id = erased_state_id;
                    res->node_id  = erased_nodes.empty() ? -1 : erased_nodes.front().node_id;
                    res->parent_node_id = erased_nodes.empty() ? -1 : erased_nodes.front().parent_node_id;
                    res->fork_id  = erased_fork_id;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                    slot->callback_on_deferred(slot->id);
                } break;
            case SERVER_TASK_TYPE_SLOT_FORK:
                {
                    if (!params_base.kv_unified) {
                        send_error(task, "Slot fork requires a unified KV cache", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    if (ctx_dft || spec) {
                        send_error(task, "Slot fork is not supported with speculative decoding", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!params_base.lora_adapters.empty()) {
                        send_error(task, "Slot fork is not supported with LoRA adapters", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    server_slot * source = resolve_slot_action_target(task, "Invalid source slot ID");
                    if (source == nullptr) {
                        break;
                    }
                    const int id_slot = source->id;
                    if (source->is_processing()) {
                        send_error(task, "Source slot is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (source->is_fork_reserved()) {
                        bool has_sibling = false;
                        for (const server_slot & slot : slots) {
                            if (slot.id != source->id &&
                                    slot.fork_source_id == source->id &&
                                    slot.fork_id == source->fork_id) {
                                has_sibling = true;
                                break;
                            }
                        }
                        if (!source->is_fork_root() ||
                                source->fork_id < 0 ||
                                source->fork_id != task.slot_action.fork_id ||
                                has_sibling) {
                            send_error(task, "Source slot is unavailable", ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                    } else if (task.slot_action.fork_id >= 0) {
                        send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (source->prompt.tokens.empty()) {
                        send_error(task, "Source slot has no cached prompt", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!source->lora.empty()) {
                        send_error(task, "Slot fork is not supported with LoRA adapters", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (task.slot_action.destinations.empty()) {
                        send_error(task, "Slot fork requires at least one destination", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    std::unordered_set<int> destination_ids;
                    std::vector<server_slot *> destinations;
                    destinations.reserve(task.slot_action.destinations.size());

                    bool valid = true;
                    for (const int destination_id : task.slot_action.destinations) {
                        if (destination_id < 0 || (size_t) destination_id >= slots.size()) {
                            send_error(task, "Invalid destination slot ID", ERROR_TYPE_INVALID_REQUEST);
                            valid = false;
                            break;
                        }
                        if (destination_id == id_slot || !destination_ids.insert(destination_id).second) {
                            send_error(task, "Slot fork destinations must be unique and exclude the source",
                                    ERROR_TYPE_INVALID_REQUEST);
                            valid = false;
                            break;
                        }

                        server_slot * destination = get_slot_by_id(destination_id);
                        GGML_ASSERT(destination != nullptr);
                        if (!destination->is_available()) {
                            send_error(task, "Destination slot is unavailable", ERROR_TYPE_UNAVAILABLE);
                            valid = false;
                            break;
                        }

                        destinations.push_back(destination);
                    }
                    if (!valid) {
                        break;
                    }

                    const int64_t t_start = ggml_time_us();
                    statetree_fork_family_result fork_res;
                    {
                        std::string fork_error;
                        if (!statetree_fork_family(source, destinations, fork_res, fork_error)) {
                            send_error(task, fork_error, ERROR_TYPE_SERVER);
                            break;
                        }
                    }
                    const int64_t parent_fork_id = fork_res.parent_fork_id;
                    const int64_t parent_node_id = fork_res.parent_node_id;
                    const int64_t state_id = fork_res.state_id;
                    const int64_t fork_id = fork_res.fork_id;
                    std::vector<server_slot *> family_members = destinations;
                    family_members.push_back(source);
                    std::sort(family_members.begin(), family_members.end(), [](const auto * left, const auto * right) {
                        return left->id < right->id;
                    });

                    std::vector<int> family_slots;
                    family_slots.reserve(family_members.size());
                    for (const server_slot * member : family_members) {
                        family_slots.push_back(member->id);
                    }
                    const auto family_nodes = get_statetree_node_refs(family_members);
                    size_t family_state_bytes = 0;
                    for (const server_slot & member : slots) {
                        if (member.state_id == state_id && member.fork_id == fork_id) {
                            family_state_bytes += member.prompt_state_bytes();
                        }
                    }
                    record_statetree_event(
                            "fork",
                            state_id,
                            fork_id,
                            id_slot,
                            family_slots,
                            family_state_bytes,
                            parent_fork_id,
                            family_nodes);

                    const int64_t t_end = ggml_time_us();
                    const auto retention = get_retention_stats();

                    auto res = std::make_unique<server_task_result_slot_fork>();
                    res->id           = task.id;
                    res->id_slot      = id_slot;
                    res->state_id     = state_id;
                    res->node_id      = source->node_id;
                    res->parent_node_id = parent_node_id;
                    res->fork_id      = fork_id;
                    res->nodes        = get_statetree_nodes_json(family_members);
                    res->destinations = task.slot_action.destinations;
                    res->n_tokens     = source->prompt.tokens.size();
                    res->t_ms         = (t_end - t_start) / 1000.0;
                    res->retention_enabled = retention_enabled();
                    res->lease_remaining_ms = lease_remaining_ms(*source, t_end);
                    res->state_bytes = retention.state_bytes;
                    res->retained_bytes = retention.retained_bytes;
                    res->state_budget_bytes = params_base.statetree_max_state_bytes;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_COMMIT:
                {
                    if (!params_base.kv_unified) {
                        send_error(task, "Slot commit requires a unified KV cache", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    if (ctx_dft || spec) {
                        send_error(task, "Slot commit is not supported with speculative decoding", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!params_base.lora_adapters.empty()) {
                        send_error(task, "Slot commit is not supported with LoRA adapters", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    server_slot * winner = resolve_slot_action_target(task, "Invalid winner slot ID");
                    if (winner == nullptr) {
                        break;
                    }
                    if (winner->is_processing()) {
                        send_error(task, "Winner slot is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (!winner->is_fork_reserved() || winner->fork_id != task.slot_action.fork_id) {
                        send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (family_is_expired(*winner, ggml_time_us())) {
                        send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (winner->prompt.tokens.empty()) {
                        send_error(task, "Winner slot has no cached prompt", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int source_id = winner->fork_source_id;
                    const int64_t state_id = winner->state_id;
                    std::vector<server_slot *> family;
                    std::vector<int> released;
                    family.reserve(slots.size());
                    released.reserve(slots.size());

                    bool valid = true;
                    for (server_slot & slot : slots) {
                        if (slot.fork_source_id != source_id || slot.fork_id != task.slot_action.fork_id) {
                            continue;
                        }
                        GGML_ASSERT(slot.state_id == state_id);
                        if (slot.is_processing()) {
                            send_error(task, "Fork family is still processing", ERROR_TYPE_UNAVAILABLE);
                            valid = false;
                            break;
                        }

                        family.push_back(&slot);
                        if (slot.id != winner->id) {
                            released.push_back(slot.id);
                        }
                    }
                    if (!valid) {
                        break;
                    }
                    GGML_ASSERT(!family.empty());

                    const int64_t t_start = ggml_time_us();
                    const auto family_nodes = get_statetree_node_refs(family);
                    for (server_slot * slot : family) {
                        if (slot != winner) {
                            slot->prompt_clear(false);
                        }
                    }

                    winner->fork_source_id = winner->id;
                    touch_family(winner->id, winner->fork_id, false);
                    const int64_t t_end = ggml_time_us();
                    const auto retention = get_retention_stats();
                    record_statetree_event(
                            "commit",
                            state_id,
                            winner->fork_id,
                            winner->id,
                            released,
                            winner->prompt_state_bytes(),
                            -1,
                            family_nodes);

                    auto res = std::make_unique<server_task_result_slot_commit>();
                    res->id        = task.id;
                    res->id_slot   = winner->id;
                    res->state_id  = state_id;
                    res->node_id   = winner->node_id;
                    res->parent_node_id = winner->parent_node_id;
                    res->source_id = source_id;
                    res->fork_id   = winner->fork_id;
                    res->nodes     = json::array();
                    for (const auto & node : family_nodes) {
                        res->nodes.push_back(node.to_json());
                    }
                    res->released  = released;
                    res->n_tokens  = winner->prompt.tokens.size();
                    res->t_ms      = (t_end - t_start) / 1000.0;
                    res->retention_enabled = retention_enabled();
                    res->lease_remaining_ms = lease_remaining_ms(*winner, t_end);
                    res->state_bytes = retention.state_bytes;
                    res->retained_bytes = retention.retained_bytes;
                    res->state_budget_bytes = params_base.statetree_max_state_bytes;
                    queue_results.send(std::move(res));

                    for (const int released_id : released) {
                        slots[released_id].callback_on_deferred(released_id);
                    }
                } break;
            case SERVER_TASK_TYPE_SLOT_RENEW:
                {
                    if (!params_base.kv_unified) {
                        send_error(task, "Slot renew requires a unified KV cache", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    server_slot * member = resolve_slot_action_target(task, "Invalid slot ID");
                    if (member == nullptr) {
                        break;
                    }
                    const int id_slot = member->id;
                    if (!member->is_fork_reserved() || member->fork_id != task.slot_action.fork_id ||
                            family_is_expired(*member, ggml_time_us())) {
                        send_error(task, "Fork transaction is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();
                    const int source_id = member->fork_source_id;
                    std::vector<int> members;
                    std::vector<server_slot *> family_members;
                    for (const server_slot & slot : slots) {
                        if (slot.fork_source_id == source_id && slot.fork_id == task.slot_action.fork_id) {
                            members.push_back(slot.id);
                            family_members.push_back(&slots[slot.id]);
                        }
                    }
                    const auto family_nodes = get_statetree_node_refs(family_members);
                    touch_family(source_id, task.slot_action.fork_id, true);
                    const int64_t t_end = ggml_time_us();
                    const auto retention = get_retention_stats();
                    record_statetree_event(
                            "renew",
                            member->state_id,
                            task.slot_action.fork_id,
                            source_id,
                            members,
                            std::accumulate(
                                slots.begin(), slots.end(), size_t(0),
                                [&](size_t total, const server_slot & slot) {
                                    return total + (slot.state_id == member->state_id &&
                                            slot.fork_id == task.slot_action.fork_id
                                        ? slot.prompt_state_bytes()
                                        : 0);
                                }),
                            -1,
                            family_nodes);

                    auto res = std::make_unique<server_task_result_slot_renew>();
                    res->id = task.id;
                    res->id_slot = id_slot;
                    res->state_id = member->state_id;
                    res->node_id = member->node_id;
                    res->parent_node_id = member->parent_node_id;
                    res->source_id = source_id;
                    res->fork_id = task.slot_action.fork_id;
                    res->nodes = get_statetree_nodes_json(family_members);
                    res->members = std::move(members);
                    res->lease_remaining_ms = lease_remaining_ms(*member, t_end);
                    res->state_bytes = retention.state_bytes;
                    res->retained_bytes = retention.retained_bytes;
                    res->state_budget_bytes = params_base.statetree_max_state_bytes;
                    res->t_ms = (t_end - t_start) / 1000.0;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_CAPTURE:
                {
                    if (params_base.statetree_max_snapshot_bytes == 0) {
                        send_error(task, "StateTree snapshots are disabled; set --statetree-max-snapshot-bytes",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    if (ctx_dft || spec) {
                        send_error(task, "StateTree snapshots are not supported with speculative decoding",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!params_base.lora_adapters.empty()) {
                        send_error(task, "StateTree snapshots are not supported with LoRA adapters",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    server_slot * source = resolve_slot_action_target(task, "Invalid source slot ID");
                    if (source == nullptr) {
                        break;
                    }
                    if (source->is_processing()) {
                        send_error(task, "Source node is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (source->prompt.tokens.empty()) {
                        send_error(task, "Source node has no cached prompt", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!source->lora.empty()) {
                        send_error(task, "StateTree snapshots are not supported with LoRA adapters",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (statetree_next_snapshot_id > (uint64_t) std::numeric_limits<int64_t>::max()) {
                        send_error(task, "StateTree snapshot identity space is exhausted", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();
                    statetree_snapshot snapshot;
                    snapshot.state_id = source->state_id;
                    snapshot.source_node_id = source->node_id;
                    snapshot.source_fork_id = source->fork_id;
                    snapshot.source_slot = source->id;
                    snapshot.captured_at_us = t_start;
                    std::shared_ptr<statetree_snapshot_payload> candidate;
                    try {
                        candidate = std::make_shared<statetree_snapshot_payload>();
                        candidate->tokens = source->prompt.tokens.get_tokens();
                    } catch (const std::exception & e) {
                        send_error(task, std::string("Failed to allocate StateTree snapshot: ") + e.what(),
                                ERROR_TYPE_SERVER);
                        break;
                    }

                    const size_t state_size = llama_state_seq_get_size_ext(
                            ctx_tgt, source->id, LLAMA_STATE_SEQ_FLAGS_NONE);
                    const size_t cvec_trailer_bytes = llama_adapter_cvec_seq_mode(ctx_tgt)
                        ? snapshot_cvec_trailer_size
                        : 0;
                    if (state_size < sizeof(uint32_t) + sizeof(llama_seq_id) ||
                            state_size > std::numeric_limits<size_t>::max() - cvec_trailer_bytes ||
                            candidate->tokens.size() >
                                (std::numeric_limits<size_t>::max() - state_size - cvec_trailer_bytes) /
                                sizeof(llama_token)) {
                        send_error(task, "StateTree snapshot payload size is invalid", ERROR_TYPE_SERVER);
                        break;
                    }
                    const size_t candidate_payload_bytes = state_size + cvec_trailer_bytes +
                        candidate->tokens.size() * sizeof(llama_token);
                    if (candidate_payload_bytes > params_base.statetree_max_snapshot_bytes) {
                        snapshot_rejected_total++;
                        send_error(task, "StateTree snapshot exceeds the per-content byte ceiling",
                                ERROR_TYPE_UNAVAILABLE);
                        break;
                    }

                    try {
                        candidate->state.resize(state_size);
                    } catch (const std::exception & e) {
                        send_error(task, std::string("Failed to allocate StateTree snapshot: ") + e.what(),
                                ERROR_TYPE_SERVER);
                        break;
                    }
                    const size_t written = llama_state_seq_get_data_ext(
                            ctx_tgt, candidate->state.data(), candidate->state.size(), source->id,
                            LLAMA_STATE_SEQ_FLAGS_NONE);
                    if (written != candidate->state.size()) {
                        send_error(task, "Failed to serialize the complete StateTree snapshot", ERROR_TYPE_SERVER);
                        break;
                    }

                    // The sequence header contains a physical slot ID. Canonicalize it so
                    // identical content captured from different fork heads has one digest.
                    const llama_seq_id canonical_seq_id = 0;
                    std::memcpy(candidate->state.data() + sizeof(uint32_t),
                            &canonical_seq_id, sizeof(canonical_seq_id));
                    if (llama_adapter_cvec_seq_mode(ctx_tgt)) {
                        snapshot_append_cvec_scale(
                                candidate->state,
                                llama_adapter_cvec_seq_get(ctx_tgt, source->id));
                    }
                    candidate->digest = snapshot_content_digest(candidate->tokens, candidate->state);

                    bool deduplicated = false;
                    bool inserted_content = false;
                    size_t inserted_content_bytes = 0;
                    const auto existing = statetree_snapshot_contents.find(candidate->digest);
                    if (existing != statetree_snapshot_contents.end()) {
                        if (existing->second->tokens != candidate->tokens ||
                                existing->second->state != candidate->state) {
                            send_error(task, "StateTree snapshot digest collision", ERROR_TYPE_SERVER);
                            break;
                        }
                        snapshot.payload = existing->second;
                        deduplicated = true;
                    } else {
                        const uint64_t budget = params_base.statetree_max_snapshot_bytes;
                        GGML_ASSERT(candidate->payload_bytes() == candidate_payload_bytes);
                        if (snapshot_bytes > budget - candidate_payload_bytes) {
                            snapshot_rejected_total++;
                            send_error(task, "StateTree snapshot budget is exhausted", ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                        snapshot.payload = candidate;
                        try {
                            const auto inserted = statetree_snapshot_contents.emplace(
                                    candidate->digest, snapshot.payload);
                            GGML_ASSERT(inserted.second);
                        } catch (const std::exception & e) {
                            send_error(task, std::string("Failed to retain StateTree snapshot content: ") + e.what(),
                                    ERROR_TYPE_SERVER);
                            break;
                        }
                        inserted_content = true;
                        inserted_content_bytes = candidate_payload_bytes;
                        snapshot_bytes += inserted_content_bytes;
                    }
                    snapshot.snapshot_id = (int64_t) statetree_next_snapshot_id;

                    const int64_t snapshot_id = snapshot.snapshot_id;
                    decltype(statetree_snapshots)::iterator inserted_snapshot;
                    try {
                        const auto inserted = statetree_snapshots.emplace(snapshot_id, std::move(snapshot));
                        GGML_ASSERT(inserted.second);
                        inserted_snapshot = inserted.first;
                    } catch (const std::exception & e) {
                        if (inserted_content) {
                            GGML_ASSERT(snapshot_bytes >= inserted_content_bytes);
                            snapshot_bytes -= inserted_content_bytes;
                            GGML_ASSERT(statetree_snapshot_contents.erase(candidate->digest) == 1);
                        }
                        send_error(task, std::string("Failed to retain StateTree snapshot handle: ") + e.what(),
                                ERROR_TYPE_SERVER);
                        break;
                    }
                    statetree_next_snapshot_id++;
                    snapshots_captured_total++;
                    snapshot_high_water_bytes = std::max(snapshot_high_water_bytes, snapshot_bytes);

                    const int64_t t_end = ggml_time_us();
                    auto res = std::make_unique<server_task_result_snapshot>();
                    res->id = task.id;
                    res->result = inserted_snapshot->second.to_json();
                    res->result["action"] = "capture";
                    res->result["deduplicated"] = deduplicated;
                    res->result["snapshot_bytes"] = snapshot_bytes;
                    res->result["snapshot_budget_bytes"] = params_base.statetree_max_snapshot_bytes;
                    res->result["snapshot_content_count"] = statetree_snapshot_contents.size();
                    res->result["timings"] = {{"capture_ms", (t_end - t_start) / 1000.0}};
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_MATERIALIZE:
                {
                    statetree_snapshot * snapshot = resolve_snapshot(task);
                    if (snapshot == nullptr) {
                        break;
                    }
                    json materialized;
                    if (!materialize_snapshot_content(
                            task,
                            snapshot->payload->tokens,
                            snapshot->payload->state,
                            snapshot->state_id,
                            snapshot->source_node_id,
                            snapshot->snapshot_id,
                            snapshot->payload->digest,
                            materialized)) {
                        break;
                    }
                    snapshots_materialized_total++;
                    auto res = std::make_unique<server_task_result_snapshot>();
                    res->id = task.id;
                    res->result = std::move(materialized);
                    res->result["action"] = "materialize";
                    res->result["snapshot_id"] = snapshot->snapshot_id;
                    res->result["cold"] = false;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_ERASE:
                {
                    statetree_snapshot * snapshot = resolve_snapshot(task);
                    if (snapshot == nullptr) {
                        break;
                    }
                    const json erased = snapshot->to_json();
                    const int64_t erased_snapshot_id = snapshot->snapshot_id;
                    const size_t erased_bytes = snapshot->payload_bytes();
                    const std::string digest = snapshot->payload->digest;
                    const bool last_handle = std::count_if(
                            statetree_snapshots.begin(), statetree_snapshots.end(),
                            [&](const auto & entry) { return entry.second.payload->digest == digest; }) == 1;
                    snapshots_erased_total++;
                    statetree_snapshots.erase(erased_snapshot_id);
                    if (last_handle) {
                        GGML_ASSERT(snapshot_bytes >= erased_bytes);
                        snapshot_bytes -= erased_bytes;
                        GGML_ASSERT(statetree_snapshot_contents.erase(digest) == 1);
                    }

                    auto res = std::make_unique<server_task_result_snapshot>();
                    res->id = task.id;
                    res->result = erased;
                    res->result["action"] = "erase";
                    res->result["content_reclaimed"] = last_handle;
                    res->result["snapshot_bytes"] = snapshot_bytes;
                    res->result["snapshot_budget_bytes"] = params_base.statetree_max_snapshot_bytes;
                    res->result["snapshot_content_count"] = statetree_snapshot_contents.size();
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_SPILL:
            case SERVER_TASK_TYPE_SNAPSHOT_PUBLISH:
            case SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE:
                {
                    if (!snapshot_store) {
                        send_error(task, "Durable StateTree snapshot content is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    statetree_snapshot * snapshot = resolve_snapshot(task);
                    if (snapshot == nullptr) {
                        break;
                    }
                    const std::string digest = snapshot->payload->digest;
                    const bool managed_publish = task.type == SERVER_TASK_TYPE_SNAPSHOT_PUBLISH ||
                        task.type == SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE;
                    if (managed_publish &&
                            snapshot->payload->payload_bytes() > params_base.statetree_max_snapshot_load_bytes) {
                        durable_rejected_total++;
                        send_error(task,
                                "managed snapshot publish exceeds the configured recovery load budget",
                                ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    if (task.type == SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE) {
                        const auto head = durable_logical_heads.find(task.slot_action.head_name);
                        const bool committed_retry = head != durable_logical_heads.end() &&
                            task.slot_action.expected_generation < std::numeric_limits<uint64_t>::max() &&
                            head->second.generation == task.slot_action.expected_generation + 1 &&
                            head->second.digest == digest &&
                            head->second.parent_digest == task.slot_action.expected_digest;
                        const bool expected = head != durable_logical_heads.end() &&
                            head->second.generation == task.slot_action.expected_generation &&
                            head->second.digest == task.slot_action.expected_digest;
                        if (!expected && !committed_retry) {
                            send_error(task, "logical head compare-and-swap conflict",
                                    ERROR_TYPE_UNAVAILABLE);
                            break;
                        }
                    }
                    uint64_t disk_reservation = 0;
                    if (durable_store_entries.find(digest) == durable_store_entries.end() &&
                            durable_io_pending_spill_digests.find(digest) == durable_io_pending_spill_digests.end()) {
                        try {
                            disk_reservation = snapshot_store->projected_file_bytes(
                                    snapshot->payload->tokens.size(), snapshot->payload->state.size());
                        } catch (const server_snapshot_store_error & error) {
                            send_error(task, error.what(), ERROR_TYPE_SERVER);
                            break;
                        }
                        const uint64_t disk_bytes = durable_store_disk_bytes;
                        const uint64_t disk_budget = durable_store_disk_budget_bytes;
                        const bool fits_without_reclamation = disk_bytes <= disk_budget &&
                            durable_io_reserved_disk_bytes <= disk_budget - disk_bytes &&
                            disk_reservation <= disk_budget - disk_bytes - durable_io_reserved_disk_bytes;
                        if (!fits_without_reclamation) {
                            if (!managed_publish) {
                                durable_rejected_total++;
                                send_error(task, "durable snapshot content disk budget is exhausted",
                                        ERROR_TYPE_UNAVAILABLE);
                                break;
                            }
                            disk_reservation = 0;
                        } else {
                            durable_io_reserved_disk_bytes += disk_reservation;
                            durable_io_reserved_disk_high_water = std::max(
                                    durable_io_reserved_disk_high_water, durable_io_reserved_disk_bytes);
                            durable_io_pending_spill_digests.insert(digest);
                        }
                    }

                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = task.type == SERVER_TASK_TYPE_SNAPSHOT_PUBLISH
                        ? durable_io_operation::publish
                        : task.type == SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE
                            ? durable_io_operation::publish_advance
                            : durable_io_operation::spill;
                    job.request = std::move(task);
                    job.hot_payload = snapshot->payload;
                    job.digest = digest;
                    job.owner = job.request.slot_action.owner;
                    job.retention_class = job.request.slot_action.retention_class;
                    job.disk_reservation_bytes = disk_reservation;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        if (disk_reservation > 0) {
                            GGML_ASSERT(durable_io_reserved_disk_bytes >= disk_reservation);
                            durable_io_reserved_disk_bytes -= disk_reservation;
                            durable_io_pending_spill_digests.erase(digest);
                        }
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_CONTENT_MATERIALIZE:
                {
                    if (!snapshot_store) {
                        send_error(task, "Durable StateTree snapshot content is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    if (ctx_dft || spec || !params_base.lora_adapters.empty()) {
                        send_error(task, "StateTree snapshot materialization is unavailable with the active model configuration",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (task.slot_action.id_slot >= 0 && get_slot_by_id(task.slot_action.id_slot) == nullptr) {
                        send_error(task, "Invalid destination slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    server_snapshot_store_entry entry;
                    const auto found_entry = durable_store_entries.find(task.slot_action.digest);
                    if (found_entry == durable_store_entries.end()) {
                        send_error(task, "durable snapshot content is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    entry = found_entry->second;
                    const uint64_t load_budget = params_base.statetree_max_snapshot_load_bytes;
                    if (entry.payload_bytes > load_budget ||
                            durable_io_reserved_load_bytes > load_budget - entry.payload_bytes) {
                        durable_rejected_total++;
                        send_error(task, "durable snapshot cold-load reservation budget is exhausted",
                                ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    durable_io_reserved_load_bytes += entry.payload_bytes;
                    durable_io_reserved_load_high_water = std::max(
                            durable_io_reserved_load_high_water, durable_io_reserved_load_bytes);

                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = durable_io_operation::load;
                    job.request = std::move(task);
                    job.digest = entry.digest;
                    job.max_load_bytes = load_budget;
                    job.load_reservation_bytes = entry.payload_bytes;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        GGML_ASSERT(durable_io_reserved_load_bytes >= entry.payload_bytes);
                        durable_io_reserved_load_bytes -= entry.payload_bytes;
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_CONTENT_ERASE:
                {
                    if (!snapshot_store) {
                        send_error(task, "Durable StateTree snapshot content is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    server_snapshot_store_entry entry;
                    const auto found_entry = durable_store_entries.find(task.slot_action.digest);
                    if (found_entry == durable_store_entries.end()) {
                        send_error(task, "durable snapshot content is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    entry = found_entry->second;
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = durable_io_operation::erase;
                    job.request = std::move(task);
                    job.digest = entry.digest;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RETAIN:
            case SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RELEASE:
                {
                    if (!snapshot_manifest) {
                        send_error(task, "Durable StateTree snapshot ownership is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = task.type == SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RETAIN
                        ? durable_io_operation::retain
                        : durable_io_operation::release;
                    job.request = std::move(task);
                    job.digest = job.request.slot_action.digest;
                    job.owner = job.request.slot_action.owner;
                    job.retention_class = job.request.slot_action.retention_class;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_COMPACT:
                {
                    if (!snapshot_manifest) {
                        send_error(task, "Durable StateTree snapshot ownership is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = durable_io_operation::compact;
                    job.request = std::move(task);
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_PRUNE:
                {
                    if (!snapshot_manifest) {
                        send_error(task, "Durable StateTree snapshot ownership is disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (task.slot_action.target_bytes > durable_store_disk_budget_bytes) {
                        send_error(task, "snapshot prune target exceeds the durable disk budget",
                                ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = durable_io_operation::prune;
                    job.request = std::move(task);
                    job.target_disk_bytes = job.request.slot_action.target_bytes;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_HEAD_CREATE:
            case SERVER_TASK_TYPE_SNAPSHOT_HEAD_ADVANCE:
            case SERVER_TASK_TYPE_SNAPSHOT_HEAD_DELETE:
                {
                    if (!snapshot_manifest) {
                        send_error(task, "Durable StateTree logical heads are disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = task.type == SERVER_TASK_TYPE_SNAPSHOT_HEAD_CREATE
                        ? durable_io_operation::head_create
                        : task.type == SERVER_TASK_TYPE_SNAPSHOT_HEAD_ADVANCE
                            ? durable_io_operation::head_advance
                            : durable_io_operation::head_delete;
                    job.request = std::move(task);
                    job.digest = job.request.slot_action.digest;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SNAPSHOT_HEAD_MATERIALIZE:
                {
                    if (!snapshot_manifest) {
                        send_error(task, "Durable StateTree logical heads are disabled",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    if (ctx_dft || spec || !params_base.lora_adapters.empty()) {
                        send_error(task, "StateTree logical head materialization is unavailable with the active model configuration",
                                ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (task.slot_action.id_slot >= 0 && get_slot_by_id(task.slot_action.id_slot) == nullptr) {
                        send_error(task, "Invalid destination slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    const auto head = durable_logical_heads.find(task.slot_action.head_name);
                    if (head == durable_logical_heads.end() ||
                            head->second.generation != task.slot_action.expected_generation ||
                            head->second.digest != task.slot_action.expected_digest) {
                        send_error(task, "logical head compare-and-swap conflict", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    const auto entry = durable_store_entries.find(head->second.digest);
                    if (entry == durable_store_entries.end()) {
                        send_error(task, "logical head content is unavailable", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    const uint64_t load_budget = params_base.statetree_max_snapshot_load_bytes;
                    if (entry->second.payload_bytes > load_budget ||
                            durable_io_reserved_load_bytes > load_budget - entry->second.payload_bytes) {
                        durable_rejected_total++;
                        send_error(task, "durable snapshot cold-load reservation budget is exhausted",
                                ERROR_TYPE_UNAVAILABLE);
                        break;
                    }
                    durable_io_reserved_load_bytes += entry->second.payload_bytes;
                    durable_io_reserved_load_high_water = std::max(
                            durable_io_reserved_load_high_water, durable_io_reserved_load_bytes);
                    durable_io_job job;
                    job.id = durable_io_next_id++;
                    job.operation = durable_io_operation::head_load;
                    job.request = std::move(task);
                    job.digest = entry->second.digest;
                    job.max_load_bytes = load_budget;
                    job.load_reservation_bytes = entry->second.payload_bytes;
                    job.enqueued_us = ggml_time_us();
                    if (!enqueue_durable_io(job)) {
                        GGML_ASSERT(durable_io_reserved_load_bytes >= entry->second.payload_bytes);
                        durable_io_reserved_load_bytes -= entry->second.payload_bytes;
                        send_error(job.request, "durable snapshot I/O worker is unavailable", ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_DURABLE_IO_COMPLETE:
                {
                    durable_io_completion completion;
                    if (!take_durable_io_completion(task.slot_action.durable_io_id, completion)) {
                        break;
                    }
                    release_durable_io_reservations(completion.job);
                    apply_durable_store_cache(completion);
                    apply_durable_manifest_cache(completion);
                    for (const auto & eviction : completion.cache_evictions) {
                        if (eviction.object_erased) {
                            durable_cache_evicted_total++;
                            durable_cache_reclaimed_bytes_total += eviction.file_bytes;
                        }
                    }
                    server_task & request = completion.job.request;
                    const int64_t t_complete = ggml_time_us();
                    const double queue_ms = (completion.started_us - completion.job.enqueued_us) / 1000.0;
                    const double io_ms = (completion.finished_us - completion.started_us) / 1000.0;
                    const double completion_wait_ms = (t_complete - completion.finished_us) / 1000.0;

                    if (!completion.success) {
                        if (completion.unavailable) {
                            durable_rejected_total++;
                        }
                        send_error(request, completion.error,
                                completion.unavailable ? ERROR_TYPE_UNAVAILABLE : ERROR_TYPE_SERVER);
                        break;
                    }

                    switch (completion.job.operation) {
                        case durable_io_operation::spill:
                            {
                                durable_spilled_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "spill"},
                                    {"snapshot_id", request.slot_action.snapshot_id},
                                    {"digest", completion.spill.entry.digest},
                                    {"deduplicated", completion.spill.deduplicated},
                                    {"n_tokens", completion.spill.entry.n_tokens},
                                    {"state_bytes", completion.spill.entry.state_bytes},
                                    {"payload_bytes", completion.spill.entry.payload_bytes},
                                    {"file_bytes", completion.spill.entry.file_bytes},
                                    {"disk_bytes", completion.disk_bytes},
                                    {"disk_budget_bytes", completion.disk_budget_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"spill_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::publish_advance:
                            {
                                durable_spilled_total++;
                                durable_retained_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                json evictions = json::array();
                                for (const auto & eviction : completion.cache_evictions) {
                                    evictions.push_back({
                                        {"digest", eviction.digest},
                                        {"released_refs", eviction.released_refs},
                                        {"file_bytes", eviction.file_bytes},
                                    });
                                }
                                const auto & committed = completion.publish_advance;
                                res->result = {
                                    {"action", "publish_advance"},
                                    {"snapshot_id", request.slot_action.snapshot_id},
                                    {"digest", completion.spill.entry.digest},
                                    {"owner", committed.ref.owner},
                                    {"retention_class", committed.ref.retention_class},
                                    {"object_deduplicated", completion.spill.deduplicated},
                                    {"transaction_deduplicated", committed.deduplicated},
                                    {"head", {
                                        {"name", committed.head.name},
                                        {"generation", committed.head.generation},
                                        {"digest", committed.head.digest},
                                        {"parent_digest", committed.head.parent_digest},
                                        {"revision", committed.head.revision},
                                    }},
                                    {"n_tokens", completion.spill.entry.n_tokens},
                                    {"state_bytes", completion.spill.entry.state_bytes},
                                    {"payload_bytes", completion.spill.entry.payload_bytes},
                                    {"file_bytes", completion.spill.entry.file_bytes},
                                    {"disk_bytes", completion.disk_bytes},
                                    {"disk_budget_bytes", completion.disk_budget_bytes},
                                    {"manifest_revision", committed.revision},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"manifest_budget_bytes", completion.manifest_budget_bytes},
                                    {"ref_revision", committed.ref.revision},
                                    {"cache_evictions", std::move(evictions)},
                                    {"cache_reclaimed_bytes", completion.prune_disk_bytes_before -
                                        completion.prune_disk_bytes_after},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::publish:
                            {
                                durable_spilled_total++;
                                durable_retained_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                json evictions = json::array();
                                for (const auto & eviction : completion.cache_evictions) {
                                    evictions.push_back({
                                        {"digest", eviction.digest},
                                        {"released_refs", eviction.released_refs},
                                        {"file_bytes", eviction.file_bytes},
                                    });
                                }
                                res->result = {
                                    {"action", "publish"},
                                    {"snapshot_id", request.slot_action.snapshot_id},
                                    {"digest", completion.spill.entry.digest},
                                    {"owner", completion.manifest.ref.owner},
                                    {"retention_class", completion.manifest.ref.retention_class},
                                    {"object_deduplicated", completion.spill.deduplicated},
                                    {"ownership_deduplicated", completion.publish_begin.deduplicated},
                                    {"n_tokens", completion.spill.entry.n_tokens},
                                    {"state_bytes", completion.spill.entry.state_bytes},
                                    {"payload_bytes", completion.spill.entry.payload_bytes},
                                    {"file_bytes", completion.spill.entry.file_bytes},
                                    {"disk_bytes", completion.disk_bytes},
                                    {"disk_budget_bytes", completion.disk_budget_bytes},
                                    {"manifest_revision", completion.manifest.revision},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"manifest_budget_bytes", completion.manifest_budget_bytes},
                                    {"ref_revision", completion.manifest.ref.revision},
                                    {"ref_count", durable_manifest_ref_count(completion.manifest.ref.digest)},
                                    {"cache_evictions", std::move(evictions)},
                                    {"cache_reclaimed_bytes", completion.prune_disk_bytes_before -
                                        completion.prune_disk_bytes_after},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::load:
                            {
                                if (!queue_results.is_waiting_task_id(request.id)) {
                                    durable_io_cancelled_loads_total++;
                                    break;
                                }
                                json materialized;
                                if (!materialize_snapshot_content(
                                        request,
                                        completion.load.tokens,
                                        completion.load.state,
                                        -1,
                                        -1,
                                        -1,
                                        completion.load.entry.digest,
                                        materialized)) {
                                    break;
                                }
                                durable_materialized_total++;
                                const int64_t t_end = ggml_time_us();
                                materialized["action"] = "materialize";
                                materialized["snapshot_id"] = nullptr;
                                materialized["cold"] = true;
                                materialized["file_bytes"] = completion.load.entry.file_bytes;
                                materialized["timings"]["queue_ms"] = queue_ms;
                                materialized["timings"]["load_verify_ms"] = io_ms;
                                materialized["timings"]["completion_wait_ms"] = completion_wait_ms;
                                materialized["timings"]["total_ms"] =
                                    (t_end - completion.job.enqueued_us) / 1000.0;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = std::move(materialized);
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::erase:
                            {
                                durable_erased_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "erase"},
                                    {"digest", completion.erase.digest},
                                    {"file_bytes", completion.erase.file_bytes},
                                    {"disk_bytes", completion.disk_bytes},
                                    {"disk_budget_bytes", completion.disk_budget_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::retain:
                            {
                                durable_retained_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "retain"},
                                    {"owner", completion.manifest.ref.owner},
                                    {"digest", completion.manifest.ref.digest},
                                    {"retention_class", completion.manifest.ref.retention_class},
                                    {"ref_revision", completion.manifest.ref.revision},
                                    {"manifest_revision", completion.manifest.revision},
                                    {"ref_count", durable_manifest_ref_count(completion.manifest.ref.digest)},
                                    {"deduplicated", completion.manifest.deduplicated},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"manifest_budget_bytes", completion.manifest_budget_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::release:
                            {
                                durable_released_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "release"},
                                    {"owner", completion.manifest.ref.owner},
                                    {"digest", completion.manifest.ref.digest},
                                    {"retention_class", completion.manifest.ref.retention_class},
                                    {"released_ref_revision", completion.manifest.ref.revision},
                                    {"manifest_revision", completion.manifest.revision},
                                    {"ref_count", durable_manifest_ref_count(completion.manifest.ref.digest)},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"manifest_budget_bytes", completion.manifest_budget_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::compact:
                            {
                                durable_compacted_total++;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "compact"},
                                    {"manifest_revision", completion.compact.revision},
                                    {"refs", completion.compact.refs},
                                    {"records_before", completion.compact.records_before},
                                    {"records_after", completion.compact.records_after},
                                    {"bytes_before", completion.compact.bytes_before},
                                    {"bytes_after", completion.compact.bytes_after},
                                    {"manifest_budget_bytes", completion.manifest_budget_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::prune:
                            {
                                json evictions = json::array();
                                uint64_t released_refs = 0;
                                for (const auto & eviction : completion.cache_evictions) {
                                    released_refs += eviction.released_refs;
                                    evictions.push_back({
                                        {"digest", eviction.digest},
                                        {"released_refs", eviction.released_refs},
                                        {"file_bytes", eviction.file_bytes},
                                    });
                                }
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", "prune"},
                                    {"target_disk_bytes", completion.job.target_disk_bytes},
                                    {"disk_bytes_before", completion.prune_disk_bytes_before},
                                    {"disk_bytes_after", completion.prune_disk_bytes_after},
                                    {"reclaimed_bytes", completion.prune_disk_bytes_before -
                                        completion.prune_disk_bytes_after},
                                    {"released_refs", released_refs},
                                    {"evictions", std::move(evictions)},
                                    {"manifest_revision", completion.manifest_revision},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::head_create:
                        case durable_io_operation::head_advance:
                        case durable_io_operation::head_delete:
                            {
                                const char * action = completion.job.operation == durable_io_operation::head_create
                                    ? "create" : completion.job.operation == durable_io_operation::head_advance
                                        ? "advance" : "delete";
                                const auto & head = completion.head.head;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = {
                                    {"action", action},
                                    {"name", head.name},
                                    {"generation", head.generation},
                                    {"digest", head.digest},
                                    {"parent_digest", head.parent_digest.empty()
                                        ? json(nullptr) : json(head.parent_digest)},
                                    {"head_revision", head.revision},
                                    {"manifest_revision", completion.head.revision},
                                    {"manifest_file_bytes", completion.manifest_file_bytes},
                                    {"deduplicated", completion.head.deduplicated},
                                    {"timings", {
                                        {"queue_ms", queue_ms},
                                        {"io_ms", io_ms},
                                        {"completion_wait_ms", completion_wait_ms},
                                        {"total_ms", (t_complete - completion.job.enqueued_us) / 1000.0},
                                    }},
                                };
                                queue_results.send(std::move(res));
                            } break;
                        case durable_io_operation::head_load:
                            {
                                if (!queue_results.is_waiting_task_id(request.id)) {
                                    durable_io_cancelled_loads_total++;
                                    break;
                                }
                                json materialized;
                                if (!materialize_snapshot_content(
                                        request,
                                        completion.load.tokens,
                                        completion.load.state,
                                        -1,
                                        -1,
                                        -1,
                                        completion.load.entry.digest,
                                        materialized)) {
                                    break;
                                }
                                durable_materialized_total++;
                                const int64_t t_end = ggml_time_us();
                                const auto & head = completion.head.head;
                                materialized["action"] = "materialize_head";
                                materialized["snapshot_id"] = nullptr;
                                materialized["cold"] = true;
                                materialized["head"] = {
                                    {"name", head.name},
                                    {"generation", head.generation},
                                    {"digest", head.digest},
                                    {"parent_digest", head.parent_digest.empty()
                                        ? json(nullptr) : json(head.parent_digest)},
                                    {"revision", head.revision},
                                };
                                materialized["file_bytes"] = completion.load.entry.file_bytes;
                                materialized["timings"]["queue_ms"] = queue_ms;
                                materialized["timings"]["load_verify_ms"] = io_ms;
                                materialized["timings"]["completion_wait_ms"] = completion_wait_ms;
                                materialized["timings"]["total_ms"] =
                                    (t_end - completion.job.enqueued_us) / 1000.0;
                                auto res = std::make_unique<server_task_result_snapshot>();
                                res->id = request.id;
                                res->result = std::move(materialized);
                                queue_results.send(std::move(res));
                            } break;
                    }
                } break;
            case SERVER_TASK_TYPE_STATETREE:
                {
                    auto res = std::make_unique<server_task_result_statetree>();
                    res->id = task.id;
                    res->states = get_statetree_states_json();
                    res->snapshots = get_statetree_snapshots_json();
                    res->snapshot_bytes = snapshot_bytes;
                    res->snapshot_budget_bytes = params_base.statetree_max_snapshot_bytes;
                    res->snapshot_high_water_bytes = snapshot_high_water_bytes;
                    res->snapshot_content_count = statetree_snapshot_contents.size();
                    res->durable_contents = get_snapshot_store_contents_json();
                    res->durable_manifest_refs = get_snapshot_manifest_refs_json();
                    res->durable_managed_digests = get_snapshot_managed_digests_json();
                    res->durable_logical_heads = get_snapshot_logical_heads_json();
                    if (snapshot_store) {
                        res->durable_disk_bytes = durable_store_disk_bytes;
                        res->durable_disk_budget_bytes = durable_store_disk_budget_bytes;
                        res->durable_disk_high_water_bytes = durable_store_disk_high_water_bytes;
                        res->durable_recovered_temp_files = durable_store_recovered_temp_files;
                        res->durable_ignored_corrupt_files = durable_store_ignored_corrupt_files;
                        res->durable_runtime_integrity_failures = durable_store_runtime_integrity_failures;
                        res->durable_orphaned_disk_bytes = durable_store_orphaned_disk_bytes;
                        res->durable_manifest_revision = durable_manifest_revision;
                        res->durable_manifest_file_bytes = durable_manifest_file_bytes;
                        res->durable_manifest_budget_bytes = durable_manifest_budget_bytes;
                        res->durable_manifest_high_water_bytes = durable_manifest_high_water_bytes;
                        res->durable_manifest_record_count = durable_manifest_record_count;
                        res->durable_manifest_recovered_temp_files = durable_manifest_recovered_temp_files;
                        res->durable_manifest_recovered_tail_bytes = durable_manifest_recovered_tail_bytes;
                        res->durable_manifest_compactions = durable_manifest_compactions;
                        res->durable_manifest_recovered_publish_commits =
                            durable_manifest_recovered_publish_commits;
                        res->durable_manifest_recovered_publish_aborts =
                            durable_manifest_recovered_publish_aborts;
                        res->durable_managed_recovered_erases = durable_managed_recovered_erases;
                        res->durable_managed_recovered_bytes = durable_managed_recovered_bytes;
                    }
                    res->durable_io_pending = durable_io_pending.load();
                    res->durable_io_queue_high_water = durable_io_queue_high_water;
                    res->durable_io_completed_total = durable_io_completed_total;
                    res->durable_io_cancelled_loads_total = durable_io_cancelled_loads_total;
                    res->durable_io_reserved_disk_bytes = durable_io_reserved_disk_bytes;
                    res->durable_io_reserved_disk_high_water = durable_io_reserved_disk_high_water;
                    res->durable_io_reserved_load_bytes = durable_io_reserved_load_bytes;
                    res->durable_io_reserved_load_high_water = durable_io_reserved_load_high_water;
                    res->journal = get_statetree_journal_json();
                    res->journal_capacity = statetree_journal_capacity;
                    res->journal_oldest_sequence = statetree_journal.empty()
                        ? statetree_journal_next_sequence
                        : statetree_journal.front().sequence;
                    res->journal_next_sequence = statetree_journal_next_sequence;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_PCBT:
                {
                    auto res = handle_pcbt(task);
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    bool uses_spec_checkpoint(
            common_context_seq_rm_type rm_type,
            llama_context * ctx,
            int32_t n_rollback) const {
        return rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
            (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
             static_cast<uint32_t>(n_rollback) > llama_n_rs_seq(ctx));
    }

    void restore_spec_context(
            server_slot & slot,
            llama_context * ctx,
            common_context_seq_rm_type rm_type,
            int32_t n_rollback,
            bool target) {
        const auto & ckpt = slot.spec_ckpt;
        if (uses_spec_checkpoint(rm_type, ctx, n_rollback)) {
            if (target) {
                ckpt.load_tgt(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } else {
                ckpt.load_dft(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
        common_context_seq_rm(ctx, slot.id, ckpt.pos_max + 1, -1);
    }

    int decode_serial_transition(
            server_slot & slot,
            llama_token token,
            llama_pos pos,
            common_sampler * smpl,
            llama_token & sampled,
            bool process_spec) {
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = slot.id;
        llama_seq_id * seq_ids[] = { &seq_id };
        int8_t logits = 1;
        llama_batch serial_batch = {
            1,
            &token,
            nullptr,
            &pos,
            &n_seq_id,
            seq_ids,
            &logits,
        };

        const int ret = llama_decode(slot.ctx_tgt, serial_batch);
        metrics.on_decoded(slots);
        if (ret != 0) {
            return ret;
        }
        if (process_spec && !common_speculative_process(spec.get(), serial_batch)) {
            return -2;
        }

        sampled = common_sampler_sample(smpl, slot.ctx_tgt, 0);
        return 0;
    }

    int build_serial_audit(server_slot & slot) {
        slot.spec_serial.clear();

        common_sampler_ptr smpl(common_sampler_clone(slot.smpl.get()));
        llama_token token = slot.sampled;
        llama_pos pos = slot.spec_ckpt.pos_max + 1;

        for (size_t i = 0; i <= slot.spec_draft.size(); ++i) {
            llama_token sampled = LLAMA_TOKEN_NULL;
            const int ret = decode_serial_transition(slot, token, pos, smpl.get(), sampled, false);
            if (ret != 0) {
                return ret;
            }

            slot.spec_serial.push_back(sampled);
            if (i == slot.spec_draft.size() || sampled != slot.spec_draft[i]) {
                break;
            }

            common_sampler_accept(smpl.get(), sampled, true);
            token = sampled;
            pos++;
        }

        return 0;
    }

    void update_slots() {
        maintain_retention();

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child() || slot.is_fork_reserved()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        {
            common_speculative_draft(spec.get());
        }

        // make checkpoints if needed
        for (auto * slot_ptr : drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();
            if (!draft.empty()) {
                slot.n_draft_per_round.push_back(static_cast<int32_t>(draft.size()));
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const int32_t n_anchor_rollback = draft.size() +
                    (slot.task->params.speculative_serial_anchor ? 1 : 0);
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                    static_cast<uint32_t>(n_anchor_rollback) > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                    static_cast<uint32_t>(n_anchor_rollback) > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        }

        // update the batch with the sampled/drafted tokens
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd &&
                                    !slot.is_fork_reserved();

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx_tgt, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft && llama_get_ctx_other(ctx_dft.get()) != ctx_tgt) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            //       [TAG_MTMD_DRAFT_PROCESSING]
                            res = input_tokens.process_chunk(ctx_dft.get(), mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed =
                            !n_before_user_known ||
                            is_on_user ||
                            (is_after_user && near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        bool serial_anchor_failed = false;
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;
            if (!slot.task->params.speculative_serial_anchor || slot.spec_draft.empty()) {
                continue;
            }

            const int ret = build_serial_audit(slot);
            if (ret != 0) {
                SLT_ERR(slot, "serial speculative audit failed, ret=%d\n", ret);
                serial_anchor_failed = true;
                break;
            }
            restore_spec_context(
                slot,
                slot.ctx_tgt,
                ctx_tgt_seq_rm_type,
                static_cast<int32_t>(slot.spec_serial.size()),
                true);
        }

        if (serial_anchor_failed) {
            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    send_error(slot, "serial speculative audit failed");
                    slot.release();
                    slot.prompt_clear(false);
                }
            }
            return;
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        struct batch_shape_key {
            int32_t active_generating;
            int32_t submitted_tokens;

            bool operator<(const batch_shape_key & other) const {
                return active_generating < other.active_generating ||
                       (active_generating == other.active_generating &&
                        submitted_tokens < other.submitted_tokens);
            }
        };
        struct batch_shape_stat {
            uint64_t calls  = 0;
            uint64_t tokens = 0;
        };
        static const bool batch_shape_prof = getenv("TREEBEARD_BATCH_SHAPE_PROF") != nullptr;
        static std::map<batch_shape_key, batch_shape_stat> batch_shape_stats;
        static uint64_t batch_shape_calls = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx_tgt, batch_view);

            metrics.on_decoded(slots);

            if (batch_shape_prof && ret == 0) {
                const batch_shape_key key {
                    static_cast<int32_t>(generating.size()),
                    batch_view.n_tokens,
                };
                auto & stat = batch_shape_stats[key];
                stat.calls++;
                stat.tokens += batch_view.n_tokens;
                batch_shape_calls++;

                if (batch_shape_calls % 100 == 0) {
                    SRV_INF("[treebeard-batch-shape] after=%" PRIu64 " successful decode calls\n",
                            batch_shape_calls);
                    for (const auto & entry : batch_shape_stats) {
                        SRV_INF("[treebeard-batch-shape] active=%d submitted=%d calls=%" PRIu64
                                " share=%.4f tokens=%" PRIu64 "\n",
                                entry.first.active_generating,
                                entry.first.submitted_tokens,
                                entry.second.calls,
                                static_cast<double>(entry.second.calls) /
                                    static_cast<double>(batch_shape_calls),
                                entry.second.tokens);
                    }
                }
            }

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            if (!common_speculative_process(spec.get(), batch_view)) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                break;
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        const size_t child_bytes = child->prompt_state_bytes();
                        const size_t parent_bytes = slot.prompt_state_bytes();
                        const bool copy_prompt_state = reserve_state_bytes(parent_bytes, child_bytes, *child);
                        if (!copy_prompt_state) {
                            SLT_WRN(*child, "%s", "copying parent state without retained checkpoints due to byte pressure\n");
                        }
                        slot.copy_state_to(*child, copy_prompt_state);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                bool serial_commit = false;

                // verify and try to accept the draft
                {
                    // save the sampler sampler state in case we need to restore it
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    if (slot.task->params.speculative_serial_anchor) {
                        GGML_ASSERT(!slot.spec_serial.empty());

                        const llama_token serial_token = slot.spec_serial.front();
                        const llama_token batched_token = accepted.front();
                        const bool anchor_match = serial_token == batched_token;

                        slot.n_draft_anchor++;
                        slot.draft_anchor_serial_tokens.push_back(serial_token);
                        slot.draft_anchor_batched_tokens.push_back(batched_token);

                        if (anchor_match) {
                            slot.n_draft_anchor_match++;
                        } else {
                            slot.n_draft_anchor_fallback++;
                        }

                        size_t first_mismatch = 0;
                        const size_t n_compared = std::min(accepted.size(), slot.spec_serial.size());
                        while (first_mismatch < n_compared &&
                               accepted[first_mismatch] == slot.spec_serial[first_mismatch]) {
                            first_mismatch++;
                        }

                        if (first_mismatch == n_compared && accepted.size() != slot.spec_serial.size()) {
                            SLT_ERR(slot,
                                    "serial audit length mismatch: serial=%zu batched=%zu draft=%zu\n",
                                    slot.spec_serial.size(), accepted.size(), n_draft);
                            send_error(slot, "serial speculative audit length mismatch");
                            slot.release();
                            slot.prompt_clear(false);
                            continue;
                        }

                        slot.n_draft_audit_tokens_matched += static_cast<int32_t>(first_mismatch);

                        if (first_mismatch == slot.spec_serial.size()) {
                            slot.n_draft_audit_tokens += static_cast<int32_t>(first_mismatch);
                            slot.draft_audit_first_mismatch.push_back(-1);
                        } else {
                            slot.n_draft_audit_tokens += static_cast<int32_t>(first_mismatch + 1);
                            const llama_token serial_mismatch = slot.spec_serial[first_mismatch];
                            const llama_token batched_mismatch = accepted[first_mismatch];

                            slot.n_draft_audit_fallback++;
                            slot.draft_audit_first_mismatch.push_back(static_cast<int32_t>(first_mismatch));
                            slot.draft_audit_serial_tokens.push_back(serial_mismatch);
                            slot.draft_audit_batched_tokens.push_back(batched_mismatch);
                            SLT_WRN(slot,
                                    "serial audit mismatch: column=%zu serial=%d batched=%d draft=%zu; falling back\n",
                                    first_mismatch, serial_mismatch, batched_mismatch, n_draft);
                        }

                        // A matching top-1 vector does not prove that the wide decode produced
                        // serial-equivalent KV state. Always rebuild the committed frontier
                        // serially so the next round starts from a trusted state.
                        serial_commit = true;
                        const size_t n_serial_commit = first_mismatch < slot.spec_serial.size()
                            ? first_mismatch + 1
                            : slot.spec_serial.size();

                        const int32_t n_rollback = static_cast<int32_t>(n_draft + 1);
                        restore_spec_context(slot, slot.ctx_tgt, ctx_tgt_seq_rm_type, n_rollback, true);
                        if (slot.ctx_dft) {
                            restore_spec_context(slot, slot.ctx_dft, ctx_dft_seq_rm_type, n_rollback, false);
                        }

                        slot.prompt.tokens.keep_first(slot.spec_ckpt.n_tokens);
                        slot.smpl = std::move(smpl_save);

                        llama_token replay_token = slot.sampled;
                        llama_pos replay_pos = slot.spec_ckpt.pos_max + 1;
                        bool replay_failed = false;
                        for (size_t j = 0; j < n_serial_commit; ++j) {
                            llama_token replayed = LLAMA_TOKEN_NULL;
                            const int ret = decode_serial_transition(
                                slot, replay_token, replay_pos, slot.smpl.get(), replayed, true);
                            if (ret != 0 || replayed != slot.spec_serial[j]) {
                                SLT_ERR(slot,
                                        "serial audit replay failed, column=%zu ret=%d expected=%d replayed=%d\n",
                                        j, ret, slot.spec_serial[j], replayed);
                                replay_failed = true;
                                break;
                            }

                            slot.prompt.tokens.push_back(replay_token);
                            common_sampler_accept(slot.smpl.get(), replayed, true);
                            replay_token = replayed;
                            replay_pos++;
                        }

                        if (replay_failed) {
                            send_error(slot, "serial speculative audit replay failed");
                            slot.release();
                            slot.prompt_clear(false);
                            continue;
                        }

                        common_speculative_accept(
                            spec.get(), slot.id, static_cast<uint16_t>(n_serial_commit - 1));

                        accepted.assign(
                            slot.spec_serial.begin(),
                            slot.spec_serial.begin() + n_serial_commit);

                        slot.spec_serial.clear();
                    }

                    if (!serial_commit) {
                        const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                        const bool use_ckpt_tgt = uses_spec_checkpoint(
                            ctx_tgt_seq_rm_type, ctx_tgt, n_rollback);

                        // check for partial draft acceptance
                        if (n_rollback > 0) {
                            if (use_ckpt_tgt) {
                                if (trace > 0) {
                                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                                }

                                // partial acceptance is not supported by the context -> truncate the draft and restore the state
                                slot.spec_draft = std::move(accepted);

                                const auto & ckpt = slot.spec_ckpt;

                                SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                                {
                                    ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                                }

                                if (slot.ctx_dft) {
                                    ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                    common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                                }

                                slot.prompt.tokens.keep_first(ckpt.n_tokens);
                                slot.smpl = std::move(smpl_save);

                                continue;
                            }
                        }

                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                        }

                        common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
                    }

                    slot.spec_draft = std::move(accepted);
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;
                slot.n_draft_accepted_per_round.push_back(static_cast<int32_t>(ids.size()) - 1);

                if (!serial_commit) {
                    // add accepted tokens to the prompt
                    slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                    slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});
                }

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                if (!serial_commit) {
                    common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                    if (slot.ctx_dft) {
                        common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                    }
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    slot.n_decoded += 1;

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        maintain_retention();
        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_MILLISECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->stop_durable_io_worker();
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* json_webui_settings    */ impl->json_webui_settings,  // Deprecated
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(
            server_queue & queue_tasks,
            server_response & queue_results,
            int sleep_idle_seconds,
            bool bypass_sleep = false,
            int polling_interval_ms = HTTP_POLLING_MILLISECONDS)
            : rd(queue_tasks, queue_results, polling_interval_ms) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);
            if (data.contains("state_id")) {
                if (!data.at("state_id").is_number_integer()) {
                    throw std::invalid_argument("state_id must be a non-negative integer");
                }
                task.state_id = data.at("state_id").get<int64_t>();
                if (task.state_id < 0) {
                    throw std::invalid_argument("state_id must be a non-negative integer");
                }
            }
            if (data.contains("node_id")) {
                if (!data.at("node_id").is_number_integer()) {
                    throw std::invalid_argument("node_id must be a non-negative integer");
                }
                task.node_id = data.at("node_id").get<int64_t>();
                if (task.node_id < 0) {
                    throw std::invalid_argument("node_id must be a non-negative integer");
                }
            }
            if (data.contains("fork_id")) {
                if (!data.at("fork_id").is_number_integer()) {
                    throw std::invalid_argument("fork_id must be a non-negative integer");
                }
                task.fork_id = data.at("fork_id").get<int64_t>();
                if (task.fork_id < 0) {
                    throw std::invalid_argument("fork_id must be a non-negative integer");
                }
            }

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req, &params](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &req, &start_time, &params]() {
                    if (req.should_stop()) {
                        return true; // should_stop condition met
                    } else if (params.sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)params.sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep, int polling_interval_ms) {
    return std::make_unique<server_res_generator>(
            queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep, polling_interval_ms);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "statetree_expired_total"},
                    {"help",  "Number of StateTree families reclaimed after lease expiry."},
                    {"value",  res_task->statetree_expired_total}
            }, {
                    {"name",  "statetree_evicted_total"},
                    {"help",  "Number of StateTree families or idle slots reclaimed under byte pressure."},
                    {"value",  res_task->statetree_evicted_total}
            }, {
                    {"name",  "statetree_reclaimed_bytes_total"},
                    {"help",  "Exact prompt-state bytes reclaimed by StateTree retention controls."},
                    {"value",  res_task->statetree_reclaimed_bytes_total}
            }, {
                    {"name",  "statetree_renewed_total"},
                    {"help",  "Number of generation-fenced StateTree lease renewals."},
                    {"value",  res_task->statetree_renewed_total}
            }, {
                    {"name",  "statetree_pressure_rejected_total"},
                    {"help",  "Number of state-pressure events with no evictable state."},
                    {"value",  res_task->statetree_pressure_rejected_total}
            }, {
                    {"name",  "statetree_snapshots_captured_total"},
                    {"help",  "Number of immutable StateTree snapshots captured."},
                    {"value",  res_task->statetree_snapshots_captured_total}
            }, {
                    {"name",  "statetree_snapshots_materialized_total"},
                    {"help",  "Number of immutable StateTree snapshots materialized into live nodes."},
                    {"value",  res_task->statetree_snapshots_materialized_total}
            }, {
                    {"name",  "statetree_snapshots_erased_total"},
                    {"help",  "Number of immutable StateTree snapshots explicitly erased."},
                    {"value",  res_task->statetree_snapshots_erased_total}
            }, {
                    {"name",  "statetree_snapshot_rejected_total"},
                    {"help",  "Number of immutable snapshot captures rejected by the payload budget."},
                    {"value",  res_task->statetree_snapshot_rejected_total}
            }, {
                    {"name",  "statetree_durable_spilled_total"},
                    {"help",  "Number of durable snapshot spill operations completed."},
                    {"value",  res_task->statetree_durable_spilled_total}
            }, {
                    {"name",  "statetree_durable_materialized_total"},
                    {"help",  "Number of durable snapshot content objects materialized."},
                    {"value",  res_task->statetree_durable_materialized_total}
            }, {
                    {"name",  "statetree_durable_erased_total"},
                    {"help",  "Number of durable snapshot content objects explicitly erased."},
                    {"value",  res_task->statetree_durable_erased_total}
            }, {
                    {"name",  "statetree_durable_rejected_total"},
                    {"help",  "Number of durable snapshot operations rejected by disk, load, or integrity admission."},
                    {"value",  res_task->statetree_durable_rejected_total}
            }, {
                    {"name",  "statetree_durable_io_completed_total"},
                    {"help",  "Number of background durable I/O completions consumed by the state thread."},
                    {"value",  res_task->statetree_durable_io_completed_total}
            }, {
                    {"name",  "statetree_durable_io_cancelled_loads_total"},
                    {"help",  "Verified cold loads discarded because their HTTP owner disconnected."},
                    {"value",  res_task->statetree_durable_io_cancelled_loads_total}
            }, {
                    {"name",  "statetree_durable_retained_total"},
                    {"help",  "Number of durable ownership retain operations completed."},
                    {"value",  res_task->statetree_durable_retained_total}
            }, {
                    {"name",  "statetree_durable_released_total"},
                    {"help",  "Number of durable ownership release operations completed."},
                    {"value",  res_task->statetree_durable_released_total}
            }, {
                    {"name",  "statetree_durable_compacted_total"},
                    {"help",  "Number of explicit durable ownership manifest compactions completed."},
                    {"value",  res_task->statetree_durable_compacted_total}
            }, {
                    {"name",  "statetree_durable_manifest_recovered_publish_commits_total"},
                    {"help",  "Pending managed publishes verified and committed during startup recovery."},
                    {"value",  res_task->statetree_durable_manifest_recovered_publish_commits}
            }, {
                    {"name",  "statetree_durable_manifest_recovered_publish_aborts_total"},
                    {"help",  "Abandoned managed publish intents durably aborted during startup recovery."},
                    {"value",  res_task->statetree_durable_manifest_recovered_publish_aborts}
            }, {
                    {"name",  "statetree_durable_cache_evicted_total"},
                    {"help",  "Managed cache objects erased by explicit or pressure-driven reconciliation."},
                    {"value",  res_task->statetree_durable_cache_evicted_total}
            }, {
                    {"name",  "statetree_durable_cache_reclaimed_bytes_total"},
                    {"help",  "Exact managed cache object bytes reclaimed."},
                    {"value",  res_task->statetree_durable_cache_reclaimed_bytes_total}
            }, {
                    {"name",  "statetree_durable_managed_recovered_erases_total"},
                    {"help",  "Unreachable managed objects erased while completing interrupted eviction at startup."},
                    {"value",  res_task->statetree_durable_managed_recovered_erases}
            }, {
                    {"name",  "statetree_durable_managed_recovered_bytes_total"},
                    {"help",  "Exact managed object bytes erased during startup eviction recovery."},
                    {"value",  res_task->statetree_durable_managed_recovered_bytes}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_idle"},
                    {"help",  "Number of slots available for automatic requests."},
                    {"value",  (uint64_t) res_task->n_idle_slots}
            },{
                    {"name",  "requests_reserved"},
                    {"help",  "Number of idle slots reserved by shared-prefix forks."},
                    {"value",  (uint64_t) res_task->n_reserved_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "statetree_state_bytes"},
                    {"help",  "Exact prompt-state bytes held by all live slots."},
                    {"value",  res_task->statetree_state_bytes}
            },{
                    {"name",  "statetree_retained_bytes"},
                    {"help",  "Exact prompt-state bytes retained by idle slots."},
                    {"value",  res_task->statetree_retained_bytes}
            },{
                    {"name",  "statetree_active_bytes"},
                    {"help",  "Exact prompt-state bytes held by processing slots."},
                    {"value",  res_task->statetree_active_bytes}
            },{
                    {"name",  "statetree_state_budget_bytes"},
                    {"help",  "Configured exact live slot prompt-state byte ceiling, or zero when unlimited."},
                    {"value",  res_task->statetree_state_budget_bytes}
            },{
                    {"name",  "statetree_state_high_water_bytes"},
                    {"help",  "Largest post-enforcement live slot prompt-state byte count."},
                    {"value",  res_task->statetree_state_high_water_bytes}
            },{
                    {"name",  "statetree_retained_high_water_bytes"},
                    {"help",  "Largest post-enforcement idle retained-state byte count."},
                    {"value",  res_task->statetree_retained_high_water_bytes}
            },{
                    {"name",  "statetree_snapshot_bytes"},
                    {"help",  "Exact serialized state and token payload bytes held by immutable snapshots."},
                    {"value",  res_task->statetree_snapshot_bytes}
            },{
                    {"name",  "statetree_snapshot_budget_bytes"},
                    {"help",  "Configured immutable snapshot payload byte ceiling, or zero when disabled."},
                    {"value",  res_task->statetree_snapshot_budget_bytes}
            },{
                    {"name",  "statetree_snapshot_high_water_bytes"},
                    {"help",  "Largest immutable snapshot payload byte count."},
                    {"value",  res_task->statetree_snapshot_high_water_bytes}
            },{
                    {"name",  "statetree_snapshot_count"},
                    {"help",  "Number of immutable snapshot provenance handles retained by this server process."},
                    {"value",  res_task->statetree_snapshot_count}
            },{
                    {"name",  "statetree_snapshot_content_count"},
                    {"help",  "Number of unique digest-addressed immutable snapshot payloads."},
                    {"value",  res_task->statetree_snapshot_content_count}
            },{
                    {"name",  "statetree_durable_disk_bytes"},
                    {"help",  "Exact durable snapshot object file bytes in the active compatibility namespace."},
                    {"value",  res_task->statetree_durable_disk_bytes}
            },{
                    {"name",  "statetree_durable_disk_budget_bytes"},
                    {"help",  "Configured durable snapshot namespace file-byte ceiling."},
                    {"value",  res_task->statetree_durable_disk_budget_bytes}
            },{
                    {"name",  "statetree_durable_disk_high_water_bytes"},
                    {"help",  "Largest durable snapshot namespace file-byte count."},
                    {"value",  res_task->statetree_durable_disk_high_water_bytes}
            },{
                    {"name",  "statetree_durable_content_count"},
                    {"help",  "Number of valid durable content objects discovered in the active namespace."},
                    {"value",  res_task->statetree_durable_content_count}
            },{
                    {"name",  "statetree_durable_recovered_temp_files"},
                    {"help",  "Crash-interrupted temporary durable objects removed during discovery."},
                    {"value",  res_task->statetree_durable_recovered_temp_files}
            },{
                    {"name",  "statetree_durable_ignored_corrupt_files"},
                    {"help",  "Malformed durable objects ignored during namespace discovery."},
                    {"value",  res_task->statetree_durable_ignored_corrupt_files}
            },{
                    {"name",  "statetree_durable_runtime_integrity_failures"},
                    {"help",  "Durable objects that failed size, envelope, or digest verification when loaded."},
                    {"value",  res_task->statetree_durable_runtime_integrity_failures}
            },{
                    {"name",  "statetree_durable_orphaned_disk_bytes"},
                    {"help",  "Durable namespace bytes occupied by malformed or unindexed content files."},
                    {"value",  res_task->statetree_durable_orphaned_disk_bytes}
            },{
                    {"name",  "statetree_durable_io_pending"},
                    {"help",  "Durable I/O jobs queued, running, or awaiting state-thread completion."},
                    {"value",  res_task->statetree_durable_io_pending}
            },{
                    {"name",  "statetree_durable_io_queue_high_water"},
                    {"help",  "Largest durable I/O pending-job count."},
                    {"value",  res_task->statetree_durable_io_queue_high_water}
            },{
                    {"name",  "statetree_durable_io_reserved_disk_bytes"},
                    {"help",  "Projected object bytes reserved by in-flight unique spills."},
                    {"value",  res_task->statetree_durable_io_reserved_disk_bytes}
            },{
                    {"name",  "statetree_durable_io_reserved_disk_high_water"},
                    {"help",  "Largest in-flight durable spill disk-byte reservation."},
                    {"value",  res_task->statetree_durable_io_reserved_disk_high_water}
            },{
                    {"name",  "statetree_durable_io_reserved_load_bytes"},
                    {"help",  "Payload bytes reserved by queued or verified cold loads."},
                    {"value",  res_task->statetree_durable_io_reserved_load_bytes}
            },{
                    {"name",  "statetree_durable_io_reserved_load_high_water"},
                    {"help",  "Largest aggregate cold-load payload-byte reservation."},
                    {"value",  res_task->statetree_durable_io_reserved_load_high_water}
            },{
                    {"name",  "statetree_durable_manifest_refs"},
                    {"help",  "Number of live owner-to-content references in the durable manifest."},
                    {"value",  res_task->statetree_durable_manifest_refs}
            },{
                    {"name",  "statetree_durable_managed_count"},
                    {"help",  "Number of durable objects governed by managed lifecycle reconciliation."},
                    {"value",  res_task->statetree_durable_managed_count}
            },{
                    {"name",  "statetree_durable_manifest_revision"},
                    {"help",  "Last committed durable ownership manifest revision."},
                    {"value",  res_task->statetree_durable_manifest_revision}
            },{
                    {"name",  "statetree_durable_manifest_file_bytes"},
                    {"help",  "Current checksummed ownership WAL/checkpoint file bytes."},
                    {"value",  res_task->statetree_durable_manifest_file_bytes}
            },{
                    {"name",  "statetree_durable_manifest_budget_bytes"},
                    {"help",  "Configured ownership manifest file-byte ceiling."},
                    {"value",  res_task->statetree_durable_manifest_budget_bytes}
            },{
                    {"name",  "statetree_durable_manifest_high_water_bytes"},
                    {"help",  "Largest ownership manifest file-byte count observed by this process."},
                    {"value",  res_task->statetree_durable_manifest_high_water_bytes}
            },{
                    {"name",  "statetree_durable_manifest_record_count"},
                    {"help",  "Number of records in the current ownership WAL/checkpoint."},
                    {"value",  res_task->statetree_durable_manifest_record_count}
            },{
                    {"name",  "statetree_durable_manifest_recovered_temp_files"},
                    {"help",  "Interrupted manifest compaction files recovered during startup."},
                    {"value",  res_task->statetree_durable_manifest_recovered_temp_files}
            },{
                    {"name",  "statetree_durable_manifest_recovered_tail_bytes"},
                    {"help",  "Interrupted manifest tail bytes truncated during startup."},
                    {"value",  res_task->statetree_durable_manifest_recovered_tail_bytes}
            },{
                    {"name",  "statetree_durable_manifest_compactions"},
                    {"help",  "Manifest compactions performed by this process, including automatic compaction."},
                    {"value",  res_task->statetree_durable_manifest_compactions}
            },{
                    {"name",  "statetree_families"},
                    {"help",  "Number of live StateTree fork families."},
                    {"value",  res_task->n_statetree_families}
            },{
                    {"name",  "statetree_active_families"},
                    {"help",  "Number of StateTree families with in-flight work."},
                    {"value",  res_task->n_statetree_active_families}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                const json & value = metric_def.at("value");
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->get_snapshots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response(
                    "This server does not support StateTree snapshots. Start it with `--slots`",
                    ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        {
            server_task task(SERVER_TASK_TYPE_STATETREE);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);
        }
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * statetree = dynamic_cast<server_task_result_statetree *>(result.get());
        GGML_ASSERT(statetree != nullptr);
        res->ok({
            {"snapshots", statetree->snapshots},
            {"snapshot_bytes", statetree->snapshot_bytes},
            {"snapshot_budget_bytes", statetree->snapshot_budget_bytes},
            {"snapshot_high_water_bytes", statetree->snapshot_high_water_bytes},
            {"snapshot_content_count", statetree->snapshot_content_count},
        });
        return res;
    };

    this->get_snapshot_contents = [this](const server_http_req & req) {
        auto res = create_response();
        {
            server_task task(SERVER_TASK_TYPE_STATETREE);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);
        }
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * statetree = dynamic_cast<server_task_result_statetree *>(result.get());
        GGML_ASSERT(statetree != nullptr);
        res->ok({
            {"contents", statetree->durable_contents},
            {"refs", statetree->durable_manifest_refs},
            {"managed_digests", statetree->durable_managed_digests},
            {"disk_bytes", statetree->durable_disk_bytes},
            {"disk_budget_bytes", statetree->durable_disk_budget_bytes},
            {"disk_high_water_bytes", statetree->durable_disk_high_water_bytes},
            {"recovered_temp_files", statetree->durable_recovered_temp_files},
            {"ignored_corrupt_files", statetree->durable_ignored_corrupt_files},
            {"runtime_integrity_failures", statetree->durable_runtime_integrity_failures},
            {"orphaned_disk_bytes", statetree->durable_orphaned_disk_bytes},
            {"io_pending", statetree->durable_io_pending},
            {"io_queue_high_water", statetree->durable_io_queue_high_water},
            {"io_completed_total", statetree->durable_io_completed_total},
            {"io_cancelled_loads_total", statetree->durable_io_cancelled_loads_total},
            {"io_reserved_disk_bytes", statetree->durable_io_reserved_disk_bytes},
            {"io_reserved_disk_high_water", statetree->durable_io_reserved_disk_high_water},
            {"io_reserved_load_bytes", statetree->durable_io_reserved_load_bytes},
            {"io_reserved_load_high_water", statetree->durable_io_reserved_load_high_water},
            {"manifest_revision", statetree->durable_manifest_revision},
            {"manifest_file_bytes", statetree->durable_manifest_file_bytes},
            {"manifest_budget_bytes", statetree->durable_manifest_budget_bytes},
            {"manifest_high_water_bytes", statetree->durable_manifest_high_water_bytes},
            {"manifest_record_count", statetree->durable_manifest_record_count},
            {"manifest_recovered_temp_files", statetree->durable_manifest_recovered_temp_files},
            {"manifest_recovered_tail_bytes", statetree->durable_manifest_recovered_tail_bytes},
            {"manifest_compactions", statetree->durable_manifest_compactions},
            {"manifest_recovered_publish_commits",
                statetree->durable_manifest_recovered_publish_commits},
            {"manifest_recovered_publish_aborts",
                statetree->durable_manifest_recovered_publish_aborts},
            {"managed_recovered_erases", statetree->durable_managed_recovered_erases},
            {"managed_recovered_bytes", statetree->durable_managed_recovered_bytes},
        });
        return res;
    };

    this->get_snapshot_heads = [this](const server_http_req & req) {
        auto res = create_response();
        server_task task(SERVER_TASK_TYPE_STATETREE);
        task.id = res->rd.get_new_id();
        res->rd.post_task(std::move(task), true);
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * statetree = dynamic_cast<server_task_result_statetree *>(result.get());
        GGML_ASSERT(statetree != nullptr);
        res->ok({
            {"heads", statetree->durable_logical_heads},
            {"manifest_revision", statetree->durable_manifest_revision},
            {"manifest_file_bytes", statetree->durable_manifest_file_bytes},
            {"manifest_budget_bytes", statetree->durable_manifest_budget_bytes},
        });
        return res;
    };

    // --- proof-carrying branch transactions (contract v1) ----------------
    // HTTP threads parse for early 400s; every mutation decision happens on
    // the state thread via SERVER_TASK_TYPE_PCBT (invariant 1).

    auto pcbt_error_type = [](const std::string & cls) {
        if (cls == "not_found")  return ERROR_TYPE_NOT_FOUND;
        if (cls == "capacity")   return ERROR_TYPE_UNAVAILABLE;
        if (cls == "conflict" || cls == "expired" || cls == "unprocessable") {
            return ERROR_TYPE_NOT_SUPPORTED;
        }
        return ERROR_TYPE_INVALID_REQUEST;
    };

    auto pcbt_roundtrip = [this, pcbt_error_type](const server_http_req & req,
                                                  server_task::pcbt_action action) {
        auto res = create_response();
        server_task task(SERVER_TASK_TYPE_PCBT);
        task.pcbt = std::move(action);
        task.id = res->rd.get_new_id();
        res->rd.post_task(std::move(task), true);
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        auto * pcbt = dynamic_cast<server_task_result_pcbt *>(result.get());
        GGML_ASSERT(pcbt != nullptr);
        if (pcbt->is_error()) {
            json err = format_error_response(pcbt->message, pcbt_error_type(pcbt->error_class));
            err["class"]       = pcbt->error_class;
            err["http_status"] = pcbt->http_status;
            res->error(err);
            return res;
        }
        res->ok(pcbt->payload);
        return res;
    };

    auto pcbt_parse_tx_id = [](const server_http_req & req, uint64_t & out) {
        const std::string str = req.get_param("transaction_id");
        try {
            size_t parsed = 0;
            const long long v = std::stoll(str, &parsed);
            if (parsed != str.size() || v < 0) {
                return false;
            }
            out = (uint64_t) v;
            return true;
        } catch (const std::exception &) {
            return false;
        }
    };

    this->post_transactions = [this, pcbt_roundtrip](const server_http_req & req) {
        const json body = json::parse(req.body, nullptr, false);
        pcbt_create_request parsed;
        const auto pr = body.is_discarded()
            ? pcbt_parse_result::fail(pcbt_error::INVALID_REQUEST, "invalid JSON")
            : pcbt_parse_create(body, parsed);
        if (!pr.ok()) {
            auto res = create_response();
            res->error(format_error_response(pr.message, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        server_task::pcbt_action action;
        action.op        = server_task::pcbt_action::CREATE;
        action.body_json = req.body;
        return pcbt_roundtrip(req, std::move(action));
    };

    this->post_transaction_action = [this, pcbt_roundtrip, pcbt_parse_tx_id](const server_http_req & req) {
        server_task::pcbt_action action;
        if (!pcbt_parse_tx_id(req, action.transaction_id)) {
            auto res = create_response();
            res->error(format_error_response("invalid transaction ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string verb = req.get_param("action");
        const json body = json::parse(req.body, nullptr, false);
        pcbt_parse_result pr;
        if (body.is_discarded()) {
            pr = pcbt_parse_result::fail(pcbt_error::INVALID_REQUEST, "invalid JSON");
        } else if (verb == "commit") {
            pcbt_commit_request parsed;
            pr = pcbt_parse_commit(body, parsed);
            action.op = server_task::pcbt_action::COMMIT;
        } else if (verb == "abort") {
            pcbt_abort_request parsed;
            pr = pcbt_parse_abort(body, parsed);
            action.op = server_task::pcbt_action::ABORT;
        } else {
            pr = pcbt_parse_result::fail(pcbt_error::INVALID_REQUEST, "invalid action");
        }
        if (!pr.ok()) {
            auto res = create_response();
            res->error(format_error_response(pr.message, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        action.body_json = req.body;
        return pcbt_roundtrip(req, std::move(action));
    };

    this->get_transactions = [this, pcbt_roundtrip, pcbt_parse_tx_id](const server_http_req & req) {
        server_task::pcbt_action action;
        if (!pcbt_parse_tx_id(req, action.transaction_id)) {
            auto res = create_response();
            res->error(format_error_response("invalid transaction ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        action.op = server_task::pcbt_action::OBSERVE;
        return pcbt_roundtrip(req, std::move(action));
    };

    this->get_transaction_events = [this, pcbt_roundtrip, pcbt_parse_tx_id](const server_http_req & req) {
        server_task::pcbt_action action;
        if (!pcbt_parse_tx_id(req, action.transaction_id)) {
            auto res = create_response();
            res->error(format_error_response("invalid transaction ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string after = req.get_param("after");
        if (!after.empty()) {
            try {
                action.after_seq = (uint64_t) std::stoull(after);
            } catch (const std::exception &) {
                auto res = create_response();
                res->error(format_error_response("invalid after", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        action.op = server_task::pcbt_action::EVENTS;
        return pcbt_roundtrip(req, std::move(action));
    };

    this->get_states = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response(
                    "This server does not support StateTree inspection. Start it with `--slots`",
                    ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        uint64_t journal_after = 0;
        const std::string journal_after_str = req.get_param("journal_after");
        if (!journal_after_str.empty()) {
            try {
                size_t parsed = 0;
                journal_after = std::stoull(journal_after_str, &parsed);
                if (parsed != journal_after_str.size()) {
                    throw std::invalid_argument("trailing journal sequence characters");
                }
            } catch (const std::exception &) {
                res->error(format_error_response(
                        "journal_after must be a non-negative integer",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }

        {
            server_task task(SERVER_TASK_TYPE_STATETREE);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true);
        }

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * statetree = dynamic_cast<server_task_result_statetree *>(result.get());
        GGML_ASSERT(statetree != nullptr);
        if (journal_after > 0) {
            json filtered = json::array();
            for (const auto & entry : statetree->journal) {
                if (entry.at("sequence").get<uint64_t>() > journal_after) {
                    filtered.push_back(entry);
                }
            }
            statetree->journal = std::move(filtered);
        }
        res->ok(statetree->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            size_t parsed = 0;
            id_slot = std::stoi(id_slot_str, &parsed);
            if (parsed != id_slot_str.size()) {
                throw std::invalid_argument("trailing slot ID characters");
            }
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (id_slot < 0 || id_slot >= params.n_parallel) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "fork") {
            return handle_slots_fork(req, id_slot);
        }
        if (action == "commit") {
            return handle_slots_commit(req, id_slot);
        }
        if (action == "renew") {
            return handle_slots_renew(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->post_nodes = [this](const server_http_req & req) {
        auto res = create_response();

        const std::string node_id_str = req.get_param("node_id");
        int64_t node_id;
        try {
            size_t parsed = 0;
            node_id = std::stoll(node_id_str, &parsed);
            if (parsed != node_id_str.size() || node_id < 0) {
                throw std::invalid_argument("invalid node ID");
            }
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid node ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string action = req.get_param("action");
        if (action == "fork") {
            return handle_slots_fork(req, -1, node_id);
        }
        if (action == "snapshot") {
            return handle_snapshot_capture(req, node_id);
        }
        if (action == "commit") {
            return handle_slots_commit(req, -1, node_id);
        }
        if (action == "renew") {
            return handle_slots_renew(req, -1, node_id);
        }
        if (action == "erase") {
            return handle_slots_erase(req, -1, node_id);
        }

        res->error(format_error_response("Invalid node action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->post_snapshots = [this](const server_http_req & req) {
        auto res = create_response();
        int64_t snapshot_id = -1;
        try {
            const std::string value = req.get_param("snapshot_id");
            size_t parsed = 0;
            snapshot_id = std::stoll(value, &parsed);
            if (parsed != value.size() || snapshot_id < 0) {
                throw std::invalid_argument("invalid snapshot ID");
            }
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid snapshot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int id_slot = -1;
        std::string digest;
        std::string owner;
        std::string retention_class;
        std::string head_name;
        std::string expected_digest;
        uint64_t expected_generation = 0;
        bool has_expected_generation = false;
        size_t request_field_count = 0;
        try {
            const json data = req.body.empty() ? json::object() : json::parse(req.body);
            if (!data.is_object()) {
                throw std::invalid_argument("request body must be an object");
            }
            request_field_count = data.size();
            if (data.contains("digest")) {
                if (!data.at("digest").is_string()) {
                    throw std::invalid_argument("digest must be a string");
                }
                digest = data.at("digest").get<std::string>();
            }
            if (data.contains("id_slot")) {
                if (!data.at("id_slot").is_number_integer()) {
                    throw std::invalid_argument("id_slot must be an integer");
                }
                id_slot = data.at("id_slot").get<int>();
                if (id_slot < 0) {
                    throw std::invalid_argument("id_slot must be non-negative");
                }
            }
            if (data.contains("owner")) {
                if (!data.at("owner").is_string()) {
                    throw std::invalid_argument("owner must be a string");
                }
                owner = data.at("owner").get<std::string>();
                if (owner.empty() || owner.size() > 128 || !std::all_of(
                        owner.begin(), owner.end(), [](unsigned char value) {
                            return (value >= 'a' && value <= 'z') ||
                                (value >= 'A' && value <= 'Z') ||
                                (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                                value == ':' || value == '/' || value == '-';
                        })) {
                    throw std::invalid_argument(
                            "owner must contain 1 to 128 portable identifier characters");
                }
            }
            if (data.contains("retention_class")) {
                if (!data.at("retention_class").is_string()) {
                    throw std::invalid_argument("retention_class must be a string");
                }
                retention_class = data.at("retention_class").get<std::string>();
                if (retention_class != "pinned" && retention_class != "cache") {
                    throw std::invalid_argument("retention_class must be pinned or cache");
                }
            }
            if (data.contains("head_name")) {
                if (!data.at("head_name").is_string()) {
                    throw std::invalid_argument("head_name must be a string");
                }
                head_name = data.at("head_name").get<std::string>();
                if (head_name.empty() || head_name.size() > 128 || !std::all_of(
                        head_name.begin(), head_name.end(), [](unsigned char value) {
                            return (value >= 'a' && value <= 'z') ||
                                (value >= 'A' && value <= 'Z') ||
                                (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                                value == ':' || value == '/' || value == '-';
                        })) {
                    throw std::invalid_argument(
                            "head_name must contain 1 to 128 portable identifier characters");
                }
            }
            if (data.contains("expected_digest")) {
                if (!data.at("expected_digest").is_string()) {
                    throw std::invalid_argument("expected_digest must be a string");
                }
                expected_digest = data.at("expected_digest").get<std::string>();
                if (expected_digest.size() != 71 || expected_digest.compare(0, 7, "sha256:") != 0 ||
                        !std::all_of(expected_digest.begin() + 7, expected_digest.end(),
                            [](unsigned char value) {
                                return (value >= '0' && value <= '9') ||
                                    (value >= 'a' && value <= 'f');
                            })) {
                    throw std::invalid_argument(
                            "expected_digest must be a canonical SHA-256 digest");
                }
            }
            if (data.contains("expected_generation")) {
                if (data.at("expected_generation").is_number_unsigned()) {
                    expected_generation = data.at("expected_generation").get<uint64_t>();
                } else if (data.at("expected_generation").is_number_integer()) {
                    const int64_t value = data.at("expected_generation").get<int64_t>();
                    if (value < 0) {
                        throw std::invalid_argument("expected_generation must be non-negative");
                    }
                    expected_generation = (uint64_t) value;
                } else {
                    throw std::invalid_argument(
                            "expected_generation must be a non-negative integer");
                }
                has_expected_generation = true;
            }
        } catch (const std::exception & e) {
            res->error(format_error_response(
                    std::string("Invalid snapshot request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string action = req.get_param("action");
        server_task_type type;
        if (action == "materialize") {
            if (!owner.empty() || !retention_class.empty() || !head_name.empty() ||
                    !expected_digest.empty() || has_expected_generation) {
                res->error(format_error_response(
                        "Snapshot materialize does not accept ownership fields",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_MATERIALIZE;
        } else if (action == "spill") {
            if (id_slot >= 0 || !owner.empty() || !retention_class.empty() || !head_name.empty() ||
                    !expected_digest.empty() || has_expected_generation) {
                res->error(format_error_response(
                        "Snapshot spill does not accept id_slot or ownership fields",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_SPILL;
        } else if (action == "publish") {
            if (id_slot >= 0 || owner.empty() || retention_class.empty() || !head_name.empty() ||
                    !expected_digest.empty() || has_expected_generation) {
                res->error(format_error_response(
                        "Snapshot publish requires owner and retention_class and does not accept id_slot",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_PUBLISH;
        } else if (action == "publish-advance") {
            if (id_slot >= 0 || owner.empty() || retention_class.empty() || head_name.empty() ||
                    expected_digest.empty() || !has_expected_generation || expected_generation == 0 ||
                    request_field_count != 6) {
                res->error(format_error_response(
                        "Snapshot publish-advance requires only digest, owner, retention_class, head_name, expected_generation, and expected_digest",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE;
        } else if (action == "erase") {
            if (id_slot >= 0 || !owner.empty() || !retention_class.empty() || !head_name.empty() ||
                    !expected_digest.empty() || has_expected_generation) {
                res->error(format_error_response(
                        "Snapshot erase does not accept id_slot or ownership fields",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_ERASE;
        } else {
            res->error(format_error_response("Invalid snapshot action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        server_task task(type);
        task.id = res->rd.get_new_id();
        task.slot_action.snapshot_id = snapshot_id;
        task.slot_action.id_slot = id_slot;
        task.slot_action.digest = std::move(digest);
        task.slot_action.owner = std::move(owner);
        task.slot_action.retention_class = std::move(retention_class);
        task.slot_action.head_name = std::move(head_name);
        task.slot_action.expected_digest = std::move(expected_digest);
        task.slot_action.expected_generation = expected_generation;
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_snapshot *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_snapshot_contents = [this](const server_http_req & req) {
        auto res = create_response(false, DURABLE_IO_POLLING_MILLISECONDS);
        const std::string digest_hex = req.get_param("digest");
        if (digest_hex.size() != 64 || !std::all_of(
                digest_hex.begin(), digest_hex.end(), [](unsigned char value) {
                    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                })) {
            res->error(format_error_response("Invalid snapshot content digest", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int id_slot = -1;
        std::string owner;
        std::string retention_class;
        try {
            const json data = req.body.empty() ? json::object() : json::parse(req.body);
            if (!data.is_object()) {
                throw std::invalid_argument("request body must be an object");
            }
            if (data.contains("id_slot")) {
                if (!data.at("id_slot").is_number_integer()) {
                    throw std::invalid_argument("id_slot must be an integer");
                }
                id_slot = data.at("id_slot").get<int>();
                if (id_slot < 0) {
                    throw std::invalid_argument("id_slot must be non-negative");
                }
            }
            if (data.contains("owner")) {
                if (!data.at("owner").is_string()) {
                    throw std::invalid_argument("owner must be a string");
                }
                owner = data.at("owner").get<std::string>();
                if (owner.empty() || owner.size() > 128 || !std::all_of(
                        owner.begin(), owner.end(), [](unsigned char value) {
                            return (value >= 'a' && value <= 'z') ||
                                (value >= 'A' && value <= 'Z') ||
                                (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                                value == ':' || value == '/' || value == '-';
                        })) {
                    throw std::invalid_argument(
                            "owner must contain 1 to 128 portable identifier characters");
                }
            }
            if (data.contains("retention_class")) {
                if (!data.at("retention_class").is_string()) {
                    throw std::invalid_argument("retention_class must be a string");
                }
                retention_class = data.at("retention_class").get<std::string>();
                if (retention_class != "pinned" && retention_class != "cache") {
                    throw std::invalid_argument("retention_class must be pinned or cache");
                }
            }
        } catch (const std::exception & e) {
            res->error(format_error_response(
                    std::string("Invalid snapshot content request: ") + e.what(),
                    ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string action = req.get_param("action");
        server_task_type type;
        if (action == "materialize") {
            if (!owner.empty() || !retention_class.empty()) {
                res->error(format_error_response(
                        "Snapshot content materialize does not accept ownership fields",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_CONTENT_MATERIALIZE;
        } else if (action == "erase") {
            if (id_slot >= 0 || !owner.empty() || !retention_class.empty()) {
                res->error(format_error_response(
                        "Snapshot content erase does not accept id_slot or ownership fields",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_CONTENT_ERASE;
        } else if (action == "retain") {
            if (id_slot >= 0 || owner.empty() || retention_class.empty()) {
                res->error(format_error_response(
                        "Snapshot content retain requires owner and retention_class and does not accept id_slot",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RETAIN;
        } else if (action == "release") {
            if (id_slot >= 0 || owner.empty() || !retention_class.empty()) {
                res->error(format_error_response(
                        "Snapshot content release requires owner and does not accept id_slot or retention_class",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RELEASE;
        } else {
            res->error(format_error_response("Invalid snapshot content action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        server_task task(type);
        task.id = res->rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.digest = "sha256:" + digest_hex;
        task.slot_action.owner = std::move(owner);
        task.slot_action.retention_class = std::move(retention_class);
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_snapshot *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_snapshot_manifest = [this](const server_http_req & req) {
        auto res = create_response(false, DURABLE_IO_POLLING_MILLISECONDS);
        const std::string action = req.get_param("action");
        server_task_type type;
        uint64_t target_disk_bytes = 0;
        try {
            const json data = req.body.empty() ? json::object() : json::parse(req.body);
            if (!data.is_object()) {
                throw std::invalid_argument("request body must be an object");
            }
            if (action == "compact") {
                if (!data.empty()) {
                    throw std::invalid_argument("compact request body must be empty");
                }
                type = SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_COMPACT;
            } else if (action == "prune") {
                if (data.size() != 1 || !data.contains("target_disk_bytes") ||
                        (!data.at("target_disk_bytes").is_number_unsigned() &&
                         !data.at("target_disk_bytes").is_number_integer())) {
                    throw std::invalid_argument(
                            "prune requires only a non-negative integer target_disk_bytes");
                }
                if (data.at("target_disk_bytes").is_number_unsigned()) {
                    target_disk_bytes = data.at("target_disk_bytes").get<uint64_t>();
                } else {
                    const int64_t signed_target = data.at("target_disk_bytes").get<int64_t>();
                    if (signed_target < 0) {
                        throw std::invalid_argument("target_disk_bytes must be non-negative");
                    }
                    target_disk_bytes = (uint64_t) signed_target;
                }
                type = SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_PRUNE;
            } else {
                throw std::invalid_argument("unknown action");
            }
        } catch (const std::exception & e) {
            res->error(format_error_response(
                    std::string("Invalid snapshot manifest request: ") + e.what(),
                    ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        server_task task(type);
        task.id = res->rd.get_new_id();
        task.slot_action.target_bytes = target_disk_bytes;
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_snapshot *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_snapshot_heads = [this](const server_http_req & req) {
        auto res = create_response(false, DURABLE_IO_POLLING_MILLISECONDS);
        const std::string head_name = req.get_param("head");
        const std::string action = req.get_param("action");
        if (head_name.empty() || head_name.size() > 128 || !std::all_of(
                head_name.begin(), head_name.end(), [](unsigned char value) {
                    return (value >= 'a' && value <= 'z') ||
                        (value >= 'A' && value <= 'Z') ||
                        (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                        value == ':' || value == '/' || value == '-';
                })) {
            res->error(format_error_response(
                    "Invalid logical head name", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string digest;
        std::string expected_digest;
        uint64_t expected_generation = 0;
        bool has_expected_generation = false;
        int id_slot = -1;
        try {
            const json data = req.body.empty() ? json::object() : json::parse(req.body);
            if (!data.is_object()) {
                throw std::invalid_argument("request body must be an object");
            }
            const auto only_keys = [&](std::initializer_list<const char *> allowed) {
                for (auto item = data.begin(); item != data.end(); ++item) {
                    if (std::none_of(allowed.begin(), allowed.end(), [&](const char * key) {
                            return item.key() == key;
                        })) {
                        return false;
                    }
                }
                return true;
            };
            if ((action == "create" && !only_keys({"digest"})) ||
                    (action == "advance" && !only_keys({
                        "digest", "expected_digest", "expected_generation"})) ||
                    (action == "delete" && !only_keys({
                        "expected_digest", "expected_generation"})) ||
                    (action == "materialize" && !only_keys({
                        "expected_digest", "expected_generation", "id_slot"}))) {
                throw std::invalid_argument("request contains fields unsupported by this action");
            }
            const auto read_digest = [&](const char * key, std::string & value) {
                if (!data.contains(key) || !data.at(key).is_string()) {
                    return false;
                }
                value = data.at(key).get<std::string>();
                if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0 ||
                        !std::all_of(value.begin() + 7, value.end(), [](unsigned char byte) {
                            return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
                        })) {
                    throw std::invalid_argument(std::string(key) + " must be a canonical SHA-256 digest");
                }
                return true;
            };
            read_digest("digest", digest);
            read_digest("expected_digest", expected_digest);
            if (data.contains("expected_generation")) {
                if (data.at("expected_generation").is_number_unsigned()) {
                    expected_generation = data.at("expected_generation").get<uint64_t>();
                } else if (data.at("expected_generation").is_number_integer()) {
                    const int64_t value = data.at("expected_generation").get<int64_t>();
                    if (value < 0) {
                        throw std::invalid_argument("expected_generation must be non-negative");
                    }
                    expected_generation = (uint64_t) value;
                } else {
                    throw std::invalid_argument("expected_generation must be a non-negative integer");
                }
                has_expected_generation = true;
            }
            if (data.contains("id_slot")) {
                if (!data.at("id_slot").is_number_integer()) {
                    throw std::invalid_argument("id_slot must be an integer");
                }
                id_slot = data.at("id_slot").get<int>();
                if (id_slot < 0) {
                    throw std::invalid_argument("id_slot must be non-negative");
                }
            }
        } catch (const std::exception & e) {
            res->error(format_error_response(
                    std::string("Invalid logical head request: ") + e.what(),
                    ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        server_task_type type;
        if (action == "create") {
            if (digest.empty() || !expected_digest.empty() || has_expected_generation || id_slot >= 0) {
                res->error(format_error_response(
                        "Head create requires only digest", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_HEAD_CREATE;
        } else if (action == "advance") {
            if (digest.empty() || expected_digest.empty() || !has_expected_generation || id_slot >= 0) {
                res->error(format_error_response(
                        "Head advance requires digest, expected_digest, and expected_generation",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_HEAD_ADVANCE;
        } else if (action == "delete") {
            if (!digest.empty() || expected_digest.empty() || !has_expected_generation || id_slot >= 0) {
                res->error(format_error_response(
                        "Head delete requires expected_digest and expected_generation",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_HEAD_DELETE;
        } else if (action == "materialize") {
            if (!digest.empty() || expected_digest.empty() || !has_expected_generation) {
                res->error(format_error_response(
                        "Head materialize requires expected_digest and expected_generation",
                        ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            type = SERVER_TASK_TYPE_SNAPSHOT_HEAD_MATERIALIZE;
        } else {
            res->error(format_error_response("Invalid logical head action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        server_task task(type);
        task.id = res->rd.get_new_id();
        task.slot_action.head_name = head_name;
        task.slot_action.digest = std::move(digest);
        task.slot_action.expected_digest = std::move(expected_digest);
        task.slot_action.expected_generation = expected_generation;
        task.slot_action.id_slot = id_slot;
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_snapshot *>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "pcbt",                        { {"contract", "v1"}, {"enabled", false} } },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "statetree",                    json {
                {"lease_ms", params.statetree_lease_ms},
                {"max_state_bytes", params.statetree_max_state_bytes},
                {"byte_scope", "all_live_slot_prompt_state"},
                {"state_identity_scope", "server_process"},
                {"node_identity_scope", "server_process"},
                {"node_semantics", "immutable_branch_incarnation"},
                {"node_mutations", json::array({"fork", "snapshot", "commit", "renew", "erase"})},
                {"snapshot_enabled", params.statetree_max_snapshot_bytes > 0},
                {"max_snapshot_bytes", params.statetree_max_snapshot_bytes},
                {"snapshot_digest", "sha256:turbo-statetree-snapshot-v1"},
                {"snapshot_admission", "reject"},
                {"snapshot_storage", "digest_deduplicated_immutable_payloads"},
                {"durable_snapshot_enabled", !params.statetree_snapshot_store.empty()},
                {"durable_snapshot_compatibility_id", params.statetree_snapshot_compat_id.empty()
                    ? json(nullptr)
                    : json(params.statetree_snapshot_compat_id)},
                {"max_snapshot_disk_bytes", params.statetree_max_snapshot_disk_bytes},
                {"max_snapshot_load_bytes", params.statetree_max_snapshot_load_bytes},
                {"max_snapshot_manifest_bytes", params.statetree_max_snapshot_manifest_bytes},
                {"snapshot_manifest_format", "turbo-statetree-manifest-v1"},
                {"snapshot_manifest_checkpoint_schema", "atomic-publish-advance-v3"},
                {"durable_snapshot_publish", "intent_object_commit_reconciled"},
                {"durable_cache_reclamation", "managed_cache_only_oldest_revision"},
                {"durable_cache_pressure_publish", true},
                {"durable_logical_heads", "generation_digest_cas"},
                {"durable_logical_head_parent_edge", "previous_digest"},
                {"durable_logical_head_delete", "retry_safe_retired_name_tombstone"},
                {"durable_publish_advance", "intent_object_atomic_owner_head_commit"},
                {"snapshot_retention_classes", json::array({"pinned", "cache"})},
                {"durable_snapshot_format", "turbo-statetree-cold-v1"},
                {"durable_snapshot_admission", "reject"},
                {"durable_snapshot_io", "single_ordered_worker_two_phase"},
                {"durable_snapshot_load_budget_scope", "aggregate_queued_and_verified_payloads"},
                {"journal_capacity", SERVER_STATETREE_JOURNAL_CAPACITY},
            } },
            // New keys
            { "ui",                           params.ui },
            { "ui_settings",                  meta->json_ui_settings },
            // Deprecated: use ui/ui_settings instead (kept for backward compat)
            { "webui",                        params.webui },
            { "webui_settings",               meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy || params.webui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    std::string filename;
    int64_t fork_id = -1;
    try {
        const json request_data = json::parse(req.body);
        filename = request_data.at("filename");
        fork_id = server_optional_fork_id(request_data);
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot save request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.fork_id  = fork_id;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    std::string filename;
    int64_t fork_id = -1;
    try {
        const json request_data = json::parse(req.body);
        filename = request_data.at("filename");
        fork_id = server_optional_fork_id(request_data);
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot restore request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.fork_id  = fork_id;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(
        const server_http_req & req, int id_slot, int64_t node_id) {
    auto res = create_response();
    int64_t fork_id = -1;
    int64_t state_id = -1;
    try {
        const json request_data = req.body.empty() ? json::object() : json::parse(req.body);
        fork_id = server_optional_fork_id(request_data);
        state_id = server_optional_state_id(request_data);
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot erase request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.state_id = state_id;
        task.slot_action.node_id = node_id;
        task.slot_action.fork_id = fork_id;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_fork(
        const server_http_req & req, int id_slot, int64_t node_id) {
    auto res = create_response();

    std::vector<int> destinations;
    int64_t fork_id = -1;
    int64_t state_id = -1;
    try {
        const json request_data = json::parse(req.body);
        if (!request_data.is_object() ||
                !request_data.contains("destinations") ||
                !request_data.at("destinations").is_array()) {
            res->error(format_error_response(
                    "Slot fork requires a destinations array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        for (const auto & destination : request_data.at("destinations")) {
            if (!destination.is_number_integer()) {
                res->error(format_error_response(
                        "Slot fork destinations must be integers", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            destinations.push_back(destination.get<int>());
        }
        if (request_data.contains("fork_id")) {
            if (!request_data.at("fork_id").is_number_integer()) {
                res->error(format_error_response(
                        "Slot fork fork_id must be an integer", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            fork_id = request_data.at("fork_id").get<int64_t>();
            if (fork_id < 0) {
                res->error(format_error_response(
                        "Slot fork fork_id must be non-negative", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        state_id = server_optional_state_id(request_data);
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot fork request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    if (destinations.empty()) {
        res->error(format_error_response(
                "Slot fork requires at least one destination", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_FORK);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.state_id = state_id;
        task.slot_action.node_id = node_id;
        task.slot_action.fork_id = fork_id;
        task.slot_action.destinations = std::move(destinations);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_fork *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_snapshot_capture(
        const server_http_req & req, int64_t node_id) {
    auto res = create_response();

    int64_t fork_id = -1;
    int64_t state_id = -1;
    try {
        const json data = req.body.empty() ? json::object() : json::parse(req.body);
        fork_id = server_optional_fork_id(data);
        state_id = server_optional_state_id(data);
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid snapshot capture request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    server_task task(SERVER_TASK_TYPE_SNAPSHOT_CAPTURE);
    task.id = res->rd.get_new_id();
    task.slot_action.node_id = node_id;
    task.slot_action.state_id = state_id;
    task.slot_action.fork_id = fork_id;
    res->rd.post_task(std::move(task));

    auto result = res->rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    GGML_ASSERT(dynamic_cast<server_task_result_snapshot *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_commit(
        const server_http_req & req, int id_slot, int64_t node_id) {
    auto res = create_response();

    int64_t fork_id = -1;
    int64_t state_id = -1;
    try {
        const json request_data = req.body.empty() ? json::object() : json::parse(req.body);
        fork_id = server_optional_fork_id(request_data);
        state_id = server_optional_state_id(request_data);
        if (node_id < 0 && fork_id < 0) {
            res->error(format_error_response(
                    "Slot commit requires an integer fork_id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot commit request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_COMMIT);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.state_id = state_id;
        task.slot_action.node_id = node_id;
        task.slot_action.fork_id = fork_id;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_commit *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_renew(
        const server_http_req & req, int id_slot, int64_t node_id) {
    auto res = create_response();

    int64_t fork_id = -1;
    int64_t state_id = -1;
    try {
        const json request_data = req.body.empty() ? json::object() : json::parse(req.body);
        fork_id = server_optional_fork_id(request_data);
        state_id = server_optional_state_id(request_data);
        if (node_id < 0 && fork_id < 0) {
            res->error(format_error_response(
                    "Slot renew requires an integer fork_id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } catch (const std::exception & e) {
        res->error(format_error_response(
                std::string("Invalid slot renew request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RENEW);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.state_id = state_id;
        task.slot_action.node_id = node_id;
        task.slot_action.fork_id = fork_id;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_renew *>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
