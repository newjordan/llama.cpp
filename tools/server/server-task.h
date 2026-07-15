#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SLOT_FORK,
    SERVER_TASK_TYPE_SLOT_COMMIT,
    SERVER_TASK_TYPE_SLOT_RENEW,
    SERVER_TASK_TYPE_SNAPSHOT_CAPTURE,
    SERVER_TASK_TYPE_SNAPSHOT_MATERIALIZE,
    SERVER_TASK_TYPE_SNAPSHOT_ERASE,
    SERVER_TASK_TYPE_SNAPSHOT_SPILL,
    SERVER_TASK_TYPE_SNAPSHOT_PUBLISH,
    SERVER_TASK_TYPE_SNAPSHOT_PUBLISH_ADVANCE,
    SERVER_TASK_TYPE_SNAPSHOT_CONTENT_MATERIALIZE,
    SERVER_TASK_TYPE_SNAPSHOT_CONTENT_ERASE,
    SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RETAIN,
    SERVER_TASK_TYPE_SNAPSHOT_CONTENT_RELEASE,
    SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_COMPACT,
    SERVER_TASK_TYPE_SNAPSHOT_MANIFEST_PRUNE,
    SERVER_TASK_TYPE_SNAPSHOT_HEAD_CREATE,
    SERVER_TASK_TYPE_SNAPSHOT_HEAD_ADVANCE,
    SERVER_TASK_TYPE_SNAPSHOT_HEAD_DELETE,
    SERVER_TASK_TYPE_SNAPSHOT_HEAD_MATERIALIZE,
    SERVER_TASK_TYPE_DURABLE_IO_COMPLETE,
    SERVER_TASK_TYPE_STATETREE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    // number of prompt tokens before the latest user message
    int32_t n_before_user = -1;

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    // Per-request cap for speculative proposals. -1 uses the server default.
    int32_t speculative_n_max = -1;
    bool speculative_serial_anchor = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;
    int64_t fork_id = -1; // expected StateTree generation for an explicit slot request
    int64_t state_id = -1; // logical StateTree lineage, independent of its physical slot
    int64_t node_id = -1; // immutable branch incarnation within a StateTree lineage

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE,
    // SERVER_TASK_TYPE_SLOT_FORK, SERVER_TASK_TYPE_SLOT_COMMIT, SERVER_TASK_TYPE_SLOT_RENEW
    struct slot_action {
        int id_slot = -1;
        int64_t state_id = -1;
        int64_t node_id = -1;
        int64_t fork_id = -1;
        int64_t snapshot_id = -1;
        std::vector<int> destinations;
        std::string digest;
        std::string owner;
        std::string retention_class;
        std::string head_name;
        std::string expected_digest;
        uint64_t expected_generation = 0;
        std::string filename;
        std::string filepath;
        uint64_t durable_io_id = 0;
        uint64_t target_bytes = 0;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    static task_params params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data);

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;
    std::vector<int32_t> draft_n_per_round;
    std::vector<int32_t> draft_n_accepted_per_round;
    int32_t draft_anchor_n = 0;
    int32_t draft_anchor_match_n = 0;
    int32_t draft_anchor_fallback_n = 0;
    std::vector<llama_token> draft_anchor_serial_tokens;
    std::vector<llama_token> draft_anchor_batched_tokens;
    int32_t draft_audit_tokens = 0;
    int32_t draft_audit_tokens_matched = 0;
    int32_t draft_audit_fallback_n = 0;
    std::vector<int32_t> draft_audit_first_mismatch;
    std::vector<llama_token> draft_audit_serial_tokens;
    std::vector<llama_token> draft_audit_batched_tokens;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    int n_idle_slots;
    int n_processing_slots;
    int n_reserved_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
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

    uint64_t statetree_state_bytes             = 0;
    uint64_t statetree_retained_bytes          = 0;
    uint64_t statetree_active_bytes            = 0;
    uint64_t statetree_state_budget_bytes      = 0;
    uint64_t statetree_state_high_water_bytes  = 0;
    uint64_t statetree_retained_high_water_bytes = 0;
    uint64_t statetree_expired_total           = 0;
    uint64_t statetree_evicted_total           = 0;
    uint64_t statetree_reclaimed_bytes_total   = 0;
    uint64_t statetree_renewed_total           = 0;
    uint64_t statetree_pressure_rejected_total = 0;
    uint64_t statetree_snapshot_bytes          = 0;
    uint64_t statetree_snapshot_budget_bytes   = 0;
    uint64_t statetree_snapshot_high_water_bytes = 0;
    uint64_t statetree_snapshot_count          = 0;
    uint64_t statetree_snapshot_content_count  = 0;
    uint64_t statetree_snapshots_captured_total = 0;
    uint64_t statetree_snapshots_materialized_total = 0;
    uint64_t statetree_snapshots_erased_total  = 0;
    uint64_t statetree_snapshot_rejected_total = 0;
    uint64_t statetree_durable_disk_bytes = 0;
    uint64_t statetree_durable_disk_budget_bytes = 0;
    uint64_t statetree_durable_disk_high_water_bytes = 0;
    uint64_t statetree_durable_content_count = 0;
    uint64_t statetree_durable_spilled_total = 0;
    uint64_t statetree_durable_materialized_total = 0;
    uint64_t statetree_durable_erased_total = 0;
    uint64_t statetree_durable_rejected_total = 0;
    uint64_t statetree_durable_recovered_temp_files = 0;
    uint64_t statetree_durable_ignored_corrupt_files = 0;
    uint64_t statetree_durable_runtime_integrity_failures = 0;
    uint64_t statetree_durable_orphaned_disk_bytes = 0;
    uint64_t statetree_durable_io_pending = 0;
    uint64_t statetree_durable_io_queue_high_water = 0;
    uint64_t statetree_durable_io_completed_total = 0;
    uint64_t statetree_durable_io_cancelled_loads_total = 0;
    uint64_t statetree_durable_io_reserved_disk_bytes = 0;
    uint64_t statetree_durable_io_reserved_disk_high_water = 0;
    uint64_t statetree_durable_io_reserved_load_bytes = 0;
    uint64_t statetree_durable_io_reserved_load_high_water = 0;
    uint64_t statetree_durable_manifest_refs = 0;
    uint64_t statetree_durable_manifest_revision = 0;
    uint64_t statetree_durable_manifest_file_bytes = 0;
    uint64_t statetree_durable_manifest_budget_bytes = 0;
    uint64_t statetree_durable_manifest_high_water_bytes = 0;
    uint64_t statetree_durable_manifest_record_count = 0;
    uint64_t statetree_durable_manifest_recovered_temp_files = 0;
    uint64_t statetree_durable_manifest_recovered_tail_bytes = 0;
    uint64_t statetree_durable_manifest_compactions = 0;
    uint64_t statetree_durable_manifest_recovered_publish_commits = 0;
    uint64_t statetree_durable_manifest_recovered_publish_aborts = 0;
    uint64_t statetree_durable_managed_count = 0;
    uint64_t statetree_durable_cache_evicted_total = 0;
    uint64_t statetree_durable_cache_reclaimed_bytes_total = 0;
    uint64_t statetree_durable_managed_recovered_erases = 0;
    uint64_t statetree_durable_managed_recovered_bytes = 0;
    uint64_t statetree_durable_retained_total = 0;
    uint64_t statetree_durable_released_total = 0;
    uint64_t statetree_durable_compacted_total = 0;
    int      n_statetree_families               = 0;
    int      n_statetree_active_families        = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    int64_t state_id;
    int64_t node_id;
    int64_t parent_node_id;
    int64_t fork_id;
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_slot_fork : server_task_result {
    int64_t state_id;
    int64_t node_id;
    int64_t parent_node_id;
    int64_t fork_id;
    json nodes = json::array();
    std::vector<int> destinations;

    size_t n_tokens;
    double t_ms;

    bool retention_enabled = false;
    int64_t lease_remaining_ms = -1;
    uint64_t state_bytes = 0;
    uint64_t retained_bytes = 0;
    uint64_t state_budget_bytes = 0;

    virtual json to_json() override;
};

struct server_task_result_slot_commit : server_task_result {
    int64_t state_id;
    int64_t node_id;
    int64_t parent_node_id;
    int source_id;
    int64_t fork_id;
    json nodes = json::array();
    std::vector<int> released;

    size_t n_tokens;
    double t_ms;

    bool retention_enabled = false;
    int64_t lease_remaining_ms = -1;
    uint64_t state_bytes = 0;
    uint64_t retained_bytes = 0;
    uint64_t state_budget_bytes = 0;

    virtual json to_json() override;
};

struct server_task_result_slot_renew : server_task_result {
    int64_t state_id;
    int64_t node_id;
    int64_t parent_node_id;
    int source_id;
    int64_t fork_id;
    json nodes = json::array();
    std::vector<int> members;

    int64_t lease_remaining_ms;
    uint64_t state_bytes;
    uint64_t retained_bytes;
    uint64_t state_budget_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_statetree : server_task_result {
    json states = json::array();
    json snapshots = json::array();
    json journal = json::array();
    uint64_t snapshot_bytes = 0;
    uint64_t snapshot_budget_bytes = 0;
    uint64_t snapshot_high_water_bytes = 0;
    uint64_t snapshot_content_count = 0;
    json durable_contents = json::array();
    json durable_manifest_refs = json::array();
    json durable_managed_digests = json::array();
    json durable_logical_heads = json::array();
    uint64_t durable_disk_bytes = 0;
    uint64_t durable_disk_budget_bytes = 0;
    uint64_t durable_disk_high_water_bytes = 0;
    uint64_t durable_recovered_temp_files = 0;
    uint64_t durable_ignored_corrupt_files = 0;
    uint64_t durable_runtime_integrity_failures = 0;
    uint64_t durable_orphaned_disk_bytes = 0;
    uint64_t durable_io_pending = 0;
    uint64_t durable_io_queue_high_water = 0;
    uint64_t durable_io_completed_total = 0;
    uint64_t durable_io_cancelled_loads_total = 0;
    uint64_t durable_io_reserved_disk_bytes = 0;
    uint64_t durable_io_reserved_disk_high_water = 0;
    uint64_t durable_io_reserved_load_bytes = 0;
    uint64_t durable_io_reserved_load_high_water = 0;
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
    uint64_t journal_capacity = 0;
    uint64_t journal_oldest_sequence = 0;
    uint64_t journal_next_sequence = 0;

    virtual json to_json() override;
};

struct server_task_result_snapshot : server_task_result {
    json result = json::object();

    virtual json to_json() override {
        return result;
    }
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt {
    server_tokens tokens;

    server_prompt_data data;

    std::list<common_prompt_checkpoint> checkpoints;

    size_t checkpoint_size() const {
        size_t res = 0;
        for (const auto & ckpt : checkpoints) {
            res += ckpt.size();
        }
        return res;
    }

    size_t size() const {
        return data.size() + checkpoint_size();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            data,
            checkpoints,
        };
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_main, llama_context * ctx_drft, int32_t id_slot);

    void update();
};
