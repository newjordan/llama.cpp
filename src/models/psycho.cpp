#include "models.h"

// pSYCho: a SYCL-targeted block-draft speculative-decode head. Original work, inspired by the
// DFlash block-diffusion concept but with our own single-cache architecture, our own distilled
// weights, and our own seeded all-/anchor-block draft. No external weights or arch are used.
//
// A small standalone Qwen3-style transformer that drafts a whole block of tokens in ONE
// parallel forward pass (bidirectional, is_causal=false). It does not own token embeddings
// or an lm_head: it borrows the target's (via cparams.ctx_other). It is conditioned on a
// concatenation of several intermediate target-layer hidden states (target_layer_ids), fused
// by `fc` and normalised once by `hidden_norm` -> the "context".
//
// Two passes, like EAGLE3:
//   encoder (is_enc=true) : context features [n_embd_inp, n] -> fc -> hidden_norm -> g  (t_h_nextn)
//   decoder (is_enc=false): block tokens + g -> N layers of two-source attention -> draft logits
//
// Per-layer attention (Qwen3DFlashAttention): Q = q_norm(q_proj(h))  [noise/block stream only];
// K = k_norm(concat(k_proj(g), k_proj(h))); V = concat(v_proj(g), v_proj(h)); bidirectional.

void llama_model_psycho::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false) || target_layer_ids.empty()) {
        throw std::runtime_error("DFlash model requires a non-empty 'target_layers' in GGUF metadata");
    }

    uint32_t n_embd_tgt = 0;
    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);

    // fused context input width = (#taps) * target hidden size  -> consumed by `fc`
    hparams.n_embd_inp_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    LLAMA_LOG_INFO("%s: DFlash taps=%zu, n_embd_tgt=%u, n_embd_inp=%u, draft n_embd=%u\n",
            __func__, target_layer_ids.size(), n_embd_tgt, hparams.n_embd_inp_impl, hparams.n_embd);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_psycho::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp();

    // feature fusion (context): [n_embd_inp -> n_embd], then a single RMSNorm
    fc          = create_tensor(tn(LLM_TENSOR_FC,                 "weight"), {n_embd_inp, n_embd}, 0);
    hidden_norm = create_tensor(tn(LLM_TENSOR_PSYCHO_HIDDEN_NORM, "weight"), {n_embd}, 0);

    // final norm; lm_head + token embeddings are normally borrowed from the target (optional here)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);

        layer.wq          = create_tensor(tn(LLM_TENSOR_ATTN_Q,      "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk          = create_tensor(tn(LLM_TENSOR_ATTN_K,      "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv          = create_tensor(tn(LLM_TENSOR_ATTN_V,      "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // Qwen3 per-head q/k norms over head_dim
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate    = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down    = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up      = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_psycho::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_psycho::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));
    return cur;
}

// Encoder: fuse the concatenated target hidden states into the context g = hidden_norm(fc(features)).
// Stored in t_h_nextn so the runtime can read it back (same channel EAGLE3/MTP use).
template <>
llama_model_psycho::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "hidden_norm", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// Decoder: draft a block in one bidirectional forward.
//   inp->tokens : block token ids        -> tok_embd -> h   (the "noise" stream, n_tokens = block)
//   inp->embd   : context g [n_embd, n]  (from the encoder pass)
// Per layer: Q from h only; K/V = concat(context, h) through the same projections; k_norm over the
// concatenated K; no causal mask (full/bidirectional). NOTE: first cut feeds context with the same
// position set as the block; incremental context KV + per-stream RoPE positions land in the runtime
// integration (P4) + numerical-parity pass (P3).
template <>
llama_model_psycho::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // borrow target token embeddings if the draft ships none (null during the memory-reserve
    // probe, before ctx_other is wired -> fall back to a shape-only placeholder)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr && cparams.ctx_other != nullptr) {
        tok_embd = llama_get_model(cparams.ctx_other)->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    // Input selection (mirrors build_inp_embd via ubatch.token): a TOKEN batch is the drafted block
    // (embed via the borrowed target table); an EMBD batch is the fused context g that process()
    // decodes to populate the committed-context K/V — the block then attends to that cached context.
    ggml_tensor * h;
    if (ubatch.token) {
        if (tok_embd != nullptr) {
            h = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        } else {
            // reserve probe (no target bound): keep inp->tokens consumed (zeroed) so it allocates
            ggml_tensor * base  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(base);
            ggml_tensor * tok_f = ggml_reshape_2d(ctx0, ggml_cast(ctx0, inp->tokens, GGML_TYPE_F32), 1, n_tokens);
            h = ggml_add(ctx0, base, ggml_scale(ctx0, tok_f, 0.0f));
        }
    } else {
        h = inp->embd;  // fused context g (cached by process())
    }
    cb(h, "inp_embd", -1);
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    auto *        inp_attn    = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));
    const bool  is_block = (ubatch.token != nullptr);  // token batch = drafted block; embd batch = context g
    ggml_tensor * ctx_out = nullptr;                    // accumulates context-pass outputs (keeps K/V writes reachable)

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = h;

        // noise/block stream gets per-layer input_layernorm; the fused context g does NOT (it is
        // already hidden_norm'd once by the encoder and fed raw to every layer's k/v_proj).
        ggml_tensor * attn_in = is_block ? build_norm(h, layer.attn_norm, NULL, LLM_NORM_RMS, il) : h;
        cb(attn_in, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, attn_in);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, attn_in);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, attn_in);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);  // k_norm over k_ctx and k_noise alike

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        ggml_tensor * cur = build_attn(inp_attn, layer.wo, NULL, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        if (!is_block) {
            // CONTEXT pass: each layer caches K/V from the FIXED g (h is not evolved), so the block's
            // layer L later cross-attends k/v_proj_L(g). Accumulate the (discarded) outputs only to
            // keep all the per-layer K/V cache writes reachable in the graph.
            ctx_out = ctx_out ? ggml_add(ctx0, ctx_out, cur) : cur;
            continue;
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        h = ggml_add(ctx0, cur, ffn_inp);
        cb(h, "layer_out", il);
    }

    // block pass -> evolved h; context pass -> the accumulated (ignored) output
    ggml_tensor * out_h = is_block ? h : ctx_out;

    // NOTE: unlike EAGLE3 (autoregressive), DFlash does not feed back a per-token pre-norm hidden,
    // so the decoder does not expose t_h_nextn (only the encoder's context output is read back).
    ggml_tensor * cur = build_norm(out_h, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head: borrow the target's if the draft ships none (placeholder during reserve probe)
    auto * output = model.output;
    if (output == nullptr && cparams.ctx_other != nullptr) {
        output = llama_get_model(cparams.ctx_other)->output;
    }
    if (output != nullptr) {
        cur = build_lora_mm(output, cur);
    } else {
        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t) model.vocab.n_tokens(), n_tokens);
        ggml_set_input(cur);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
