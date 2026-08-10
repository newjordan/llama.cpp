#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#include "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
int ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);


int ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// MUL_MAT_ID + MUL_MAT_ID + GLU(swiglu) dual expert path (serial decode).
// Implemented in ggml-sycl.cpp.
int ggml_sycl_fuse_moe_dual_swiglu(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// MUL_MAT + MUL_MAT + GLU(swiglu) dense shared-expert path.
// Implemented in ggml-sycl.cpp.
int ggml_sycl_fuse_dense_dual_swiglu(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// MoE down: MUL_MAT_ID → MUL(w) → VIEW×k → ADD×(k-1) weighted reduce.
// Implemented in ggml-sycl.cpp.
int ggml_sycl_fuse_moe_down_weighted(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
// [lx-qkv] Fuse Q/V/K attention projections into one reorder-MMVQ launch (serial decode).
int ggml_sycl_fuse_qkv(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);


// RMS_NORM → MUL(weight). Default ON. Kill: GGML_SYCL_DISABLE_RMS_NORM_FUSE=1
int ggml_sycl_fuse_rms_norm_mul(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// ROPE → VIEW → SET_ROWS (KV write). Default ON. Kill: GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE=1
int ggml_sycl_fuse_rope_set_rows(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// [lx-norm-rope] RMS_NORM+MUL+ROPE+VIEW+SET_ROWS (K-cache normed rope) in one launch.
// Default OFF: GGML_SYCL_FUSE_NORM_ROPE=1
int ggml_sycl_fuse_norm_rope(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_sycl_fuse_norm_rope_q(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// [lx-qk-rope] Merged Q+K normed rope: RMS+MUL+ROPE (Q, contiguous dst) and
// RMS+MUL+ROPE+VIEW+SET_ROWS (K, cache scatter) in ONE launch at serial decode.
// Kill switch: GGML_SYCL_DISABLE_QK_ROPE_MERGE=1 (falls back to the two
// separate fuses above). Engagement print: [lx-norm-rope-qk].
int ggml_sycl_fuse_norm_rope_qk(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// Decode-only Q6_K V projection -> indexed F16 V-cache write.
int ggml_sycl_fuse_v_mmvq_set_rows(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// Softplus × mul attn gate (Laguna XS.2). Default ON. Kill: GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE=1
int ggml_sycl_fuse_softplus_mul(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
// [lx-gate-softplus] attn_gate_proj GEMV + softplus + per-head MUL in one launch.
int ggml_sycl_fuse_gate_softplus_mul(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// ADD → ADD residual (moe+shexp then +ffn_inp). Default ON. Kill: GGML_SYCL_DISABLE_ADD_ADD_FUSE=1
int ggml_sycl_fuse_add_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// MUL_MAT → ADD residual (optional second ADD) via reorder MMVQ row_addend epilogue.
// Quality-safe default: decode only (src1.ne[1]==1). Implemented in ggml-sycl.cpp.
int ggml_sycl_fuse_mul_mat_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
