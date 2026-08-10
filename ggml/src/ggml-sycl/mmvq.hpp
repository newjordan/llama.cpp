//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused Q/V/K/G decode launch: one grid covers the Q (q4_K), V (q6_K), K (q4_K)
// and per-head attention gate (q4_K) segments against one q8_1 activation row
// (see mmvq.cpp).
void reorder_mul_mat_vec_q4_q6_q4_k_q8_1_qkv_sycl(
    const void * vx_q, const void * vx_v, const void * vx_k, const void * vx_g,
    const void * vy, float * dst_q, float * dst_v, float * dst_k, float * dst_g,
    const int ncols, const int nrows_q, const int nrows_v, const int nrows_k,
    const int nrows_g,
    dpct::queue_ptr stream);

// Same fused Q/V/K/G launch with a q4_K V segment (Q4_K_M stores some
// attention.wv tensors at q4_K).
void reorder_mul_mat_vec_q4_q4_q4_k_q8_1_qkv_sycl(
    const void * vx_q, const void * vx_v, const void * vx_k, const void * vx_g,
    const void * vy, float * dst_q, float * dst_v, float * dst_k, float * dst_g,
    const int ncols, const int nrows_q, const int nrows_v, const int nrows_k,
    const int nrows_g,
    dpct::queue_ptr stream);

// Returns false if src0_type isn't handled; caller should fall back.
// n_tokens>1: grid dim0 over tokens; ids_row_stride/src1_token_stride/dst_token_stride index per-token.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1)
    const int32_t *    ids_dev,             // device-side int32, n_experts_used per token
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst expert rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Dual gate+up MoE MMVQ with fused SwiGLU. Writes silu(gate)*up into dst.
// Serial decode path (one activation vector, shared or per-slot). Returns false
// if type unsupported.
bool ggml_sycl_mul_mat_vec_q_id_dual_swiglu_reorder(
    enum ggml_type     src0_type,
    const void *       vx_gate_base,
    const void *       vx_up_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             gate_expert_stride,
    size_t             up_expert_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Dense shared-expert gate/up (ordinary MUL_MAT, not MUL_MAT_ID). Writes silu(gate)*up.
// Q4_K/Q5_K/Q6_K reorder MMVQ. ncols_dst is activation batch (1 for decode; cap 32).
bool ggml_sycl_mul_mat_vec_q_dense_dual_swiglu_reorder(
    enum ggml_type     src0_type,
    const void *       vx_gate,
    const void *       vx_up,
    const void *       vy,
    float *            dst,
    int                ncols,
    int                nrows,
    int                ncols_dst,
    size_t             src1_col_stride_bytes,
    size_t             dst_col_stride,
    dpct::queue_ptr    stream);

// Integrated MoE down: per-row sum over experts of (GEMV * route_weight).
// Decode-first (n_tokens typically 1); Q4/Q5/Q6_K reorder. Writes only final dst.
// n_experts_used must be 8 for the unrolled path (Laguna); other k fall back false.
bool ggml_sycl_mul_mat_vec_q_id_weighted_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    const float *      weights,
    float *            dst,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             src1_row_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             weights_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Host-side row residual for reorder MMVQ epilogue (MUL_MAT+ADD fuse).
// Pointers are USM device floats, length = nrows (per column for multi-col).
// Cleared after each fused launch. Thread-local to the calling host thread.
void ggml_sycl_mmvq_set_row_addend(const float * addend);
void ggml_sycl_mmvq_set_row_addend2(const float * addend2);
const float * ggml_sycl_mmvq_get_row_addend();
const float * ggml_sycl_mmvq_get_row_addend2();

// [lx-gate-softplus] XS.2 per-head attention-gate epilogue (see mmvq.cpp).
// Set non-null before ggml_sycl_mul_mat for the gate GEMV to also write the
// UNARY(SOFTPLUS) dst and the per-head MUL dst; clear (nullptr) after.
struct lx_gate_epilogue {
    float *       softplus_dst;
    const float * attn;
    float *       mul_dst;
    int64_t       hd, nh, nt;
};
void ggml_sycl_mmvq_set_gate_epilogue(const struct lx_gate_epilogue * ep);
const struct lx_gate_epilogue * ggml_sycl_mmvq_get_gate_epilogue();

// Optional decode-only V-cache publication for a single-column reordered Q6_K
// MMVQ. The ordinary F32 destination is still written; the finalized scalar is
// also converted to F16 and stored at cache_rows[cache_idx[0]].
void ggml_sycl_mmvq_set_vcache_f16(sycl::half * cache, const int64_t * cache_idx,
                                   int64_t cache_row_stride);

#endif // GGML_SYCL_MMVQ_HPP

