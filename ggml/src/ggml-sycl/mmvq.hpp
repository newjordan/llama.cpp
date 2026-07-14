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

// Chunk-of-8 wrapper: handles any src1_ncols by looping the multi-column MMVQ kernels,
// keeping quantized weights on the fast path for batch sizes past the 8-column templates.
void ggml_sycl_op_mul_mat_vec_q_wide(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused routed-expert epilogue for MoE down projections.  Consumes the
// [n_embd, n_experts_used, n_tokens] MUL_MAT_ID result and routing weights and
// writes the weighted expert sum directly, replacing the broadcast MUL plus
// expert-view ADD chain with one kernel.
void ggml_sycl_op_moe_weighted_sum(
    const ggml_tensor * experts,
    const ggml_tensor * weights,
    ggml_tensor *       dst,
    const dpct::queue_ptr & stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1)
    const int32_t *    ids_dev,             // device-side int32, length n_experts_used
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    int                n_tokens,             // batched decode: grid dim 0, one ids row per token
    int                ids_row_stride,       // int32 elems between tokens' ids rows
    size_t             src1_token_stride,    // bytes between tokens in vy
    size_t             dst_token_stride,     // bytes between tokens in dst_base
    dpct::queue_ptr    stream);

// Standard-layout Q8_0 production down-projection variant with direct ordered
// routing-weight accumulation.
bool ggml_sycl_mul_mat_vec_q_id_weighted(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    const float *      weights,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             src1_row_stride,
    size_t             weights_slot_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             weights_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Grouped (phase-2) reorder variant: buckets (token, slot) pairs by expert on device, then
// reads each distinct expert's weights once per batch, dotting all its routed tokens per pass.
// scratch: device int32 buffer of at least (2*n_as + 2 + n_tokens*n_experts_used) elements.
// Returns false (stream untouched) if src0_type/n_as/shape isn't handled; caller falls back.
bool ggml_sycl_mul_mat_vec_q_id_grouped_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    int32_t *          scratch,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_as,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
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

// Fused separate gate/up expert projections. Both matrices use the same
// routed activations and expert ids. Each subgroup computes the matching gate
// and up row and stores silu(gate) * up directly, avoiding both full MMID
// intermediates and the following GLU submission.
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
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

bool ggml_sycl_mul_mat_vec_q_id_dual_swiglu_grouped_reorder(
    enum ggml_type     src0_type,
    const void *       vx_gate_base,
    const void *       vx_up_base,
    const void *       vy,
    const int32_t *    ids_dev,
    int32_t *          scratch,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_as,
    size_t             gate_expert_stride,
    size_t             up_expert_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Production MoE down-projection path. Computes every routed expert dot
// product for a token, applies routing weights in slot order, and writes only
// the final reduced output. This avoids materializing the expert-result tensor.
bool ggml_sycl_mul_mat_vec_q_id_weighted_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    const float *      weights,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             src1_row_stride,
    size_t             weights_slot_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             weights_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Batched ordered down projection with cross-token expert weight reuse. A
// routing pre-pass groups token/slot pairs by expert; the main row workgroup
// retains weighted contributions in local memory and reduces slots in their
// original order.
bool ggml_sycl_mul_mat_vec_q_id_weighted_grouped_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    const float *      weights,
    int32_t *          scratch,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_as,
    size_t             expert_weight_stride,
    size_t             src1_row_stride,
    size_t             weights_slot_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             weights_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

// Complete batched MoE expert pipeline for separate gate/up matrices.  One
// routing pre-pass feeds a paired gate/up SwiGLU kernel that writes reordered
// Q8_1 blocks directly, followed by the grouped ordered down projection.
bool ggml_sycl_mul_mat_vec_q_id_swiglu_down_grouped_reorder(
    enum ggml_type     gate_type,
    enum ggml_type     down_type,
    const void *       vx_gate_base,
    const void *       vx_up_base,
    const void *       vx_down_base,
    const void *       vy,
    const int32_t *    ids_dev,
    const float *      weights,
    int32_t *          scratch,
    void *             q8_intermediate,
    float *            dst_base,
    int                gate_ncols,
    int                gate_nrows,
    int                gate_nrows_padded,
    int                down_nrows,
    int                n_experts_used,
    int                n_as,
    size_t             gate_expert_stride,
    size_t             up_expert_stride,
    size_t             down_expert_stride,
    size_t             src1_row_stride,
    size_t             weights_slot_stride,
    int                n_tokens,
    int                ids_row_stride,
    size_t             src1_token_stride,
    size_t             q8_row_stride,
    size_t             q8_token_stride,
    size_t             weights_token_stride,
    size_t             dst_token_stride,
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
