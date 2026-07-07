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

#endif // GGML_SYCL_MMVQ_HPP
