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

#ifndef GGML_SYCL_ROPE_HPP
#define GGML_SYCL_ROPE_HPP

#include "common.hpp"

#define SYCL_ROPE_BLOCK_SIZE 256

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_rope_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// [lx-norm-rope] RMS_NORM(+MUL) + NEOX rope + SET_ROWS in one launch (K-cache path).
void ggml_sycl_rope_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * rms,
                               ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// [lx-qk-rope] Merged Q+K normed rope: ONE launch for the Q chain
// (RMS+MUL+ROPE, contiguous dst) and the K chain (RMS+MUL+ROPE+VIEW+SET_ROWS
// scatter), removing the second per-layer dispatch at serial decode. Each
// chain's arithmetic is byte-for-byte the separate ggml_sycl_rope_norm_fused
// calls (same kernel body, same params, same order) — only the dispatch count
// changes. view_k is the K chain's VIEW node (validated, kept for the pattern).
void ggml_sycl_rope_norm_fused_qk(ggml_backend_sycl_context & ctx,
                                  ggml_tensor * rms_q, ggml_tensor * mul_q, ggml_tensor * rope_q,
                                  ggml_tensor * rms_k, ggml_tensor * mul_k, ggml_tensor * rope_k,
                                  ggml_tensor * view_k, ggml_tensor * set_rows_k);

#endif // GGML_SYCL_ROPE_HPP
