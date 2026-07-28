#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_state_io(
        ggml_backend_sycl_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * state_cache,
        const ggml_tensor * state_ids,
        const ggml_tensor * state_scratch,
        const ggml_tensor * token,
        ggml_tensor * state_dst,
        int64_t n_rs);
void ggml_sycl_state_io_gather_conflicts(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * state_cache,
        const ggml_tensor * state_ids,
        ggml_tensor * state_scratch,
        const ggml_tensor * state_dst,
        int64_t n_rs);
