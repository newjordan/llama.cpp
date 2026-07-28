#include "ssm_conv.hpp"
#include "common.hpp"

#include <cstdio>

using namespace sycl;

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(global_range, local_range),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int channel = static_cast<int>(idx % d_inner);
                const int token   = static_cast<int>((idx / d_inner) % n_t);
                const int seq     = static_cast<int>(idx / (static_cast<size_t>(d_inner) * static_cast<size_t>(n_t)));

                const float *s = src_data
                    + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
                    + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
                    + static_cast<size_t>(token);

                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

                float sumf = 0.0f;
                for (int i0 = 0; i0 < d_conv; ++i0) {
                    sumf += s[i0] * c[i0];
                }

                const size_t dst_idx =
                    static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
                    static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
                    static_cast<size_t>(channel);

                dst_data[dst_idx] = sumf;
            }
        );
    });
}

static void kernel_ssm_conv_state_io(
    queue &q,
    const float *state_cache,
    const int32_t *state_ids,
    const float *state_scratch,
    const float *token,
    const float *weights,
    float *dst_data,
    float *state_dst,
    int d_conv,
    int d_inner,
    int n_s,
    size_t state_cache_stride,
    int64_t state_dst_row0,
    size_t token_stride_inner,
    size_t token_stride_seq,
    size_t state_dst_stride
) {
    constexpr int max_d_conv = 16;
    GGML_ASSERT(d_conv >= 2 && d_conv <= max_d_conv);

    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(range<1>(num_work_groups * work_group_size), range<1>(work_group_size)),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int channel = static_cast<int>(idx % d_inner);
                const int seq     = static_cast<int>(idx / static_cast<size_t>(d_inner));
                const size_t state_channel = static_cast<size_t>(channel) * static_cast<size_t>(d_conv - 1);
                const int64_t state_src_row = state_ids[seq];
                const bool staged = state_src_row != state_dst_row0 + seq;
                const float *state_src = (staged
                    ? state_scratch + static_cast<size_t>(seq) * static_cast<size_t>((d_conv - 1) * d_inner)
                    : state_cache + static_cast<size_t>(state_src_row) * state_cache_stride) + state_channel;
                float *state_out = state_dst
                    + static_cast<size_t>(seq) * state_dst_stride + state_channel;

                float old_state[max_d_conv - 1];
                float sumf = 0.0f;
                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);
                for (int i0 = 0; i0 < d_conv - 1; ++i0) {
                    old_state[i0] = state_src[i0];
                    sumf += old_state[i0] * c[i0];
                }
                const float token_value = token[
                    static_cast<size_t>(seq) * token_stride_seq +
                    static_cast<size_t>(channel) * token_stride_inner];
                sumf += token_value * c[d_conv - 1];
                dst_data[idx] = sumf;

                for (int i0 = 0; i0 < d_conv - 2; ++i0) {
                    state_out[i0] = old_state[i0 + 1];
                }
                state_out[d_conv - 2] = token_value;
            });
    });
}

inline void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        float *dst_data       = static_cast<float *>(dst->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq
        );

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

void ggml_sycl_ssm_conv_state_io(
        ggml_backend_sycl_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * state_cache,
        const ggml_tensor * state_ids,
        const ggml_tensor * state_scratch,
        const ggml_tensor * token,
        ggml_tensor * state_dst,
        int64_t n_rs) {
    const ggml_tensor * weights = dst->src[1];
    const int d_conv  = weights->ne[0];
    const int d_inner = weights->ne[1];
    const int n_s     = dst->ne[2];
    const int64_t state_size = (int64_t) (d_conv - 1) * d_inner;

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(state_cache->type == GGML_TYPE_F32 && token->type == GGML_TYPE_F32 &&
                state_scratch->type == GGML_TYPE_F32 && state_dst->type == GGML_TYPE_F32 &&
                state_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == d_inner && dst->ne[1] == 1 && dst->ne[2] == n_s);
    GGML_ASSERT(state_cache->ne[0] == state_size && state_dst->ne[0] == state_size &&
                state_dst->ne[1] == n_s && state_ids->ne[0] == n_s &&
                state_scratch->ne[0] == state_size && state_scratch->ne[1] == n_s);
    GGML_ASSERT(token->ne[0] == 1 && token->ne[1] == d_inner && token->ne[2] == n_s);
    GGML_ASSERT(state_cache->nb[0] == sizeof(float) && state_dst->nb[0] == sizeof(float) &&
                weights->nb[0] == sizeof(float));

    const uintptr_t cache_ptr = (uintptr_t) state_cache->data;
    const uintptr_t dst_ptr   = (uintptr_t) state_dst->data;
    GGML_ASSERT(dst_ptr >= cache_ptr && state_cache->nb[1] != 0 &&
                (dst_ptr - cache_ptr) % state_cache->nb[1] == 0);
    const int64_t state_dst_row0 = (dst_ptr - cache_ptr) / state_cache->nb[1];
    GGML_ASSERT(n_rs >= n_s && state_dst_row0 + n_rs <= state_cache->ne[1]);

    kernel_ssm_conv_state_io(
        *ctx.stream(),
        (const float *) state_cache->data,
        (const int32_t *) state_ids->data,
        (const float *) state_scratch->data,
        (const float *) token->data,
        (const float *) weights->data,
        (float *) dst->data,
        (float *) state_dst->data,
        d_conv,
        d_inner,
        n_s,
        state_cache->nb[1] / sizeof(float),
        state_dst_row0,
        token->nb[1] / sizeof(float),
        token->nb[2] / sizeof(float),
        state_dst->nb[1] / sizeof(float));
}

void ggml_sycl_state_io_gather_conflicts(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * state_cache,
        const ggml_tensor * state_ids,
        ggml_tensor * state_scratch,
        const ggml_tensor * state_dst,
        int64_t n_rs) {
    GGML_ASSERT(state_cache->type == GGML_TYPE_F32 && state_scratch->type == GGML_TYPE_F32);
    GGML_ASSERT(state_ids->type == GGML_TYPE_I32 && state_ids->ne[0] == state_scratch->ne[1]);
    GGML_ASSERT(state_cache->ne[0] == state_scratch->ne[0]);
    GGML_ASSERT(state_cache->nb[0] == sizeof(float) && state_scratch->nb[0] == sizeof(float));

    const uintptr_t cache_ptr = (uintptr_t) state_cache->data;
    const uintptr_t dst_ptr   = (uintptr_t) state_dst->data;
    GGML_ASSERT(dst_ptr >= cache_ptr && state_cache->nb[1] != 0 &&
                (dst_ptr - cache_ptr) % state_cache->nb[1] == 0);
    const int64_t dst_row0 = (dst_ptr - cache_ptr) / state_cache->nb[1];
    const int64_t state_size = state_scratch->ne[0];
    const int64_t n_seqs = state_scratch->ne[1];
    GGML_ASSERT(n_rs >= n_seqs && dst_row0 + n_rs <= state_cache->ne[1]);

    const float * cache = (const float *) state_cache->data;
    const int32_t * ids = (const int32_t *) state_ids->data;
    float * scratch = (float *) state_scratch->data;
    const size_t cache_stride = state_cache->nb[1] / sizeof(float);
    const size_t scratch_stride = state_scratch->nb[1] / sizeof(float);
    const size_t total = (size_t) state_size * (size_t) n_seqs;
    constexpr size_t local_size = 256;
    const size_t global_size = ((total + local_size - 1) / local_size) * local_size;

    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            const size_t index = item.get_global_id(0);
            if (index >= total) {
                return;
            }
            const int64_t seq = index / (size_t) state_size;
            const int64_t elem = index % (size_t) state_size;
            const int64_t src_row = ids[seq];
            if (src_row != dst_row0 + seq) {
                scratch[(size_t) seq * scratch_stride + (size_t) elem] =
                    cache[(size_t) src_row * cache_stride + (size_t) elem];
            }
        });
}
