//
// SYCL port of ggml-cuda/topk-moe.cu — see topk-moe.hpp for rationale.
// The compute kernel is a faithful SYCL translation of topk_moe_cuda (no-bias path); the
// fusion-detection helpers are ported near-verbatim from ggml-cuda.cu (they are pure graph /
// pointer inspection and backend-agnostic). Bias and groups are NOT handled here: if detected,
// we decline the fusion so the eager path runs unchanged.
//
#include "topk-moe.hpp"
#include "ggml.h"
#include "ggml-impl.h"

#include <cfloat>
#include <vector>
#include <initializer_list>

struct sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

static inline float topk_warp_reduce_max(float x, const sycl::sub_group & sg) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(sg, x, mask));
    }
    return x;
}

static inline float topk_warp_reduce_sum(float x, const sycl::sub_group & sg) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(sg, x, mask);
    }
    return x;
}

// One sub-group (== one work-group of WARP_SIZE) handles one row/token, mirroring topk_moe_cuda.
template <int n_experts>
static void topk_moe_kernel(const float * __restrict__ logits,
                            float * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            const int                  n_rows,
                            const int                  n_expert_used,
                            const float                clamp_val,
                            const float                scale_val,
                            const bool                 with_norm,
                            const bool                 use_sigmoid,
                            const bool                 delayed_softmax,
                            const sycl::nd_item<1> &   item) {
    const sycl::sub_group sg  = item.get_sub_group();
    const int             row = item.get_group(0);
    if (row >= n_rows) {
        return;
    }
    const int lane = item.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr int ept = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[ept];
#pragma unroll
    for (int i = 0; i < ept; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert    = i + lane;
        wt[i / WARP_SIZE]   = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!delayed_softmax) {
        if (use_sigmoid) {
#pragma unroll
            for (int i = 0; i < ept; i++) {
                wt[i] = 1.0f / (1.0f + sycl::exp(-wt[i]));
            }
        } else {
            float max_val = -INFINITY;
#pragma unroll
            for (int i = 0; i < ept; i++) {
                max_val = sycl::fmax(max_val, wt[i]);
            }
            max_val = topk_warp_reduce_max(max_val, sg);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < ept; i++) {
                wt[i] = sycl::exp(wt[i] - max_val);
                sum += wt[i];
            }
            sum               = topk_warp_reduce_sum(sum, sg);
            const float inv   = 1.0f / sum;
#pragma unroll
            for (int i = 0; i < ept; i++) {
                wt[i] *= inv;
            }
        }
    }

    // Sanitize NaN -> -FLT_MAX so the iterative argmax produces unique expert IDs (NaN compares false).
#pragma unroll
    for (int i = 0; i < ept; i++) {
        if (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;
    float output_weights[ept];
#pragma unroll
    for (int i = 0; i < ept; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < ept; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            const int   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            if (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (with_norm) {
        wt_sum            = topk_warp_reduce_sum(wt_sum, sg);
        wt_sum            = sycl::fmax(wt_sum, clamp_val);
        const float inv   = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < ept; i++) {
            output_weights[i] *= inv;
        }
    }

    if (delayed_softmax) {
        // softmax over just the selected weights (limit = n_expert_used)
        float max_val = -INFINITY;
#pragma unroll
        for (int i = 0; i < ept; i++) {
            const int idx = lane + i * WARP_SIZE;
            if (idx < n_expert_used) {
                max_val = sycl::fmax(max_val, output_weights[i]);
            }
        }
        max_val = topk_warp_reduce_max(max_val, sg);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < ept; i++) {
            const int idx = lane + i * WARP_SIZE;
            if (idx < n_expert_used) {
                output_weights[i] = sycl::exp(output_weights[i] - max_val);
                sum += output_weights[i];
            } else {
                output_weights[i] = 0.f;
            }
        }
        sum             = topk_warp_reduce_sum(sum, sg);
        const float inv = 1.0f / sum;
#pragma unroll
        for (int i = 0; i < ept; i++) {
            output_weights[i] *= inv;
        }
    }

#pragma unroll
    for (int i = 0; i < ept; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int n_experts>
static void launch_topk_moe(queue_ptr stream, const float * logits, float * weights, int32_t * ids,
                            int n_rows, int n_expert_used, float clamp_val, float scale_val,
                            bool with_norm, bool use_sigmoid, bool delayed_softmax) {
    const sycl::range<1> global((size_t) n_rows * WARP_SIZE);
    const sycl::range<1> local(WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global, local),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used,
                                                        clamp_val, scale_val, with_norm, use_sigmoid,
                                                        delayed_softmax, item);
                         });
    });
}

static void ggml_sycl_op_topk_moe(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor *         logits,
                                  ggml_tensor *               weights,
                                  ggml_tensor *               ids,
                                  const ggml_tensor *         clamp,
                                  const ggml_tensor *         scale,
                                  const sycl_topk_moe_args &  args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    const int n_experts     = logits->ne[0];
    const int n_rows        = logits->ne[1];
    const int n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    const bool  with_norm = clamp != nullptr;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    switch (n_experts) {
        case 1:   launch_topk_moe<1>  (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 2:   launch_topk_moe<2>  (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 4:   launch_topk_moe<4>  (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 8:   launch_topk_moe<8>  (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 16:  launch_topk_moe<16> (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 32:  launch_topk_moe<32> (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 64:  launch_topk_moe<64> (stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 128: launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 256: launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        case 512: launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val, with_norm, args.sigmoid, args.delayed_softmax); break;
        default:  GGML_ASSERT(false && "topk-moe: unsupported n_experts"); break;
    }
}

static bool sycl_should_use_topk_moe(const ggml_tensor * gating_op, const ggml_tensor * weights,
                                     const ggml_tensor * logits, const ggml_tensor * ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512)) {
        return false;
    }
    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }
    if (gating_op->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    (const float *) gating_op->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) gating_op->op_params + 1, sizeof(float));
        if (!ggml_is_contiguous(gating_op->src[0])) {
            return false;
        }
        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }
        if (gating_op->src[1] || gating_op->src[2]) {  // masks/sinks
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }
    return true;
}

// Ported from ggml_cuda_topk_moe_fusion — pure graph inspection.
static bool sycl_topk_moe_fusion(const ggml_cgraph * cgraph, int node_idx, sycl_topk_moe_args & args) {
    args = sycl_topk_moe_args{};

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }
    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        args.sigmoid = true;
    }
    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.softmax) {
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;
        if (node_idx >= n_nodes) {
            return false;
        }
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }
        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }
        node_idx++;
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;
        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };
        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };
        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;
        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }
        node_idx++;
    }

    if (node_idx < n_nodes && nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// Ported from ggml_cuda_check_fusion_memory_ranges — pure pointer/range inspection.
static bool sycl_check_fusion_memory_ranges(const ggml_cgraph * cgraph, int node_idx, int node_count,
                                            const int * out_nodes, int out_count, bool is_topk_moe) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(a->buffer), a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(b->buffer), b);
        return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
    };

    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) == 1) {
        return true;
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];
        for (int j = node_idx; j < node_idx + node_count; ++j) {
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];
                if (!src || src->op == GGML_OP_NONE) {
                    continue;
                }
                if (nodes_overlap(dst, src)) {
                    bool found = false;
                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) { found = true; break; }
                    }
                    if (!found) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int ggml_sycl_try_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];

    if (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT) {
        return 0;
    }

    sycl_topk_moe_args args;
    if (!sycl_topk_moe_fusion(cgraph, i, args)) {
        return 0;
    }

    // This kernel implements the no-bias path only; decline anything with a routing bias.
    if (args.prob_bias) {
        return 0;
    }

    const ggml_tensor * logits = node->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    int out_nodes[2];

    if (!args.delayed_softmax) {
        const ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        if (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (int) ops.size() - 3];
        }
        if (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (int) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (int) ops.size() - 1];
        out_nodes[1] = i + (int) ops.size() - 1;

        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            sycl_should_use_topk_moe(node, weights, logits, ids) &&
            sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    } else if (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0] = i + 1;
        out_nodes[1] = i + 5;
        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    }

    return 0;
}
