#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. Kernel mirrors CUDA topk_moe_cuda including has_bias
// (select with wt+bias, emit unbiased wt). Fusion helpers port graph inspection from
// ggml-cuda.cu; Laguna score-correction bias path is enabled (see fuse_topk_moe).

struct ggml_sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <int experts_per_thread, bool use_limit>
static inline void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        if (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
static inline void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::exp(-vals[i])) : -INFINITY;
    }
}

/*
    Fused router: gating activation -> optional score-correction bias -> top-k ids
    -> optional weight norm/scale. One warp (sub-group) per token.

    has_bias=false: iterative warp argmax (CUDA no-bias path).
    has_bias=true:  bitonic argsort on selection scores matching SYCL k_argsort
                    (GGML_SORT_ORDER_DESC) so golden matches the eager GPU path;
                    emit unbiased wt for selected experts (Laguna / DeepSeek-V3).
*/
template <typename T>
static inline void topk_swap(T & a, T & b) {
    T t = a;
    a   = b;
    b   = t;
}

// No-bias path: iterative top-k (CUDA parity).
template <int n_experts>
static void topk_moe_kernel_nobias(const float * __restrict__ logits,
                                   float * __restrict__       weights,
                                   int32_t * __restrict__     ids,
                                   const int                  n_rows,
                                   const int                  n_expert_used,
                                   const float                clamp_val,
                                   const float                scale_val,
                                   const topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int row       = item_ct1.get_group(0);
    if (row >= n_rows) {
        return;
    }
    const int lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } else {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
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
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        const float inv = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Hybrid Laguna bias path.
// Stock-oracle (all compute_forward): golden OK, no kernel win.
// This fire: stock selection (sigmoid/add/argsort) + fused gather into GET_ROWS buffer
// + stock norm/scale chain. Isolates gather bitexact; if OK, fuse norm next.

// Exported from ggml-sycl.cpp
bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);
void ggml_sycl_argsort_f32_i32(ggml_backend_sycl_context & ctx, const float * x, int * dst, int ncols, int nrows,
                               ggml_sort_order order);

static void ggml_sycl_op_topk_moe_hybrid_bias_stock(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i,
                                                    int nops) {
    for (int j = 0; j < nops; ++j) {
        ggml_tensor * t = cgraph->nodes[i + j];
        if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t)) {
            continue;
        }
        const bool ok = ggml_sycl_compute_forward(ctx, t);
        GGML_ASSERT(ok && "hybrid stock-oracle: compute_forward failed");
    }
}

// Fused sigmoid + score-correction bias ADD.
// Writes both sigmoid buffer (unbiased probs for gather) and add buffer (argsort input).
// f32: s = 1/(1+exp(-x)) then add = s + bias[i % n_experts] — matches stock op_sigmoid + binbcast ADD.
// Kill: GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD=1
static void router_sigmoid_add_kernel(const float * __restrict__ logits, const float * __restrict__ bias,
                                      float * __restrict__ sig_out, float * __restrict__ add_out,
                                      const int n_experts, const int n_elts) {
    auto      item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int tid  = (int) item.get_global_id(0);
    if (tid >= n_elts) {
        return;
    }
    const float x = logits[tid];
    const float s = 1.0f / (1.0f + sycl::exp(-x));
    sig_out[tid]  = s;
    add_out[tid]  = s + bias[tid % n_experts];
}

static bool ggml_sycl_router_sigmoid_add_enabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return !disabled;
}

// F32 router GEMV + sigmoid + score-correction bias (Laguna ffn_gate_inp).
// Replaces stock MUL_MAT(F32) + fused sigmoid+add with one launch.
// Default ON. Kill: GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE=1
static bool ggml_sycl_router_gemv_fuse_enabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return !disabled;
}

// Router GEMV+sigmoid+bias — decode path (n_rows==1) as shipped.
// Prefill multi-row was CLOSED under tip (20260730): expert×row thrash ~3230 pp |
// expert-outer ~3330 | SLM W ~2814 | MKL tip ~3730 (pp tok/s on that config).
// [lx-router-multirow] 2026-08-09: reopen n_rows>1 env-gated
// (GGML_SYCL_ROUTER_MULTIROW=1, default off) so the scored pp512 path can be
// A/B'd against the oneMKL F32 GEMM it currently uses. Rows chunked
// ROWS_PER_WG=64 to bound serial work; W row stays in registers; x re-read per
// expert WG from L2 ("expert-outer" structure). Dot order matches the decode
// kernel (scalar, k = lid; k < ncols; k += WG, reduce_over_group).
// Dot = WG reduce_over_group (not bitexact vs MKL). Scalar loop (float4 changed
// golden tokens with flat score — keep scalar for tip oracle stability).
static void router_gemv_sigmoid_add_kernel(const float * __restrict__ W,
                                           const float * __restrict__ x,
                                           const float * __restrict__ bias,
                                           float * __restrict__ sig_out,
                                           float * __restrict__ add_out,
                                           const int n_experts, const int ncols) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    constexpr int WG = 256;
    const int lid    = (int) item.get_local_id(0);
    const int expert = (int) item.get_group(0);
    if (expert >= n_experts) {
        return;
    }

    const float * w_row = W + (size_t) expert * (size_t) ncols;
    float         sum   = 0.f;
#pragma unroll 8
    for (int k = lid; k < ncols; k += WG) {
        sum += w_row[k] * x[k];
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    if (lid == 0) {
        // Stock op_sigmoid f32: 1 / (1 + exp(-x)).
        const float s   = 1.f / (1.f + sycl::exp(-sum));
        sig_out[expert] = s;
        add_out[expert] = s + bias[expert];
    }
}

// [lx-router-multirow] 2026-08-09: env-gated multi-row router GEMV
// (GGML_SYCL_ROUTER_MULTIROW=1, default off). One WG per (expert,row) —
// "expert-outer" structure, W row in registers, x re-read from L2 per WG.
// Same dot order as the decode kernel (scalar, k=lid; k<ncols; k+=WG,
// reduce_over_group). sig/add layout: [n_experts][n_rows] contiguous, so
// index = row*n_experts + expert. A/B target: the oneMKL F32 GEMM the scored
// pp512 path uses for the 256×2048 @ 2048×512 router multiply.
static void router_gemv_sigmoid_add_kernel_multi(const float * __restrict__ W,
                                                 const float * __restrict__ x,
                                                 const float * __restrict__ bias,
                                                 float * __restrict__ sig_out,
                                                 float * __restrict__ add_out,
                                                 const int n_experts, const int ncols,
                                                 const int n_rows) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    constexpr int WG = 256;
    const int lid    = (int) item.get_local_id(0);
    const int gid    = (int) item.get_group(0);
    const int expert = gid / n_rows;
    const int row    = gid % n_rows;
    if (expert >= n_experts) {
        return;
    }

    const float * w_row = W + (size_t) expert * (size_t) ncols;
    const float * x_row = x + (size_t) row * (size_t) ncols;
    float         sum   = 0.f;
#pragma unroll 8
    for (int k = lid; k < ncols; k += WG) {
        sum += w_row[k] * x_row[k];
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    if (lid == 0) {
        const size_t idx  = (size_t) row * (size_t) n_experts + (size_t) expert;
        const float  s    = 1.f / (1.f + sycl::exp(-sum));
        sig_out[idx]      = s;
        add_out[idx]      = s + bias[expert];
    }
}

// [lx-router-multirow mode 2] 2026-08-09: per-row WG, expert-outer loop.
// One WG (256 lanes) per row; lane l keeps x[l], x[l+WG], …, x[l+7*WG] in
// registers (ncols/WG = 8). For each expert (8 per reduction batch), each lane
// forms a partial dot over its 8 x values against the same 8 W entries, then a
// group reduce per expert. reduce_over_group count = n_experts/8 per WG (32 for
// Laguna), vs 131072 group reduces for mode 1. sig/add layout as mode 1:
// [n_experts][n_rows], idx = row*n_experts + expert. Same scalar dot order per
// (expert,row) as the decode kernel and mode 1.
static void router_gemv_sigmoid_add_kernel_row(const float * __restrict__ W,
                                               const float * __restrict__ x,
                                               const float * __restrict__ bias,
                                               float * __restrict__ sig_out,
                                               float * __restrict__ add_out,
                                               const int n_experts, const int ncols,
                                               const int n_rows) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    constexpr int WG     = 256;
    constexpr int UNROLL = 8; // ncols / WG == 8 for Laguna
    const int lid    = (int) item.get_local_id(0);
    const int row    = (int) item.get_group(0);
    if (row >= n_rows) {
        return;
    }
    const float * x_row = x + (size_t) row * (size_t) ncols;
    float xv[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xv[i] = x_row[lid + i * WG];
    }
    for (int eb = 0; eb < n_experts; eb += UNROLL) {
        float sum[UNROLL] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float * w_row = W + (size_t) (eb + i) * (size_t) ncols;
            for (int k = lid; k < ncols; k += WG) {
                sum[i] += w_row[k] * xv[(k - lid) / WG];
            }
        }
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            float s = sycl::reduce_over_group(item.get_group(), sum[i], sycl::plus<float>());
            if (lid == 0) {
                const size_t idx = (size_t) row * (size_t) n_experts + (size_t) (eb + i);
                const float  v   = 1.f / (1.f + sycl::exp(-s));
                sig_out[idx]     = v;
                add_out[idx]     = v + bias[eb + i];
            }
        }
    }
}

static bool ggml_sycl_launch_router_gemv_sigmoid_add(
    queue_ptr stream, const float * W, const float * x, const float * bias, float * sig_out,
    float * add_out, int n_experts, int ncols, int n_rows) {
    static const int multirow = []() {
        const char * e = getenv("GGML_SYCL_ROUTER_MULTIROW");
        return e != nullptr ? std::atoi(e) : 0; // 0=off, 1=per-expert WG, 2=per-row WG
    }();
    if (n_experts <= 0 || ncols <= 0 || n_rows <= 0) {
        return false;
    }
    if (multirow == 0 && n_rows != 1) {
        return false; // decode only unless env-gated reopen
    }
    constexpr int WG = 256;
    if ((ncols % WG) != 0) {
        return false; // Laguna gate_inp: 2048
    }
    if (multirow == 2 && n_rows > 1) {
        // Per-row WG, expert-outer loop, 8 experts per reduction. 512 WGs vs
        // 131072 for mode 1; W (2MB) streams from L2 per row; x read once per row.
        // One WG per row: lane l holds x[l], x[l+WG], … (ncols/WG = 8 floats).
        if (ncols != 8 * WG) {
            return false; // kernel_row assumes ncols/WG == 8; fall back to stock
        }
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_rows * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> /*item*/) {
                router_gemv_sigmoid_add_kernel_row(W, x, bias, sig_out, add_out,
                                                   n_experts, ncols, n_rows);
            });
        return true;
    }
    if (multirow != 0 && n_rows > 1) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_experts * (size_t) n_rows * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> /*item*/) {
                router_gemv_sigmoid_add_kernel_multi(W, x, bias, sig_out, add_out,
                                                     n_experts, ncols, n_rows);
            });
        return true;
    }
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_experts * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> /*item*/) {
            router_gemv_sigmoid_add_kernel(W, x, bias, sig_out, add_out, n_experts, ncols);
        });
    return true;
}

// True top-k selection into full ARGSORT buffer layout (only first k slots written).
// Replaces stock k_argsort full DESC of n_experts when only VIEW top-k is consumed.
// Bitexact aim vs stable DESC bitonic: on ties prefer lower expert index (bitonic
// never swaps equals; init order 0..n-1 ⇒ lower index first among ties).
// Default ON. Kill: GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1
static bool ggml_sycl_router_true_topk_enabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return !disabled;
}

// One warp (sub-group) per token. n_experts must be divisible by WARP_SIZE.
// Optional: gather (get_rows), sum (k_sum_rows order), clamp+div+scale (final weights).
// do_norm requires sum_out + weights_out; writes clamp_out[row] when non-null.
template <int n_experts>
static void router_true_topk_ids_kernel(const float * __restrict__ scores, int32_t * __restrict__ ids,
                                        const float * __restrict__ probs, float * __restrict__ gather_out,
                                        float * __restrict__ sum_out, float * __restrict__ weights_out,
                                        float * __restrict__ clamp_out, const float clamp_min,
                                        const float clamp_max, const float scale_val, const float scale_bias,
                                        const bool do_norm, const int n_rows, const int n_expert_used) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int row      = item_ct1.get_group(0);
    if (row >= n_rows) {
        return;
    }
    const int lane = item_ct1.get_local_id(0);
    static_assert(n_experts % WARP_SIZE == 0, "router true top-k requires n_experts % WARP_SIZE == 0");
    constexpr int experts_per_thread = n_experts / WARP_SIZE;

    scores += (size_t) n_experts * row;
    ids    += (size_t) n_experts * row;
    if (probs) {
        probs += (size_t) n_experts * row;
    }
    float * gr_row = gather_out ? (gather_out + (size_t) n_expert_used * row) : nullptr;
    float * w_row  = weights_out ? (weights_out + (size_t) n_expert_used * row) : nullptr;

    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        wt[i]            = scores[expert];
    }

    // Private copy of selected unbiased probs for optional warp-sum (k_sum_rows order).
    float sel[WARP_SIZE];
#pragma unroll
    for (int i = 0; i < WARP_SIZE; i++) {
        sel[i] = 0.f;
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            // Strict-gt or equal with lower index — match stable DESC argsort top-k.
            if (wt[i] > max_val || (wt[i] == max_val && expert < max_expert)) {
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

        // Mask selected expert in its owning lane's register (no cross-lane write).
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        // All lanes agree on max_expert; load unbiased prob for gather + sum.
        const float p = (probs != nullptr) ? probs[max_expert] : 0.f;
        if (k < WARP_SIZE) {
            sel[k] = p;
        }
        if (lane == 0) {
            ids[k] = max_expert;
            if (gr_row) {
                gr_row[k] = p;
            }
        }
    }

    // Optional sum_rows: match k_sum_rows_f32 (strided + warp_reduce_sum).
    if ((sum_out != nullptr || do_norm) && probs != nullptr) {
        float sum = 0.f;
        for (int i = lane; i < n_expert_used; i += WARP_SIZE) {
            sum += sel[i];
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
        }
        if (lane == 0 && sum_out) {
            sum_out[row] = sum; // pre-clamp (stock SUM_ROWS)
        }
        // Norm: match mode8 elementwise path — reload gr/sum from global after fence
        // (register-only sel[] path golden-failed 2026-07-30).
        if (do_norm && w_row && gr_row && sum_out) {
            item_ct1.barrier(sycl::access::fence_space::global_and_local);
            float den = sum_out[row];
            den       = den < clamp_min ? clamp_min : (den > clamp_max ? clamp_max : den);
            if (lane == 0 && clamp_out) {
                clamp_out[row] = den;
            }
            for (int i = lane; i < n_expert_used; i += WARP_SIZE) {
                const float q = gr_row[i] / den;
                w_row[i]      = q * scale_val + scale_bias;
            }
        }
    }
}

template <int n_experts>
static void launch_router_true_topk_ids(queue_ptr stream, const float * scores, int32_t * ids,
                                        const float * probs, float * gather_out, float * sum_out,
                                        float * weights_out, float * clamp_out, float clamp_min, float clamp_max,
                                        float scale_val, float scale_bias, bool do_norm, int n_rows,
                                        int n_expert_used) {
    const sycl::range<1> block_dims(WARP_SIZE);
    const sycl::range<1> block_nums((size_t) n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             router_true_topk_ids_kernel<n_experts>(
                                 scores, ids, probs, gather_out, sum_out, weights_out, clamp_out, clamp_min,
                                 clamp_max, scale_val, scale_bias, do_norm, n_rows, n_expert_used);
                             GGML_UNUSED(item_ct1);
                         });
}

// Dispatch true top-k. Optional gather / sum / clamp+div+scale into same launch.
// Returns false if shape unsupported → caller falls back to stock full argsort.
static bool ggml_sycl_launch_router_true_topk(
    queue_ptr stream, const float * scores, int32_t * ids, int n_experts, int n_rows, int n_expert_used,
    const float * probs = nullptr, float * gather_out = nullptr, float * sum_out = nullptr,
    float * weights_out = nullptr, float * clamp_out = nullptr, float clamp_min = -INFINITY,
    float clamp_max = INFINITY, float scale_val = 1.f, float scale_bias = 0.f, bool do_norm = false) {
    if (!scores || !ids || n_rows <= 0 || n_expert_used <= 0 || n_expert_used > n_experts ||
        n_expert_used > WARP_SIZE || (n_experts % WARP_SIZE) != 0) {
        return false;
    }
    // Gather fuse requires both or neither.
    if ((probs == nullptr) != (gather_out == nullptr)) {
        return false;
    }
    // Sum fuse requires gather.
    if (sum_out != nullptr && gather_out == nullptr) {
        return false;
    }
    // Norm fuse requires gather + weights.
    if (do_norm && (gather_out == nullptr || weights_out == nullptr)) {
        return false;
    }
    switch (n_experts) {
        case 32:
            launch_router_true_topk_ids<32>(stream, scores, ids, probs, gather_out, sum_out, weights_out, clamp_out,
                                            clamp_min, clamp_max, scale_val, scale_bias, do_norm, n_rows,
                                            n_expert_used);
            return true;
        case 64:
            launch_router_true_topk_ids<64>(stream, scores, ids, probs, gather_out, sum_out, weights_out, clamp_out,
                                            clamp_min, clamp_max, scale_val, scale_bias, do_norm, n_rows,
                                            n_expert_used);
            return true;
        case 128:
            launch_router_true_topk_ids<128>(stream, scores, ids, probs, gather_out, sum_out, weights_out, clamp_out,
                                             clamp_min, clamp_max, scale_val, scale_bias, do_norm, n_rows,
                                             n_expert_used);
            return true;
        case 256:
            launch_router_true_topk_ids<256>(stream, scores, ids, probs, gather_out, sum_out, weights_out, clamp_out,
                                             clamp_min, clamp_max, scale_val, scale_bias, do_norm, n_rows,
                                             n_expert_used);
            return true;
        case 512:
            launch_router_true_topk_ids<512>(stream, scores, ids, probs, gather_out, sum_out, weights_out, clamp_out,
                                             clamp_min, clamp_max, scale_val, scale_bias, do_norm, n_rows,
                                             n_expert_used);
            return true;
        default:
            return false;
    }
}

// Gather only: weights_out[e + t*k] = probs[ids[e + t*n_experts] + t*n_experts]
// Matches stock get_rows on probs reshaped [1,n_experts,n_tokens] with ids VIEW [k,n_tokens]
// (parent stride n_experts). Output layout matches get_rows dst [1,k,n_tokens] contiguous.
static void router_gather_kernel(const float * __restrict__ probs, const int32_t * __restrict__ ids_full,
                                 float * __restrict__ weights_out, const int n_experts, const int n_rows,
                                 const int n_expert_used) {
    auto      item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int tid  = (int) item.get_global_id(0);
    const int total = n_expert_used * n_rows;
    if (tid >= total) {
        return;
    }
    const int e   = tid % n_expert_used;
    const int row = tid / n_expert_used;
    const int expert = ids_full[(size_t) row * (size_t) n_experts + (size_t) e];
    weights_out[tid] = probs[(size_t) row * (size_t) n_experts + (size_t) expert];
}

// Norm+scale only: read get_rows buffer [k * n_rows], write final weights.
// One work-item per token (sequential sum over k) — matches left-to-right sum_rows
// more closely than a warp tree reduce for small k.
static void router_norm_scale_kernel(const float * __restrict__ getrows, float * __restrict__ weights,
                                     const int n_rows, const int n_expert_used, const float clamp_min,
                                     const float scale_val, const float scale_bias, const bool with_norm) {
    auto      item  = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int row   = (int) item.get_global_id(0);
    if (row >= n_rows) {
        return;
    }

    const float * gr_row = getrows + (size_t) n_expert_used * row;
    float *       w_row  = weights + (size_t) n_expert_used * row;

    if (with_norm) {
        float wt_sum = 0.f;
        for (int e = 0; e < n_expert_used; ++e) {
            wt_sum += gr_row[e];
        }
        wt_sum = sycl::fmax(wt_sum, clamp_min);
        const float inv = 1.0f / wt_sum;
        for (int e = 0; e < n_expert_used; ++e) {
            w_row[e] = gr_row[e] * inv * scale_val + scale_bias;
        }
    } else {
        for (int e = 0; e < n_expert_used; ++e) {
            w_row[e] = gr_row[e] * scale_val + scale_bias;
        }
    }
}

// Hybrid stages (GGML_SYCL_TOPK_MOE_HYBRID_MODE):
//   0 = full stock-oracle
//   1 = fused gather + stock norm/scale
//   2 = fused gather + stock sum/div + fused scale
//   3 = stock get_rows + stock norm (wiring control)
//   4 = stock all after sigmoid (same as 3)
//   5 = stock through div + fused scale
//   6 = fused gather + warp-sum norm (research; golden FAIL)
//   7 = fused gather + stock sum/clamp + fused DIV+SCALE
//   8 = fused gather + stock sum + fused CLAMP+DIV+SCALE  (prior tip)
//   9 = fused gather + fused sum (k_sum_rows order) + fused CLAMP+DIV+SCALE
static int ggml_sycl_topk_moe_hybrid_mode() {
    static const int mode = []() {
        const char * env = getenv("GGML_SYCL_TOPK_MOE_HYBRID_MODE");
        if (env == nullptr) {
            return 8;  // promote to 9 only after formal beat
        }
        return std::atoi(env);
    }();
    return mode;
}

// Bitexact vs k_sum_rows_f32: strided load + warp_reduce_sum, lane0 stores.
static void router_sum_rows_kernel(const float * __restrict__ x, float * __restrict__ dst,
                                   const int ncols, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);
    float sum = 0.0f;
    for (int i = col; i < ncols; i += (int) item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }
    sum = warp_reduce_sum(sum, item_ct1);
    if (col == 0) {
        dst[row] = sum;
    }
}

// Match k_sum_rows_f32 reduction order (strided load + xor-butterfly warp_reduce_sum),
// then clamp + div + scale in-register. One warp per token. Writes final weights only.
static void router_norm_warp_kernel(const float * __restrict__ getrows, float * __restrict__ weights,
                                    const int n_rows, const int n_expert_used, const float clamp_min,
                                    const float scale_val, const float scale_bias,
                                    const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);
    if (row >= n_rows) {
        return;
    }

    const float * gr_row = getrows + (size_t) row * (size_t) n_expert_used;
    float *       w_row  = weights + (size_t) row * (size_t) n_expert_used;

    // Same accumulation as k_sum_rows_f32 (ncols = n_expert_used, block WARP_SIZE).
    float sum = 0.0f;
    for (int i = col; i < n_expert_used; i += (int) item_ct1.get_local_range(2)) {
        sum += gr_row[i];
    }
    sum = warp_reduce_sum(sum, item_ct1);
    sum = sycl::fmax(sum, clamp_min);

    // Broadcast sum is already on every lane after butterfly; div+scale per expert.
    for (int i = col; i < n_expert_used; i += (int) item_ct1.get_local_range(2)) {
        w_row[i] = (gr_row[i] / sum) * scale_val + scale_bias;
    }
}

static void ggml_sycl_op_topk_moe_hybrid_bias(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i, int nops,
                                              ggml_tensor * sigmoid_node, ggml_tensor * add_node,
                                              ggml_tensor * argsort_node, ggml_tensor * getrows_node,
                                              ggml_tensor * weights_final, const ggml_tensor * clamp,
                                              const ggml_tensor * scale,
                                              ggml_tensor * mul_mat_node = nullptr) {
    const int mode = ggml_sycl_topk_moe_hybrid_mode();
    // Relative index of first post-GET_ROWS op (norm/scale tail).
    // SIGMOID-start: GET_ROWS at +5 → tail at 6. MUL_MAT-start: GET_ROWS at +6 → tail at 7.
    const int j_tail = mul_mat_node ? 7 : 6;
    if (mode <= 0) {
        // Stock path runs every real op in [i, i+nops); mul_mat is included when present.
        ggml_sycl_op_topk_moe_hybrid_bias_stock(ctx, cgraph, i, nops);
        return;
    }

    // Stage A: selection. Prefer fused GEMV+sigmoid+add when fuse starts at MUL_MAT;
    // else fused sigmoid+add on precomputed logits; else stock.
    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    const ggml_tensor * logits = sigmoid_node->src[0];
    const ggml_tensor * bias   = add_node->src[1];
    const int n_experts        = (int) sigmoid_node->ne[0];
    const int n_rows           = (int) ggml_nrows(sigmoid_node);
    const int64_t n_elts       = ggml_nelements(sigmoid_node);

    bool did_gemv_fuse = false;
    // Decode-only GEMV (launch rejects n_rows!=1). Prefill: stock MKL + sigmoid+add.
    if (mul_mat_node && ggml_sycl_router_gemv_fuse_enabled() &&
        mul_mat_node->op == GGML_OP_MUL_MAT && logits == mul_mat_node &&
        mul_mat_node->src[0] && mul_mat_node->src[1] &&
        mul_mat_node->src[0]->type == GGML_TYPE_F32 && mul_mat_node->src[1]->type == GGML_TYPE_F32 &&
        mul_mat_node->type == GGML_TYPE_F32 && sigmoid_node->type == GGML_TYPE_F32 &&
        add_node->type == GGML_TYPE_F32 && bias && bias->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(mul_mat_node->src[0]) && ggml_is_contiguous(mul_mat_node->src[1]) &&
        ggml_is_contiguous(sigmoid_node) && ggml_is_contiguous(add_node) &&
        ggml_is_contiguous(bias) &&
        (int) mul_mat_node->src[0]->ne[1] == n_experts &&
        (int) mul_mat_node->src[0]->ne[0] == (int) mul_mat_node->src[1]->ne[0] &&
        (int) ggml_nrows(mul_mat_node) == n_rows &&
        bias->ne[0] == n_experts && ggml_nelements(bias) == n_experts &&
        add_node->src[0] == sigmoid_node &&
        ggml_nelements(sigmoid_node) == n_elts && ggml_nelements(add_node) == n_elts) {
        const int ncols = (int) mul_mat_node->src[0]->ne[0];
        const float * W_d    = (const float *) mul_mat_node->src[0]->data;
        const float * x_d    = (const float *) mul_mat_node->src[1]->data;
        const float * bias_d = (const float *) bias->data;
        float *       sig_d  = (float *) sigmoid_node->data;
        float *       add_d  = (float *) add_node->data;
        // gemv epilogue writes sig+add; mul_mat dst unused when fused.
        did_gemv_fuse = ggml_sycl_launch_router_gemv_sigmoid_add(
            stream, W_d, x_d, bias_d, sig_d, add_d, n_experts, ncols, n_rows);
        if (did_gemv_fuse) {
            static std::atomic<int> once{0};
            if (once.fetch_add(1) == 0) {
                fprintf(stderr,
                        "[lx-control-topk-moe] fused gemv+sigmoid+add n_experts=%d ncols=%d n_rows=%d\n",
                        n_experts, ncols, n_rows);
            }
        }
    }

    if (!did_gemv_fuse) {
        // Own mul_mat but gemv fuse declined → stock GEMV first.
        if (mul_mat_node) {
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, mul_mat_node));
        }

        const bool fuse_sig_add =
            ggml_sycl_router_sigmoid_add_enabled() &&
            logits && bias &&
            logits->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
            sigmoid_node->type == GGML_TYPE_F32 && add_node->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(logits) && ggml_is_contiguous(sigmoid_node) &&
            ggml_is_contiguous(add_node) && ggml_is_contiguous(bias) &&
            ggml_nelements(logits) == n_elts && ggml_nelements(add_node) == n_elts &&
            bias->ne[0] == n_experts && ggml_nelements(bias) == n_experts &&
            add_node->src[0] == sigmoid_node;

        if (fuse_sig_add) {
            const float * logits_d = (const float *) logits->data;
            const float * bias_d   = (const float *) bias->data;
            float *       sig_d    = (float *) sigmoid_node->data;
            float *       add_d    = (float *) add_node->data;
            constexpr int BS = 256;
            const int nblocks = (int) ((n_elts + BS - 1) / BS);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
                [=](sycl::nd_item<1> item) {
                    router_sigmoid_add_kernel(logits_d, bias_d, sig_d, add_d, n_experts, (int) n_elts);
                    GGML_UNUSED(item);
                });
            {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    fprintf(stderr,
                            "[lx-control-topk-moe] fused sigmoid+add n_experts=%d n_rows=%d\n",
                            n_experts, n_rows);
                }
            }
        } else {
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, sigmoid_node));
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, add_node));
        }
    }

    // n_expert_used from get_rows dst: [1, k, n_tokens] → ne[1]=k
    // Needed before selection so true top-k can write only first k of ARGSORT buffer.
    const int n_expert_used = (int) getrows_node->ne[1];
    GGML_ASSERT(n_expert_used > 0 && n_expert_used <= n_experts);
    GGML_ASSERT(n_expert_used <= WARP_SIZE && "gather-norm assumes k <= WARP_SIZE");

    // Stage A2: selection. Prefer true top-k (k iters) over full bitonic of n_experts.
    // Optionally fuses gather + sum + clamp+div+scale (mode8 full router tail).
    bool did_true_topk      = false;
    bool did_topk_gather    = false;  // wrote get_rows → skip separate gather
    bool did_topk_sum       = false;  // wrote sum_rows → skip stock SUM
    bool did_topk_norm      = false;  // wrote final weights → skip clamp+div+scale kernel
    const float * probs_d   = (const float *) sigmoid_node->data;
    const int32_t * ids_d   = (const int32_t *) argsort_node->data;
    float *       gr_d      = (float *) getrows_node->data;

    float clamp_min  = -INFINITY;
    float clamp_max  = INFINITY;
    float scale_val  = 1.0f;
    float scale_bias = 0.0f;
    if (clamp) {
        clamp_min = ggml_get_op_params_f32(clamp, 0);
        clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    if (scale) {
        scale_val  = ggml_get_op_params_f32(scale, 0);
        scale_bias = ggml_get_op_params_f32(scale, 1);
    }
    const bool with_norm = clamp != nullptr;
    float *    weights_d = (float *) weights_final->data;

    // Locate SUM_ROWS / CLAMP early when mode8/9 so true top-k can write them in-kernel.
    ggml_tensor * sum_node_early   = nullptr;
    ggml_tensor * clamp_node_early = nullptr;
    if (mode == 8 || mode == 9) {
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t) {
                continue;
            }
            if (t->op == GGML_OP_SUM_ROWS && !sum_node_early) {
                sum_node_early = t;
            }
            if (t->op == GGML_OP_CLAMP && !clamp_node_early) {
                clamp_node_early = t;
            }
        }
    }

    {
        if (ggml_sycl_router_true_topk_enabled() && argsort_node->type == GGML_TYPE_I32 &&
            argsort_node->src[0] == add_node && ggml_is_contiguous(add_node) &&
            ggml_is_contiguous(argsort_node) && add_node->type == GGML_TYPE_F32 &&
            (int) argsort_node->ne[0] == n_experts && (int) ggml_nrows(argsort_node) == n_rows) {
            const float * scores_d = (const float *) add_node->data;
            int32_t *     ids_buf  = (int32_t *) argsort_node->data;
            // Fuse gather when mode will use fused gather path (default mode8).
            // get_rows layout is [1, k, n_tokens] → ne[1]=k, ne[2]=n_tokens.
            const bool want_gather =
                mode != 3 && mode != 4 && mode != 5 && mode > 0 &&
                probs_d && gr_d && getrows_node->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(getrows_node) && ggml_is_contiguous(sigmoid_node) &&
                (int) getrows_node->ne[1] == n_expert_used &&
                (int) getrows_node->ne[2] == n_rows &&
                ggml_nelements(getrows_node) == (int64_t) n_expert_used * (int64_t) n_rows;
            const bool want_sum =
                want_gather && sum_node_early && sum_node_early->data &&
                sum_node_early->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(sum_node_early) &&
                (int) ggml_nelements(sum_node_early) == n_rows;
            // Full norm (clamp+div+scale in top-k): DEFAULT ON under GEMV tip.
            // Saves separate mode8 clamp+div+scale launch (k=8) after topk+sum.
            // Kill: GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM=1
            // (Legacy enable env still accepted.)
            static const bool enable_topk_norm = []() {
                const char * dis = getenv("GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM");
                if (dis != nullptr && std::atoi(dis) != 0) {
                    return false;
                }
                const char * e = getenv("GGML_SYCL_ENABLE_ROUTER_TRUE_TOPK_NORM");
                if (e != nullptr) {
                    return std::atoi(e) != 0;
                }
                return true; // default ON
            }();
            const bool want_norm =
                enable_topk_norm && want_sum && (mode == 8 || mode == 9) && with_norm && scale &&
                weights_d && weights_final->type == GGML_TYPE_F32 && ggml_is_contiguous(weights_final) &&
                ggml_nelements(weights_final) == (int64_t) n_expert_used * (int64_t) n_rows;
            float * sum_d    = want_sum ? (float *) sum_node_early->data : nullptr;
            float * clamp_d  = nullptr;
            if (want_norm && clamp_node_early && clamp_node_early->data &&
                clamp_node_early->type == GGML_TYPE_F32 &&
                (int) ggml_nelements(clamp_node_early) == n_rows) {
                clamp_d = (float *) clamp_node_early->data;
            }
            did_true_topk = ggml_sycl_launch_router_true_topk(
                stream, scores_d, ids_buf, n_experts, n_rows, n_expert_used,
                want_gather ? probs_d : nullptr, want_gather ? gr_d : nullptr, sum_d,
                want_norm ? weights_d : nullptr, clamp_d, clamp_min, clamp_max, scale_val, scale_bias,
                want_norm);
            if (did_true_topk) {
                did_topk_gather = want_gather;
                did_topk_sum    = want_sum;
                did_topk_norm   = want_norm;
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    fprintf(stderr,
                            "[lx-control-topk-moe] true top-k%s%s%s n_experts=%d k=%d n_rows=%d\n",
                            did_topk_gather ? "+gather" : " (not full argsort)",
                            did_topk_sum ? "+sum" : "", did_topk_norm ? "+norm" : "", n_experts,
                            n_expert_used, n_rows);
                }
            }
        }
        if (!did_true_topk) {
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, argsort_node));
        }
    }

    auto launch_norm = [&]() {
        constexpr int BS = 256;
        const int     nb = (n_rows + BS - 1) / BS;
        const int     nr = n_rows;
        const int     nk = n_expert_used;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * BS), sycl::range<1>(BS)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 router_norm_scale_kernel(gr_d, weights_d, nr, nk, clamp_min, scale_val, scale_bias,
                                                          with_norm);
                                 GGML_UNUSED(item_ct1);
                             });
    };

    if (mode == 3) {
        // Stock get_rows + stock norm (control for mode3 path wiring).
        // Set GGML_SYCL_TOPK_MOE_HYBRID_MODE=4 for stock get_rows + fused norm.
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, getrows_node));
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t)) {
                continue;
            }
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
        }
        return;
    }

    if (mode == 4) {
        // Stock everything except fuse DIV+SCALE by calling stock DIV then stock SCALE
        // via compute_forward — should match mode3. If we need a real fuse later, compare
        // elementwise to this. (Previous pure-math fuse golden-failed.)
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, getrows_node));
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t)) {
                continue;
            }
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
        }
        return;
    }

    if (mode == 5) {
        // Stock through DIV (incl. sum/clamp/div); fuse SCALE only as x*s+b.
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, getrows_node));
        ggml_tensor * scale_node = nullptr;
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t) {
                continue;
            }
            if (t->op == GGML_OP_SCALE) {
                scale_node = t;
                continue;
            }
            if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE) {
                continue;
            }
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
        }
        if (!scale_node) {
            return;
        }
        const float * src = (const float *) scale_node->src[0]->data;
        float *       dst = (float *) scale_node->data;
        const int64_t nelt = ggml_nelements(scale_node);
        const float   s    = scale_val;
        const float   b    = scale_bias;
        constexpr int BS   = 256;
        const int nblocks  = (int) ((nelt + BS - 1) / BS);
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = (int64_t) item.get_global_id(0);
                if (idx < nelt) {
                    dst[idx] = src[idx] * s + b;
                }
            });
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr, "[lx-control-topk-moe] mode=5 stock-div+fused-scale nelt=%" PRId64 " s=%g same=%d\n",
                    nelt, (double) s, (int) (src == dst));
        }
        return;
    }

    // mode 1/2/6/7/8/9: fused gather into get_rows (skip if true top-k already wrote it)
    if (!did_topk_gather) {
        const int total = n_expert_used * n_rows;
        constexpr int BS = 256;
        const int     nb = (total + BS - 1) / BS;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * BS), sycl::range<1>(BS)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 router_gather_kernel(probs_d, ids_d, gr_d, n_experts, n_rows, n_expert_used);
                                 GGML_UNUSED(item_ct1);
                             });
    }

    if (mode == 1) {
        // Stock norm/scale for remaining real ops after GET_ROWS.
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t)) {
                continue;
            }
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
        }
        return;
    }

    if (mode == 6) {
        // Warp-sum norm matching stock sum_rows reduction + clamp + div + scale → final weights.
        // Research only — still golden-fails (2026-07-30); keep mode 2 as tip.
        const int nk = n_expert_used;
        const int nr = n_rows;
        const float cmin_use = with_norm ? clamp_min : 0.0f;
        const float s  = scale_val;
        const float b  = scale_bias;
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        const sycl::range<3> block_nums(1, (size_t) nr, 1);
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                router_norm_warp_kernel(gr_d, weights_d, nr, nk, cmin_use, s, b, item_ct1);
            });
        {
            static std::atomic<int> once{0};
            if (once.fetch_add(1) == 0) {
                fprintf(stderr,
                        "[lx-control-topk-moe] hybrid mode=6 gather+warp-sum-norm n_experts_used=%d n_rows=%d\n",
                        nk, nr);
            }
        }
        return;
    }

    if (mode == 7 || mode == 8 || mode == 9) {
        // mode7: stock through CLAMP; fuse DIV+SCALE with stock clamp as divisor.
        // mode8: stock through SUM_ROWS (or true top-k+sum); fuse CLAMP+DIV+SCALE.
        // mode9: fused SUM + fused CLAMP+DIV+SCALE (no stock sum launch).
        ggml_tensor * div_node   = nullptr;
        ggml_tensor * scale_node = nullptr;
        ggml_tensor * clamp_node = nullptr;
        ggml_tensor * sum_node   = sum_node_early;
        for (int j = j_tail; j < nops; ++j) {
            ggml_tensor * t = cgraph->nodes[i + j];
            if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW) {
                continue;
            }
            if (t->op == GGML_OP_SUM_ROWS) {
                sum_node = t;
                // Skip stock sum if true top-k wrote it, or mode9 will fuse, or mode8+topk sum.
                if (mode == 9 || did_topk_sum) {
                    continue;
                }
            }
            if (t->op == GGML_OP_CLAMP) {
                clamp_node = t;
                if (mode == 8 || mode == 9) {
                    continue; // fuse clamp into div kernel
                }
            }
            if (t->op == GGML_OP_DIV) {
                div_node = t;
                continue;
            }
            if (t->op == GGML_OP_SCALE) {
                scale_node = t;
                continue;
            }
            // Skip reshape after DIV (linear-identical when contiguous).
            if (t->op == GGML_OP_RESHAPE && div_node && t->src[0] == div_node) {
                continue;
            }
            // No-op reshape/view sharing storage with src — no device work.
            if (t->op == GGML_OP_RESHAPE && t->src[0] && t->data == t->src[0]->data) {
                continue;
            }
            GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
        }
        if (!div_node || !scale_node || !div_node->src[0] || !div_node->src[1]) {
            for (int j = j_tail; j < nops; ++j) {
                ggml_tensor * t = cgraph->nodes[i + j];
                if (!t || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
                    continue;
                }
                GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
            }
            return;
        }

        // mode9 without true top-k sum: write sum_rows with stock reduction from get_rows.
        if (mode == 9 && !did_topk_sum && sum_node && sum_node->data && n_expert_used > 0 && n_rows > 0) {
            float * sum_d = (float *) sum_node->data;
            const int nk_sum = n_expert_used;
            const int nr_sum = n_rows;
            const sycl::range<3> block_dims(1, 1, WARP_SIZE);
            const sycl::range<3> block_nums(1, (size_t) nr_sum, 1);
            stream->parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    router_sum_rows_kernel(gr_d, sum_d, nk_sum, item_ct1);
                });
        }

        // True top-k already wrote final weights (clamp+div+scale in-kernel).
        if (did_topk_norm) {
            static std::atomic<int> once{0};
            if (once.fetch_add(1) == 0) {
                fprintf(stderr,
                        "[lx-control-topk-moe] hybrid mode=%d topk full-norm (skip clamp+div+scale kernel) "
                        "k=%d n_rows=%d\n",
                        mode, n_expert_used, n_rows);
            }
            return;
        }

        const float * num = (const float *) div_node->src[0]->data;
        // mode7: denom is clamp buffer; mode8/9: denom is sum_rows, clamp in-kernel.
        const float * denom_raw =
            ((mode == 8 || mode == 9) && sum_node) ? (const float *) sum_node->data
                                    : (const float *) div_node->src[1]->data;
        float *       dstd  = (float *) div_node->data;
        float *       dsts  = (float *) scale_node->data;
        float *       dclamp = ((mode == 8 || mode == 9) && clamp_node) ? (float *) clamp_node->data : nullptr;
        const int64_t nelt  = ggml_nelements(div_node);
        const int64_t nk    = n_expert_used;
        const float   s     = scale_val;
        const float   b     = scale_bias;
        const float   cmin  = clamp_min;
        const float   cmax  = clamp_max;
        const bool    do_clamp = (mode == 8 || mode == 9);
        // Fuse claims DIV as intermediate (only SCALE + VIEW ids are outs). Skip writing
        // the DIV buffer when it is a distinct allocation — half the stores, same final weights.
        // If DIV and SCALE share storage (inplace), write only the scaled value.
        const bool write_div = (dstd != dsts);
        const bool skip_div_store = write_div; // always skip distinct DIV store (mode7/8/9)
        const int64_t nd = do_clamp ? n_rows : ggml_nelements(div_node->src[1]);
        GGML_ASSERT(nelt == nk * n_rows && nd == n_rows);
        constexpr int BS = 256;
        const int nblocks = (int) ((nelt + BS - 1) / BS);
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = (int64_t) item.get_global_id(0);
                if (idx >= nelt) {
                    return;
                }
                // Contiguous [ne0=k, ne1=tokens]: idx = expert + token*k
                const int64_t tok = idx / nk;
                float         den = denom_raw[tok];
                if (do_clamp) {
                    // Match stock op_clamp(x, min, max).
                    den = den < cmin ? cmin : (den > cmax ? cmax : den);
                    if (dclamp && (idx % nk) == 0) {
                        dclamp[tok] = den;
                    }
                }
                const float q = num[idx] / den;
                // Only SCALE (route weights) is a fuse output. DIV is elided intermediate.
                if (!skip_div_store) {
                    dstd[idx] = q;
                }
                dsts[idx] = q * s + b;
            });
        {
            static std::atomic<int> once{0};
            if (once.fetch_add(1) == 0) {
                const char * sum_kind =
                    did_topk_sum ? "topk-" : (mode == 9 ? "fused-" : "stock-");
                fprintf(stderr,
                        "[lx-control-topk-moe] hybrid mode=%d %ssum+fused-%sdiv-scale nelt=%" PRId64
                        " skip_div_store=%d\n",
                        mode, sum_kind, do_clamp ? "clamp+" : "", nelt, (int) skip_div_store);
            }
        }
        return;
    }

    // mode >= 2 (default 2): fused gather + stock sum/clamp/div + fused scale.
    // Full sequential fused norm golden-fails; keep sum/div stock unless mode 6/7.
    ggml_tensor * scale_node = nullptr;
    for (int j = j_tail; j < nops; ++j) {
        ggml_tensor * t = cgraph->nodes[i + j];
        if (!t) {
            continue;
        }
        if (t->op == GGML_OP_SCALE) {
            scale_node = t;
            continue;
        }
        if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE) {
            continue;
        }
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, t));
    }
    if (scale_node) {
        const float * src = (const float *) scale_node->src[0]->data;
        float *       dst = (float *) scale_node->data;
        const int64_t nelt = ggml_nelements(scale_node);
        const float   s    = scale_val;
        const float   b    = scale_bias;
        constexpr int BS   = 256;
        const int nblocks  = (int) ((nelt + BS - 1) / BS);
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = (int64_t) item.get_global_id(0);
                if (idx < nelt) {
                    dst[idx] = src[idx] * s + b;
                }
            });
    }
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr, "[lx-control-topk-moe] hybrid mode=2 gather+stock-div+fused-scale\n");
        }
    }
}

template <int n_experts>
static void launch_topk_moe_nobias(queue_ptr stream, const float * logits, float * weights, int32_t * ids, int n_rows,
                                   int n_expert_used, float clamp_val, float scale_val,
                                   const topk_moe_config & config) {
    const sycl::range<1> block_dims(WARP_SIZE);
    const sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel_nobias<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                               scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

static void launch_topk_moe_dispatch(queue_ptr stream, const float * logits, float * weights, int32_t * ids,
                                     const float * bias, int n_rows, int n_experts, int n_expert_used, float clamp_val,
                                     float scale_val, const topk_moe_config & config) {
    // Bias path is handled by hybrid (stock argsort) from fuse_topk_moe; do not re-enter bitonic here.
    GGML_ASSERT(bias == nullptr && "use hybrid bias path (ggml_sycl_op_topk_moe_hybrid_bias)");

    switch (n_experts) {
        case 1:
            launch_topk_moe_nobias<1>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                      config);
            break;
        case 2:
            launch_topk_moe_nobias<2>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                      config);
            break;
        case 4:
            launch_topk_moe_nobias<4>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                      config);
            break;
        case 8:
            launch_topk_moe_nobias<8>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                      config);
            break;
        case 16:
            launch_topk_moe_nobias<16>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                       config);
            break;
        case 32:
            launch_topk_moe_nobias<32>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                       config);
            break;
        case 64:
            launch_topk_moe_nobias<64>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                       config);
            break;
        case 128:
            launch_topk_moe_nobias<128>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                        config);
            break;
        case 256:
            launch_topk_moe_nobias<256>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                        config);
            break;
        case 512:
            launch_topk_moe_nobias<512>(stream, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val,
                                        config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

static void ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  const ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);
    // Bias Laguna path uses hybrid (stock argsort); this is the no-bias fused kernel only.
    GGML_ASSERT(!args.prob_bias);

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

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    launch_topk_moe_dispatch(stream, logits_d, weights_d, ids_d, /*bias=*/nullptr, n_rows, n_experts, n_expert_used,
                             clamp_val, scale_val, config);
}

static bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op, const ggml_tensor * weights,
                                          const ggml_tensor * logits, const ggml_tensor * ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;

        memcpy(&scale, (const float *) gating_op->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) gating_op->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(gating_op->src[0])) {
            return false;
        }
        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }
        // don't fuse when masks or sinks are present
        if (gating_op->src[1] || gating_op->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }

    return true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
static bool ggml_sycl_topk_moe_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

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
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT -> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
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

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
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

        // DIV <- CLAMP, RESHAPE
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

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
static bool ggml_sycl_check_fusion_memory_ranges(const ggml_cgraph * cgraph, const int node_idx,
                                                 const int node_count, const int * out_nodes, const int out_count,
                                                 const bool is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    if (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
        return true;
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

int ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    // Q/V/K attention projections: three GEMVs sharing one activation -> one launch.
    const int qkv_skip = ggml_sycl_fuse_qkv(ctx, cgraph, i);
    if (qkv_skip != 0) {
        return qkv_skip;
    }

    // attn_gate_proj GEMV + softplus + per-head MUL (XS.2 decode) -> one launch.
    const int gate_skip = ggml_sycl_fuse_gate_softplus_mul(ctx, cgraph, i);
    if (gate_skip != 0) {
        return gate_skip;
    }

    // Dual MoE SwiGLU (serial decode) before topk fusion.
    const int dual_skip = ggml_sycl_fuse_moe_dual_swiglu(ctx, cgraph, i);
    if (dual_skip != 0) {
        return dual_skip;
    }

    // Dense shared-expert dual SwiGLU (MUL_MAT+MUL_MAT+GLU).
    const int dense_skip = ggml_sycl_fuse_dense_dual_swiglu(ctx, cgraph, i);
    if (dense_skip != 0) {
        return dense_skip;
    }

    // MoE down weighted reduce.
    const int down_skip = ggml_sycl_fuse_moe_down_weighted(ctx, cgraph, i);
    if (down_skip != 0) {
        return down_skip;
    }

    // [lx-qk-rope] merged Q+K normed rope must win over the separate Q/K fuses.
    const int qk_rope_skip = ggml_sycl_fuse_norm_rope_qk(ctx, cgraph, i);
    if (qk_rope_skip != 0) {
        return qk_rope_skip;
    }

    // [lx-norm-rope] normed K rope: RMS+MUL+ROPE+SET_ROWS must win over the
    // 2-node rms_norm_mul fuse, so try it before rms_norm_mul.
    const int norm_rope_skip = ggml_sycl_fuse_norm_rope(ctx, cgraph, i);
    if (norm_rope_skip != 0) {
        return norm_rope_skip;
    }
    // [lx-norm-rope-q] Q-path twin (RMS+MUL+ROPE, no set_rows) right after the
    // K pattern so the 5-node fuse always wins.
    const int norm_rope_q_skip = ggml_sycl_fuse_norm_rope_q(ctx, cgraph, i);
    if (norm_rope_q_skip != 0) {
        return norm_rope_q_skip;
    }

    // RMS_NORM+MUL (Laguna pre-norm every layer).
    const int rms_skip = ggml_sycl_fuse_rms_norm_mul(ctx, cgraph, i);
    if (rms_skip != 0) {
        return rms_skip;
    }

    // ADD+ADD residual (moe+shexp then +ffn_inp).
    const int add_skip = ggml_sycl_fuse_add_add(ctx, cgraph, i);
    if (add_skip != 0) {
        return add_skip;
    }

    // Softplus × mul attention gate.
    const int sp_skip = ggml_sycl_fuse_softplus_mul(ctx, cgraph, i);
    if (sp_skip != 0) {
        return sp_skip;
    }

    // MUL_MAT+ADD residual (decode-only quality-safe; any-batch opt-in).
    const int mmadd_skip = ggml_sycl_fuse_mul_mat_add(ctx, cgraph, i);
    if (mmadd_skip != 0) {
        return mmadd_skip;
    }

    // Decode V MMVQ -> indexed F16 V-cache write.
    const int kv_skip = ggml_sycl_fuse_v_mmvq_set_rows(ctx, cgraph, i);
    if (kv_skip != 0) {
        return kv_skip;
    }

    // ROPE → VIEW → SET_ROWS into KV cache.
    const int rope_skip = ggml_sycl_fuse_rope_set_rows(ctx, cgraph, i);
    if (rope_skip != 0) {
        return rope_skip;
    }

    // Router GEMV+full-norm tip lives in this file (topk-moe).
    return ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

// Laguna bias hybrid fuse — DEFAULT ON (mode1: fused gather + stock norm).
// Measured 2026-07-30 formal: dual+hybrid-m1 ~tg110.3 / +2.25% vs pin; dual-only ~tg109.9 / +1.97%.
// Mode2 gather-norm still golden-FAIL (research via GGML_SYCL_TOPK_MOE_HYBRID_MODE=2).
// Kill: GGML_SYCL_DISABLE_TOPK_MOE=1  or  GGML_SYCL_ENABLE_TOPK_MOE_BIAS=0
static bool ggml_sycl_topk_moe_disabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_TOPK_MOE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return disabled;
}

static bool ggml_sycl_topk_moe_bias_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_TOPK_MOE_BIAS");
        // Default ON when unset (mode1 is golden-safe). Explicit 0 disables.
        if (env == nullptr) {
            return true;
        }
        return std::atoi(env) != 0;
    }();
    return enabled;
}

// Append optional weight-norm + scale after GET_ROWS. Returns false if graph ends early
// but still valid without them (caller keeps current ops).
static void ggml_sycl_topk_append_norm_scale(const ggml_cgraph * cgraph, int i, std::vector<ggml_op> & ops,
                                             const ggml_tensor *& clamp, const ggml_tensor *& scale) {
    int idx = i + (int) ops.size();
    if (idx >= cgraph->n_nodes) {
        return;
    }
    // RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
    if (cgraph->nodes[idx]->op == GGML_OP_RESHAPE && cgraph->nodes[idx]->src[0] == cgraph->nodes[idx - 1]) {
        static const ggml_op norm_ops[] = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV,
                                            GGML_OP_RESHAPE };
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            if (idx + k >= cgraph->n_nodes || cgraph->nodes[idx + k]->op != norm_ops[k] ||
                cgraph->nodes[idx + k]->src[0] != cgraph->nodes[idx + k - 1]) {
                ok = false;
                break;
            }
        }
        if (ok && idx + 3 < cgraph->n_nodes && cgraph->nodes[idx + 3]->op == GGML_OP_DIV &&
            cgraph->nodes[idx + 3]->src[1] == cgraph->nodes[idx + 2] &&
            cgraph->nodes[idx + 3]->src[0] == cgraph->nodes[idx] && idx + 4 < cgraph->n_nodes &&
            cgraph->nodes[idx + 4]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[idx + 4]->src[0] == cgraph->nodes[idx + 3]) {
            for (int k = 0; k < 5; ++k) {
                ops.push_back(norm_ops[k]);
            }
            clamp = cgraph->nodes[i + (int) ops.size() - 3];
            idx   = i + (int) ops.size();
        }
    }
    if (idx < cgraph->n_nodes && cgraph->nodes[idx]->op == GGML_OP_SCALE &&
        cgraph->nodes[idx]->src[0] == cgraph->nodes[idx - 1]) {
        ops.push_back(GGML_OP_SCALE);
        scale = cgraph->nodes[i + (int) ops.size() - 1];
    }
}

int ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (ggml_sycl_topk_moe_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT &&
        node->op != GGML_OP_MUL_MAT) {
        return 0;
    }

    // -------------------------------------------------------------------------
    // Prefer fuse starting at F32 router MUL_MAT when present:
    //   MUL_MAT -> SIGMOID -> RESHAPE -> ADD(bias) -> ARGSORT -> VIEW -> GET_ROWS
    // Falls back to SIGMOID-start (GEMV already executed as prior node).
    // -------------------------------------------------------------------------
    if (node->op == GGML_OP_MUL_MAT && i + 6 < cgraph->n_nodes && ggml_sycl_topk_moe_bias_enabled()) {
        ggml_tensor * sigmoid = cgraph->nodes[i + 1];
        ggml_tensor * reshape = cgraph->nodes[i + 2];
        ggml_tensor * add     = cgraph->nodes[i + 3];
        ggml_tensor * argsort = cgraph->nodes[i + 4];
        ggml_tensor * view    = cgraph->nodes[i + 5];
        ggml_tensor * getrows = cgraph->nodes[i + 6];

        const bool laguna_mm =
            sigmoid->op == GGML_OP_UNARY && ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            sigmoid->src[0] == node &&
            reshape->op == GGML_OP_RESHAPE && reshape->src[0] == sigmoid &&
            add->op == GGML_OP_ADD && add->src[0] == sigmoid && add->src[1] != nullptr &&
            argsort->op == GGML_OP_ARGSORT && argsort->src[0] == add &&
            view->op == GGML_OP_VIEW && view->src[0] == argsort &&
            getrows->op == GGML_OP_GET_ROWS && getrows->src[0] == reshape && getrows->src[1] == view &&
            node->type == GGML_TYPE_F32 && node->src[0] && node->src[0]->type == GGML_TYPE_F32 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32;

        if (laguna_mm) {
            std::vector<ggml_op> ops = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD,
                                         GGML_OP_ARGSORT, GGML_OP_VIEW,  GGML_OP_GET_ROWS };
            const ggml_tensor * clamp = nullptr;
            const ggml_tensor * scale = nullptr;
            ggml_sycl_topk_append_norm_scale(cgraph, i, ops, clamp, scale);

            int out_nodes[2] = { i + 5, i + (int) ops.size() - 1 };  // VIEW ids, final weights
            ggml_tensor *       ids     = cgraph->nodes[out_nodes[0]];
            ggml_tensor *       weights = cgraph->nodes[out_nodes[1]];
            const ggml_tensor * logits  = node;
            const ggml_tensor * bias    = add->src[1];

            const bool can_sub = ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2);
            const bool can_use = ggml_sycl_should_use_topk_moe(sigmoid, weights, logits, ids);
            const bool bias_ok = bias && ggml_is_contiguous(bias) && bias->type == GGML_TYPE_F32;
            const bool can_mem = ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2,
                                                                     /*is_topk_moe=*/true);
            if (can_sub && can_use && bias_ok && can_mem && node->data && sigmoid->data && add->data &&
                argsort->data && getrows->data && weights->data) {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    const int m = ggml_sycl_topk_moe_hybrid_mode();
                    fprintf(stderr,
                            "[lx-control-topk-moe] laguna bias fuse HIT (mul_mat+hybrid mode=%d)\n", m);
                }
                GGML_UNUSED(logits);
                GGML_UNUSED(ids);
                GGML_UNUSED(bias);
                // i/nops span includes MUL_MAT; hybrid owns GEMV when fuse enabled.
                ggml_sycl_op_topk_moe_hybrid_bias(ctx, cgraph, i, (int) ops.size(), /*sigmoid=*/sigmoid,
                                                  /*add=*/add, /*argsort=*/argsort, /*getrows=*/getrows,
                                                  /*weights_final=*/weights, clamp, scale, /*mul_mat=*/node);
                return (int) ops.size() - 1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Laguna live graph (measured): SIGMOID -> RESHAPE -> ADD(bias) -> ARGSORT
    //   -> VIEW -> GET_ROWS [-> norm/scale]
    // ADD.src[0] is the sigmoid (not the reshape); GET_ROWS reads reshaped probs.
    // -------------------------------------------------------------------------
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
        i + 5 < cgraph->n_nodes) {
        ggml_tensor * reshape = cgraph->nodes[i + 1];
        ggml_tensor * add     = cgraph->nodes[i + 2];
        ggml_tensor * argsort = cgraph->nodes[i + 3];
        ggml_tensor * view    = cgraph->nodes[i + 4];
        ggml_tensor * getrows = cgraph->nodes[i + 5];

        const bool laguna_ok =
            reshape->op == GGML_OP_RESHAPE && reshape->src[0] == node &&
            add->op == GGML_OP_ADD && add->src[0] == node && add->src[1] != nullptr &&
            argsort->op == GGML_OP_ARGSORT && argsort->src[0] == add &&
            view->op == GGML_OP_VIEW && view->src[0] == argsort &&
            getrows->op == GGML_OP_GET_ROWS && getrows->src[0] == reshape && getrows->src[1] == view;

        if (laguna_ok && ggml_sycl_topk_moe_bias_enabled()) {
            std::vector<ggml_op> ops = { GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT,
                                         GGML_OP_VIEW,  GGML_OP_GET_ROWS };
            const ggml_tensor * clamp = nullptr;
            const ggml_tensor * scale = nullptr;
            ggml_sycl_topk_append_norm_scale(cgraph, i, ops, clamp, scale);

            int out_nodes[2] = { i + 4, i + (int) ops.size() - 1 };  // VIEW ids, final weights
            ggml_tensor *       ids     = cgraph->nodes[out_nodes[0]];
            ggml_tensor *       weights = cgraph->nodes[out_nodes[1]];
            const ggml_tensor * logits  = node->src[0];
            const ggml_tensor * bias    = add->src[1];

            const bool can_sub = ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2);
            const bool can_use = ggml_sycl_should_use_topk_moe(node, weights, logits, ids);
            const bool bias_ok = ggml_is_contiguous(bias) && bias->type == GGML_TYPE_F32;
            const bool can_mem = ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2,
                                                                     /*is_topk_moe=*/true);
            if (can_sub && can_use && bias_ok && can_mem && node->data && add->data && argsort->data &&
                getrows->data && weights->data) {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    const int m = ggml_sycl_topk_moe_hybrid_mode();
                    fprintf(stderr,
                            "[lx-control-topk-moe] laguna bias fuse HIT (hybrid mode=%d: 0=stock 1=gather 2=gather-norm)\n",
                            m);
                }
                GGML_UNUSED(logits);
                GGML_UNUSED(ids);
                GGML_UNUSED(bias);
                ggml_sycl_op_topk_moe_hybrid_bias(ctx, cgraph, i, (int) ops.size(), /*sigmoid=*/node, /*add=*/add,
                                                  /*argsort=*/argsort, /*getrows=*/getrows, /*weights_final=*/weights,
                                                  clamp, scale);
                return (int) ops.size() - 1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Legacy patterns (CUDA-style reshape-before-argsort, gpt-oss delayed softmax)
    // -------------------------------------------------------------------------
    ggml_sycl_topk_moe_args args;
    if (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        return 0;
    }

    const ggml_tensor * logits  = node->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    int                  out_nodes[2];

    if (!args.delayed_softmax) {
        const ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.push_back(gating_op);
        const int i_probs = i + (int) ops.size() - 1;

        if (args.prob_bias) {
            // CUDA-style reshape-before-add is not Laguna; hybrid Laguna path handles live bias above.
            // Decline here — no bitonic / no untested CUDA-layout hybrid this fire.
            return 0;
        }
        ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i_probs + 3;
        ids          = cgraph->nodes[out_nodes[0]];

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
            ggml_sycl_should_use_topk_moe(node, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    } else if (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (int) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            return (int) ops.size() - 1;
        }
    }

    return 0;
}
