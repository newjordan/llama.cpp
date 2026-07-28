//
// SYCL port of the CUDA topk-moe fusion (ggml-cuda/topk-moe.cu).
// Fuses the MoE router tail softmax/sigmoid -> top-k (argmax) -> get_rows [-> norm] [-> scale]
// into a single kernel, eliding the serial chain of tiny ops that otherwise blocks the expert GEMVs.
// On this Arc B70 hybrid-MoE workload that serial chain is submit-bound (not hidden behind a big op),
// so collapsing it is the lever that actually moves decode. See DECODE_2X_PLAN.md (2026-06-21 update).
//
#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#include "common.hpp"

// Detect a fusable topk-moe subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel and return the number of *following* nodes consumed (so the caller does i += return; continue).
// Returns 0 if no topk-moe fusion applies at i.
int ggml_sycl_try_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

#endif // GGML_SYCL_TOPK_MOE_HPP
