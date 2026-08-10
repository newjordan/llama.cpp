/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#pragma once

#include <sycl/nd_item.hpp>

#include "ggml-sycl/dpct/helper.hpp"

template <int ElementsPerWI>
__dpct_inline__ static void quantize_q8_1_impl(const float * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_values, float & d,
                                               float & sum, const sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<float, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);

    float amax = 0.0f;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_values[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <int ElementsPerWI> struct no_quantize_q8_1 {
    void operator()(const float *, void *, int, int, const sycl::nd_item<1> &) const {}
};

template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// [lx-norm-qkv] quantize variant with an inline RMS_NORM(*w) prologue over the raw
// activation row. Computes scale bit-identically to the standalone rms_norm_f32
// (ncols >= 1024 path, one row): same per-virtual-thread strided accumulation,
// same 16-lane butterfly order, same cross-warp s_sum combine — emulated in
// registers by each subgroup (no global sync needed, all subgroups agree on the
// same scale). The scaled row values (scale*x)*w then feed the stock per-block
// q8_1 quantization, so the resulting SoA q8_1 blocks are bit-identical to
// quantize(norm(x)*w) from the standalone kernels.
template <int ElementsPerWI>
struct quantize_norm_reorder_q8_1_soa {
    __dpct_inline__ void operator()(const float * __restrict__ x, const float * __restrict__ w,
                                    const float eps, const int ncols, const int wg_size,
                                    void * reordered_q8_tensor, const int kx, const int kx_padded,
                                    const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);
        constexpr int WARP = WARP_SIZE;

        // ---- emulated rms_norm_f32 reduction (single row, ncols >= 1024) ----
        // stock thread tid accumulates cols tid, tid+wg_size, ... sequentially;
        // butterfly warp reduce; cross-warp combine via s_sum[wi + i*WARP].
        const int nwarps = wg_size / WARP;
        float p[64];  // one partial per virtual warp (wg_size <= 1024 -> <= 64)
        for (int vw = 0; vw < nwarps; ++vw) {
            const int tid = vw * WARP + wi_id;
            float acc = 0.0f;
            for (int col = tid; col < ncols; col += wg_size) {
                const float xi = x[col];
                acc += xi * xi;
            }
            p[vw] = acc;
        }
        for (int vw = 0; vw < nwarps; ++vw) {
#pragma unroll
            for (int mask = WARP / 2; mask > 0; mask >>= 1) {
                p[vw] += dpct::permute_sub_group_by_xor(it.get_sub_group(), p[vw], mask);
            }
        }
        float t = 0.0f;
        for (int i = 0; i < nwarps; i += WARP) {
            t += p[wi_id + i];
        }
#pragma unroll
        for (int mask = WARP / 2; mask > 0; mask >>= 1) {
            t += dpct::permute_sub_group_by_xor(it.get_sub_group(), t, mask);
        }
        const float mean  = t / (float) ncols;
        const float scale = sycl::rsqrt(mean + eps);

        // ---- stock q8_1 per-block quantization of the scaled row ----
        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float d = 0.0f, sum = 0.0f;
        float amax = 0.0f;
        const int base = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
        sycl::vec<float, ElementsPerWI> v;
#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            v[i] = (scale * x[base + i]) * w[base + i];  // matches (scale*x)*w order
            sum += v[i];
            amax = sycl::fmax(amax, sycl::fabs(v[i]));
            quantized_values[i] = 0;
        }
        sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
        amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
        d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            quantized_values[i] = sycl::round(v[i] / d);
        }
        d = amax == 0 ? 0 : d;

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <int ElementsPerWI> struct quantize_q8_1 {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * q8_tensor, const int kx, const int kx_padded,
                                    const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        const int pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % num_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_values;
        if (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <int> typename quantize_f>
void quantize_row_q8_1_sycl(const float * x, void * vy, const int kx, const int ky, const int kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}

// [lx-norm-qkv] Single-row inline-RMS_NORM(*w) + q8_1 quantize for the fused
// QKV activation path. Same launch shape as quantize_row_q8_1_sycl; each
// subgroup redundantly computes the row scale (bit-exact vs rms_norm_f32) and
// then quantizes its own QK8_1 block. Decode-only (ky == 1) by contract.
inline void quantize_norm_row_q8_1_sycl(const float * x, const float * w, const float eps, const int ncols,
                                 const int wg_size, void * vy, const int kx, const int ky, const int kx_padded,
                                 dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_norm_reorder_q8_1_soa<QK8_1 / WARP_SIZE>()(
                                 x, w, eps, ncols, wg_size, vy, kx, kx_padded, it);
                         });
}
