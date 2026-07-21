#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst>
static void mul_mat_vec_q_reorder_ncols(const void * __restrict__ vx, const void * __restrict__ vy,
                                        float * __restrict__ dst, const int ncols, const int nrows,
                                        const int stride_col_y_bytes, const int stride_col_dst,
                                        const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum[ncols_dst] = {0.0f};
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
        const int  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const char       * vy_j           = (const char *)vy + j * stride_col_y_bytes;
                const int8_t     * q8_1_quant_ptr = (const int8_t *)vy_j + iby * QK8_1;
                const sycl::half2* q8_1_ds_ptr    = (const sycl::half2 *)(vy_j + ncols + iby * sizeof(sycl::half2));

                partial_sum[j] += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j], std::plus<>());

        if (sg.leader()) {
            dst[j * stride_col_dst + row] = sum;
        }
    }
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_reorder_ncols_hoisted(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * blocks_per_row;
    const auto *  base                        = static_cast<const uint8_t *>(vx);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum[ncols_dst] = {0.0f};
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
         i < blocks_per_row; i += blocks_per_subgroup) {
        const int  ibx       = row * blocks_per_row + i;
        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
        const int  iby       = i * block_type::block_to_q8_1_ratio();
        const auto * qs      = reinterpret_cast<const int8_t *>(base + bx_offset.first);
        const float d        = static_cast<float>(
            *reinterpret_cast<const ggml_half *>(base + d_offset.first));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq *
                (sg.get_local_linear_id() % block_elements_per_subgroup);
            int v[block_traits::vdr_mmvq];

#pragma unroll
            for (size_t q = 0; q < block_traits::vdr_mmvq; ++q) {
                v[q] = get_int_from_int8(qs, iqs + q);
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const char * vy_j = static_cast<const char *>(vy) + j * stride_col_y_bytes;
                const auto * q8_1_quant_ptr = reinterpret_cast<const int8_t *>(vy_j) + iby * QK8_1;
                const auto * q8_1_ds_ptr = reinterpret_cast<const sycl::half2 *>(
                    vy_j + ncols + iby * sizeof(sycl::half2));
                int u[block_traits::vdr_mmvq];

#pragma unroll
                for (size_t q = 0; q < block_traits::vdr_mmvq; ++q) {
                    u[q] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + q);
                }

                int sumi = 0;
#pragma unroll
                for (size_t q = 0; q < block_traits::vdr_mmvq; ++q) {
                    sumi = dpct::dp4a(v[q], u[q], sumi);
                }
                const sycl::half2 ds_values = *q8_1_ds_ptr;
                partial_sum[j] += d * static_cast<float>(ds_values[0]) * sumi;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        const float sum = sycl::reduce_over_group(sg, partial_sum[j], std::plus<>());
        if (sg.leader()) {
            dst[j * stride_col_dst + row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static void mul_mat_vec_q_ncols(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols,
        const int nrows,
        const int stride_col_y,
        const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    float tmp[ncols_dst] = {0.0f};

    const block_q_t  * x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, block_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, block_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, block_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_nvfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, block_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, block_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, block_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

static int ggml_sycl_q8_mmvq_subgroups() {
    // Default 32: B70 product A/B (2026-07-20 post-reboot ABA) beat 16 by ~+0.9% tg128
    // on Qwen3.6-35B-A3B Q5_K_XL short-ctx. Multi-col Q8 reorder still hardcodes 16.
    // Override: GGML_SYCL_Q8_MMVQ_SUBGROUPS=1|2|4|8|16|32.
    static const int n = []() {
        const char * env = getenv("GGML_SYCL_Q8_MMVQ_SUBGROUPS");
        const int value = env == nullptr ? 32 : atoi(env);
        switch (value) {
            case 1: case 2: case 4: case 8: case 16: case 32: return value;
            default: return 32;
        }
    }();
    return n;
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    const size_t num_subgroups = (size_t) ggml_sycl_q8_mmvq_subgroups();
    const int    block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q8_0_reorder_ncols_hoisted<ncols_dst>(
                                 vx, vy, dst, ncols, nrows,
                                 stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static bool q8_0_ncols_weight_hoist_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (q8_0_ncols_weight_hoist_enabled()) {
        static std::atomic<bool> traced { false };
        if (!traced.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr, "[treebeard-q8-hoist] activated ncols=%d nrows=%d k=%d\n",
                    ncols_dst, nrows, ncols);
        }
        switch (ncols_dst) {
            case 2:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<2>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 3:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<3>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 4:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<4>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 5:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<5>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 6:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<6>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 7:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<7>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            case 8:
                reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols_hoisted<8>(
                    vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
                return;
            default:
                break;
        }
    }
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 12: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<12>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, block_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q2_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, block_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

template <int ncols_dst>
static void mul_mat_vec_q3_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, block_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, block_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, block_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups) * (int) num_subgroups;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q6_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, block_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, block_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q8_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && (src1_ncols <= 8 || src1_ncols == 12)) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_K:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q3_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_MXFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_NVFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            default:
                GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// Wide-batch MMVQ: the ncols kernels amortize one weight-matrix pass over up to 8 dst columns.
// For src1_ncols > 8 (batched decode with more than 8 parallel sequences), process in chunks of 8
// instead of falling back to the dequantize+GEMM path, which streams ~6x the weight bytes for
// small-token batches. Chunk k reads src1/dst at a column offset; strides are unchanged.
void ggml_sycl_op_mul_mat_vec_q_wide(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                     const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i,
                                     const float * src1_ddf_i, const char * src1_ddq_i, float * dst_dd_i,
                                     const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
                                     const int64_t src1_padded_col_size, const dpct::queue_ptr & stream) {
    static const bool disable_12col = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMVQ_12COL");
        return env != nullptr && atoi(env) != 0;
    }();
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (!disable_12col && src1_ncols == 12 && src0->type == GGML_TYPE_Q8_0 &&
        extra && extra->optimized_feature.reorder) {
        ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                                   row_low, row_high, src1_ncols, src1_padded_col_size, stream);
        return;
    }

    constexpr int64_t chunk = 8;  // max ncols_dst the templated multi-column kernels support
    for (int64_t col0 = 0; col0 < src1_ncols; col0 += chunk) {
        const int64_t ncols_chunk  = std::min(src1_ncols - col0, chunk);
        const char *  src1_ddq_col = src1_ddq_i + col0 * src1_padded_col_size * sizeof(block_q8_1) / QK8_1;
        float *       dst_dd_col   = dst_dd_i + col0 * dst->ne[0];
        ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_col, dst_dd_col,
                                   row_low, row_high, ncols_chunk, src1_padded_col_size, stream);
    }
}

void ggml_sycl_op_moe_weighted_sum(
    const ggml_tensor * experts,
    const ggml_tensor * weights,
    ggml_tensor *       dst,
    const dpct::queue_ptr & stream) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(experts->ne[0] == dst->ne[0]);
    GGML_ASSERT(experts->ne[1] == weights->ne[1]);
    GGML_ASSERT(experts->ne[2] == weights->ne[2]);
    GGML_ASSERT(weights->ne[0] == 1);
    GGML_ASSERT(dst->ne[1] == experts->ne[2]);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);

    const int64_t nrows     = experts->ne[0];
    const int64_t n_experts = experts->ne[1];
    const int64_t n_tokens  = experts->ne[2];
    const int64_t total     = nrows * n_tokens;

    const char * experts_data = (const char *) experts->data;
    const char * weights_data = (const char *) weights->data;
    char *       dst_data     = (char *) dst->data;

    const size_t experts_row_stride   = experts->nb[0];
    const size_t experts_slot_stride  = experts->nb[1];
    const size_t experts_token_stride = experts->nb[2];
    const size_t weights_slot_stride  = weights->nb[1];
    const size_t weights_token_stride = weights->nb[2];
    const size_t dst_row_stride       = dst->nb[0];
    const size_t dst_token_stride     = dst->nb[1];

    constexpr int block_size = 256;
    const int64_t global_size = GGML_PAD(total, block_size);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) global_size),
                              sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item) {
                const int64_t linear = (int64_t) item.get_global_linear_id();
                if (linear >= total) {
                    return;
                }
                const int64_t token = linear / nrows;
                const int64_t row   = linear - token * nrows;
                float sum = 0.0f;
                for (int64_t expert = 0; expert < n_experts; ++expert) {
                    const float value = *(const float *) (
                        experts_data + token * experts_token_stride +
                        expert * experts_slot_stride + row * experts_row_stride);
                    const float weight = *(const float *) (
                        weights_data + token * weights_token_stride +
                        expert * weights_slot_stride);

                    // The unfused graph materializes GGML_OP_MUL before the ordered
                    // expert ADD chain. Preserve that float rounding point instead of
                    // allowing contraction into an FMA.
                    volatile float weighted = value * weight;
                    sum += weighted;
                }
                *(float *) (dst_data + token * dst_token_stride + row * dst_row_stride) = sum;
            });
    });
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
// Batched over tokens via grid dim 0: each token reads its own ids row / activation / dst slice,
// so a whole batched-decode MUL_MAT_ID is ONE launch with no host readback of the routing ids.
static int ggml_sycl_mmid_wg_subgroups(const int nrows) {
    static const int override_subgroups = []() {
        const char * env = getenv("GGML_SYCL_MMID_WG_SUBGROUPS");
        const int value = env == nullptr ? 0 : atoi(env);
        switch (value) {
            case 0:
            case 1:
            case 2:
            case 4:
            case 8:
            case 16:
            case 32:
                return value;
            default:
                return 0;
        }
    }();
    if (override_subgroups != 0) {
        return override_subgroups;
    }
    // Product MoE down is nrows=2048 (embd). 4 was conservative; 8 fills the WG
    // better for the serial top-k expert loop without ballooning register pressure
    // (measured A/B 2026-07-20: env 8/16/32 — keep 8 as new default for large).
    if (nrows >= 2048) {
        return 8;
    }
    if (nrows >= 1024) {
        return 4;
    }
    return 1;
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_moe(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const int ids_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {

    const auto sg         = item_ct1.get_sub_group();
    const int  token      = item_ct1.get_group(0);
    const int  expert_idx = item_ct1.get_group(1);
    const int  i02        = ids_dev[token * ids_row_stride + expert_idx];

    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id();

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg.get_local_linear_id() / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (sg.get_local_linear_id() % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(sg, tmp, mask);
    }

    if (sg.leader()) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const int n_tokens, const int ids_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int            num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int            block_num_y = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride,
                    ids_row_stride, src1_token_stride, dst_token_stride, item);
            });
    });
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_moe_weighted(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    const int32_t * __restrict__ ids_dev, const float * __restrict__ weights,
    float * __restrict__ dst_base, const int ncols, const int nrows,
    const int n_experts_used, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const int ids_row_stride, const size_t src1_token_stride,
    const size_t weights_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    const auto sg    = item_ct1.get_sub_group();
    const int  token = item_ct1.get_group(0);
    const int  row   = item_ct1.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    float reduced = 0.0f;

    for (int expert_idx = 0; expert_idx < n_experts_used; ++expert_idx) {
        const int i02 = ids_dev[token * ids_row_stride + expert_idx];
        const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
        const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride +
                          (size_t) expert_idx * src1_row_stride;
        const block_q_t *  x = (const block_q_t *) vx;
        const block_q8_1 * y = (const block_q8_1 *) vy;

        float tmp = 0.0f;
        for (int i = sg.get_local_linear_id() / (qi / vdr);
             i < blocks_per_row; i += blocks_per_warp) {
            const int ibx = row * blocks_per_row + i;
            const int iby = i * (qk / QK8_1);

            for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
                const int iqs = elem + vdr * (sg.get_local_linear_id() % (qi / vdr));
                tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp += dpct::permute_sub_group_by_xor(sg, tmp, mask);
        }

        if (sg.leader()) {
            const float weight = *(const float *) ((const char *) weights +
                (size_t) token * weights_token_stride +
                (size_t) expert_idx * weights_slot_stride);
            volatile float weighted = tmp * weight;
            reduced += weighted;
        }
    }

    if (sg.leader()) {
        float * dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride);
        dst[row] = reduced;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_weighted(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    const float * weights, float * dst_base, const int ncols, const int nrows,
    const int n_experts_used, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const int n_tokens, const int ids_row_stride, const size_t src1_token_stride,
    const size_t weights_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int            num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int            block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums((unsigned) n_tokens, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_weighted<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_base, vy, ids_dev, weights, dst_base, ncols, nrows,
                    n_experts_used, expert_weight_stride, src1_row_stride,
                    weights_slot_stride, ids_row_stride, src1_token_stride,
                    weights_token_stride, dst_token_stride, item);
            });
    });
}

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
    dpct::queue_ptr    stream) {
    if (src0_type != GGML_TYPE_Q8_0) {
        return false;
    }
    launch_mul_mat_vec_q_moe_weighted<
        QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
            vx_base, vy, ids_dev, weights, dst_base, ncols, nrows, n_experts_used,
            expert_weight_stride, src1_row_stride, weights_slot_stride,
            n_tokens, ids_row_stride, src1_token_stride, weights_token_stride,
            dst_token_stride, stream);
    return true;
}

bool ggml_sycl_mul_mat_vec_q_id(
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
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const int ids_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg         = item_ct1.get_sub_group();
    const int  token      = item_ct1.get_group(0);
    const int  expert_idx = item_ct1.get_group(1);
    const int  i02        = ids_dev[token * ids_row_stride + expert_idx];

    const char * vx  = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy  = (const char *) vy_base + (size_t) token * src1_token_stride + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        const int           iby            = i * block_type::block_to_q8_1_ratio();
        const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_reorder(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const int n_tokens, const int ids_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int            num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int            block_num_y = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride,
                    ids_row_stride, src1_token_stride, dst_token_stride, item);
            });
    });
}

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_weighted_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    const int32_t * __restrict__ ids_dev, const float * __restrict__ weights,
    float * __restrict__ dst_base, const int ncols, const int nrows,
    const int n_experts_used, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const int ids_row_stride, const size_t src1_token_stride,
    const size_t weights_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg    = item_ct1.get_sub_group();
    const int  token = item_ct1.get_group(0);
    const int  row   = item_ct1.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    // Prefetch routing for this token (top-k is typically 8 on product Qwen3.6).
    // Keeps ids/weights out of the inner k-loop; expert_idx still serial for
    // volatile ordered sum contract (matches MUL + ADD chain).
    constexpr int k_max_prefetch = 16;
    int   expert_ids_local[k_max_prefetch];
    float expert_w_local[k_max_prefetch];
    const int n_use = n_experts_used < k_max_prefetch ? n_experts_used : k_max_prefetch;
    for (int e = 0; e < n_use; ++e) {
        expert_ids_local[e] = ids_dev[token * ids_row_stride + e];
        if (sg.leader()) {
            expert_w_local[e] = *(const float *) ((const char *) weights +
                (size_t) token * weights_token_stride +
                (size_t) e * weights_slot_stride);
        }
    }

    float reduced = 0.0f;
    // Product top-k is 8: fully unroll that case so the compiler can schedule
    // eight expert MMVQs inside the serial ordered-sum loop.
    if (n_experts_used == 8) {
#pragma unroll
        for (int expert_idx = 0; expert_idx < 8; ++expert_idx) {
            const int i02 = expert_ids_local[expert_idx];
            const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
            const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride +
                              (size_t) expert_idx * src1_row_stride;
            float partial_sum = 0.0f;
            for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
                 i < blocks_per_row; i += blocks_per_subgroup) {
                const int ibx = row * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const int           iby            = i * block_type::block_to_q8_1_ratio();
                const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
                const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *)
                    ((const char *) vy + ncols + iby * sizeof(sycl::half2));
#pragma unroll
                for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                    const int iqs = elem + block_traits::vdr_mmvq *
                        (sg.get_local_linear_id() % block_elements_per_subgroup);
                    partial_sum += reorder_vec_dot_q_sycl()(
                        vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                }
            }
            const float expert_value = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
            if (sg.leader()) {
                volatile float weighted = expert_value * expert_w_local[expert_idx];
                reduced += weighted;
            }
        }
    } else {
        for (int expert_idx = 0; expert_idx < n_experts_used; ++expert_idx) {
            const int i02 = (expert_idx < n_use)
                ? expert_ids_local[expert_idx]
                : ids_dev[token * ids_row_stride + expert_idx];
            const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
            const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride +
                              (size_t) expert_idx * src1_row_stride;
            float partial_sum = 0.0f;
            for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
                 i < blocks_per_row; i += blocks_per_subgroup) {
                const int ibx = row * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const int           iby            = i * block_type::block_to_q8_1_ratio();
                const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
                const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *)
                    ((const char *) vy + ncols + iby * sizeof(sycl::half2));
#pragma unroll
                for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                    const int iqs = elem + block_traits::vdr_mmvq *
                        (sg.get_local_linear_id() % block_elements_per_subgroup);
                    partial_sum += reorder_vec_dot_q_sycl()(
                        vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                }
            }
            const float expert_value = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
            if (sg.leader()) {
                const float weight = (expert_idx < n_use)
                    ? expert_w_local[expert_idx]
                    : *(const float *) ((const char *) weights +
                        (size_t) token * weights_token_stride +
                        (size_t) expert_idx * weights_slot_stride);
                volatile float weighted = expert_value * weight;
                reduced += weighted;
            }
        }
    }

    if (sg.leader()) {
        float * dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride);
        dst[row] = reduced;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_weighted_reorder(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    const float * weights, float * dst_base, const int ncols, const int nrows,
    const int n_experts_used, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const int n_tokens, const int ids_row_stride, const size_t src1_token_stride,
    const size_t weights_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int            num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int            block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums((unsigned) n_tokens, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_weighted_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, ids_dev, weights, dst_base, ncols, nrows,
                    n_experts_used, expert_weight_stride, src1_row_stride,
                    weights_slot_stride, ids_row_stride, src1_token_stride,
                    weights_token_stride, dst_token_stride, item);
            });
    });
}

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
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_weighted_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, weights, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, src1_row_stride, weights_slot_stride,
                n_tokens, ids_row_stride, src1_token_stride, weights_token_stride,
                dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_weighted_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, weights, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, src1_row_stride, weights_slot_stride,
                n_tokens, ids_row_stride, src1_token_stride, weights_token_stride,
                dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_weighted_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, weights, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, src1_row_stride, weights_slot_stride,
                n_tokens, ids_row_stride, src1_token_stride, weights_token_stride,
                dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

// ------- grouped expert GEMM (phase 2): read each expert's weights once per batch -------
//
// The batched kernel above launches one workgroup-column per (token, expert-slot) pair, so two
// pairs routed to the same expert stream its weight matrix from VRAM twice. The grouped path
// first buckets the pairs by expert on device (one tiny single-workgroup kernel), then launches
// the GEMV grid over *distinct* experts, with each row-workgroup dotting all of its expert's
// routed tokens per weight pass (up to MMID_GROUP_TCHUNK at a time).
//
// Scratch layout (int32): [0..n_as] offsets | [n_as+1 .. 2*n_as] active expert list |
// [2*n_as+1] n_active | [2*n_as+2 ..) packed pairs (token<<16 | slot), bucketed by expert.

#define MMID_GROUP_WG     256  // routing pre-pass workgroup size; also the n_as ceiling
#define MMID_GROUP_TCHUNK 4    // tokens dotted per weight pass (register-resident accumulators)

static void k_mmid_group_pairs(
    const int32_t * __restrict__ ids_dev, int32_t * __restrict__ scratch,
    const int n_tokens, const int n_ids, const int ids_row_stride, const int n_as,
    const sycl::nd_item<1> & it,
    int32_t * counts, int32_t * cursors, int32_t * n_active_l) {
    const int lid     = it.get_local_id(0);
    const int n_pairs = n_tokens * n_ids;

    int32_t * offsets  = scratch;
    int32_t * active   = scratch + n_as + 1;
    int32_t * n_active = scratch + 2 * n_as + 1;
    int32_t * pairs    = scratch + 2 * n_as + 2;

    if (lid < n_as) {
        counts[lid] = 0;
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int p = lid; p < n_pairs; p += MMID_GROUP_WG) {
        const int e = ids_dev[(p / n_ids) * ids_row_stride + (p % n_ids)];
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space> cnt(counts[e]);
        cnt.fetch_add(1);
    }
    it.barrier(sycl::access::fence_space::local_space);

    if (lid == 0) {
        int off = 0;
        for (int e = 0; e < n_as; ++e) {
            offsets[e]  = off;
            cursors[e]  = off;
            off        += counts[e];
        }
        offsets[n_as] = off;
        *n_active_l   = 0;
    }
    it.barrier(sycl::access::fence_space::local_space);

    if (lid < n_as && counts[lid] > 0) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space> na(*n_active_l);
        active[na.fetch_add(1)] = lid;
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int p = lid; p < n_pairs; p += MMID_GROUP_WG) {
        const int tok  = p / n_ids;
        const int slot = p % n_ids;
        const int e    = ids_dev[tok * ids_row_stride + slot];
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space> cur(cursors[e]);
        pairs[cur.fetch_add(1)] = (tok << 16) | slot;
    }
    if (lid == 0) {
        *n_active = *n_active_l;
    }
}

static void launch_mmid_group_pairs(
    const int32_t * ids_dev, int32_t * scratch,
    const int n_tokens, const int n_ids, const int ids_row_stride, const int n_as,
    dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> counts(sycl::range<1>(MMID_GROUP_WG), cgh);
        sycl::local_accessor<int32_t, 1> cursors(sycl::range<1>(MMID_GROUP_WG), cgh);
        sycl::local_accessor<int32_t, 1> n_active_l(sycl::range<1>(1), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(MMID_GROUP_WG), sycl::range<1>(MMID_GROUP_WG)),
            [=](sycl::nd_item<1> it) {
                k_mmid_group_pairs(ids_dev, scratch, n_tokens, n_ids, ids_row_stride, n_as, it,
                                   counts.get_multi_ptr<sycl::access::decorated::no>().get(),
                                   cursors.get_multi_ptr<sycl::access::decorated::no>().get(),
                                   n_active_l.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });

    // Diagnostic-only occupancy sensor. The grouped kernel's speedup ceiling is set by the
    // number of distinct experts in a routed batch, but that count normally stays on device.
    // Opting in deliberately serializes this launch and copies one int so production-shaped
    // routing can falsify expert-weight-reuse proposals before another kernel is built.
    static const bool profile_reuse = []() {
        const char * env = std::getenv("GGML_SYCL_MOE_REUSE_PROFILE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    static std::atomic<int> profile_samples { 0 };
    if (profile_reuse) {
        const int sample = profile_samples.fetch_add(1, std::memory_order_relaxed);
        if (sample >= 4096) {
            return;
        }
        int32_t n_active = 0;
        stream->memcpy(&n_active, scratch + 2 * n_as + 1, sizeof(n_active)).wait();
        std::fprintf(stderr,
                     "[treebeard-moe-reuse] sample=%d tokens=%d slots=%d routes=%d"
                     " experts=%d active=%d duplicate_routes=%d ideal_weight_read_reduction=%.6f\n",
                     sample, n_tokens, n_ids, n_tokens * n_ids, n_as, n_active,
                     n_tokens * n_ids - n_active,
                     n_tokens * n_ids > 0 ?
                         1.0 - (double) n_active / (double) (n_tokens * n_ids) : 0.0);
    }
}

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_weighted_grouped_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    const float * __restrict__ weights, float * __restrict__ dst_base,
    const int32_t * __restrict__ scratch, float * __restrict__ contributions,
    const int ncols, const int nrows, const int n_experts_used,
    const int n_tokens, const int n_as,
    const int num_subgroups, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const size_t src1_token_stride, const size_t weights_token_stride,
    const size_t dst_token_stride, const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    // Each subgroup owns one output row and its token/slot contribution table.
    // Initialize cooperatively before invalid tail-row subgroups return.
    const int route_count = n_tokens * n_experts_used;
    const int local_count = num_subgroups * route_count;
    for (int i = item_ct1.get_local_linear_id(); i < local_count;
         i += item_ct1.get_local_range().size()) {
        contributions[i] = 0.0f;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const auto sg          = item_ct1.get_sub_group();
    const int subgroup_idx = sg.get_group_linear_id();
    const int row = item_ct1.get_group(2) * num_subgroups + subgroup_idx;
    if (row >= nrows) {
        return;
    }

    const int32_t * offsets  = scratch;
    const int32_t * active   = scratch + n_as + 1;
    const int32_t   n_active = scratch[2 * n_as + 1];
    const int32_t * pairs    = scratch + 2 * n_as + 2;

    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float * row_contributions = contributions + subgroup_idx * route_count;
    for (int g = 0; g < n_active; ++g) {
        const int e     = active[g];
        const int start = offsets[e];
        const int count = offsets[e + 1] - start;
        const char * vx = (const char *) vx_base + (size_t) e * expert_weight_stride;

        for (int t0 = 0; t0 < count; t0 += MMID_GROUP_TCHUNK) {
            const int tc = sycl::min(count - t0, MMID_GROUP_TCHUNK);
            const char * vys[MMID_GROUP_TCHUNK];
            int tokens[MMID_GROUP_TCHUNK];
            int slots[MMID_GROUP_TCHUNK];
            for (int j = 0; j < tc; ++j) {
                const int32_t pair = pairs[start + t0 + j];
                tokens[j] = pair >> 16;
                slots[j]  = pair & 0xffff;
                vys[j] = (const char *) vy_base +
                    (size_t) tokens[j] * src1_token_stride +
                    (size_t) slots[j] * src1_row_stride;
            }

            float partial[MMID_GROUP_TCHUNK] = { 0.0f };
            for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
                 i < blocks_per_row; i += blocks_per_subgroup) {
                const int ibx = row * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const int iby = i * block_type::block_to_q8_1_ratio();

#pragma unroll
                for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                    const int iqs = elem + block_traits::vdr_mmvq *
                        (sg.get_local_linear_id() % block_elements_per_subgroup);
                    for (int j = 0; j < tc; ++j) {
                        const int8_t * q8_1_quant_ptr =
                            (const int8_t *) vys[j] + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr = (const sycl::half2 *)
                            ((const char *) vys[j] + ncols + iby * sizeof(sycl::half2));
                        partial[j] += reorder_vec_dot_q_sycl()(
                            vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }

            for (int j = 0; j < tc; ++j) {
                const float expert_value =
                    sycl::reduce_over_group(sg, partial[j], std::plus<>());
                if (sg.leader()) {
                    const float weight = *(const float *) ((const char *) weights +
                        (size_t) tokens[j] * weights_token_stride +
                        (size_t) slots[j] * weights_slot_stride);
                    volatile float weighted = expert_value * weight;
                    row_contributions[tokens[j] * n_experts_used + slots[j]] = weighted;
                }
            }
        }
    }

    if (sg.leader()) {
        for (int token = 0; token < n_tokens; ++token) {
            float reduced = 0.0f;
            for (int slot = 0; slot < n_experts_used; ++slot) {
                reduced += row_contributions[token * n_experts_used + slot];
            }
            float * dst = (float *) ((char *) dst_base +
                                     (size_t) token * dst_token_stride);
            dst[row] = reduced;
        }
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_weighted_grouped_reorder(
    const void * vx_base, const void * vy, const float * weights,
    const int32_t * scratch, float * dst_base, const int ncols,
    const int nrows, const int n_experts_used, const int n_tokens,
    const int n_as, const size_t expert_weight_stride,
    const size_t src1_row_stride, const size_t weights_slot_stride,
    const size_t src1_token_stride, const size_t weights_token_stride,
    const size_t dst_token_stride, dpct::queue_ptr stream) {
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums(1, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    const size_t local_floats =
        (size_t) num_subgroups * n_tokens * n_experts_used;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> contributions(
            sycl::range<1>(local_floats), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_weighted_grouped_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, weights, dst_base, scratch,
                    contributions.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ncols, nrows, n_experts_used, n_tokens, n_as, num_subgroups,
                    expert_weight_stride, src1_row_stride, weights_slot_stride,
                    src1_token_stride, weights_token_stride, dst_token_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id_weighted_grouped_reorder(
    enum ggml_type src0_type, const void * vx_base, const void * vy,
    const int32_t * ids_dev, const float * weights, int32_t * scratch,
    float * dst_base, int ncols, int nrows, int n_experts_used, int n_as,
    size_t expert_weight_stride, size_t src1_row_stride,
    size_t weights_slot_stride, int n_tokens, int ids_row_stride,
    size_t src1_token_stride, size_t weights_token_stride,
    size_t dst_token_stride, dpct::queue_ptr stream) {
    if (n_as > MMID_GROUP_WG || n_tokens > 0xffff ||
        n_experts_used > 0xffff || n_tokens < 1) {
        return false;
    }
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }

    launch_mmid_group_pairs(
        ids_dev, scratch, n_tokens, n_experts_used, ids_row_stride, n_as, stream);

    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_weighted_grouped_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, weights, scratch, dst_base, ncols, nrows,
                n_experts_used, n_tokens, n_as, expert_weight_stride,
                src1_row_stride, weights_slot_stride, src1_token_stride,
                weights_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_weighted_grouped_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, weights, scratch, dst_base, ncols, nrows,
                n_experts_used, n_tokens, n_as, expert_weight_stride,
                src1_row_stride, weights_slot_stride, src1_token_stride,
                weights_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_weighted_grouped_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, weights, scratch, dst_base, ncols, nrows,
                n_experts_used, n_tokens, n_as, expert_weight_stride,
                src1_row_stride, weights_slot_stride, src1_token_stride,
                weights_token_stride, dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_grouped_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ scratch,
    const int ncols, const int nrows, const int n_as,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int32_t * offsets  = scratch;
    const int32_t * active   = scratch + n_as + 1;
    const int32_t   n_active = scratch[2 * n_as + 1];
    const int32_t * pairs    = scratch + 2 * n_as + 2;

    const int g = item_ct1.get_group(1);
    if (g >= n_active) {
        return;
    }
    const int e     = active[g];
    const int start = offsets[e];
    const int cnt   = offsets[e + 1] - start;

    const char * vx = (const char *) vx_base + (size_t) e * expert_weight_stride;

    const auto sg = item_ct1.get_sub_group();
    const int row = item_ct1.get_group(2) * sg.get_group_linear_range() + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    for (int t0 = 0; t0 < cnt; t0 += MMID_GROUP_TCHUNK) {
        const int tc = sycl::min(cnt - t0, MMID_GROUP_TCHUNK);

        const char * vys [MMID_GROUP_TCHUNK];
        float *      dsts[MMID_GROUP_TCHUNK];
        for (int j = 0; j < tc; ++j) {
            const int32_t pair = pairs[start + t0 + j];
            const int     tok  = pair >> 16;
            const int     slot = pair & 0xffff;
            vys[j]  = (const char *) vy_base + (size_t) tok * src1_token_stride + (size_t) slot * src1_row_stride;
            dsts[j] = (float *) ((char *) dst_base + (size_t) tok * dst_token_stride + (size_t) slot * dst_row_stride);
        }

        float partial[MMID_GROUP_TCHUNK] = { 0.0f };
        for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
            const int ibx = row * blocks_per_row + i;

            const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
            const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

            const int iby = i * block_type::block_to_q8_1_ratio();

#pragma unroll
            for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
                // one weight-block fetch feeds every routed token in the chunk
                for (int j = 0; j < tc; ++j) {
                    const int8_t *      q8_1_quant_ptr = (const int8_t *) vys[j] + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vys[j] + ncols + iby * sizeof(sycl::half2));
                    partial[j] += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                }
            }
        }

        for (int j = 0; j < tc; ++j) {
            const auto sum = sycl::reduce_over_group(sg, partial[j], std::plus<>());
            if (sg.leader()) {
                dsts[j][row] = sum;
            }
        }
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_grouped_reorder(
    const void * vx_base, const void * vy, const int32_t * scratch,
    float * dst_base, const int ncols, const int nrows, const int n_as, const int max_groups,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int            num_subgroups = ggml_sycl_mmid_wg_subgroups(nrows);
    const int            block_num_y = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums(1, (unsigned) max_groups, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_grouped_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, scratch, ncols, nrows, n_as,
                    expert_weight_stride, dst_row_stride, src1_row_stride,
                    src1_token_stride, dst_token_stride, item);
            });
    });
}

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
    dpct::queue_ptr    stream) {
    if (n_as > MMID_GROUP_WG || n_tokens > 0xffff || n_experts_used > 0xffff) {
        return false;
    }
    // dispatchability check BEFORE submitting the pre-pass, so a bail leaves the stream untouched
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }

    launch_mmid_group_pairs(ids_dev, scratch, n_tokens, n_experts_used, ids_row_stride, n_as, stream);

    const int max_groups = std::min(n_tokens * n_experts_used, n_as);
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_grouped_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, scratch, dst_base, ncols, nrows, n_as, max_groups,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_grouped_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, scratch, dst_base, ncols, nrows, n_as, max_groups,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_grouped_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, scratch, dst_base, ncols, nrows, n_as, max_groups,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                src1_token_stride, dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

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
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                n_tokens, ids_row_stride, src1_token_stride, dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

// Qwen3.6 stores routed gate and up projections as separate expert matrices.
// Pair matching rows from those matrices in one subgroup and apply SwiGLU
// before storage. This removes one complete MMID output and the standalone GLU
// read/write pass while retaining the grouped expert-reuse path.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_dual_swiglu_reorder(
    const void * __restrict__ vx_gate_base,
    const void * __restrict__ vx_up_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t dst_row_stride, const size_t src1_row_stride,
    const int ids_row_stride, const size_t src1_token_stride,
    const size_t dst_token_stride, const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg         = item_ct1.get_sub_group();
    const int  token      = item_ct1.get_group(0);
    const int  expert_idx = item_ct1.get_group(1);
    const int  expert     = ids_dev[token * ids_row_stride + expert_idx];
    const int  row        = item_ct1.get_group(2) * sg.get_group_linear_range() +
                            sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const char * vx_gate = (const char *) vx_gate_base +
                           (size_t) expert * gate_expert_stride;
    const char * vx_up = (const char *) vx_up_base +
                         (size_t) expert * up_expert_stride;
    const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride +
                      (size_t) expert_idx * src1_row_stride;
    float * dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride +
                             (size_t) expert_idx * dst_row_stride);

    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
         i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;
        const auto bx = block_type::get_block_offset(ibx, nblocks);
        const auto d  = block_type::get_d_offset(nrows, ncols, ibx);
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t * q8 = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_ds = (const sycl::half2 *)
            ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq *
                (sg.get_local_linear_id() % block_elements_per_subgroup);
            gate_partial += reorder_vec_dot_q_sycl()(vx_gate, bx, d, q8, q8_ds, iqs);
            up_partial   += reorder_vec_dot_q_sycl()(vx_up,   bx, d, q8, q8_ds, iqs);
        }
    }

    const float gate = sycl::reduce_over_group(sg, gate_partial, std::plus<>());
    const float up   = sycl::reduce_over_group(sg, up_partial,   std::plus<>());
    if (sg.leader()) {
        dst[row] = (gate / (1.0f + sycl::native::exp(-gate))) * up;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_dual_swiglu_reorder(
    const void * vx_gate_base, const void * vx_up_base, const void * vy,
    const int32_t * ids_dev, float * dst_base, const int ncols,
    const int nrows, const int n_experts_used,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t dst_row_stride, const size_t src1_row_stride,
    const int n_tokens, const int ids_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(2 * nrows);
    const int block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums(
        (unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_dual_swiglu_reorder<reorder_vec_dot_q_sycl>(
                    vx_gate_base, vx_up_base, vy, dst_base, ids_dev, ncols, nrows,
                    gate_expert_stride, up_expert_stride, dst_row_stride,
                    src1_row_stride, ids_row_stride, src1_token_stride,
                    dst_token_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id_dual_swiglu_reorder(
    enum ggml_type src0_type, const void * vx_gate_base,
    const void * vx_up_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, int ncols, int nrows, int n_experts_used,
    size_t gate_expert_stride, size_t up_expert_stride,
    size_t dst_row_stride, size_t src1_row_stride, int n_tokens,
    int ids_row_stride, size_t src1_token_stride, size_t dst_token_stride,
    dpct::queue_ptr stream) {
    switch (src0_type) {
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                    vx_gate_base, vx_up_base, vy, ids_dev, dst_base, ncols,
                    nrows, n_experts_used, gate_expert_stride, up_expert_stride,
                    dst_row_stride, src1_row_stride, n_tokens, ids_row_stride,
                    src1_token_stride, dst_token_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                    vx_gate_base, vx_up_base, vy, ids_dev, dst_base, ncols,
                    nrows, n_experts_used, gate_expert_stride, up_expert_stride,
                    dst_row_stride, src1_row_stride, n_tokens, ids_row_stride,
                    src1_token_stride, dst_token_stride, stream);
            return true;
        default:
            return false;
    }
}

// Dense (non-ID) dual-SwiGLU: shared-expert gate/up are ordinary MUL_MATs that
// share one activation vector. Pair both projections per row and store
// silu(gate)*up — collapses two MMVQ submits + GLU into one kernel. Decode
// (ncols_dst=1) is the ship target; multi-col is supported for prefill microbatch.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_dense_dual_swiglu_reorder(
    const void * __restrict__ vx_gate,
    const void * __restrict__ vx_up,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const int ncols, const int nrows,
    const size_t src1_col_stride_bytes, const size_t dst_col_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg  = item_ct1.get_sub_group();
    const int  col = item_ct1.get_group(0);
    const int  row = item_ct1.get_group(2) * sg.get_group_linear_range() +
                     sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const char * vy = (const char *) vy_base + (size_t) col * src1_col_stride_bytes;
    float * dst = dst_base + (size_t) col * dst_col_stride;

    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
         i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;
        const auto bx = block_type::get_block_offset(ibx, nblocks);
        const auto d  = block_type::get_d_offset(nrows, ncols, ibx);
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t * q8 = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_ds = (const sycl::half2 *)
            ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq *
                (sg.get_local_linear_id() % block_elements_per_subgroup);
            gate_partial += reorder_vec_dot_q_sycl()(vx_gate, bx, d, q8, q8_ds, iqs);
            up_partial   += reorder_vec_dot_q_sycl()(vx_up,   bx, d, q8, q8_ds, iqs);
        }
    }

    const float gate = sycl::reduce_over_group(sg, gate_partial, std::plus<>());
    const float up   = sycl::reduce_over_group(sg, up_partial,   std::plus<>());
    if (sg.leader()) {
        dst[row] = (gate / (1.0f + sycl::native::exp(-gate))) * up;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_dense_dual_swiglu_reorder(
    const void * vx_gate, const void * vx_up, const void * vy,
    float * dst, const int ncols, const int nrows, const int ncols_dst,
    const size_t src1_col_stride_bytes, const size_t dst_col_stride,
    dpct::queue_ptr stream) {
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(2 * nrows);
    const int block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums(
        (unsigned) ncols_dst, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_dense_dual_swiglu_reorder<reorder_vec_dot_q_sycl>(
                    vx_gate, vx_up, vy, dst, ncols, nrows,
                    src1_col_stride_bytes, dst_col_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_dense_dual_swiglu_reorder(
    enum ggml_type src0_type, const void * vx_gate, const void * vx_up,
    const void * vy, float * dst, int ncols, int nrows, int ncols_dst,
    size_t src1_col_stride_bytes, size_t dst_col_stride,
    dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 32) {
        return false;
    }
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
            // UD-Q5_K_XL stores shared-expert gate/up as Q8_0 (measured 2026-07-20).
            launch_mul_mat_vec_q_dense_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                    vx_gate, vx_up, vy, dst, ncols, nrows, ncols_dst,
                    src1_col_stride_bytes, dst_col_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_dense_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                    vx_gate, vx_up, vy, dst, ncols, nrows, ncols_dst,
                    src1_col_stride_bytes, dst_col_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_dense_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                    vx_gate, vx_up, vy, dst, ncols, nrows, ncols_dst,
                    src1_col_stride_bytes, dst_col_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_dense_dual_swiglu_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                    vx_gate, vx_up, vy, dst, ncols, nrows, ncols_dst,
                    src1_col_stride_bytes, dst_col_stride, stream);
            return true;
        default:
            return false;
    }
}

// Dual dense MMVQ: share one quantized activation across two weight matrices.
// Used for GDN pairs (attn_qkv+attn_gate, ssm_alpha+ssm_beta) that share `cur`.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_dense_dual_mmvq_reorder(
    const void * __restrict__ vx_a,
    const void * __restrict__ vx_b,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_a_base,
    float * __restrict__ dst_b_base,
    const int ncols, const int nrows_a, const int nrows_b,
    const size_t src1_col_stride_bytes,
    const size_t dst_a_col_stride, const size_t dst_b_col_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg  = item_ct1.get_sub_group();
    const int  col = item_ct1.get_group(0);
    const int  row = item_ct1.get_group(2) * sg.get_group_linear_range() +
                     sg.get_group_linear_id();
    const bool do_a = row < nrows_a;
    const bool do_b = row < nrows_b;
    if (!do_a && !do_b) {
        return;
    }

    const char * vy = (const char *) vy_base + (size_t) col * src1_col_stride_bytes;
    float * dst_a = dst_a_base + (size_t) col * dst_a_col_stride;
    float * dst_b = dst_b_base + (size_t) col * dst_b_col_stride;

    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks_a = nrows_a * blocks_per_row;
    const int nblocks_b = nrows_b * blocks_per_row;

    float partial_a = 0.0f;
    float partial_b = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
         i < blocks_per_row; i += blocks_per_subgroup) {
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t * q8 = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_ds = (const sycl::half2 *)
            ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq *
                (sg.get_local_linear_id() % block_elements_per_subgroup);
            if (do_a) {
                const int ibx = row * blocks_per_row + i;
                const auto bx = block_type::get_block_offset(ibx, nblocks_a);
                const auto d  = block_type::get_d_offset(nrows_a, ncols, ibx);
                partial_a += reorder_vec_dot_q_sycl()(vx_a, bx, d, q8, q8_ds, iqs);
            }
            if (do_b) {
                const int ibx = row * blocks_per_row + i;
                const auto bx = block_type::get_block_offset(ibx, nblocks_b);
                const auto d  = block_type::get_d_offset(nrows_b, ncols, ibx);
                partial_b += reorder_vec_dot_q_sycl()(vx_b, bx, d, q8, q8_ds, iqs);
            }
        }
    }

    if (do_a) {
        const float sum = sycl::reduce_over_group(sg, partial_a, std::plus<>());
        if (sg.leader()) {
            dst_a[row] = sum;
        }
    }
    if (do_b) {
        const float sum = sycl::reduce_over_group(sg, partial_b, std::plus<>());
        if (sg.leader()) {
            dst_b[row] = sum;
        }
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_dense_dual_mmvq_reorder(
    const void * vx_a, const void * vx_b, const void * vy,
    float * dst_a, float * dst_b, const int ncols,
    const int nrows_a, const int nrows_b, const int ncols_dst,
    const size_t src1_col_stride_bytes,
    const size_t dst_a_col_stride, const size_t dst_b_col_stride,
    dpct::queue_ptr stream) {
    const int nrows_max = nrows_a > nrows_b ? nrows_a : nrows_b;
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(2 * nrows_max);
    const int block_num_y   = ceil_div(nrows_max, num_subgroups);
    const sycl::range<3> block_nums(
        (unsigned) ncols_dst, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_dense_dual_mmvq_reorder<reorder_vec_dot_q_sycl>(
                    vx_a, vx_b, vy, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    src1_col_stride_bytes, dst_a_col_stride, dst_b_col_stride,
                    item);
            });
    });
}

// Dual F32 GEMV: one workgroup-row per output row index (shared across both
// destinations when nrows match). Loads activation once; dots both weight rows.
static void mul_mat_vec_f32_dense_dual_kernel(
    const float * __restrict__ wa,
    const float * __restrict__ wb,
    const float * __restrict__ x_base,
    float * __restrict__ dst_a_base,
    float * __restrict__ dst_b_base,
    const int ncols, const int nrows_a, const int nrows_b,
    const size_t x_col_stride,
    const size_t dst_a_col_stride, const size_t dst_b_col_stride,
    const size_t wa_row_stride, const size_t wb_row_stride,
    const sycl::nd_item<3> & item) {
    const auto sg  = item.get_sub_group();
    const int  col = item.get_group(0);
    const int  row = item.get_group(2) * sg.get_group_linear_range() +
                     sg.get_group_linear_id();
    const bool do_a = row < nrows_a;
    const bool do_b = row < nrows_b;
    if (!do_a && !do_b) {
        return;
    }

    const float * x     = x_base + (size_t) col * x_col_stride;
    float *       dst_a = dst_a_base + (size_t) col * dst_a_col_stride;
    float *       dst_b = dst_b_base + (size_t) col * dst_b_col_stride;
    const float * ra    = do_a ? wa + (size_t) row * wa_row_stride : nullptr;
    const float * rb    = do_b ? wb + (size_t) row * wb_row_stride : nullptr;

    float partial_a = 0.0f;
    float partial_b = 0.0f;
    const int lane  = sg.get_local_linear_id();
    // Vectorized path when K is a multiple of 4.
    if ((ncols & 3) == 0) {
        const int n4 = ncols >> 2;
        for (int i = lane; i < n4; i += WARP_SIZE) {
            const sycl::float4 xv = *reinterpret_cast<const sycl::float4 *>(x + (i << 2));
            if (do_a) {
                const sycl::float4 wv = *reinterpret_cast<const sycl::float4 *>(ra + (i << 2));
                partial_a += xv.x() * wv.x() + xv.y() * wv.y() + xv.z() * wv.z() + xv.w() * wv.w();
            }
            if (do_b) {
                const sycl::float4 wv = *reinterpret_cast<const sycl::float4 *>(rb + (i << 2));
                partial_b += xv.x() * wv.x() + xv.y() * wv.y() + xv.z() * wv.z() + xv.w() * wv.w();
            }
        }
    } else {
        for (int i = lane; i < ncols; i += WARP_SIZE) {
            const float xv = x[i];
            if (do_a) {
                partial_a += ra[i] * xv;
            }
            if (do_b) {
                partial_b += rb[i] * xv;
            }
        }
    }

    if (do_a) {
        const float sum = sycl::reduce_over_group(sg, partial_a, std::plus<>());
        if (sg.leader()) {
            dst_a[row] = sum;
        }
    }
    if (do_b) {
        const float sum = sycl::reduce_over_group(sg, partial_b, std::plus<>());
        if (sg.leader()) {
            dst_b[row] = sum;
        }
    }
}

bool ggml_sycl_mul_mat_vec_f32_dense_dual(
    const float * wa, const float * wb, const float * x,
    float * dst_a, float * dst_b, int ncols, int nrows_a, int nrows_b,
    int ncols_dst, size_t x_col_stride, size_t dst_a_col_stride,
    size_t dst_b_col_stride, size_t wa_row_stride, size_t wb_row_stride,
    dpct::queue_ptr stream) {
    // Decode-first. Cap multi-col modestly: prefill uses oneDNN for these shapes
    // already (gemm-route); dual is for tiny MKL GEMV pairs on the decode path.
    if (ncols_dst < 1 || ncols_dst > 8 || nrows_a < 1 || nrows_b < 1 || ncols < 1) {
        return false;
    }
    const int nrows_max = nrows_a > nrows_b ? nrows_a : nrows_b;
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(2 * nrows_max);
    const int block_num_y   = ceil_div(nrows_max, num_subgroups);
    const sycl::range<3> block_nums(
        (unsigned) ncols_dst, 1, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_f32_dense_dual_kernel(
                    wa, wb, x, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    x_col_stride, dst_a_col_stride, dst_b_col_stride,
                    wa_row_stride, wb_row_stride, item);
            });
    });
    return true;
}

bool ggml_sycl_mul_mat_vec_q_dense_dual_mmvq_reorder(
    enum ggml_type src0_type, const void * vx_a, const void * vx_b,
    const void * vy, float * dst_a, float * dst_b, int ncols,
    int nrows_a, int nrows_b, int ncols_dst,
    size_t src1_col_stride_bytes, size_t dst_a_col_stride,
    size_t dst_b_col_stride, dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 4 || nrows_a < 1 || nrows_b < 1) {
        return false;
    }
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_dense_dual_mmvq_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                    vx_a, vx_b, vy, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    ncols_dst, src1_col_stride_bytes, dst_a_col_stride,
                    dst_b_col_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_dense_dual_mmvq_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                    vx_a, vx_b, vy, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    ncols_dst, src1_col_stride_bytes, dst_a_col_stride,
                    dst_b_col_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_dense_dual_mmvq_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                    vx_a, vx_b, vy, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    ncols_dst, src1_col_stride_bytes, dst_a_col_stride,
                    dst_b_col_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_dense_dual_mmvq_reorder<
                reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                    vx_a, vx_b, vy, dst_a, dst_b, ncols, nrows_a, nrows_b,
                    ncols_dst, src1_col_stride_bytes, dst_a_col_stride,
                    dst_b_col_stride, stream);
            return true;
        default:
            return false;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_dual_swiglu_grouped_reorder(
    const void * __restrict__ vx_gate_base,
    const void * __restrict__ vx_up_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const int32_t * __restrict__ scratch,
    const int ncols, const int nrows, const int n_as,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t dst_row_stride, const size_t src1_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int32_t * offsets  = scratch;
    const int32_t * active   = scratch + n_as + 1;
    const int32_t   n_active = scratch[2 * n_as + 1];
    const int32_t * pairs    = scratch + 2 * n_as + 2;
    const int group = item_ct1.get_group(1);
    if (group >= n_active) {
        return;
    }

    const int expert = active[group];
    const int start  = offsets[expert];
    const int count  = offsets[expert + 1] - start;
    const auto sg    = item_ct1.get_sub_group();
    const int row    = item_ct1.get_group(2) * sg.get_group_linear_range() +
                       sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const char * vx_gate = (const char *) vx_gate_base +
                           (size_t) expert * gate_expert_stride;
    const char * vx_up = (const char *) vx_up_base +
                         (size_t) expert * up_expert_stride;
    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    for (int t0 = 0; t0 < count; t0 += MMID_GROUP_TCHUNK) {
        const int tc = sycl::min(count - t0, MMID_GROUP_TCHUNK);
        const char * vys[MMID_GROUP_TCHUNK];
        float * dsts[MMID_GROUP_TCHUNK];
        for (int j = 0; j < tc; ++j) {
            const int32_t pair = pairs[start + t0 + j];
            const int token = pair >> 16;
            const int slot  = pair & 0xffff;
            vys[j] = (const char *) vy_base + (size_t) token * src1_token_stride +
                     (size_t) slot * src1_row_stride;
            dsts[j] = (float *) ((char *) dst_base +
                                 (size_t) token * dst_token_stride +
                                 (size_t) slot * dst_row_stride);
        }

        float gate_partial[MMID_GROUP_TCHUNK] = { 0.0f };
        float up_partial[MMID_GROUP_TCHUNK]   = { 0.0f };
        for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
             i < blocks_per_row; i += blocks_per_subgroup) {
            const int ibx = row * blocks_per_row + i;
            const auto bx = block_type::get_block_offset(ibx, nblocks);
            const auto d  = block_type::get_d_offset(nrows, ncols, ibx);
            const int iby = i * block_type::block_to_q8_1_ratio();

#pragma unroll
            for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                const int iqs = elem + block_traits::vdr_mmvq *
                    (sg.get_local_linear_id() % block_elements_per_subgroup);
                for (int j = 0; j < tc; ++j) {
                    const int8_t * q8 = (const int8_t *) vys[j] + iby * QK8_1;
                    const sycl::half2 * q8_ds = (const sycl::half2 *)
                        ((const char *) vys[j] + ncols + iby * sizeof(sycl::half2));
                    gate_partial[j] += reorder_vec_dot_q_sycl()(
                        vx_gate, bx, d, q8, q8_ds, iqs);
                    up_partial[j] += reorder_vec_dot_q_sycl()(
                        vx_up, bx, d, q8, q8_ds, iqs);
                }
            }
        }

        for (int j = 0; j < tc; ++j) {
            const float gate = sycl::reduce_over_group(
                sg, gate_partial[j], std::plus<>());
            const float up = sycl::reduce_over_group(
                sg, up_partial[j], std::plus<>());
            if (sg.leader()) {
                dsts[j][row] = (gate / (1.0f + sycl::native::exp(-gate))) * up;
            }
        }
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_dual_swiglu_grouped_reorder(
    const void * vx_gate_base, const void * vx_up_base, const void * vy,
    const int32_t * scratch, float * dst_base, const int ncols,
    const int nrows, const int n_as, const int max_groups,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t dst_row_stride, const size_t src1_row_stride,
    const size_t src1_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    const int num_subgroups = ggml_sycl_mmid_wg_subgroups(2 * nrows);
    const int block_num_y   = ceil_div(nrows, num_subgroups);
    const sycl::range<3> block_nums(1, (unsigned) max_groups, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_dual_swiglu_grouped_reorder<
                    reorder_vec_dot_q_sycl>(
                        vx_gate_base, vx_up_base, vy, dst_base, scratch,
                        ncols, nrows, n_as, gate_expert_stride,
                        up_expert_stride, dst_row_stride, src1_row_stride,
                        src1_token_stride, dst_token_stride, item);
            });
    });
}

// Composite MoE activation/down pipeline.  Four subgroups cooperatively
// compute a 32-row gate/up tile for up to four routed token/slot pairs.  The
// activated values live only in workgroup memory; the same subgroups then
// quantize each pair directly into the reordered Q8_1 layout consumed by the
// down projection.  This removes the global F32 SwiGLU tensor and its separate
// quantization submission.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_dual_swiglu_grouped_q8_reorder(
    const void * __restrict__ vx_gate_base,
    const void * __restrict__ vx_up_base,
    const void * __restrict__ vy_base,
    void * __restrict__ q8_dst_base,
    const int32_t * __restrict__ scratch,
    float * __restrict__ activations,
    const int ncols, const int nrows, const int nrows_padded,
    const int n_experts_used, const int n_as,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t src1_row_stride, const size_t src1_token_stride,
    const size_t q8_row_stride, const size_t q8_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    constexpr int tile_rows = QK8_1;
    constexpr int num_subgroups = MMID_GROUP_TCHUNK;
    constexpr int quant_values_per_lane = tile_rows / WARP_SIZE;
    static_assert(tile_rows % WARP_SIZE == 0);

    const int32_t * offsets  = scratch;
    const int32_t * active   = scratch + n_as + 1;
    const int32_t   n_active = scratch[2 * n_as + 1];
    const int32_t * pairs    = scratch + 2 * n_as + 2;
    const int group = item_ct1.get_group(1);
    if (group >= n_active) {
        return;
    }

    const int expert = active[group];
    const int start  = offsets[expert];
    const int count  = offsets[expert + 1] - start;
    const auto sg    = item_ct1.get_sub_group();
    const int subgroup_idx = sg.get_group_linear_id();
    const int lane = sg.get_local_linear_id();
    const int row_base = item_ct1.get_group(2) * tile_rows;

    const char * vx_gate = (const char *) vx_gate_base +
                           (size_t) expert * gate_expert_stride;
    const char * vx_up = (const char *) vx_up_base +
                         (size_t) expert * up_expert_stride;
    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup =
        ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup =
        block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    for (int t0 = 0; t0 < count; t0 += MMID_GROUP_TCHUNK) {
        const int tc = sycl::min(count - t0, MMID_GROUP_TCHUNK);
        const char * vys[MMID_GROUP_TCHUNK];
        int tokens[MMID_GROUP_TCHUNK];
        int slots[MMID_GROUP_TCHUNK];
        for (int j = 0; j < tc; ++j) {
            const int32_t pair = pairs[start + t0 + j];
            tokens[j] = pair >> 16;
            slots[j]  = pair & 0xffff;
            vys[j] = (const char *) vy_base +
                     (size_t) tokens[j] * src1_token_stride +
                     (size_t) slots[j] * src1_row_stride;
        }

        // The four subgroups cover one complete Q8 block.  Each subgroup owns
        // every fourth output row, keeping the token chunk resident while the
        // expert's matching gate/up rows are streamed.
        for (int tile_row = subgroup_idx; tile_row < tile_rows;
             tile_row += num_subgroups) {
            const int row = row_base + tile_row;
            float gate_partial[MMID_GROUP_TCHUNK] = { 0.0f };
            float up_partial[MMID_GROUP_TCHUNK]   = { 0.0f };
            for (int i = lane / block_elements_per_subgroup;
                 i < blocks_per_row; i += blocks_per_subgroup) {
                const int ibx = row * blocks_per_row + i;
                const auto bx = block_type::get_block_offset(ibx, nblocks);
                const auto d  = block_type::get_d_offset(nrows, ncols, ibx);
                const int iby = i * block_type::block_to_q8_1_ratio();

#pragma unroll
                for (int elem = 0; elem < block_elements_per_subgroup;
                     elem += WARP_SIZE) {
                    const int iqs = elem + block_traits::vdr_mmvq *
                        (lane % block_elements_per_subgroup);
                    for (int j = 0; j < tc; ++j) {
                        const int8_t * q8 = (const int8_t *) vys[j] + iby * QK8_1;
                        const sycl::half2 * q8_ds = (const sycl::half2 *)
                            ((const char *) vys[j] + ncols +
                             iby * sizeof(sycl::half2));
                        gate_partial[j] += reorder_vec_dot_q_sycl()(
                            vx_gate, bx, d, q8, q8_ds, iqs);
                        up_partial[j] += reorder_vec_dot_q_sycl()(
                            vx_up, bx, d, q8, q8_ds, iqs);
                    }
                }
            }

            for (int j = 0; j < tc; ++j) {
                const float gate = sycl::reduce_over_group(
                    sg, gate_partial[j], std::plus<>());
                const float up = sycl::reduce_over_group(
                    sg, up_partial[j], std::plus<>());
                if (sg.leader()) {
                    activations[j * tile_rows + tile_row] =
                        (gate / (1.0f + sycl::native::exp(-gate))) * up;
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // One subgroup quantizes each routed pair.  The address calculation is
        // identical to quantize_and_reorder_q8_1_soa: quant bytes first, then
        // one half2(scale,sum) per logical 32-value block.
        if (subgroup_idx < tc) {
            float values[quant_values_per_lane];
            float sum = 0.0f;
            float amax = 0.0f;
#pragma unroll
            for (int elem = 0; elem < quant_values_per_lane; ++elem) {
                values[elem] = activations[
                    subgroup_idx * tile_rows +
                    lane * quant_values_per_lane + elem];
                sum += values[elem];
                amax = sycl::fmax(amax, sycl::fabs(values[elem]));
            }
            sum = sycl::reduce_over_group(sg, sum, std::plus<>());
            amax = sycl::reduce_over_group(
                sg, amax, sycl::maximum<float>());
            const float divisor = amax == 0.0f ? 1.0f : amax / 127.0f;
            char * q8_row = (char *) q8_dst_base +
                (size_t) tokens[subgroup_idx] * q8_token_stride +
                (size_t) slots[subgroup_idx] * q8_row_stride;
#pragma unroll
            for (int elem = 0; elem < quant_values_per_lane; ++elem) {
                ((int8_t *) q8_row)[
                    row_base + lane * quant_values_per_lane + elem] =
                    (int8_t) sycl::round(values[elem] / divisor);
            }
            if (sg.leader()) {
                sycl::half2 * ds = (sycl::half2 *)
                    (q8_row + nrows +
                     (row_base / tile_rows) * sizeof(sycl::half2));
                *ds = sycl::half2(
                    sycl::half(amax == 0.0f ? 0.0f : divisor),
                    sycl::half(sum));
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    GGML_UNUSED(nrows_padded);
    GGML_UNUSED(n_experts_used);
}

template <typename gate_reorder_vec_dot_q_sycl,
          typename down_reorder_vec_dot_q_sycl>
static void launch_moe_swiglu_down_pipeline_reorder(
    const void * vx_gate_base, const void * vx_up_base,
    const void * vx_down_base, const void * vy,
    const float * weights, const int32_t * scratch,
    void * q8_intermediate, float * dst_base,
    const int gate_ncols, const int gate_nrows,
    const int gate_nrows_padded, const int down_nrows,
    const int n_experts_used, const int n_tokens, const int n_as,
    const size_t gate_expert_stride, const size_t up_expert_stride,
    const size_t down_expert_stride, const size_t src1_row_stride,
    const size_t weights_slot_stride, const size_t src1_token_stride,
    const size_t q8_row_stride, const size_t q8_token_stride,
    const size_t weights_token_stride, const size_t dst_token_stride,
    dpct::queue_ptr stream) {
    constexpr int num_subgroups = MMID_GROUP_TCHUNK;
    const int max_groups = std::min(n_tokens * n_experts_used, n_as);
    const int row_blocks = gate_nrows / QK8_1;
    const sycl::range<3> block_nums(
        1, (unsigned) max_groups, (unsigned) row_blocks);
    const sycl::range<3> block_dims(
        1, 1, (unsigned) num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> activations(
            sycl::range<1>(MMID_GROUP_TCHUNK * QK8_1), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_dual_swiglu_grouped_q8_reorder<
                    gate_reorder_vec_dot_q_sycl>(
                        vx_gate_base, vx_up_base, vy, q8_intermediate,
                        scratch,
                        activations.get_multi_ptr<sycl::access::decorated::no>().get(),
                        gate_ncols, gate_nrows, gate_nrows_padded,
                        n_experts_used, n_as, gate_expert_stride,
                        up_expert_stride, src1_row_stride, src1_token_stride,
                        q8_row_stride, q8_token_stride, item);
            });
    });

    launch_mul_mat_vec_q_moe_weighted_grouped_reorder<
        down_reorder_vec_dot_q_sycl>(
            vx_down_base, q8_intermediate, weights, scratch, dst_base,
            gate_nrows, down_nrows, n_experts_used, n_tokens, n_as,
            down_expert_stride, q8_row_stride, weights_slot_stride,
            q8_token_stride, weights_token_stride, dst_token_stride, stream);
}

bool ggml_sycl_mul_mat_vec_q_id_dual_swiglu_grouped_reorder(
    enum ggml_type src0_type, const void * vx_gate_base,
    const void * vx_up_base, const void * vy, const int32_t * ids_dev,
    int32_t * scratch, float * dst_base, int ncols, int nrows,
    int n_experts_used, int n_as, size_t gate_expert_stride,
    size_t up_expert_stride, size_t dst_row_stride,
    size_t src1_row_stride, int n_tokens, int ids_row_stride,
    size_t src1_token_stride, size_t dst_token_stride,
    dpct::queue_ptr stream) {
    if (n_as > MMID_GROUP_WG || n_tokens > 0xffff ||
        n_experts_used > 0xffff) {
        return false;
    }
    if (src0_type != GGML_TYPE_Q5_K && src0_type != GGML_TYPE_Q6_K) {
        return false;
    }

    launch_mmid_group_pairs(ids_dev, scratch, n_tokens, n_experts_used,
                            ids_row_stride, n_as, stream);
    const int max_groups = std::min(n_tokens * n_experts_used, n_as);
    if (src0_type == GGML_TYPE_Q5_K) {
        launch_mul_mat_vec_q_moe_dual_swiglu_grouped_reorder<
            reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_gate_base, vx_up_base, vy, scratch, dst_base, ncols,
                nrows, n_as, max_groups, gate_expert_stride,
                up_expert_stride, dst_row_stride, src1_row_stride,
                src1_token_stride, dst_token_stride, stream);
    } else {
        launch_mul_mat_vec_q_moe_dual_swiglu_grouped_reorder<
            reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_gate_base, vx_up_base, vy, scratch, dst_base, ncols,
                nrows, n_as, max_groups, gate_expert_stride,
                up_expert_stride, dst_row_stride, src1_row_stride,
                src1_token_stride, dst_token_stride, stream);
    }
    return true;
}

template <typename gate_reorder_vec_dot_q_sycl>
static bool launch_moe_swiglu_down_pipeline_for_gate(
    enum ggml_type down_type, const void * vx_gate_base,
    const void * vx_up_base, const void * vx_down_base, const void * vy,
    const float * weights, const int32_t * scratch, void * q8_intermediate,
    float * dst_base, int gate_ncols, int gate_nrows,
    int gate_nrows_padded, int down_nrows, int n_experts_used,
    int n_tokens, int n_as, size_t gate_expert_stride,
    size_t up_expert_stride, size_t down_expert_stride,
    size_t src1_row_stride, size_t weights_slot_stride,
    size_t src1_token_stride, size_t q8_row_stride,
    size_t q8_token_stride, size_t weights_token_stride,
    size_t dst_token_stride, dpct::queue_ptr stream) {
    switch (down_type) {
        case GGML_TYPE_Q4_K:
            launch_moe_swiglu_down_pipeline_reorder<
                gate_reorder_vec_dot_q_sycl,
                reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                    vx_gate_base, vx_up_base, vx_down_base, vy, weights,
                    scratch, q8_intermediate, dst_base, gate_ncols,
                    gate_nrows, gate_nrows_padded, down_nrows,
                    n_experts_used, n_tokens, n_as, gate_expert_stride,
                    up_expert_stride, down_expert_stride, src1_row_stride,
                    weights_slot_stride, src1_token_stride, q8_row_stride,
                    q8_token_stride, weights_token_stride, dst_token_stride,
                    stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_moe_swiglu_down_pipeline_reorder<
                gate_reorder_vec_dot_q_sycl,
                reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                    vx_gate_base, vx_up_base, vx_down_base, vy, weights,
                    scratch, q8_intermediate, dst_base, gate_ncols,
                    gate_nrows, gate_nrows_padded, down_nrows,
                    n_experts_used, n_tokens, n_as, gate_expert_stride,
                    up_expert_stride, down_expert_stride, src1_row_stride,
                    weights_slot_stride, src1_token_stride, q8_row_stride,
                    q8_token_stride, weights_token_stride, dst_token_stride,
                    stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_moe_swiglu_down_pipeline_reorder<
                gate_reorder_vec_dot_q_sycl,
                reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                    vx_gate_base, vx_up_base, vx_down_base, vy, weights,
                    scratch, q8_intermediate, dst_base, gate_ncols,
                    gate_nrows, gate_nrows_padded, down_nrows,
                    n_experts_used, n_tokens, n_as, gate_expert_stride,
                    up_expert_stride, down_expert_stride, src1_row_stride,
                    weights_slot_stride, src1_token_stride, q8_row_stride,
                    q8_token_stride, weights_token_stride, dst_token_stride,
                    stream);
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_mul_mat_vec_q_id_swiglu_down_grouped_reorder(
    enum ggml_type gate_type, enum ggml_type down_type,
    const void * vx_gate_base, const void * vx_up_base,
    const void * vx_down_base, const void * vy, const int32_t * ids_dev,
    const float * weights, int32_t * scratch, void * q8_intermediate,
    float * dst_base, int gate_ncols, int gate_nrows,
    int gate_nrows_padded, int down_nrows, int n_experts_used, int n_as,
    size_t gate_expert_stride, size_t up_expert_stride,
    size_t down_expert_stride, size_t src1_row_stride,
    size_t weights_slot_stride, int n_tokens, int ids_row_stride,
    size_t src1_token_stride, size_t q8_row_stride,
    size_t q8_token_stride, size_t weights_token_stride,
    size_t dst_token_stride, dpct::queue_ptr stream) {
    if (n_as > MMID_GROUP_WG || n_tokens < 4 || n_tokens > 0xffff ||
        n_experts_used < 1 || n_experts_used > 0xffff ||
        gate_ncols % QK8_1 != 0 || gate_nrows % QK8_1 != 0 ||
        gate_nrows_padded < gate_nrows) {
        return false;
    }
    if (gate_type != GGML_TYPE_Q5_K && gate_type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (down_type != GGML_TYPE_Q4_K && down_type != GGML_TYPE_Q5_K &&
        down_type != GGML_TYPE_Q6_K) {
        return false;
    }

    launch_mmid_group_pairs(ids_dev, scratch, n_tokens, n_experts_used,
                            ids_row_stride, n_as, stream);
    if (gate_type == GGML_TYPE_Q5_K) {
        return launch_moe_swiglu_down_pipeline_for_gate<
            reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                down_type, vx_gate_base, vx_up_base, vx_down_base, vy,
                weights, scratch, q8_intermediate, dst_base, gate_ncols,
                gate_nrows, gate_nrows_padded, down_nrows, n_experts_used,
                n_tokens, n_as, gate_expert_stride, up_expert_stride,
                down_expert_stride, src1_row_stride, weights_slot_stride,
                src1_token_stride, q8_row_stride, q8_token_stride,
                weights_token_stride, dst_token_stride, stream);
    }
    return launch_moe_swiglu_down_pipeline_for_gate<
        reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
            down_type, vx_gate_base, vx_up_base, vx_down_base, vy, weights,
            scratch, q8_intermediate, dst_base, gate_ncols, gate_nrows,
            gate_nrows_padded, down_nrows, n_experts_used, n_tokens, n_as,
            gate_expert_stride, up_expert_stride, down_expert_stride,
            src1_row_stride, weights_slot_stride, src1_token_stride,
            q8_row_stride, q8_token_stride, weights_token_stride,
            dst_token_stride, stream);
}
