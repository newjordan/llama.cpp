#include "rope.hpp"
#include "convert.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "norm.hpp"

#include <atomic>
#include <cstdio>

struct rope_corr_dims {
    float v[2];
};

struct mrope_sections {
    int v[4];
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / sycl::max(0.001f, high - low);
    return 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <bool forward>
static void rope_yarn(const float theta_extrap, const float freq_scale,
                      const rope_corr_dims corr_dims, const int64_t i0,
                      const float ext_factor, float mscale, float &cos_theta,
                      float &sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static void rope_norm(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto &store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 >= n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const float theta_base = pos[i2] * dpct::pow(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static void rope_neox(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 >= n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const float theta_base = pos[i2] * dpct::pow(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static void rope_multi(const T *x, T *dst, const int ne00, const int ne01,
                       const int ne02, const int s01, const int s02,
                       const int s03, const int s1, const int s2, const int s3,
                       const int n_dims, const int32_t *pos,
                       const float freq_scale, const float ext_factor,
                       const float attn_factor, const rope_corr_dims corr_dims,
                       const float theta_scale, const float *freq_factors,
                       const mrope_sections sections, const bool is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (i0 >= n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        return;
    }

    const int sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, i0 / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, i0 / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, i0 / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, i0 / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, bool has_ff, typename T>
static void rope_vision(const T *x, T *dst, const int ne00, const int ne01,
                        const int ne02, const int s01, const int s02,
                        const int s03, const int s1, const int s2, const int s3,
                        const int n_dims, const int32_t *pos,
                        const float freq_scale, const float ext_factor,
                        const float attn_factor, const rope_corr_dims corr_dims,
                        const float theta_scale, const float *freq_factors,
                        const mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, typename T, typename D>
static void
rope_norm_sycl(const T *x, D *dst, const int ne00, const int ne01,
               const int ne02, const int s01, const int s02, const int s03,
               const int s1, const int s2, const int s3, const int n_dims,
               const int nr, const int32_t *pos, const float freq_scale,
               const float freq_base, const float ext_factor,
               const float attn_factor, const rope_corr_dims corr_dims,
               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <bool forward, typename T, typename D>
static void
rope_neox_sycl(const T *x, D *dst, const int ne00, const int ne01,
               const int ne02, const int s01, const int s02, const int s03,
               const int s1, const int s2, const int s3, const int n_dims,
               const int nr, const int32_t *pos, const float freq_scale,
               const float freq_base, const float ext_factor,
               const float attn_factor, const rope_corr_dims corr_dims,
               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const int block_size =
        ne00 == 128 && nr == ne01 * ne02 ? 64 : SYCL_ROPE_BLOCK_SIZE;
    const dpct::dim3 block_dims(1, block_size, 1);
    const int n_blocks_x = (ne00 + 2 * block_size - 1) / (2 * block_size);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <bool forward, typename T>
static void
rope_multi_sycl(const T *x, T *dst, const int ne00, const int ne01,
                const int ne02, const int s01, const int s02, const int s03,
                const int s1, const int s2, const int s3, const int n_dims,
                const int nr, const int32_t *pos, const float freq_scale,
                const float freq_base, const float ext_factor,
                const float attn_factor, const rope_corr_dims corr_dims,
                const float *freq_factors, const mrope_sections sections,
                const bool is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <bool forward, typename T>
static void
rope_vision_sycl(const T *x, T *dst, const int ne00, const int ne01,
                 const int ne02, const int s01, const int s02, const int s03,
                 const int s1, const int s2, const int s3, const int n_dims,
                 const int nr, const int32_t *pos, const float freq_scale,
                 const float freq_base, const float ext_factor,
                 const float attn_factor, const rope_corr_dims corr_dims,
                 const float *freq_factors, const mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <bool forward>
void ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            const ggml_tensor *set_rows = nullptr) {
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const ggml_tensor *src2 = dst->src[2];

    const float *src0_d = (const float *)src0->data;
    const float *src1_d = (const float *)src1->data;

    void *dst_d = dst->data;
    const int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    int set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (const int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    const int n_dims = ((int32_t *)dst->op_params)[1];
    const int mode = ((int32_t *)dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *)dst->op_params)[4];
    mrope_sections sections;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, sizeof(float));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, sizeof(int) * 4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
    }

    const int32_t *pos = (const int32_t *)src1_d;

    const float *freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, float, sycl::half>(
                (const float *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else if (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((const float *)src0_d, (float *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else if (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, float, sycl::half>(
                (const float *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

void ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<true>(ctx, dst);
}

void ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<false>(ctx, dst);
}

void ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<true>(ctx, rope, set_rows);
}

// [lx-norm-rope] Fused RMS_NORM(+MUL scale) + NEOX rope + SET_ROWS scatter for the
// K-cache path (Kcur -> rms -> mul -> rope -> view -> set_rows in ONE launch).
// Indexing mirrors rope_neox exactly: i0 is the element-pair index; lanes with
// j = i0/2 < n_dims/2 rotate the pair (j, j + n_dims/2); tail lanes (j >= n_dims/2)
// copy elements (2j, 2j+1) unchanged; together all ne00 elements are covered once.
// The launch is gated to ne00 == 128 so one 64-lane workgroup covers a whole row
// and the workgroup reduction below is exact. Env-gated at the fuse site
// (GGML_SYCL_FUSE_NORM_ROPE=1); the kernel is dead code otherwise.
template <bool forward, bool has_ff, typename T, typename D>
static void rope_neox_normed(const T *x, D *dst, const int ne00, const int ne01,
                             const int ne02, const int s01, const int s02,
                             const int s03, const int s1, const int s2,
                             const int s3, const int n_dims, const int32_t *pos,
                             const float freq_scale, const float ext_factor,
                             const float attn_factor,
                             const rope_corr_dims corr_dims,
                             const float theta_scale, const float *freq_factors,
                             const float eps, const float *norm_w,
                             const int64_t *row_indices,
                             const int set_rows_stride, const bool no_norm,
                             float * s_sum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    const int j = i0 / 2;
    int idst = j + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = j + i1 * s01 + i2 * s02 + i3 * s03;

    if (set_rows_stride != 0) {
        idst = i1 * s1 + j;
        idst += row_indices[i2] * set_rows_stride;
    }

    // RMS over the row: replicate rms_norm_f32's EXACT summation order for
    // bit-identical numerics — WARP_SIZE=16 lanes, each summing 8 elements
    // (col = l, l+16, ..., l+112) sequentially, then warp_reduce_sum over the
    // 16-lane subgroup. Lanes 16..63 contribute 0; the subgroup sums are
    // combined through a 4-entry local buffer.
    const int e0 = j < n_dims / 2 ? j : 2 * j;
    const int e1 = j < n_dims / 2 ? j + n_dims / 2 : 2 * j + 1;
    const float x0 = (float) x[ix - j + e0];
    const float x1 = (float) x[ix - j + e1];
    float scale = 1.0f;
    if (!no_norm) {
        const int lane = (int) item_ct1.get_local_id(1);
        float tmp = 0.0f;
        if (lane < WARP_SIZE) {
            const float * rowx = x + (ix - j);
            for (int c = lane; c < ne00; c += WARP_SIZE) {
                tmp += rowx[c] * rowx[c];
            }
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = (int) sg.get_group_linear_id();
        if (sg.get_local_linear_id() == 0) {
            s_sum[sg_id] = tmp;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        float total = 0.0f;
        for (int k = 0; k < 4; ++k) {
            total += s_sum[k];
        }
        const float mean = total / (float) ne00;
        scale = sycl::rsqrt(mean + eps);
    }
    const float n0 = !no_norm && norm_w != nullptr ? scale * x0 * norm_w[e0] : x0;
    const float n1 = !no_norm && norm_w != nullptr ? scale * x1 * norm_w[e1] : x1;

    if (j >= n_dims / 2) {
        // tail: copy elements (2j, 2j+1) unchanged (normed), matching rope_neox
        // (dst base is j, so dst[j + j] = element 2j).
        dst[idst + j + 0] = (D) n0;
        dst[idst + j + 1] = (D) n1;
        return;
    }

    const float theta_base = pos[i2] * dpct::pow(theta_scale, j);

    const float freq_factor = has_ff ? freq_factors[j] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    dst[idst + 0] = (D) (n0 * cos_theta - n1 * sin_theta);
    dst[idst + n_dims / 2] = (D) (n0 * sin_theta + n1 * cos_theta);
}

// [lx-norm-rope] launcher: neox-mode rope with an RMS_NORM(+MUL) pre-scale.
template <bool forward, typename T, typename D>
static void rope_neox_normed_sycl(const T *x, D *dst, const int ne00,
                                  const int ne01, const int ne02, const int s01,
                                  const int s02, const int s03, const int s1,
                                  const int s2, const int s3, const int n_dims,
                                  const int nr, const int32_t *pos,
                                  const float freq_scale, const float freq_base,
                                  const float ext_factor,
                                  const float attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float *freq_factors, const float eps,
                                  const float *norm_w,
                                  const int64_t *row_indices,
                                  const int set_rows_stride, const bool no_norm,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    GGML_ASSERT(ne00 == 128);  // one 64-lane workgroup per row (reduction)
    const int block_size = 64;
    const dpct::dim3 block_dims(1, block_size, 1);
    const int n_blocks_x = (ne00 + 2 * block_size - 1) / (2 * block_size);
    GGML_ASSERT(n_blocks_x == 1);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(4), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rope_neox_normed<forward, false>(
                        x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                        n_dims, pos, freq_scale, ext_factor, attn_factor,
                        corr_dims, theta_scale, freq_factors, eps, norm_w,
                        row_indices, set_rows_stride, no_norm,
                        s_sum_acc.get_pointer());
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(4), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rope_neox_normed<forward, true>(
                        x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
                        n_dims, pos, freq_scale, ext_factor, attn_factor,
                        corr_dims, theta_scale, freq_factors, eps, norm_w,
                        row_indices, set_rows_stride, no_norm,
                        s_sum_acc.get_pointer());
                });
        });
    }
}

// [lx-norm-rope] host entry: RMS_NORM -> MUL(scale) -> ROPE -> SET_ROWS in one
// launch. x = rms->src[0] (F32 pre-norm), w = the MUL weight (F32, ne[0]==ne00),
// dst = set_rows target (F16/F32). Replicates ggml_sycl_op_rope_impl param
// handling (NEOX-mode rope) plus the rms_norm_f32_sycl mul semantics.
void ggml_sycl_rope_norm_fused(ggml_backend_sycl_context &ctx,
                               ggml_tensor *rms, ggml_tensor *mul,
                               ggml_tensor *rope, ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/4);
    const ggml_tensor *x_src = rms->src[0];

    // [lx-norm-rope debug3] compute the norm via the ORIGINAL rms+mul kernel
    // (writing the real Kcur_normed buffer), then run MY rope kernel over that
    // buffer with no_norm=1: isolates the inline norm from the rope/scatter.
    static const bool dbg3 = []() {
        const char * e = getenv("GGML_SYCL_NR_DEBUG3");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (dbg3) {
        ggml_sycl_op_rms_norm_fused(ctx, rms, mul);
        x_src = mul;  // read the normed output buffer (shape [128,8])
    }

    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));

    const ggml_tensor *w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const float *norm_w = (const float *) w->data;

    // [lx-q-norm-rope] dst target: set_rows (K scatter) or the rope's own dst
    // (Q contiguous). Declared here because the debug print below uses it.
    const ggml_tensor * dst_t0 = set_rows != nullptr ? set_rows : rope;
    const ggml_type dst_type = dst_t0->type;

    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            const char * dbg = getenv("GGML_SYCL_DEBUG_NORM_ROPE");
            if (dbg != nullptr && std::atoi(dbg) != 0) {
                fprintf(stderr,
                        "[lx-norm-rope] dbg q=%d sr=%p x_n=%lld,%lld,%lld w_n=%lld,%lld,%lld src1=%p dst=%s\n",
                        set_rows == nullptr ? 1 : 0, (void*) set_rows,
                        (long long) x_src->ne[0], (long long) x_src->ne[1],
                        (long long) x_src->ne[2],
                        (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                        (void*) rope->src[1], ggml_type_name(dst_type));
            }
        }
    }

    // [lx-q-norm-rope] When set_rows == nullptr the fused kernel writes the
    // rope's own dst contiguously (the Q path: no view/set_rows scatter). The
    // kernel's set_rows_stride==0 branch then uses plain row-major idst, which
    // is exactly the original rope_neox non-scatter indexing.
    const float *src0_d = (const float *) x_src->data;
    const ggml_tensor * dst_t = set_rows != nullptr ? set_rows : rope;
    void *dst_d = dst_t->data;
    const int64_t *row_indices =
        set_rows != nullptr ? (const int64_t *) set_rows->src[1]->data : nullptr;
    const int set_rows_stride = set_rows != nullptr
        ? (int) (set_rows->nb[1] / ggml_type_size(set_rows->type))
        : 0;

    // Rope operates on the KV-head slice of the normed tensor (GQA: 8 heads
    // here), NOT on the full 48-head norm input. Use the ROPE input's dims for
    // the grid (nr, ne01, ne02) and the RMS input's stride for reads.
    const ggml_tensor * r_src = rope->src[0];
    const int64_t ne00 = r_src->ne[0];
    const int64_t ne01 = r_src->ne[1];
    const int64_t ne02 = r_src->ne[2];
    const int64_t ne03 = r_src->ne[3];
    const int64_t nr = ggml_nrows(r_src);

    const size_t s01 = x_src->nb[1] / ggml_type_size(x_src->type);
    const size_t s02 = x_src->nb[2] / ggml_type_size(x_src->type);
    const size_t s03 = x_src->nb[3] / ggml_type_size(x_src->type);
    // strides of the ROPE node's output, divided by the ROPE's own element size
    // (F32), exactly like ggml_sycl_op_rope_impl line 498 — the cache head
    // stride is in F32-scaled units, NOT the f16 cache element size.
    const size_t s1 = rope->nb[1] / ggml_type_size(rope->type);
    const size_t s2 = rope->nb[2] / ggml_type_size(rope->type);
    const size_t s3 = rope->nb[3] / ggml_type_size(rope->type);

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base, (const int32_t *) rope->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *) rope->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (const int32_t *) rope->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (const int32_t *) rope->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (const int32_t *) rope->op_params + 10, sizeof(float));

    const int32_t *pos = (const int32_t *) rope->src[1]->data;
    const float *freq_factors =
        rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    dpct::queue_ptr stream = ctx.stream();

    // [lx-norm-rope debug2] route through the two original launches (rms+mul
    // fused, then rope+set_rows fused) to isolate graph-consumption vs kernel.
    {
        static const bool dbg2 = []() {
            const char * e = getenv("GGML_SYCL_NR_DEBUG2");
            return e != nullptr && std::atoi(e) != 0;
        }();
        if (dbg2) {
            const ggml_tensor * o_src0 = rope->src[0];
            fprintf(stderr,
                    "[lx-norm-rope] dbg2 src0_n=%lld,%lld,%lld,%lld nr=%lld s01=%zu s1=%zu "
                    "sstride=%zu w_n=%lld,%lld,%lld pos_p=%d row_p=%d\n",
                    (long long) o_src0->ne[0], (long long) o_src0->ne[1],
                    (long long) o_src0->ne[2], (long long) o_src0->ne[3],
                    (long long) ggml_nrows(o_src0),
                    o_src0->nb[1] / ggml_type_size(o_src0->type),
                    rope->nb[1] / ggml_type_size(set_rows->type),
                    set_rows->nb[1] / ggml_type_size(set_rows->type),
                    (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                    rope->src[1] != nullptr,
                    set_rows->src[1] != nullptr);
            ggml_sycl_op_rms_norm_fused(ctx, rms, mul);
            ggml_sycl_op_rope_impl<true>(ctx, rope, set_rows);
            return;
        }
    }

    static const bool no_norm = []() {
        const char * e = getenv("GGML_SYCL_NR_NO_NORM");
        return e != nullptr && std::atoi(e) != 0;
    }();
    const bool skip_norm = no_norm || dbg3;

    if (dst_type == GGML_TYPE_F16) {
        rope_neox_normed_sycl<true, float, sycl::half>(
            src0_d, (sycl::half *) dst_d, (int) ne00, (int) ne01, (int) ne02,
            (int) s01, (int) s02, (int) s03, (int) s1, (int) s2, (int) s3,
            n_dims, (int) nr, pos, freq_scale, freq_base, ext_factor,
            attn_factor, corr_dims, freq_factors, eps, norm_w, row_indices,
            set_rows_stride, skip_norm, stream);
    } else {
        GGML_ASSERT(dst_type == GGML_TYPE_F32);
        rope_neox_normed_sycl<true, float, float>(
            src0_d, (float *) dst_d, (int) ne00, (int) ne01, (int) ne02,
            (int) s01, (int) s02, (int) s03, (int) s1, (int) s2, (int) s3,
            n_dims, (int) nr, pos, freq_scale, freq_base, ext_factor,
            attn_factor, corr_dims, freq_factors, eps, norm_w, row_indices,
            set_rows_stride, skip_norm, stream);
    }
}

// [lx-qk-rope] Per-chain parameters for the merged Q+K normed-rope launch.
struct lx_rope_chain_params {
    const float * x;                 // pre-norm input (F32)
    const void *  dst;               // Q: rope's own dst (contiguous); K: set_rows target
    int ne00, ne01, ne02;            // rope input dims (chain-local rows)
    int s01, s02, s03;               // x strides in elements
    int s1, s2, s3;                  // rope-output strides in rope-element units (F32-scaled)
    const float * norm_w;            // MUL weight vector
    const int64_t * row_indices;     // K set_rows indices (nullptr for Q)
    int set_rows_stride;             // 0 for Q (contiguous write)
    int nr;                          // rows this chain owns (grid groups)
};

// [lx-qk-rope] Merged kernel: rows [0, q.nr) run the Q chain (contiguous dst,
// set_rows_stride == 0), rows [q.nr, q.nr+k.nr) run the K chain (scatter).
// The per-row body is the same rope_neox_normed logic; each chain keeps its own
// norm weight, strides, and scatter state, so numerics are identical to the two
// separate launches.
template <bool forward, bool has_ff, typename T, typename DQ, typename DK>
static void rope_neox_normed_qk(
    const lx_rope_chain_params & q, const lx_rope_chain_params & k,
    const int n_dims, const int32_t *pos, const float freq_scale,
    const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale,
    const float *freq_factors, const float eps, const bool no_norm,
    float * s_sum, const sycl::nd_item<3> & item_ct1) {
    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);
    const bool is_q = row_dst < q.nr;
    const lx_rope_chain_params & p = is_q ? q : k;
    const int row = is_q ? row_dst : row_dst - q.nr;

    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));
    if (i0 >= p.ne00) {
        return;
    }

    const uint32_t i3 = row / (p.ne01 * p.ne02);
    const uint32_t i2 = (row - i3 * p.ne01 * p.ne02) / p.ne01;
    const uint32_t i1 = row - i3 * p.ne01 * p.ne02 - i2 * p.ne01;

    const int j = i0 / 2;
    int idst = j + i1 * p.s1 + i2 * p.s2 + i3 * p.s3;
    const int ix = j + i1 * p.s01 + i2 * p.s02 + i3 * p.s03;

    if (p.set_rows_stride != 0) {
        idst = i1 * p.s1 + j;
        idst += p.row_indices[i2] * p.set_rows_stride;
    }

    // RMS over the row: replicate rms_norm_f32's EXACT summation order
    // (WARP_SIZE=16 lanes, 8 elements each, warp_reduce_sum, 4-entry combine).
    const int e0 = j < n_dims / 2 ? j : 2 * j;
    const int e1 = j < n_dims / 2 ? j + n_dims / 2 : 2 * j + 1;
    const float x0 = (float) p.x[ix - j + e0];
    const float x1 = (float) p.x[ix - j + e1];
    float scale = 1.0f;
    if (!no_norm) {
        const int lane = (int) item_ct1.get_local_id(1);
        float tmp = 0.0f;
        if (lane < WARP_SIZE) {
            const float * rowx = p.x + (ix - j);
            for (int c = lane; c < p.ne00; c += WARP_SIZE) {
                tmp += rowx[c] * rowx[c];
            }
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = (int) sg.get_group_linear_id();
        if (sg.get_local_linear_id() == 0) {
            s_sum[sg_id] = tmp;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        float total = 0.0f;
        for (int kk = 0; kk < 4; ++kk) {
            total += s_sum[kk];
        }
        const float mean = total / (float) p.ne00;
        scale = sycl::rsqrt(mean + eps);
    }
    const float n0 = !no_norm && p.norm_w != nullptr ? scale * x0 * p.norm_w[e0] : x0;
    const float n1 = !no_norm && p.norm_w != nullptr ? scale * x1 * p.norm_w[e1] : x1;

    if (j >= n_dims / 2) {
        // tail: copy elements (2j, 2j+1) unchanged (normed)
        if (is_q) {
            DQ * dst = (DQ *) p.dst;
            dst[idst + j + 0] = (DQ) n0;
            dst[idst + j + 1] = (DQ) n1;
        } else {
            DK * dst = (DK *) p.dst;
            dst[idst + j + 0] = (DK) n0;
            dst[idst + j + 1] = (DK) n1;
        }
        return;
    }

    const float theta_base = pos[i2] * dpct::pow(theta_scale, j);
    const float freq_factor = has_ff ? freq_factors[j] : 1.0f;
    float cos_theta, sin_theta;
    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    if (is_q) {
        DQ * dst = (DQ *) p.dst;
        dst[idst + 0] = (DQ) (n0 * cos_theta - n1 * sin_theta);
        dst[idst + n_dims / 2] = (DQ) (n0 * sin_theta + n1 * cos_theta);
    } else {
        DK * dst = (DK *) p.dst;
        dst[idst + 0] = (DK) (n0 * cos_theta - n1 * sin_theta);
        dst[idst + n_dims / 2] = (DK) (n0 * sin_theta + n1 * cos_theta);
    }
}

// [lx-qk-rope] launcher: one submission covering both chains.
template <bool forward, typename T, typename DQ, typename DK>
static void rope_neox_normed_qk_sycl(
    const lx_rope_chain_params & q, const lx_rope_chain_params & k,
    const int n_dims, const int32_t *pos, const float freq_scale,
    const float freq_base, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float *freq_factors,
    const float eps, const bool no_norm, dpct::queue_ptr stream) {
    GGML_ASSERT(q.ne00 == 128 && k.ne00 == 128);
    const int block_size = 64;
    const dpct::dim3 block_dims(1, block_size, 1);
    const int nr_total = q.nr + k.nr;
    const dpct::dim3 block_nums(nr_total, 1, 1);
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    if (freq_factors == nullptr) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(4), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rope_neox_normed_qk<forward, false, T, DQ, DK>(
                        q, k, n_dims, pos, freq_scale, ext_factor, attn_factor,
                        corr_dims, theta_scale, freq_factors, eps, no_norm,
                        s_sum_acc.get_pointer(), item_ct1);
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(4), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rope_neox_normed_qk<forward, true, T, DQ, DK>(
                        q, k, n_dims, pos, freq_scale, ext_factor, attn_factor,
                        corr_dims, theta_scale, freq_factors, eps, no_norm,
                        s_sum_acc.get_pointer(), item_ct1);
                });
        });
    }
}

// [lx-qk-rope] host entry: merged Q+K normed rope in one launch. Q chain writes
// rope_q's own dst contiguously (set_rows == nullptr semantics); K chain writes
// the cache via set_rows_k scatter. Shared rope params (position, frequencies)
// are read from the Q chain; the two chains' eps must agree.
void ggml_sycl_rope_norm_fused_qk(ggml_backend_sycl_context &ctx,
                                  ggml_tensor *rms_q, ggml_tensor *mul_q, ggml_tensor *rope_q,
                                  ggml_tensor *rms_k, ggml_tensor *mul_k, ggml_tensor *rope_k,
                                  ggml_tensor *view_k, ggml_tensor *set_rows_k) {
    GGML_UNUSED(view_k);
    auto chain_params = [](ggml_tensor * rms, ggml_tensor * mul, ggml_tensor * rope,
                           ggml_tensor * set_rows) -> lx_rope_chain_params {
        const ggml_tensor * x_src = rms->src[0];
        const ggml_tensor * dst_t = set_rows != nullptr ? set_rows : rope;
        const ggml_tensor * r_src = rope->src[0];
        const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        lx_rope_chain_params p;
        p.x = (const float *) x_src->data;
        p.dst = dst_t->data;
        p.ne00 = (int) r_src->ne[0];
        p.ne01 = (int) r_src->ne[1];
        p.ne02 = (int) r_src->ne[2];
        p.s01 = (int) (x_src->nb[1] / ggml_type_size(x_src->type));
        p.s02 = (int) (x_src->nb[2] / ggml_type_size(x_src->type));
        p.s03 = (int) (x_src->nb[3] / ggml_type_size(x_src->type));
        // rope-output strides in the ROPE's own element size (F32-scaled),
        // exactly like ggml_sycl_op_rope_impl and the K-fuse host entry.
        p.s1 = (int) (rope->nb[1] / ggml_type_size(rope->type));
        p.s2 = (int) (rope->nb[2] / ggml_type_size(rope->type));
        p.s3 = (int) (rope->nb[3] / ggml_type_size(rope->type));
        p.norm_w = (const float *) w->data;
        p.row_indices =
            set_rows != nullptr ? (const int64_t *) set_rows->src[1]->data : nullptr;
        p.set_rows_stride = set_rows != nullptr
            ? (int) (set_rows->nb[1] / ggml_type_size(set_rows->type))
            : 0;
        p.nr = (int) ggml_nrows(r_src);
        return p;
    };
    lx_rope_chain_params q = chain_params(rms_q, mul_q, rope_q, nullptr);
    lx_rope_chain_params k = chain_params(rms_k, mul_k, rope_k, set_rows_k);

    float eps_q = 0.0f;
    float eps_k = 0.0f;
    memcpy(&eps_q, rms_q->op_params, sizeof(float));
    memcpy(&eps_k, rms_k->op_params, sizeof(float));
    GGML_ASSERT(eps_q == eps_k);
    const int n_dims_q = ((const int32_t *) rope_q->op_params)[1];
    const int n_dims_k = ((const int32_t *) rope_k->op_params)[1];
    GGML_ASSERT(n_dims_q == n_dims_k);
    const int n_ctx_orig = ((const int32_t *) rope_q->op_params)[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base, (const int32_t *) rope_q->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *) rope_q->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (const int32_t *) rope_q->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope_q->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (const int32_t *) rope_q->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (const int32_t *) rope_q->op_params + 10, sizeof(float));
    const int32_t *pos = (const int32_t *) rope_q->src[1]->data;
    const float *freq_factors =
        rope_q->src[2] != nullptr ? (const float *) rope_q->src[2]->data : nullptr;
    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims_q, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    const ggml_type dst_type_q = rope_q->type;
    const ggml_type dst_type_k = set_rows_k->type;
    const bool no_norm = false;
    dpct::queue_ptr stream = ctx.stream();
    if (dst_type_q == GGML_TYPE_F32 && dst_type_k == GGML_TYPE_F16) {
        rope_neox_normed_qk_sycl<true, float, float, sycl::half>(
            q, k, n_dims_q, pos, freq_scale, freq_base, ext_factor, attn_factor,
            corr_dims, freq_factors, eps_q, no_norm, stream);
    } else if (dst_type_q == GGML_TYPE_F32 && dst_type_k == GGML_TYPE_F32) {
        rope_neox_normed_qk_sycl<true, float, float, float>(
            q, k, n_dims_q, pos, freq_scale, freq_base, ext_factor, attn_factor,
            corr_dims, freq_factors, eps_q, no_norm, stream);
    } else if (dst_type_q == GGML_TYPE_F16 && dst_type_k == GGML_TYPE_F16) {
        rope_neox_normed_qk_sycl<true, float, sycl::half, sycl::half>(
            q, k, n_dims_q, pos, freq_scale, freq_base, ext_factor, attn_factor,
            corr_dims, freq_factors, eps_q, no_norm, stream);
    } else {
        GGML_ASSERT(false && "lx-qk-rope: unhandled Q/K dst type combo");
    }
}
