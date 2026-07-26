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

#include <algorithm>
#include <chrono>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <optional>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
#include <level_zero/ze_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_USE_VMM
#endif
#include <sycl/half_type.hpp>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_disable_optimize = 0;
int g_ggml_sycl_disable_graph = 0;
int g_ggml_sycl_disable_dnn = 0;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_enable_level_zero = 0;
int g_ggml_sycl_enable_flash_attention = 1;


static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    info.device_count = dpct::dev_mgr::instance().device_count();
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#if !defined(GGML_SYCL_USE_VMM)
        info.devices[i].vmm = 0;
#else
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }
    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_disable_optimize = get_sycl_env("GGML_SYCL_DISABLE_OPT", 0);
        g_ggml_sycl_disable_graph = get_sycl_env("GGML_SYCL_DISABLE_GRAPH", 1);
        g_ggml_sycl_disable_dnn = get_sycl_env("GGML_SYCL_DISABLE_DNN", 0);
        g_ggml_sycl_enable_vmm = get_sycl_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_prioritize_dmmv = get_sycl_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
        g_ggml_sycl_enable_level_zero = get_sycl_env("GGML_SYCL_ENABLE_LEVEL_ZERO", ggml_sycl_info().ext_oneapi_level_zero);
#else
        g_ggml_sycl_enable_level_zero = 0;
#endif

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = get_sycl_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif
#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif
#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO: no\n");
#endif
#if defined(GGML_SYCL_USE_VMM)
        GGML_LOG_INFO("  GGML_SYCL_USE_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_VMM: no\n");
#endif

        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_OPT: %d\n", g_ggml_sycl_disable_optimize);
#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: %d\n", g_ggml_sycl_disable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: graph disabled by compile flag\n");
#endif
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_LEVEL_ZERO: %d\n", g_ggml_sycl_enable_level_zero);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_LEVEL_ZERO: Level Zero disabled by compile flag\n");
#endif
#if GGML_SYCL_DNNL
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: %d\n", g_ggml_sycl_disable_dnn);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: DNN disabled by compile flag\n");
#endif
#if defined(GGML_SYCL_USE_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);
        g_ggml_sycl_use_async_mem_op_requested = get_sycl_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || !g_ggml_sycl_disable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream) :
        device(device), dev_ptr(dev_ptr), stream(stream) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (!g_ggml_sycl_disable_optimize) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:{
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
static bool ggml_sycl_is_l0_discrete_gpu(sycl::queue &q) {
    if (!q.get_device().is_gpu() || q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        return false;
    }

    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_device_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
    return r == ZE_RESULT_SUCCESS && !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
}
#endif

static void dev2dev_memcpy(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
    // Use Level Zero direct copy for dGPU-to-dGPU transfers.
    const bool l0_copy_supported =
        ggml_sycl_is_l0_discrete_gpu(q_dst) && ggml_sycl_is_l0_discrete_gpu(q_src);
    if (g_ggml_sycl_enable_level_zero && l0_copy_supported) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
        auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
        ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                           0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_list_handle_t cl;
        ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
        if (r == ZE_RESULT_SUCCESS) {
            r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
            zeCommandListDestroy(cl);
            if (r == ZE_RESULT_SUCCESS) {
                return;
            }
        }
    }
#endif
    // Host-staged copy
    char *host_buf = (char *)malloc(size);
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(*stream_dst, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0

    void * dev_ptr;
    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream)));
    if (!dev_ptr) {
      GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device\n", __func__, size);
      return nullptr;
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return dpct::get_current_device().get_max_mem_alloc_size();

    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    ctx->streams.push_back(&(dpct::get_current_device().default_queue()));

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %lu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free_aligned_mem_host((void *)buffer->context);
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = aligned_malloc_host(TENSOR_ALIGNMENT, size);
    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%lu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_USE_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_USE_VMM)

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    inline static int counter{ 0 };

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
        counter = 0;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            ggml_sycl_buffer b               = buffer_pool[0];
            void *           ptr             = b.ptr;
            *actual_size                     = b.size;
            counter                          = 1;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];

        if (b.ptr == nullptr) {
            void * ptr;

            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size += size;
            *actual_size = size;
            counter      = counter + 1;
            return ptr;
        } else {
            ++counter;
            b.size = size;
            return b.ptr;
        }
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise do nothing, pointers will be freed once the pool is deallocated.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
#if defined(GGML_SYCL_USE_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_USE_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}


std::unique_ptr<ggml_sycl_fattn_kv_buffers> ggml_backend_sycl_context::new_fattn_kv_buffers(queue_ptr qptr, int device) {
    return std::unique_ptr<ggml_sycl_fattn_kv_buffers>(new ggml_sycl_fattn_kv_buffers(qptr, device));
}

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


template <typename Ti, typename To>
static  void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if (idx >= parallel_elements) {
            return;
        }

        const int I_HW = ih * iw;
        const int O_HW = oh * ow;
        const int nc = idx / O_HW;
        const int cur_oh = idx % O_HW / ow;
        const int cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const int start_h = cur_oh * sh - ph;
        const int bh = sycl::max(0, start_h);
        const int eh = sycl::min(ih, start_h + kh);
        const int start_w = cur_ow * sw - pw;
        const int bw = sycl::max(0, start_w);
        const int ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                break;
        }

        for (int i = bh; i < eh; i += 1) {
            for (int j = bw; j < ew; j += 1) {
#if DPCT_COMPATIBILITY_TEMP >= 350
                /*
                DPCT1098:106: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                Ti cur = *(i_ptr + i * iw + j);
#else
                Ti cur = i_ptr[i * iw + j];
#endif
                switch (op) {
                    case GGML_OP_POOL_AVG: res += (cur / (kh * kw)); break;
                    case GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); break;
                    default:
                        res = (To) sycl::nan(uint32_t(0));
                        break;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);
    GGML_ASSERT(shared_mem<=ggml_sycl_info().devices[device].smpbo);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

static void top_k_f32_sycl(
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>(block_size * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>(block_size * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);
                const int tid = item_ct1.get_local_id(0);

                if (row >= nrows) return;

                const float * src_row = src + row * ncols;
                int32_t * dst_idx_row = dst_indices + row * k;

                float local_vals[32];
                int local_idx[32];

                for (int i = 0; i < k; i++) {
                    local_vals[i] = -FLT_MAX;
                    local_idx[i] = -1;
                }

                for (int col = tid; col < ncols; col += block_size) {
                    float val = src_row[col];

                    if (val > local_vals[k-1]) {
                        int pos = k - 1;
                        while (pos > 0 && val > local_vals[pos - 1]) {
                            pos--;
                        }

                        for (int i = k - 1; i > pos; i--) {
                            local_vals[i] = local_vals[i - 1];
                            local_idx[i] = local_idx[i - 1];
                        }
                        local_vals[pos] = val;
                        local_idx[pos] = col;
                    }
                }

                for (int i = 0; i < k; i++) {
                    shared_vals[tid * k + i] = local_vals[i];
                    shared_idx[tid * k + i] = local_idx[i];
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);

                if (tid == 0) {
                    float final_vals[32];
                    int final_idx[32];

                    for (int i = 0; i < k; i++) {
                        final_vals[i] = -FLT_MAX;
                        final_idx[i] = -1;
                    }

                    for (int t = 0; t < block_size; t++) {
                        for (int i = 0; i < k; i++) {
                            float val = shared_vals[t * k + i];
                            int idx = shared_idx[t * k + i];

                            if (val > final_vals[k-1]) {
                                int pos = k - 1;
                                while (pos > 0 && val > final_vals[pos - 1]) {
                                    pos--;
                                }

                                for (int j = k - 1; j > pos; j--) {
                                    final_vals[j] = final_vals[j - 1];
                                    final_idx[j] = final_idx[j - 1];
                                }
                                final_vals[pos] = val;
                                final_idx[pos] = idx;
                            }
                        }
                    }

                    for (int i = 0; i < k; i++) {
                        dst_idx_row[i] = final_idx[i];
                    }

                    if (k > 1) {
                        int32_t temp = dst_idx_row[0];
                        dst_idx_row[0] = dst_idx_row[1];
                        dst_idx_row[1] = temp;
                    }
                }
            });
    });
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                if (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && !g_ggml_sycl_disable_dnn && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

#if GGML_SYCL_DNNL
        if (!g_ggml_sycl_disable_dnn) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols);

            const sycl::half alpha_f16 = 1.0f;
            const sycl::half beta_f16  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta_f16,
                dst_f16.get(), dpct::library_data_t::real_half, ldc,
                dpct::library_data_t::real_half)));
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting dst to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16, dst);
            to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        {
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            // oneDNN's FP32 router GEMM is not bit-reproducible on Intel B70 across
            // repeated graph executions. Tiny router drift can change the selected
            // experts and amplify into large output differences. Keep this semantic
            // projection device-resident, but use oneMKL's reproducible FP32 path.
            //
            // For non-router GEMMs on B70 MoE, prefer oneDNN for mid/large shapes:
            // measured Qwen3.6-35B-A3B Q5_K_XL pp512 +~1.3% vs native MKL when
            // GGML_SYCL_DISABLE_DNN is unset (ship speed path). Default FLOP cutoff
            // is 128^3 so only tiny mats stay on the direct MKL path. Heavy-push
            // dense MUL_MAT A/B: override with GGML_SYCL_MKL_FLOP_CUTOFF=<int64 flops>
            // (e.g. 262144=64^3, 0=all non-router oneDNN). Router always MKL.
            static const int64_t mkl_flop_cutoff = []() {
                const char * env = getenv("GGML_SYCL_MKL_FLOP_CUTOFF");
                if (env != nullptr && env[0] != '\0') {
                    return (int64_t) std::atoll(env);
                }
                return (int64_t) 128 * 128 * 128;
            }();
            const bool is_moe_router = src0->type == GGML_TYPE_F32 &&
                    std::strstr(src0->name, ".ffn_gate_inp.weight") != nullptr;
            const bool use_mkl_direct = is_moe_router || gemm_flops < mkl_flop_cutoff;
            // Diagnostic: TREEBEARD_SYCL_GEMM_ROUTE=1 logs FP32 dense GEMM backend choice
            // every 256 calls (stderr). Used by heavy-push-1 dense MUL_MAT excavation.
            static const bool gemm_route_log = []() {
                const char * env = getenv("TREEBEARD_SYCL_GEMM_ROUTE");
                return env != nullptr && std::atoi(env) != 0;
            }();
            if (gemm_route_log) {
                static std::atomic<long> route_n { 0 };
                const long n = route_n.fetch_add(1, std::memory_order_relaxed);
                if ((n % 256) == 0) {
                    const char * backend = (is_moe_router) ? "mkl-router"
                        : (g_ggml_sycl_disable_dnn || use_mkl_direct) ? "mkl" : "onednn";
                    fprintf(stderr,
                            "[treebeard-gemm-route] n=%ld backend=%s flops=%lld m=%lld n_cols=%lld k=%lld "
                            "cutoff=%lld name=%s\n",
                            n, backend, (long long) gemm_flops, (long long) row_diff,
                            (long long) src1_ncols, (long long) ne10, (long long) mkl_flop_cutoff,
                            src0->name[0] ? src0->name : "?");
                }
            }
#if GGML_SYCL_DNNL
            if (!g_ggml_sycl_disable_dnn && !use_mkl_direct) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;
    const int num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device);
}

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0 && k <= 32);
    GGML_ASSERT(k <= ncols);

    top_k_f32_sycl(src0_dd, dst_dd, ncols, nrows, k, main_stream);
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int nrows0 = ggml_nrows(dst->src[0]);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                            src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(*stream, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
    const size_t       type_size_src0 = ggml_type_size(src0->type);
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
#if GGML_SYCL_DNNL
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_disable_dnn) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            if (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // The DP4A-tiled MMQ GEMM (mmq.cpp) was disabled upstream with an unquantified
    // "accuracy issues" note. Keep this opt-in and limited to Qwen3.6 Q5_K_XL's
    // tensor formats; broad enablement still fails MUL_MAT (q4_0 large shape -> NaN).
    // GGML_SYCL_ENABLE_MMQ=1 turns it on; accuracy is gated by test-backend-ops MUL_MAT.
    static const bool enable_mmq = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_MMQ");
        return env != nullptr && atoi(env) != 0;
    }();
    if (!enable_mmq) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        return syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0* x = (const block_q8_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q3_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q3_K) == 0);

    const int nblocks = size / sizeof(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q3_K * x  = (const block_q3_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        for (int j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q5_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q5_K) == 0);

    const int nblocks = size / sizeof(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q5_K * x  = (const block_q5_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        default:
            return false;
    }
}

static bool ggml_sycl_mmvq_12col_disabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMVQ_12COL");
        return env != nullptr && atoi(env) != 0;
    }();
    return disabled;
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    const bool supported_ncols = dst->src[1]->ne[1] <= 8 ||
        (!ggml_sycl_mmvq_12col_disabled() && dst->src[1]->ne[1] == 12 && dst->src[0]->type == GGML_TYPE_Q8_0);
    return !g_ggml_sycl_disable_optimize && //allow optimize, controlled by $GGML_SYCL_DISABLE_OPT
            ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
            dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
            // Q8_0 also has a direct 12-column kernel.
            supported_ncols && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (g_ggml_sycl_disable_optimize || !ctx->opt_feature.reorder) {
        return;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Wide-batch: past the 8-column kernel templates, chunk-of-8 MMVQ still beats the
    // dequantize+GEMM fallback for small token counts (batched decode at np<=32) — the
    // fallback streams the dequantized f16 weights per op.
    // GGML_SYCL_DISABLE_MMVQ_WIDE_BATCH=1 restores the old ne11<=8 cap.
    static const bool disable_wide_batch = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMVQ_WIDE_BATCH");
        return env != nullptr && atoi(env) != 0;
    }();
    const int64_t max_batch = disable_wide_batch ? MMVQ_MAX_BATCH_SIZE : MMVQ_MAX_BATCH_SIZE_WIDE;
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= max_batch;
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    const bool supports_mmq = ggml_sycl_supports_mmq(src0->type);
    bool use_mul_mat_q = supports_mmq && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    // The XMX batch cap starves MMQ (MMVQ already owns ne11<=32), leaving prefill on the
    // f16 dequant GEMM. When MMQ is explicitly enabled, drop the cap so the tiled int8 GEMM
    // takes the large-ne11 dense path (prefill / big batch); MMVQ still wins ne11<=32 by
    // dispatch precedence below.
    if (!supports_mmq) {
        use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
    }
#endif // SYCL_USE_XMX

    // Dispatch becomes obscure with the reorder, MMVQ when the reorder optimization
    // is enabled takes precedence over DMMV, the current if-else implementation
    // requires disabling DMMV if both conditions are met

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
      // Arc770 get benefit with Q4_0 by skipping it.
      if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                gpu_arch::intel_gpu_acm_g10 &&
            src0->type == GGML_TYPE_Q4_0)) {
        use_dequantize_mul_mat_vec =
            use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
      }
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q_wide);
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q_wide);
        }
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}


struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// A/B gate for the batched fused MUL_MAT_ID extension: =1 restores the old behavior
// (fused only for ne12==1, sorted host path otherwise).
static bool ggml_sycl_mmid_fused_batch_disabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMID_FUSED_BATCH");
        return env != nullptr && atoi(env) != 0;
    }();
    return disabled;
}

// True when ggml_sycl_mul_mat_id is guaranteed to take the fused device-routed path for this
// node — i.e. no blocking host sync, which makes MUL_MAT_ID safe inside a SYCL graph record.
// Must stay in lockstep with the bail checks in ggml_sycl_mul_mat_id_mmvq_fused and the type
// switches in ggml_sycl_mul_mat_vec_q_id[_reorder] (reorder is decided lazily at exec time, so
// a type is only eligible if BOTH switches dispatch it — Q4_K/Q5_K/Q6_K may go either way,
// everything else always takes the plain switch because opt_for_reorder_id skips it).
static bool ggml_sycl_mul_mat_id_fused_eligible(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    if (ne12 < 1 || ne12 > 64) return false;
    if (ne12 > 1 && ggml_sycl_mmid_fused_batch_disabled()) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;
    if (ids->ne[1] != ne12) return false;
    if (ids->nb[0] != sizeof(int32_t)) return false;
    if (ne11 != 1 && ne11 != ids->ne[0]) return false;
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    // Batched decode (ne12 tokens) stays fused: one launch, ids read on device — no per-layer
    // host sync / per-expert launch loop. Above the cap, expert reuse across tokens makes the
    // sorted contiguous-GEMM path win (GEMV re-reads expert weights per routed row).
    if (ne12 < 1 || ne12 > 64) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;

    const int64_t n_ids_per_group = ids->ne[0];
    if (ids->ne[1] != ne12) return false;
    if (ids->nb[0] != sizeof(int32_t)) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    // Lazily reorder the (Q4_K) expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * ne12 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) (ne11 * ne12),
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) (ne11 * ne12),
            src1_padded_cols, stream);
    }

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride   = (ne11 == 1) ? 0 : bytes_per_qrow;
    const size_t src1_token_stride = (size_t) ne11 * bytes_per_qrow;
    const int    ids_row_stride    = (int) (ids->nb[1] / sizeof(int32_t));

    // Grouped phase-2 path: for multi-token batches, bucket the routed (token, slot) pairs by
    // expert on device and read each distinct expert's weights once. Below 4 tokens collisions
    // are rare enough that the extra routing launch isn't worth it.
    // A/B gate: GGML_SYCL_DISABLE_MMID_GROUPED=1 keeps the per-pair batched kernel.
    static const bool disable_grouped = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMID_GROUPED");
        return env != nullptr && atoi(env) != 0;
    }();
    const int64_t n_as = src0->ne[2];
    if (use_reorder && !disable_grouped && ne12 >= 4) {
        ggml_sycl_pool_alloc<int32_t> mmid_scratch(ctx.pool(),
            (size_t) (2 * n_as + 2 + ne12 * n_experts_used));
        if (ggml_sycl_mul_mat_vec_q_id_grouped_reorder(
                src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
                mmid_scratch.get(), (float *) dst->data, (int) ne10, nrows, n_experts_used,
                (int) n_as,
                /*expert_weight_stride=*/ src0->nb[2],
                /*dst_row_stride=*/ dst->nb[1],
                src1_row_stride, (int) ne12, ids_row_stride,
                src1_token_stride, /*dst_token_stride=*/ dst->nb[2], stream)) {
            return true;
        }
    }

    if (use_reorder) {
        return ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
            (float *) dst->data, (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, (int) ne12, ids_row_stride,
            src1_token_stride, /*dst_token_stride=*/ dst->nb[2], stream);
    }
    return ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
        (float *) dst->data, (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, (int) ne12, ids_row_stride,
        src1_token_stride, /*dst_token_stride=*/ dst->nb[2], stream);
}

// Production Qwen3.6 gate/up path. The model stores two separate expert
// matrices, but both MMIDs consume the same routed activations and ids. Compute
// matching rows together and apply SwiGLU before storage so neither MMID result
// is materialized and no standalone GLU kernel is submitted.
static bool ggml_sycl_mul_mat_id_dual_swiglu_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * gate,
    const ggml_tensor * up, ggml_tensor * glu) {
    const ggml_tensor * gate_weights = gate->src[0];
    const ggml_tensor * up_weights   = up->src[0];
    const ggml_tensor * src1         = gate->src[1];
    const ggml_tensor * ids          = gate->src[2];
    static const bool trace_reject = []() {
        const char * env = getenv("GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();

    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
        gate->src[1] != up->src[1] || gate->src[2] != up->src[2] ||
        gate_weights->type != up_weights->type ||
        (gate_weights->type != GGML_TYPE_Q5_K &&
         gate_weights->type != GGML_TYPE_Q6_K) ||
        src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        if (trace_reject) {
            fprintf(stderr, "[treebeard-moe-dual-swiglu] dispatcher=type-reject\n");
        }
        return false;
    }

    const int64_t ncols    = src1->ne[0];
    const int64_t nslots   = src1->ne[1];
    const int64_t n_tokens = src1->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t nrows    = gate_weights->ne[1];
    const int64_t n_as     = gate_weights->ne[2];
    if (n_tokens < 1 || n_tokens > 64 || ncols != gate_weights->ne[0] ||
        ncols != up_weights->ne[0] || ncols % QK8_1 != 0 ||
        up_weights->ne[1] != nrows || up_weights->ne[2] != n_as ||
        ids->ne[1] != n_tokens || ids->nb[0] != sizeof(int32_t) ||
        (nslots != 1 && nslots != n_used) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(glu) || gate->ne[0] != nrows ||
        up->ne[0] != nrows || gate->ne[1] != n_used ||
        up->ne[1] != n_used || gate->ne[2] != n_tokens ||
        up->ne[2] != n_tokens || !ggml_are_same_shape(gate, up) ||
        !ggml_are_same_shape(gate, glu)) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-moe-dual-swiglu] dispatcher=shape-reject"
                    " ncols=%" PRId64 " nslots=%" PRId64
                    " tokens=%" PRId64 " used=%" PRId64
                    " rows=%" PRId64 " experts=%" PRId64 "\n",
                    ncols, nslots, n_tokens, n_used, nrows, n_as);
        }
        return false;
    }

    opt_for_reorder_id(&ctx, gate_weights);
    opt_for_reorder_id(&ctx, up_weights);
    const ggml_tensor_extra_gpu * gate_extra =
        static_cast<const ggml_tensor_extra_gpu *>(gate_weights->extra);
    const ggml_tensor_extra_gpu * up_extra =
        static_cast<const ggml_tensor_extra_gpu *>(up_weights->extra);
    if (!gate_extra || !up_extra ||
        !gate_extra->optimized_feature.reorder ||
        !up_extra->optimized_feature.reorder) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-moe-dual-swiglu] dispatcher=reorder-reject"
                    " gate_extra=%d gate_reorder=%d up_extra=%d up_reorder=%d\n",
                    gate_extra != nullptr,
                    gate_extra != nullptr && gate_extra->optimized_feature.reorder,
                    up_extra != nullptr,
                    up_extra != nullptr && up_extra->optimized_feature.reorder);
        }
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ncols, MATRIX_ROW_PADDING);
    ggml_sycl_pool_alloc<char> src1_q8_alloc(
        ctx.pool(), (size_t) nslots * n_tokens * src1_padded_cols *
                        sizeof(block_q8_1) / QK8_1);
    char * src1_q8 = src1_q8_alloc.get();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) src1->data, src1_q8, (int) ncols,
        (int) (nslots * n_tokens), src1_padded_cols, stream);

    const size_t bytes_per_qrow =
        (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = nslots == 1 ? 0 : bytes_per_qrow;
    const size_t src1_token_stride = (size_t) nslots * bytes_per_qrow;
    const int ids_row_stride = (int) (ids->nb[1] / sizeof(int32_t));

    static const bool disable_grouped = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMID_GROUPED");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!disable_grouped && n_tokens >= 4) {
        ggml_sycl_pool_alloc<int32_t> scratch(
            ctx.pool(), (size_t) (2 * n_as + 2 + n_tokens * n_used));
        if (ggml_sycl_mul_mat_vec_q_id_dual_swiglu_grouped_reorder(
                gate_weights->type, gate_weights->data, up_weights->data,
                src1_q8, (const int32_t *) ids->data, scratch.get(),
                (float *) glu->data, (int) ncols, (int) nrows, (int) n_used,
                (int) n_as, gate_weights->nb[2], up_weights->nb[2],
                glu->nb[1], src1_row_stride, (int) n_tokens, ids_row_stride,
                src1_token_stride, glu->nb[2], stream)) {
            return true;
        }
    }

    return ggml_sycl_mul_mat_vec_q_id_dual_swiglu_reorder(
        gate_weights->type, gate_weights->data, up_weights->data, src1_q8,
        (const int32_t *) ids->data, (float *) glu->data, (int) ncols,
        (int) nrows, (int) n_used, gate_weights->nb[2], up_weights->nb[2],
        glu->nb[1], src1_row_stride, (int) n_tokens, ids_row_stride,
        src1_token_stride, glu->nb[2], stream);
}

// Shared-expert dense path: MUL_MAT gate + MUL_MAT up + GLU(SwiGLU). CUDA already
// fuses this; SYCL only had the MUL_MAT_ID dual. Target: Q5_K/Q6_K reorder MMVQ,
// decode ncols_dst=1 and small prefill batches. Disable:
// GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1. Debug: GGML_SYCL_DENSE_DUAL_SWIGLU_DEBUG=1.
static bool ggml_sycl_mul_mat_dense_dual_swiglu_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * gate,
    const ggml_tensor * up, ggml_tensor * glu) {
    const ggml_tensor * gate_weights = gate->src[0];
    const ggml_tensor * up_weights   = up->src[0];
    const ggml_tensor * src1         = gate->src[1];
    static const bool trace_reject = []() {
        const char * env = getenv("GGML_SYCL_DENSE_DUAL_SWIGLU_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();

    const bool type_ok =
        gate_weights->type == up_weights->type &&
        (gate_weights->type == GGML_TYPE_Q8_0 ||
         gate_weights->type == GGML_TYPE_Q4_K ||
         gate_weights->type == GGML_TYPE_Q5_K ||
         gate_weights->type == GGML_TYPE_Q6_K);
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        gate->src[1] != up->src[1] || !type_ok ||
        src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        if (trace_reject) {
            static std::atomic<int> tr{0};
            if (tr.fetch_add(1, std::memory_order_relaxed) < 12) {
                fprintf(stderr,
                        "[treebeard-dense-dual-swiglu] dispatcher=type-reject"
                        " gate_op=%s up_op=%s same_src1=%d"
                        " wgate=%s wup=%s src1=%s gate=%s up=%s glu=%s"
                        " wgate_name=%s wup_name=%s\n",
                        ggml_op_name(gate->op), ggml_op_name(up->op),
                        gate->src[1] == up->src[1],
                        ggml_type_name(gate_weights->type),
                        ggml_type_name(up_weights->type),
                        ggml_type_name(src1->type),
                        ggml_type_name(gate->type), ggml_type_name(up->type),
                        ggml_type_name(glu->type),
                        gate_weights->name, up_weights->name);
            }
        }
        return false;
    }

    // MMVQ shapes only: 2D weights, activation batch in ne[1], no higher dims.
    const int64_t ncols     = src1->ne[0];
    const int64_t ncols_dst = src1->ne[1];
    const int64_t nrows     = gate_weights->ne[1];
    if (ncols_dst < 1 || ncols_dst > 32 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        gate_weights->ne[0] != ncols || up_weights->ne[0] != ncols ||
        up_weights->ne[1] != nrows ||
        gate_weights->ne[2] != 1 || gate_weights->ne[3] != 1 ||
        up_weights->ne[2] != 1 || up_weights->ne[3] != 1 ||
        ncols % QK8_1 != 0 ||
        !ggml_is_contiguous(src1) || !ggml_is_contiguous(glu) ||
        !ggml_is_contiguous(gate_weights) || !ggml_is_contiguous(up_weights) ||
        gate->ne[0] != nrows || up->ne[0] != nrows ||
        gate->ne[1] != ncols_dst || up->ne[1] != ncols_dst ||
        !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, glu) ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(glu, 1) != 0 /* swapped */) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-dense-dual-swiglu] dispatcher=shape-reject"
                    " ncols=%" PRId64 " ncols_dst=%" PRId64 " rows=%" PRId64 "\n",
                    ncols, ncols_dst, nrows);
        }
        return false;
    }

    // Shared-expert Q8_0 needs MMVQ reorder (opt_for_reorder_id is K-quants only).
    auto ensure_reorder_mmvq = [&](const ggml_tensor * w) {
        if (g_ggml_sycl_disable_optimize || !ctx.opt_feature.reorder) {
            return false;
        }
        if (!ggml_sycl_supports_reorder_mmvq(w->type)) {
            return false;
        }
        ggml_tensor_extra_gpu * extra =
            static_cast<ggml_tensor_extra_gpu *>(w->extra);
        if (!extra) {
            return false;
        }
        if (extra->optimized_feature.reorder) {
            return true;
        }
        if (reorder_qw(w, ctx.stream())) {
            extra->optimized_feature.reorder = true;
            return true;
        }
        return false;
    };
    if (!ensure_reorder_mmvq(gate_weights) || !ensure_reorder_mmvq(up_weights)) {
        if (trace_reject) {
            const ggml_tensor_extra_gpu * ge =
                static_cast<const ggml_tensor_extra_gpu *>(gate_weights->extra);
            const ggml_tensor_extra_gpu * ue =
                static_cast<const ggml_tensor_extra_gpu *>(up_weights->extra);
            fprintf(stderr,
                    "[treebeard-dense-dual-swiglu] dispatcher=reorder-reject"
                    " gate_reorder=%d up_reorder=%d type=%s\n",
                    ge != nullptr && ge->optimized_feature.reorder,
                    ue != nullptr && ue->optimized_feature.reorder,
                    ggml_type_name(gate_weights->type));
        }
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ncols, MATRIX_ROW_PADDING);
    ggml_sycl_pool_alloc<char> src1_q8_alloc(
        ctx.pool(),
        (size_t) ncols_dst * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_q8 = src1_q8_alloc.get();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) src1->data, src1_q8, (int) ncols, (int) ncols_dst,
        src1_padded_cols, stream);

    const size_t bytes_per_qrow =
        (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t dst_col_stride = glu->nb[1] / sizeof(float);

    const bool ok = ggml_sycl_mul_mat_vec_q_dense_dual_swiglu_reorder(
        gate_weights->type, gate_weights->data, up_weights->data, src1_q8,
        (float *) glu->data, (int) ncols, (int) nrows, (int) ncols_dst,
        bytes_per_qrow, dst_col_stride, stream);
    if (ok && trace_reject) {
        static std::atomic<int> hits{0};
        const int n = hits.fetch_add(1, std::memory_order_relaxed);
        if (n < 8) {
            fprintf(stderr,
                    "[treebeard-dense-dual-swiglu] hit type=%s rows=%" PRId64
                    " cols=%" PRId64 " ncols_dst=%" PRId64 "\n",
                    ggml_type_name(gate_weights->type), nrows, ncols, ncols_dst);
        }
    }
    return ok;
}

// Dense dual MMVQ: two ordinary MUL_MATs sharing one activation (no GLU).
// Targets GDN pairs that share `cur`: attn_qkv+attn_gate (z), ssm_alpha+ssm_beta.
// Disable: GGML_SYCL_DISABLE_DENSE_DUAL_MMVQ=1.
// Debug:   GGML_SYCL_DENSE_DUAL_MMVQ_DEBUG=1.
static bool ggml_sycl_mul_mat_dense_dual_mmvq_fused(
    ggml_backend_sycl_context & ctx, ggml_tensor * mm_a, ggml_tensor * mm_b) {
    const ggml_tensor * wa   = mm_a->src[0];
    const ggml_tensor * wb   = mm_b->src[0];
    const ggml_tensor * src1 = mm_a->src[1];
    static const bool trace = []() {
        const char * env = getenv("GGML_SYCL_DENSE_DUAL_MMVQ_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();

    const bool type_ok =
        wa->type == wb->type &&
        (wa->type == GGML_TYPE_Q8_0 || wa->type == GGML_TYPE_Q4_K ||
         wa->type == GGML_TYPE_Q5_K || wa->type == GGML_TYPE_Q6_K);
    if (mm_a->op != GGML_OP_MUL_MAT || mm_b->op != GGML_OP_MUL_MAT ||
        mm_a->src[1] != mm_b->src[1] || !type_ok ||
        src1->type != GGML_TYPE_F32 ||
        mm_a->type != GGML_TYPE_F32 || mm_b->type != GGML_TYPE_F32) {
        if (trace) {
            fprintf(stderr,
                    "[treebeard-dense-dual-mmvq] type-reject wa=%s wb=%s "
                    "same_src1=%d\n",
                    ggml_type_name(wa->type), ggml_type_name(wb->type),
                    mm_a->src[1] == mm_b->src[1]);
        }
        return false;
    }

    const int64_t ncols     = src1->ne[0];
    const int64_t ncols_dst = src1->ne[1];
    const int64_t nrows_a   = wa->ne[1];
    const int64_t nrows_b   = wb->ne[1];
    // Decode-first: ncols_dst=1 is the ship path. Small multi-col kept for
    // microbatch; large prefill batches (64+) measured regress (pp -38%).
    if (ncols_dst < 1 || ncols_dst > 4 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        wa->ne[0] != ncols || wb->ne[0] != ncols ||
        wa->ne[2] != 1 || wa->ne[3] != 1 ||
        wb->ne[2] != 1 || wb->ne[3] != 1 ||
        ncols % QK8_1 != 0 ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(wa) || !ggml_is_contiguous(wb) ||
        !ggml_is_contiguous(mm_a) || !ggml_is_contiguous(mm_b) ||
        mm_a->ne[0] != nrows_a || mm_b->ne[0] != nrows_b ||
        mm_a->ne[1] != ncols_dst || mm_b->ne[1] != ncols_dst) {
        if (trace) {
            fprintf(stderr,
                    "[treebeard-dense-dual-mmvq] shape-reject ncols=%" PRId64
                    " ncols_dst=%" PRId64 " rows_a=%" PRId64 " rows_b=%" PRId64
                    " cont_src1=%d cont_a=%d cont_b=%d cont_wa=%d cont_wb=%d\n",
                    ncols, ncols_dst, nrows_a, nrows_b,
                    (int) ggml_is_contiguous(src1),
                    (int) ggml_is_contiguous(mm_a),
                    (int) ggml_is_contiguous(mm_b),
                    (int) ggml_is_contiguous(wa),
                    (int) ggml_is_contiguous(wb));
        }
        return false;
    }

    auto ensure_reorder_mmvq = [&](const ggml_tensor * w) {
        if (g_ggml_sycl_disable_optimize || !ctx.opt_feature.reorder) {
            return false;
        }
        if (!ggml_sycl_supports_reorder_mmvq(w->type)) {
            return false;
        }
        ggml_tensor_extra_gpu * extra =
            static_cast<ggml_tensor_extra_gpu *>(w->extra);
        if (!extra) {
            return false;
        }
        if (extra->optimized_feature.reorder) {
            return true;
        }
        if (reorder_qw(w, ctx.stream())) {
            extra->optimized_feature.reorder = true;
            return true;
        }
        return false;
    };
    if (!ensure_reorder_mmvq(wa) || !ensure_reorder_mmvq(wb)) {
        if (trace) {
            fprintf(stderr, "[treebeard-dense-dual-mmvq] reorder-reject\n");
        }
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ncols, MATRIX_ROW_PADDING);
    ggml_sycl_pool_alloc<char> src1_q8_alloc(
        ctx.pool(),
        (size_t) ncols_dst * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_q8 = src1_q8_alloc.get();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) src1->data, src1_q8, (int) ncols, (int) ncols_dst,
        src1_padded_cols, stream);

    const size_t bytes_per_qrow =
        (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t dst_a_col_stride = mm_a->nb[1] / sizeof(float);
    const size_t dst_b_col_stride = mm_b->nb[1] / sizeof(float);

    const bool ok = ggml_sycl_mul_mat_vec_q_dense_dual_mmvq_reorder(
        wa->type, wa->data, wb->data, src1_q8,
        (float *) mm_a->data, (float *) mm_b->data,
        (int) ncols, (int) nrows_a, (int) nrows_b, (int) ncols_dst,
        bytes_per_qrow, dst_a_col_stride, dst_b_col_stride, stream);
    if (ok && trace) {
        static std::atomic<int> hits{0};
        if (hits.fetch_add(1, std::memory_order_relaxed) < 8) {
            fprintf(stderr,
                    "[treebeard-dense-dual-mmvq] hit type=%s rows_a=%" PRId64
                    " rows_b=%" PRId64 " cols=%" PRId64 " ncols_dst=%" PRId64
                    " wa=%s wb=%s\n",
                    ggml_type_name(wa->type), nrows_a, nrows_b, ncols,
                    ncols_dst, wa->name, wb->name);
        }
    }
    return ok;
}

// Dense dual F32 GEMV for equal-K projections sharing one activation
// (ssm_alpha + ssm_beta on the GDN path). Default OFF; opt-in with
// GGML_SYCL_ENABLE_DENSE_DUAL_F32=1. Debug: GGML_SYCL_DENSE_DUAL_F32_DEBUG=1.
static bool ggml_sycl_mul_mat_dense_dual_f32_fused(
    ggml_backend_sycl_context & ctx, ggml_tensor * mm_a, ggml_tensor * mm_b) {
    const ggml_tensor * wa   = mm_a->src[0];
    const ggml_tensor * wb   = mm_b->src[0];
    const ggml_tensor * src1 = mm_a->src[1];
    static const bool trace = []() {
        const char * env = getenv("GGML_SYCL_DENSE_DUAL_F32_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();

    if (mm_a->op != GGML_OP_MUL_MAT || mm_b->op != GGML_OP_MUL_MAT ||
        mm_a->src[1] != mm_b->src[1] ||
        wa->type != GGML_TYPE_F32 || wb->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 ||
        mm_a->type != GGML_TYPE_F32 || mm_b->type != GGML_TYPE_F32) {
        if (trace) {
            fprintf(stderr,
                    "[treebeard-dense-dual-f32] type-reject wa=%s wb=%s "
                    "same_src1=%d\n",
                    ggml_type_name(wa->type), ggml_type_name(wb->type),
                    mm_a->src[1] == mm_b->src[1]);
        }
        return false;
    }

    const int64_t ncols     = src1->ne[0];
    const int64_t ncols_dst = src1->ne[1];
    const int64_t nrows_a   = wa->ne[1];
    const int64_t nrows_b   = wb->ne[1];
    // Decode-first: ship path is ncols_dst=1. Cap at 8 (prefill for these
    // small F32 pairs is already oneDNN; dual is for MKL GEMV submit savings).
    if (ncols_dst < 1 || ncols_dst > 8 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        wa->ne[0] != ncols || wb->ne[0] != ncols ||
        wa->ne[2] != 1 || wa->ne[3] != 1 ||
        wb->ne[2] != 1 || wb->ne[3] != 1 ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(wa) || !ggml_is_contiguous(wb) ||
        !ggml_is_contiguous(mm_a) || !ggml_is_contiguous(mm_b) ||
        mm_a->ne[0] != nrows_a || mm_b->ne[0] != nrows_b ||
        mm_a->ne[1] != ncols_dst || mm_b->ne[1] != ncols_dst) {
        if (trace) {
            fprintf(stderr,
                    "[treebeard-dense-dual-f32] shape-reject ncols=%" PRId64
                    " ncols_dst=%" PRId64 " rows_a=%" PRId64 " rows_b=%" PRId64
                    "\n",
                    ncols, ncols_dst, nrows_a, nrows_b);
        }
        return false;
    }

    // Reject MoE router weights — must stay on oneMKL for bit-stable expert choice.
    if (std::strstr(wa->name, ".ffn_gate_inp.weight") != nullptr ||
        std::strstr(wb->name, ".ffn_gate_inp.weight") != nullptr) {
        if (trace) {
            fprintf(stderr, "[treebeard-dense-dual-f32] router-reject\n");
        }
        return false;
    }

    const size_t x_col_stride      = src1->nb[1] / sizeof(float);
    const size_t dst_a_col_stride  = mm_a->nb[1] / sizeof(float);
    const size_t dst_b_col_stride  = mm_b->nb[1] / sizeof(float);
    const size_t wa_row_stride     = wa->nb[1] / sizeof(float);
    const size_t wb_row_stride     = wb->nb[1] / sizeof(float);

    // Custom dual GEMV only. oneMKL gemm_batch(2) pointer-array path was tried
    // 2026-07-20 and hung on product decode after activation hits — do not re-enable
    // without a new bound. Custom kernel completes but measured −2.7% tg (park).
    const bool ok = ggml_sycl_mul_mat_vec_f32_dense_dual(
        (const float *) wa->data, (const float *) wb->data,
        (const float *) src1->data,
        (float *) mm_a->data, (float *) mm_b->data,
        (int) ncols, (int) nrows_a, (int) nrows_b, (int) ncols_dst,
        x_col_stride, dst_a_col_stride, dst_b_col_stride,
        wa_row_stride, wb_row_stride, ctx.stream());
    if (ok && trace) {
        static std::atomic<int> hits{0};
        if (hits.fetch_add(1, std::memory_order_relaxed) < 8) {
            fprintf(stderr,
                    "[treebeard-dense-dual-f32] hit rows_a=%" PRId64
                    " rows_b=%" PRId64 " cols=%" PRId64 " ncols_dst=%" PRId64
                    " wa=%s wb=%s\n",
                    nrows_a, nrows_b, ncols, ncols_dst, wa->name, wb->name);
        }
    }
    return ok;
}

// Diagnostic (TREEBEARD_MOE_ROUTE_HIST=1): how many DISTINCT experts does a layer
// actually touch per decode step? The default MoE-down path serializes the routed
// experts per token with no cross-token reuse, so the entire upside of any
// expert-reuse kernel is bounded by draws/distinct. Nothing else measures this.
// Host-syncs on the ids buffer, so it is a measurement build only - never a perf arm.
static void ggml_sycl_moe_route_hist(const ggml_tensor * ids, const ggml_tensor * src0,
                                     int64_t n_tokens, int n_experts_used,
                                     const queue_ptr & stream) {
    static const bool enabled = []() {
        const char * env = getenv("TREEBEARD_MOE_ROUTE_HIST");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }
    const int n_as = (int) src0->ne[2];
    if (n_as <= 0) {
        return;
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids->data, ggml_nbytes(ids)).wait()));

    std::vector<int> counts((size_t) n_as, 0);
    int draws = 0;
    int distinct = 0;
    int top_count = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_experts_used; ++s) {
            const int32_t e = *(const int32_t *) (ids_host.data() + t * ids->nb[1] + s * ids->nb[0]);
            if (e < 0 || e >= n_as) {
                continue;
            }
            ++draws;
            if (counts[(size_t) e]++ == 0) {
                ++distinct;
            }
            if (counts[(size_t) e] > top_count) {
                top_count = counts[(size_t) e];
            }
        }
    }

    // Global monotonic call index; the aggregator buckets by tensor name and
    // recovers per-layer step order from the emission order.
    static std::atomic<long> call_index{0};
    const long call = call_index.fetch_add(1, std::memory_order_relaxed);

    fprintf(stderr,
            "[treebeard-moe-route] call=%ld tensor=%s n_tokens=%d draws=%d"
            " distinct=%d top_expert_count=%d\n",
            call, src0->name, (int) n_tokens, draws, distinct, top_count);
}

// MoE down-projection specialization: compute routed expert dots, multiply by
// routing weights in slot order, and write only the final token output.
static bool ggml_sycl_mul_mat_id_mmvq_weighted(
    ggml_backend_sycl_context & ctx, const ggml_tensor * mmid,
    const ggml_tensor * weights, ggml_tensor * dst) {
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * ids  = mmid->src[2];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    // ne12 is the token batch on src1. Allow single-token (decode) so the
    // integrated weighted path can replace separate mul_mat_id + weighted_sum.
    if (!ggml_sycl_mul_mat_id_fused_eligible(mmid) || ne12 < 1 || ne12 > 64) {
        return false;
    }
    const int64_t n_ids_per_group = ids->ne[0];
    if (ne11 != n_ids_per_group || ids->ne[1] != ne12 ||
        weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
        weights->ne[0] != 1 || weights->ne[1] != n_ids_per_group ||
        weights->ne[2] != ne12 || weights->ne[3] != 1 ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != src0->ne[1] || dst->ne[1] != ne12 ||
        dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    ggml_sycl_moe_route_hist(ids, src0, ne12, n_experts_used, stream);

    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool reorder_type = src0->type == GGML_TYPE_Q4_K ||
                              src0->type == GGML_TYPE_Q5_K ||
                              src0->type == GGML_TYPE_Q6_K;
    const bool use_reorder = reorder_type && src0_extra != nullptr &&
                             src0_extra->optimized_feature.reorder;
    if (!use_reorder && src0->type != GGML_TYPE_Q8_0) {
        return false;
    }

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * ne12 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) (ne11 * ne12),
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) (ne11 * ne12),
            src1_padded_cols, stream);
    }

    const size_t bytes_per_qrow =
        (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride   = bytes_per_qrow;
    const size_t src1_token_stride = (size_t) ne11 * bytes_per_qrow;
    const int    ids_row_stride    = (int) (ids->nb[1] / sizeof(int32_t));

    if (use_reorder) {
        static const bool enable_grouped = []() {
            const char * enable = getenv("GGML_SYCL_ENABLE_MOE_DOWN_GROUPED");
            const char * disable_all = getenv("GGML_SYCL_DISABLE_MMID_GROUPED");
            const char * disable_down = getenv("GGML_SYCL_DISABLE_MOE_DOWN_GROUPED");
            return enable != nullptr && std::atoi(enable) != 0 &&
                   !(disable_all != nullptr && std::atoi(disable_all) != 0) &&
                   !(disable_down != nullptr && std::atoi(disable_down) != 0);
        }();
        if (enable_grouped && ne12 >= 4) {
            const int n_as = (int) src0->ne[2];
            ggml_sycl_pool_alloc<int32_t> scratch(
                ctx.pool(), (size_t) (2 * n_as + 2 + ne12 * n_experts_used));
            if (ggml_sycl_mul_mat_vec_q_id_weighted_grouped_reorder(
                    src0->type, src0->data, src1_ddq,
                    (const int32_t *) ids->data, (const float *) weights->data,
                    scratch.get(), (float *) dst->data, (int) ne10, nrows,
                    n_experts_used, n_as, src0->nb[2], src1_row_stride,
                    weights->nb[1], (int) ne12, ids_row_stride,
                    src1_token_stride, weights->nb[2], dst->nb[1], stream)) {
                static const bool trace_grouped = []() {
                    const char * env = getenv("GGML_SYCL_MOE_DOWN_REDUCE_DEBUG");
                    return env != nullptr && std::atoi(env) != 0;
                }();
                static std::atomic<int> trace_counts[65];
                if (trace_grouped &&
                    trace_counts[ne12].fetch_add(1, std::memory_order_relaxed) < 4) {
                    fprintf(stderr,
                            "[treebeard-moe-down-reduce] event=batched-grouped-hit"
                            " rows=%d experts=%d tokens=%" PRId64 "\n",
                            nrows, n_experts_used, ne12);
                }
                return true;
            }
        }
        return ggml_sycl_mul_mat_vec_q_id_weighted_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
            (const float *) weights->data, (float *) dst->data,
            (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2], src1_row_stride,
            /*weights_slot_stride=*/ weights->nb[1], (int) ne12, ids_row_stride,
            src1_token_stride, /*weights_token_stride=*/ weights->nb[2],
            /*dst_token_stride=*/ dst->nb[1], stream);
    }
    return ggml_sycl_mul_mat_vec_q_id_weighted(
        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
        (const float *) weights->data, (float *) dst->data,
        (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2], src1_row_stride,
        /*weights_slot_stride=*/ weights->nb[1], (int) ne12, ids_row_stride,
        src1_token_stride, /*weights_token_stride=*/ weights->nb[2],
        /*dst_token_stride=*/ dst->nb[1], stream);
}

// Complete batched MoE expert pipeline. The graph still exposes separate
// gate/up MMIDs, SwiGLU, down MMID, routing MUL, and ordered expert reduction;
// this dispatcher consumes the entire region with one routing pre-pass and two
// compute submissions. The gate/up kernel emits reordered Q8_1 directly, so no
// global F32 expert activation or standalone quantization kernel is produced.
static bool ggml_sycl_moe_swiglu_down_pipeline_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * gate,
    const ggml_tensor * up, const ggml_tensor * glu,
    const ggml_tensor * down, const ggml_tensor * route_weights,
    ggml_tensor * dst) {
    const ggml_tensor * gate_weights = gate->src[0];
    const ggml_tensor * up_weights   = up->src[0];
    const ggml_tensor * down_weights = down->src[0];
    const ggml_tensor * src1         = gate->src[1];
    const ggml_tensor * ids          = gate->src[2];
    static const bool trace_reject = []() {
        const char * env = getenv("GGML_SYCL_MOE_PIPELINE_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();

    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
        glu->op != GGML_OP_GLU || down->op != GGML_OP_MUL_MAT_ID ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        glu->src[0] != gate || glu->src[1] != up ||
        gate->src[1] != up->src[1] || gate->src[2] != up->src[2] ||
        down->src[1] != glu || down->src[2] != ids ||
        gate_weights->type != up_weights->type ||
        (gate_weights->type != GGML_TYPE_Q5_K &&
         gate_weights->type != GGML_TYPE_Q6_K) ||
        (down_weights->type != GGML_TYPE_Q4_K &&
         down_weights->type != GGML_TYPE_Q5_K &&
         down_weights->type != GGML_TYPE_Q6_K) ||
        src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
        down->type != GGML_TYPE_F32 || route_weights->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-moe-pipeline] dispatcher=type-or-edge-reject"
                    " gate=%s up=%s down=%s glu_op=%d\n",
                    ggml_type_name(gate_weights->type),
                    ggml_type_name(up_weights->type),
                    ggml_type_name(down_weights->type),
                    (int) ggml_get_glu_op(glu));
        }
        return false;
    }

    const int64_t gate_ncols = src1->ne[0];
    const int64_t nslots     = src1->ne[1];
    const int64_t n_tokens   = src1->ne[2];
    const int64_t n_used     = ids->ne[0];
    const int64_t gate_nrows = gate_weights->ne[1];
    const int64_t down_nrows = down_weights->ne[1];
    const int64_t n_as       = gate_weights->ne[2];
    if (n_tokens < 4 || n_tokens > 64 || n_used < 2 || n_used > 16 ||
        n_as > 256 || gate_ncols != gate_weights->ne[0] ||
        gate_ncols != up_weights->ne[0] || gate_ncols % QK8_1 != 0 ||
        gate_nrows < QK8_1 || gate_nrows % QK8_1 != 0 ||
        up_weights->ne[1] != gate_nrows || up_weights->ne[2] != n_as ||
        down_weights->ne[0] != gate_nrows || down_weights->ne[2] != n_as ||
        ids->ne[1] != n_tokens || ids->nb[0] != sizeof(int32_t) ||
        (nslots != 1 && nslots != n_used) ||
        !ggml_is_contiguous(src1) || !ggml_is_contiguous(route_weights) ||
        !ggml_is_contiguous(dst) ||
        gate->ne[0] != gate_nrows || gate->ne[1] != n_used ||
        gate->ne[2] != n_tokens || !ggml_are_same_shape(gate, up) ||
        !ggml_are_same_shape(gate, glu) ||
        down->ne[0] != down_nrows || down->ne[1] != n_used ||
        down->ne[2] != n_tokens || down->ne[3] != 1 ||
        route_weights->ne[0] != 1 || route_weights->ne[1] != n_used ||
        route_weights->ne[2] != n_tokens || route_weights->ne[3] != 1 ||
        dst->ne[0] != down_nrows || dst->ne[1] != n_tokens ||
        dst->ne[2] != 1 || dst->ne[3] != 1) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-moe-pipeline] dispatcher=shape-reject"
                    " gate_cols=%" PRId64 " gate_rows=%" PRId64
                    " down_rows=%" PRId64 " slots=%" PRId64
                    " used=%" PRId64 " tokens=%" PRId64
                    " experts=%" PRId64 " down_ne=%" PRId64 ",%" PRId64
                    ",%" PRId64 ",%" PRId64 " dst_ne=%" PRId64 ",%" PRId64
                    ",%" PRId64 ",%" PRId64 "\n",
                    gate_ncols, gate_nrows, down_nrows, nslots, n_used,
                    n_tokens, n_as, down->ne[0], down->ne[1], down->ne[2],
                    down->ne[3], dst->ne[0], dst->ne[1], dst->ne[2],
                    dst->ne[3]);
        }
        return false;
    }

    opt_for_reorder_id(&ctx, gate_weights);
    opt_for_reorder_id(&ctx, up_weights);
    opt_for_reorder_id(&ctx, down_weights);
    const ggml_tensor_extra_gpu * gate_extra =
        static_cast<const ggml_tensor_extra_gpu *>(gate_weights->extra);
    const ggml_tensor_extra_gpu * up_extra =
        static_cast<const ggml_tensor_extra_gpu *>(up_weights->extra);
    const ggml_tensor_extra_gpu * down_extra =
        static_cast<const ggml_tensor_extra_gpu *>(down_weights->extra);
    if (!gate_extra || !up_extra || !down_extra ||
        !gate_extra->optimized_feature.reorder ||
        !up_extra->optimized_feature.reorder ||
        !down_extra->optimized_feature.reorder) {
        if (trace_reject) {
            fprintf(stderr,
                    "[treebeard-moe-pipeline] dispatcher=reorder-reject"
                    " gate=%d up=%d down=%d\n",
                    gate_extra != nullptr && gate_extra->optimized_feature.reorder,
                    up_extra != nullptr && up_extra->optimized_feature.reorder,
                    down_extra != nullptr && down_extra->optimized_feature.reorder);
        }
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols =
        GGML_PAD((int) gate_ncols, MATRIX_ROW_PADDING);
    ggml_sycl_pool_alloc<char> src1_q8_alloc(
        ctx.pool(), (size_t) nslots * n_tokens * src1_padded_cols *
                        sizeof(block_q8_1) / QK8_1);
    char * src1_q8 = src1_q8_alloc.get();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) src1->data, src1_q8, (int) gate_ncols,
        (int) (nslots * n_tokens), src1_padded_cols, stream);

    const int gate_nrows_padded =
        GGML_PAD((int) gate_nrows, MATRIX_ROW_PADDING);
    const size_t src1_qrow_bytes =
        (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t glu_qrow_bytes =
        (size_t) gate_nrows_padded * sizeof(block_q8_1) / QK8_1;
    ggml_sycl_pool_alloc<char> glu_q8_alloc(
        ctx.pool(), (size_t) n_used * n_tokens * glu_qrow_bytes);
    ggml_sycl_pool_alloc<int32_t> scratch(
        ctx.pool(), (size_t) (2 * n_as + 2 + n_tokens * n_used));

    return ggml_sycl_mul_mat_vec_q_id_swiglu_down_grouped_reorder(
        gate_weights->type, down_weights->type, gate_weights->data,
        up_weights->data, down_weights->data, src1_q8,
        (const int32_t *) ids->data, (const float *) route_weights->data,
        scratch.get(), glu_q8_alloc.get(), (float *) dst->data,
        (int) gate_ncols, (int) gate_nrows, gate_nrows_padded,
        (int) down_nrows, (int) n_used, (int) n_as,
        gate_weights->nb[2], up_weights->nb[2], down_weights->nb[2],
        nslots == 1 ? 0 : src1_qrow_bytes, route_weights->nb[1],
        (int) n_tokens, (int) (ids->nb[1] / sizeof(int32_t)),
        (size_t) nslots * src1_qrow_bytes, glu_qrow_bytes,
        (size_t) n_used * glu_qrow_bytes, route_weights->nb[2], dst->nb[1],
        stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    for (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    // Fused device-routed expert GEMV: handles ne12 up to its batch cap (see the bail checks).
    if (ne12 == 1 || !ggml_sycl_mmid_fused_batch_disabled()) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> routed_row_src;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*sizeof(mmid_row_mapping))));

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                const char *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                const mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t *free,
                                                   size_t *total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ggml_sycl_set_device(device);

    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Detect a fusable op subgraph starting at node i and, if found, dispatch a single fused kernel.
// Returns the number of *following* nodes consumed (0 = no fusion, dispatch node i normally).
// Decode on this backend is host-submit bound, so eliding glue submits (norm-weight MUL, residual
// ADD) directly buys throughput. Mirrors the CUDA backend's ggml_cuda_try_fuse for the patterns
// this model actually emits. Toggle off with GGML_SYCL_DISABLE_FUSION=1.
static bool ggml_sycl_tensor_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    GGML_ASSERT(a != nullptr && b != nullptr);
    GGML_ASSERT(a->data != nullptr && b->data != nullptr);

    const size_t a_size = ggml_nbytes(a);
    const size_t b_size = ggml_nbytes(b);
    if (a_size == 0 || b_size == 0) {
        return false;
    }

    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b->data);

    // Difference-based half-open interval test avoids end-address overflow.
    return a_begin <= b_begin ? b_begin - a_begin < a_size
                              : a_begin - b_begin < b_size;
}

enum ggml_sycl_moe_down_reduce_trace_event {
    GGML_SYCL_MOE_DOWN_REDUCE_ELIGIBLE,
    GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_SUBGRAPH_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_OUTPUT_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_HIT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_BOUNDS_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_SUBGRAPH_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_VIEWS_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_CHAIN_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_OUTPUT_REJECT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_LIVENESS_ANNOTATED,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_WEIGHTS_SNAPSHOT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_INTEGRATED_HIT,
    GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_HIT,
    GGML_SYCL_MOE_DOWN_REDUCE_TRACE_EVENT_COUNT,
};

static void ggml_sycl_trace_moe_down_reduce(
        ggml_sycl_moe_down_reduce_trace_event event,
        const ggml_tensor * experts,
        const ggml_tensor * dst = nullptr,
        unsigned int detail = 0) {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_MOE_DOWN_REDUCE_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }

    static constexpr const char * event_names[] = {
        "eligible", "single-subgraph-reject", "single-output-reject", "single-hit",
        "batched-bounds-reject", "batched-subgraph-reject", "batched-views-reject",
        "batched-chain-reject", "batched-output-reject", "batched-liveness-annotated",
        "batched-weights-snapshot", "batched-integrated-hit", "batched-hit",
    };
    static_assert(GGML_SYCL_MOE_DOWN_REDUCE_TRACE_EVENT_COUNT ==
                  (int) (sizeof(event_names) / sizeof(event_names[0])));
    static std::atomic<int> event_counts[GGML_SYCL_MOE_DOWN_REDUCE_TRACE_EVENT_COUNT][65] = {};

    GGML_ASSERT(event >= 0 && event < GGML_SYCL_MOE_DOWN_REDUCE_TRACE_EVENT_COUNT);
    const int token_index = experts->ne[2] >= 0 && experts->ne[2] <= 64 ? (int) experts->ne[2] : 0;
    const int occurrence = event_counts[event][token_index].fetch_add(1, std::memory_order_relaxed);
    if (occurrence < 4) {
        fprintf(stderr,
                "[treebeard-moe-down-reduce] event=%s occurrence=%d rows=%" PRId64
                " experts=%" PRId64 " tokens=%" PRId64
                " detail=0x%x experts_data=%p dst_data=%p\n",
                event_names[event], occurrence + 1, experts->ne[0], experts->ne[1], experts->ne[2],
                detail, experts->data, dst != nullptr ? dst->data : nullptr);
    }
}

static bool ggml_sycl_moe_down_reduce_disabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MOE_DOWN_REDUCE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return disabled;
}

static bool ggml_sycl_moe_dual_swiglu_disabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return disabled;
}

static bool ggml_sycl_moe_pipeline_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_MOE_PIPELINE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static void ggml_sycl_trace_moe_pipeline(
        const char * event, const ggml_tensor * gate,
        const ggml_tensor * dst = nullptr) {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_MOE_PIPELINE_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }
    int event_index = 4;
    if (strcmp(event, "liveness-annotated") == 0) {
        event_index = 0;
    } else if (strcmp(event, "candidate") == 0) {
        event_index = 1;
    } else if (strcmp(event, "integrated-hit") == 0 ||
               strcmp(event, "integrated-hit-views-first") == 0) {
        event_index = 2;
    } else if (strcmp(event, "dispatch-reject") == 0) {
        event_index = 3;
    }
    static std::atomic<int> trace_count[5][65] = {};
    const int token_index = gate->ne[2] >= 1 && gate->ne[2] <= 64 ?
        (int) gate->ne[2] : 0;
    const int occurrence = trace_count[event_index][token_index].fetch_add(
        1, std::memory_order_relaxed);
    if (occurrence < 8) {
        fprintf(stderr,
                "[treebeard-moe-pipeline] event=%s occurrence=%d gate_type=%s"
                " rows=%" PRId64 " used=%" PRId64 " tokens=%" PRId64
                " gate_data=%p dst_data=%p\n",
                event, occurrence + 1, ggml_type_name(gate->src[0]->type),
                gate->ne[0], gate->ne[1], gate->ne[2], gate->data,
                dst != nullptr ? dst->data : nullptr);
    }
}

static void ggml_sycl_trace_moe_dual_swiglu(
        const char * event, const ggml_tensor * gate, const ggml_tensor * dst) {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }
    static std::atomic<int> event_counts[4][65] = {};
    int event_index = 3;
    if (strcmp(event, "liveness-annotated") == 0) {
        event_index = 0;
    } else if (strcmp(event, "candidate") == 0) {
        event_index = 1;
    } else if (strcmp(event, "dual-integrated-hit") == 0) {
        event_index = 2;
    }
    const int token_index = gate->ne[2] >= 1 && gate->ne[2] <= 64 ?
        (int) gate->ne[2] : 0;
    const int occurrence = event_counts[event_index][token_index].fetch_add(
        1, std::memory_order_relaxed);
    if (occurrence < 4) {
        fprintf(stderr,
                "[treebeard-moe-dual-swiglu] event=%s occurrence=%d type=%s"
                " rows=%" PRId64 " experts=%" PRId64 " tokens=%" PRId64
                " gate_data=%p dst_data=%p\n",
                event, occurrence + 1, ggml_type_name(gate->src[0]->type),
                gate->ne[0], gate->ne[1], gate->ne[2], gate->data, dst->data);
    }
}

static bool ggml_sycl_state_io_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_STATE_IO_FUSION");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

static bool ggml_sycl_state_io_op_enabled(ggml_op op) {
    if (!ggml_sycl_state_io_enabled()) {
        return false;
    }
    static const std::string mode = []() {
        const char * env = getenv("GGML_SYCL_STATE_IO_MODE");
        return env != nullptr ? std::string(env) : std::string("all");
    }();
    if (mode == "all") {
        return true;
    }
    return (mode == "ssm" && op == GGML_OP_SSM_CONV) ||
           (mode == "gdn" && op == GGML_OP_GATED_DELTA_NET);
}

static ggml_tensor * ggml_sycl_view_base(ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static const ggml_tensor * ggml_sycl_view_base(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static ggml_tensor * ggml_sycl_find_state_copy(
        ggml_cgraph * cgraph, ggml_tensor * producer, int * count_out = nullptr) {
    ggml_tensor * found = nullptr;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_CPY && node->src[0] != nullptr &&
            ggml_sycl_view_base(node->src[0]) == producer) {
            found = node;
            ++count;
        }
    }
    if (count_out != nullptr) {
        *count_out = count;
    }
    return count == 1 ? found : nullptr;
}

static bool ggml_sycl_reserve_state_io_src(
        ggml_tensor * node, int index, ggml_tensor * value) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_SRC);
    if (node->src[index] != nullptr && node->src[index] != value) {
        return false;
    }
    node->src[index] = value;
    return true;
}

// Annotate the exact AR recurrent-state pattern before allocation. The spare
// sources are allocator liveness edges; the ordinary backend implementations
// continue to consume only their declared sources when the opt-in path is not
// runtime-safe for a particular recurrent copy map.
static void ggml_sycl_annotate_state_io(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_sycl_state_io_op_enabled(GGML_OP_SSM_CONV) &&
            node->op == GGML_OP_SSM_CONV && node->src[0] != nullptr &&
            node->src[0]->op == GGML_OP_CONCAT && node->src[1] != nullptr &&
            node->ne[1] == 1) {
            ggml_tensor * concat = node->src[0];
            ggml_tensor * gather = concat->src[0] != nullptr
                ? ggml_sycl_view_base(concat->src[0]) : nullptr;
            ggml_tensor * token = concat->src[1];
            int copy_count = 0;
            ggml_tensor * copy = ggml_sycl_find_state_copy(cgraph, concat, &copy_count);
            if (gather == nullptr || gather->op != GGML_OP_GET_ROWS ||
                gather->src[0] == nullptr || gather->src[1] == nullptr ||
                token == nullptr || copy == nullptr || copy_count != 1) {
                continue;
            }
            ggml_tensor * state_cache = gather->src[0];
            ggml_tensor * state_ids   = gather->src[1];
            ggml_tensor * state_dst   = copy->src[1];
            const int64_t d_conv = node->src[1]->ne[0];
            const int64_t d_inner = node->src[1]->ne[1];
            const int64_t state_size = (d_conv - 1) * d_inner;
            if (state_cache->type != GGML_TYPE_F32 || state_ids->type != GGML_TYPE_I32 ||
                token->type != GGML_TYPE_F32 || state_dst == nullptr ||
                state_dst->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 ||
                d_conv < 2 || d_conv > 16 || state_cache->ne[0] != state_size ||
                state_dst->ne[0] != state_size || state_dst->ne[1] != node->ne[2] ||
                token->ne[0] != 1 || token->ne[1] != d_inner || token->ne[2] != node->ne[2] ||
                ggml_sycl_view_base(state_cache) != ggml_sycl_view_base(state_dst)) {
                continue;
            }
            if (!ggml_sycl_reserve_state_io_src(node, 2, state_cache) ||
                !ggml_sycl_reserve_state_io_src(node, 3, state_ids) ||
                !ggml_sycl_reserve_state_io_src(node, 4, token) ||
                !ggml_sycl_reserve_state_io_src(node, 5, state_dst) ||
                !ggml_sycl_reserve_state_io_src(node, 6, gather)) {
                continue;
            }
        }

        if (ggml_sycl_state_io_op_enabled(GGML_OP_GATED_DELTA_NET) &&
            node->op == GGML_OP_GATED_DELTA_NET && node->src[2] != nullptr &&
            node->src[2]->ne[2] == 1 && ggml_get_op_params_i32(node, 0) == 1) {
            ggml_tensor * gather = node->src[5] != nullptr
                ? ggml_sycl_view_base(node->src[5]) : nullptr;
            int copy_count = 0;
            ggml_tensor * copy = ggml_sycl_find_state_copy(cgraph, node, &copy_count);
            if (gather == nullptr || gather->op != GGML_OP_GET_ROWS ||
                gather->src[0] == nullptr || gather->src[1] == nullptr ||
                copy == nullptr || copy_count != 1) {
                continue;
            }
            ggml_tensor * state_cache = gather->src[0];
            ggml_tensor * state_ids   = gather->src[1];
            ggml_tensor * state_dst   = copy->src[1];
            const int64_t state_size =
                node->src[2]->ne[0] * node->src[2]->ne[0] * node->src[2]->ne[1];
            if (state_cache->type != GGML_TYPE_F32 || state_ids->type != GGML_TYPE_I32 ||
                state_dst == nullptr || state_dst->type != GGML_TYPE_F32 ||
                state_cache->ne[0] != state_size || state_dst->ne[0] != state_size ||
                state_dst->ne[1] != node->src[2]->ne[3] ||
                ggml_sycl_view_base(state_cache) != ggml_sycl_view_base(state_dst)) {
                continue;
            }
            if (!ggml_sycl_reserve_state_io_src(node, 6, state_cache) ||
                !ggml_sycl_reserve_state_io_src(node, 7, state_ids) ||
                !ggml_sycl_reserve_state_io_src(node, 8, state_dst) ||
                !ggml_sycl_reserve_state_io_src(node, 9, gather)) {
                continue;
            }
        }
    }
}

// Extend the routing-weight live range before graph allocation. The production
// batched MoE graph expands all expert views before its ordered ADD chain. When
// that complete tail is present, a non-compute dependency on the final ADD
// prevents gallocr from reusing the routing-weight buffer for the destination.
// The ADD backend reads only src[0] and src[1]; the last source is allocator
// metadata. The runtime overlap check and tiny snapshot remain the safety net.
static void ggml_backend_sycl_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    if (ggml_sycl_state_io_enabled()) {
        ggml_sycl_annotate_state_io(cgraph);
    }

    const bool pipeline_enabled = ggml_sycl_moe_pipeline_enabled();
    const bool disable_down =
        ggml_sycl_moe_down_reduce_disabled() && !pipeline_enabled;
    const bool disable_dual_swiglu =
        ggml_sycl_moe_dual_swiglu_disabled() && !pipeline_enabled;
    if (disable_down && disable_dual_swiglu && !pipeline_enabled) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        // Keep shared activations and ids alive through the fused output. The
        // normal graph considers their last uses to be the two MMID nodes,
        // while the fused kernel consumes them when dispatching the GLU node's
        // destination allocation.
        if (!disable_dual_swiglu && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
            cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            ggml_tensor * mmid0 = cgraph->nodes[i];
            ggml_tensor * mmid1 = cgraph->nodes[i + 1];
            ggml_tensor * glu   = cgraph->nodes[i + 2];
            const int output = i + 2;
            if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                ggml_can_fuse_subgraph(
                    cgraph, i,
                    { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU },
                    { output }) &&
                ((glu->src[0] == mmid0 && glu->src[1] == mmid1) ||
                 (glu->src[0] == mmid1 && glu->src[1] == mmid0)) &&
                mmid0->src[1] == mmid1->src[1] &&
                mmid0->src[2] == mmid1->src[2]) {
                ggml_tensor *& activation_liveness = glu->src[GGML_MAX_SRC - 2];
                ggml_tensor *& ids_liveness        = glu->src[GGML_MAX_SRC - 1];
                if ((activation_liveness == nullptr ||
                     activation_liveness == mmid0->src[1]) &&
                    (ids_liveness == nullptr || ids_liveness == mmid0->src[2])) {
                    activation_liveness = mmid0->src[1];
                    ids_liveness        = mmid0->src[2];
                    ggml_sycl_trace_moe_dual_swiglu(
                        "liveness-annotated", glu->src[0], glu);
                    i += 2;
                    continue;
                }
            }
        }

        if (disable_down) {
            continue;
        }
        ggml_tensor * experts = cgraph->nodes[i];
        if (experts->op != GGML_OP_MUL_MAT_ID ||
            i + 1 >= cgraph->n_nodes) {
            continue;
        }

        ggml_tensor * weighted = cgraph->nodes[i + 1];
        if (weighted->op != GGML_OP_MUL ||
            (weighted->src[0] != experts && weighted->src[1] != experts)) {
            continue;
        }

        ggml_tensor * weights = weighted->src[0] == experts ? weighted->src[1] : weighted->src[0];
        const int64_t nrows            = experts->ne[0];
        const int64_t n_tensor_experts = experts->ne[1];
        const int64_t n_tokens         = experts->ne[2];
        if (experts->type != GGML_TYPE_F32 || weighted->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(experts, weighted) || experts->ne[3] != 1 ||
            weights == nullptr || weights->type != GGML_TYPE_F32 ||
            weights->ne[0] != 1 || weights->ne[1] != n_tensor_experts ||
            weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
            !ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) ||
            n_tensor_experts < 2 || n_tokens < 2) {
            continue;
        }

        // Allocation plans are normally reserved from a warmup graph. Warmup
        // can use every model expert and a much larger token count while still
        // constructing only the production top-k view/ADD tail. Discover that
        // tail instead of applying the <=64-token runtime-fusion bounds here,
        // otherwise decode reuses an unannotated warmup address plan.
        int n_reduce_experts = 0;
        while (i + 2 + n_reduce_experts < cgraph->n_nodes &&
               cgraph->nodes[i + 2 + n_reduce_experts]->op == GGML_OP_VIEW &&
               n_reduce_experts < 16) {
            ++n_reduce_experts;
        }
        if (n_reduce_experts < 2 ||
            (n_reduce_experts == 16 && i + 2 + n_reduce_experts < cgraph->n_nodes &&
             cgraph->nodes[i + 2 + n_reduce_experts]->op == GGML_OP_VIEW)) {
            continue;
        }

        const int count = 2 * n_reduce_experts + 1;
        if (count >= 32 || i + count > cgraph->n_nodes) {
            continue;
        }

        std::vector<ggml_op> ops;
        ops.reserve((size_t) count);
        ops.push_back(GGML_OP_MUL_MAT_ID);
        ops.push_back(GGML_OP_MUL);
        for (int expert = 0; expert < n_reduce_experts; ++expert) {
            ops.push_back(GGML_OP_VIEW);
        }
        for (int expert = 1; expert < n_reduce_experts; ++expert) {
            ops.push_back(GGML_OP_ADD);
        }
        GGML_ASSERT((int) ops.size() == count);

        const int output = i + count - 1;
        if (!ggml_can_fuse_subgraph(cgraph, i, count, ops.data(), &output, 1)) {
            continue;
        }

        ggml_tensor * add = cgraph->nodes[i + 2 + n_reduce_experts];
        ggml_tensor * view0 = cgraph->nodes[i + 2];
        ggml_tensor * view1 = cgraph->nodes[i + 3];
        if (view0->src[0] != weighted || view1->src[0] != weighted ||
            !((add->src[0] == view0 && add->src[1] == view1) ||
              (add->src[1] == view0 && add->src[0] == view1))) {
            continue;
        }
        bool chain_ok = true;
        for (int expert = 2; expert < n_reduce_experts; ++expert) {
            ggml_tensor * view = cgraph->nodes[i + 2 + expert];
            ggml_tensor * next = cgraph->nodes[i + 1 + n_reduce_experts + expert];
            if (view->src[0] != weighted ||
                !((next->src[0] == add && next->src[1] == view) ||
                  (next->src[1] == add && next->src[0] == view))) {
                chain_ok = false;
                break;
            }
            add = next;
        }
        if (!chain_ok || add != cgraph->nodes[output]) {
            continue;
        }

        ggml_tensor * dst = cgraph->nodes[output];
        if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            dst->ne[0] != nrows || dst->ne[1] != n_tokens ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
            continue;
        }
        ggml_tensor * ids = experts->src[2];
        ggml_tensor * activations = experts->src[1];
        ggml_tensor * pipeline_input = nullptr;
        if (pipeline_enabled && activations->op == GGML_OP_GLU &&
            ggml_get_glu_op(activations) == GGML_GLU_OP_SWIGLU &&
            activations->src[0] != nullptr && activations->src[1] != nullptr &&
            activations->src[0]->op == GGML_OP_MUL_MAT_ID &&
            activations->src[1]->op == GGML_OP_MUL_MAT_ID &&
            activations->src[0]->src[1] == activations->src[1]->src[1] &&
            activations->src[0]->src[2] == ids &&
            activations->src[1]->src[2] == ids) {
            pipeline_input = activations->src[0]->src[1];
        }
        ggml_tensor *& pipeline_input_liveness = dst->src[GGML_MAX_SRC - 4];
        ggml_tensor *& ids_liveness         = dst->src[GGML_MAX_SRC - 3];
        ggml_tensor *& activation_liveness  = dst->src[GGML_MAX_SRC - 2];
        ggml_tensor *& weights_liveness     = dst->src[GGML_MAX_SRC - 1];
        if ((pipeline_input_liveness != nullptr &&
             pipeline_input_liveness != pipeline_input) ||
            (ids_liveness != nullptr && ids_liveness != ids) ||
            (activation_liveness != nullptr && activation_liveness != activations) ||
            (weights_liveness != nullptr && weights_liveness != weights)) {
            continue;
        }
        if (pipeline_input != nullptr) {
            pipeline_input_liveness = pipeline_input;
            ggml_sycl_trace_moe_pipeline("liveness-annotated",
                                         activations->src[0], dst);
        }
        ids_liveness        = ids;
        activation_liveness = activations;
        weights_liveness    = weights;
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_LIVENESS_ANNOTATED,
            experts, dst, (unsigned int) n_reduce_experts);
        i += count - 1;
    }
}

static int ggml_sycl_try_fuse_moe_pipeline(
        ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_moe_pipeline_enabled() || i + 4 >= cgraph->n_nodes) {
        return 0;
    }

    ggml_tensor * mmid0 = cgraph->nodes[i];
    ggml_tensor * mmid1 = cgraph->nodes[i + 1];
    ggml_tensor * glu   = cgraph->nodes[i + 2];
    ggml_tensor * down  = cgraph->nodes[i + 3];
    ggml_tensor * weighted = cgraph->nodes[i + 4];
    if (mmid0->op != GGML_OP_MUL_MAT_ID ||
        mmid1->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU ||
        down->op != GGML_OP_MUL_MAT_ID || weighted->op != GGML_OP_MUL ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        !((glu->src[0] == mmid0 && glu->src[1] == mmid1) ||
          (glu->src[0] == mmid1 && glu->src[1] == mmid0)) ||
        mmid0->src[1] != mmid1->src[1] ||
        mmid0->src[2] != mmid1->src[2] || down->src[1] != glu ||
        down->src[2] != mmid0->src[2] ||
        (weighted->src[0] != down && weighted->src[1] != down)) {
        return 0;
    }

    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    ggml_tensor * route_weights =
        weighted->src[0] == down ? weighted->src[1] : weighted->src[0];
    const int64_t n_experts = down->ne[1];
    const int64_t n_tokens  = down->ne[2];
    if (n_experts < 2 || n_experts > 16 || n_tokens < 4 || n_tokens > 64 ||
        weighted->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(down, weighted) ||
        route_weights == nullptr || route_weights->type != GGML_TYPE_F32 ||
        route_weights->ne[0] != 1 || route_weights->ne[1] != n_experts ||
        route_weights->ne[2] != n_tokens || route_weights->ne[3] != 1) {
        return 0;
    }
    ggml_sycl_trace_moe_pipeline("candidate", gate);

    const int down_count = (int) (2 * n_experts + 1);
    const int count = 3 + down_count;
    if (count >= 32 || i + count > cgraph->n_nodes) {
        ggml_sycl_trace_moe_pipeline("bounds-reject", gate);
        return 0;
    }

    std::vector<ggml_op> ops;
    ops.reserve((size_t) count);
    ops.push_back(GGML_OP_MUL_MAT_ID);
    ops.push_back(GGML_OP_MUL_MAT_ID);
    ops.push_back(GGML_OP_GLU);
    ops.push_back(GGML_OP_MUL_MAT_ID);
    ops.push_back(GGML_OP_MUL);
    ops.push_back(GGML_OP_VIEW);
    ops.push_back(GGML_OP_VIEW);
    ops.push_back(GGML_OP_ADD);
    for (int64_t expert = 2; expert < n_experts; ++expert) {
        ops.push_back(GGML_OP_VIEW);
        ops.push_back(GGML_OP_ADD);
    }
    GGML_ASSERT((int) ops.size() == count);
    const int output = i + count - 1;
    // Graph optimization keeps GLU live through the final reduction by adding
    // it as allocator metadata on dst. That edge is inside this exact region,
    // but it was added after cgraph use_counts were built, so the generic
    // subgraph validator observes one more in-region use than its cached count.
    // Validate that GLU has no use outside the candidate, then list it as a
    // checked boundary output solely to bypass that stale-count comparison.
    bool glu_external_use = (glu->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
    for (int node_idx = 0; node_idx < cgraph->n_nodes && !glu_external_use;
         ++node_idx) {
        if (node_idx >= i && node_idx < i + count) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            if (cgraph->nodes[node_idx]->src[src_idx] == glu) {
                glu_external_use = true;
                break;
            }
        }
    }
    if (glu_external_use) {
        ggml_sycl_trace_moe_pipeline("glu-external-use-reject", gate);
        return 0;
    }
    const int checked_outputs[] = { i + 2, output };
    bool views_first = false;
    if (!ggml_can_fuse_subgraph(
            cgraph, i, count, ops.data(), checked_outputs, 2)) {
        ops.clear();
        ops.push_back(GGML_OP_MUL_MAT_ID);
        ops.push_back(GGML_OP_MUL_MAT_ID);
        ops.push_back(GGML_OP_GLU);
        ops.push_back(GGML_OP_MUL_MAT_ID);
        ops.push_back(GGML_OP_MUL);
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            ops.push_back(GGML_OP_VIEW);
        }
        for (int64_t expert = 1; expert < n_experts; ++expert) {
            ops.push_back(GGML_OP_ADD);
        }
        GGML_ASSERT((int) ops.size() == count);
        if (!ggml_can_fuse_subgraph(
                cgraph, i, count, ops.data(), checked_outputs, 2)) {
            ggml_sycl_trace_moe_pipeline("subgraph-reject", gate);
            return 0;
        }
        views_first = true;
    }

    const int down_i = i + 3;
    const auto get_view = [&](int64_t expert) {
        const int node = views_first ? down_i + 2 + (int) expert
                                     : expert < 2 ? down_i + 2 + (int) expert
                                                  : down_i + 2 * (int) expert + 1;
        return cgraph->nodes[node];
    };
    const auto get_add = [&](int64_t expert) {
        GGML_ASSERT(expert >= 1 && expert < n_experts);
        const int node = views_first ? down_i + 1 + (int) n_experts + (int) expert
                                     : down_i + 2 * (int) expert + 2;
        return cgraph->nodes[node];
    };
    ggml_tensor * view0 = get_view(0);
    ggml_tensor * view1 = get_view(1);
    ggml_tensor * add   = get_add(1);
    if (view0->src[0] != weighted || view1->src[0] != weighted ||
        !((add->src[0] == view0 && add->src[1] == view1) ||
          (add->src[1] == view0 && add->src[0] == view1))) {
        ggml_sycl_trace_moe_pipeline("views-reject", gate);
        return 0;
    }
    for (int64_t expert = 2; expert < n_experts; ++expert) {
        ggml_tensor * view = get_view(expert);
        ggml_tensor * next = get_add(expert);
        if (view->src[0] != weighted ||
            !((next->src[0] == add && next->src[1] == view) ||
              (next->src[1] == add && next->src[0] == view))) {
            ggml_sycl_trace_moe_pipeline("chain-reject", gate);
            return 0;
        }
        add = next;
    }

    ggml_tensor * dst = cgraph->nodes[output];
    if (add != dst || !ggml_is_contiguous(route_weights) ||
        !ggml_is_contiguous(dst) ||
        ggml_sycl_tensor_ranges_overlap(gate->src[1], dst) ||
        ggml_sycl_tensor_ranges_overlap(gate->src[2], dst) ||
        ggml_sycl_tensor_ranges_overlap(route_weights, dst)) {
        ggml_sycl_trace_moe_pipeline("overlap-reject", gate, dst);
        return 0;
    }

    if (!ggml_sycl_moe_swiglu_down_pipeline_fused(
            ctx, gate, up, glu, down, route_weights, dst)) {
        ggml_sycl_trace_moe_pipeline("dispatch-reject", gate, dst);
        return 0;
    }
    ggml_sycl_trace_moe_pipeline(
        views_first ? "integrated-hit-views-first" : "integrated-hit",
        gate, dst);
    return count - 1;
}

static int ggml_sycl_try_fuse_moe_down_reduce(
        ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * experts = cgraph->nodes[i];
    if (experts->op != GGML_OP_MUL_MAT_ID ||
        !ggml_sycl_mul_mat_id_fused_eligible(experts)) {
        return 0;
    }

    if (ggml_sycl_moe_down_reduce_disabled() || i + 1 >= cgraph->n_nodes) {
        return 0;
    }

    ggml_tensor * weighted = cgraph->nodes[i + 1];
    if (weighted->op != GGML_OP_MUL ||
        (weighted->src[0] != experts && weighted->src[1] != experts)) {
        return 0;
    }
    ggml_tensor * weights = weighted->src[0] == experts ? weighted->src[1] : weighted->src[0];
    const int64_t nrows     = experts->ne[0];
    const int64_t n_experts = experts->ne[1];
    const int64_t n_tokens  = experts->ne[2];
    if (experts->type != GGML_TYPE_F32 || weighted->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(experts, weighted) || experts->ne[3] != 1 ||
        weights == nullptr || weights->type != GGML_TYPE_F32 ||
        weights->ne[0] != 1 || weights->ne[1] != n_experts ||
        weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        !ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) ||
        n_experts < 2 || n_experts > 16 ||
        n_tokens < 1 || n_tokens > 64) {
        return 0;
    }
    ggml_sycl_trace_moe_down_reduce(GGML_SYCL_MOE_DOWN_REDUCE_ELIGIBLE, experts);

    // Single-token graphs already express the reduction as
    // PERMUTE -> CONT -> SUM_ROWS -> RESHAPE.
    if (n_tokens == 1 && i + 5 < cgraph->n_nodes) {
        constexpr ggml_op ops[] = {
            GGML_OP_MUL_MAT_ID, GGML_OP_MUL, GGML_OP_PERMUTE,
            GGML_OP_CONT, GGML_OP_SUM_ROWS, GGML_OP_RESHAPE,
        };
        const int output = i + 5;
        if (ggml_can_fuse_subgraph(cgraph, i, 6, ops, &output, 1)) {
            ggml_tensor * permute = cgraph->nodes[i + 2];
            ggml_tensor * cont    = cgraph->nodes[i + 3];
            ggml_tensor * sum     = cgraph->nodes[i + 4];
            ggml_tensor * dst     = cgraph->nodes[i + 5];
            // Product decode: routing weights almost always alias the final
            // reshape/sum output (allocator reuse). Snapshotting + fusing was
            // measured −3.5% tg (two-step) to −6.6% tg (integrated) vs unfused
            // on B70 Qwen3.6-35B-A3B (2026-07-21). Hard-reject on weight
            // overlap so decode stays on the faster unfused permute/sum path.
            // Batched path still snapshots and fuses (prefill).
            if (permute->src[0] == weighted && cont->src[0] == permute &&
                sum->src[0] == cont && dst->src[0] == sum &&
                dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
                !ggml_sycl_tensor_ranges_overlap(experts, dst) &&
                !ggml_sycl_tensor_ranges_overlap(weights, dst) &&
                dst->ne[0] == nrows &&
                dst->ne[1] == n_tokens && dst->ne[2] == 1 && dst->ne[3] == 1) {
                // Prefer integrated weighted MMVQ when live ranges are clean.
                // Opt out of integrated: default two-step not needed here —
                // clean-overlap case is rare on product; keep historical path.
                if (ggml_sycl_mul_mat_id_mmvq_weighted(ctx, experts, weights, dst)) {
                    ggml_sycl_trace_moe_down_reduce(
                        GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_HIT, experts, dst, 0x10u);
                    return 5;
                }
                ggml_sycl_mul_mat_id(ctx, experts);
                ggml_sycl_op_moe_weighted_sum(experts, weights, dst, ctx.stream());
                ggml_sycl_trace_moe_down_reduce(
                    GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_HIT, experts, dst);
                return 5;
            }
            ggml_sycl_trace_moe_down_reduce(
                GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_OUTPUT_REJECT, experts, nullptr, 0x1u);
        } else {
            ggml_sycl_trace_moe_down_reduce(
                GGML_SYCL_MOE_DOWN_REDUCE_SINGLE_SUBGRAPH_REJECT, experts, nullptr, 0x1u);
        }
    }

    // Batched graphs use two expert views followed by the first ADD, then one
    // VIEW+ADD pair for every remaining routed expert.  Fuse the complete
    // reduction only when every intermediate use is confined to this subgraph.
    const int count = (int) (2 * n_experts + 1);
    if (count >= 32 || i + count > cgraph->n_nodes) {
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_BOUNDS_REJECT, experts, nullptr,
            count >= 32 ? 0x01u : 0x02u);
        return 0;
    }
    std::vector<ggml_op> ops;
    ops.reserve((size_t) count);
    ops.push_back(GGML_OP_MUL_MAT_ID);
    ops.push_back(GGML_OP_MUL);
    ops.push_back(GGML_OP_VIEW);
    ops.push_back(GGML_OP_VIEW);
    ops.push_back(GGML_OP_ADD);
    for (int64_t expert = 2; expert < n_experts; ++expert) {
        ops.push_back(GGML_OP_VIEW);
        ops.push_back(GGML_OP_ADD);
    }
    GGML_ASSERT((int) ops.size() == count);
    const int output = i + count - 1;
    bool views_first = false;
    if (!ggml_can_fuse_subgraph(cgraph, i, count, ops.data(), &output, 1)) {
        // Production explicitly expands every expert view before constructing
        // the ordered ADD chain. Fixtures that only return the final tensor
        // are recursively expanded into the interleaved layout above.
        ops.clear();
        ops.push_back(GGML_OP_MUL_MAT_ID);
        ops.push_back(GGML_OP_MUL);
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            ops.push_back(GGML_OP_VIEW);
        }
        for (int64_t expert = 1; expert < n_experts; ++expert) {
            ops.push_back(GGML_OP_ADD);
        }
        GGML_ASSERT((int) ops.size() == count);
        if (!ggml_can_fuse_subgraph(cgraph, i, count, ops.data(), &output, 1)) {
            ggml_sycl_trace_moe_down_reduce(
                GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_SUBGRAPH_REJECT, experts);
            return 0;
        }
        views_first = true;
    }

    const auto get_view = [&](int64_t expert) {
        const int node = views_first ? i + 2 + (int) expert
                                     : expert < 2 ? i + 2 + (int) expert
                                                  : i + 2 * (int) expert + 1;
        return cgraph->nodes[node];
    };
    const auto get_add = [&](int64_t expert) {
        GGML_ASSERT(expert >= 1 && expert < n_experts);
        const int node = views_first ? i + 1 + (int) n_experts + (int) expert
                                     : i + 2 * (int) expert + 2;
        return cgraph->nodes[node];
    };

    ggml_tensor * view0 = get_view(0);
    ggml_tensor * view1 = get_view(1);
    ggml_tensor * add   = get_add(1);
    if (view0->src[0] != weighted || view1->src[0] != weighted ||
        !((add->src[0] == view0 && add->src[1] == view1) ||
          (add->src[1] == view0 && add->src[0] == view1))) {
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_VIEWS_REJECT, experts);
        return 0;
    }
    for (int64_t expert = 2; expert < n_experts; ++expert) {
        ggml_tensor * view = get_view(expert);
        ggml_tensor * next = get_add(expert);
        if (view->src[0] != weighted ||
            !((next->src[0] == add && next->src[1] == view) ||
              (next->src[1] == add && next->src[0] == view))) {
            ggml_sycl_trace_moe_down_reduce(
                GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_CHAIN_REJECT, experts, nullptr, (unsigned int) expert);
            return 0;
        }
        add = next;
    }
    ggml_tensor * dst = cgraph->nodes[output];
    const bool dst_contiguous = ggml_is_contiguous(dst);
    const bool experts_overlap = dst_contiguous && ggml_sycl_tensor_ranges_overlap(experts, dst);
    const bool weights_overlap = dst_contiguous && ggml_sycl_tensor_ranges_overlap(weights, dst);
    const bool activations_overlap = dst_contiguous &&
        ggml_sycl_tensor_ranges_overlap(experts->src[1], dst);
    const bool ids_overlap = dst_contiguous &&
        ggml_sycl_tensor_ranges_overlap(experts->src[2], dst);
    const bool dst_shape = dst->ne[0] == nrows && dst->ne[1] == n_tokens &&
                           dst->ne[2] == 1 && dst->ne[3] == 1;
    if (add != dst || dst->type != GGML_TYPE_F32 || !dst_contiguous ||
        experts_overlap || !dst_shape) {
        const unsigned int detail =
            (add != dst                  ? 0x01u : 0u) |
            (dst->type != GGML_TYPE_F32 ? 0x02u : 0u) |
            (!dst_contiguous             ? 0x04u : 0u) |
            (experts_overlap             ? 0x08u : 0u) |
            (!dst_shape                  ? 0x20u : 0u);
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_OUTPUT_REJECT, experts, dst, detail);
        return 0;
    }

    // The graph allocator may reuse the routing-weight allocation for an ADD
    // output after the materialized MUL has consumed it. Fusion extends the
    // weight live range into the reduction, so direct reads would race with
    // destination writes. Snapshot only this tiny tensor (<= 4 KiB for the
    // accepted top-16, 64-token bounds) on the in-order queue before reducing.
    ggml_sycl_pool_alloc<float> weights_snapshot(ctx.pool());
    ggml_tensor safe_weights = *weights;
    if (weights_overlap) {
        weights_snapshot.alloc((size_t) ggml_nelements(weights));
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(weights_snapshot.get(), weights->data, ggml_nbytes(weights))));
        safe_weights.data = weights_snapshot.get();
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_WEIGHTS_SNAPSHOT, experts, dst);
    }

    const ggml_tensor * safe_weights_ptr = weights_overlap ? &safe_weights : weights;
    if (!activations_overlap && !ids_overlap &&
        ggml_sycl_mul_mat_id_mmvq_weighted(ctx, experts, safe_weights_ptr, dst)) {
        ggml_sycl_trace_moe_down_reduce(
            GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_INTEGRATED_HIT,
            experts, dst, views_first ? 0x01u : 0u);
        return count - 1;
    }

    ggml_sycl_mul_mat_id(ctx, experts);
    ggml_sycl_op_moe_weighted_sum(
        experts, safe_weights_ptr, dst, ctx.stream());
    ggml_sycl_trace_moe_down_reduce(
        GGML_SYCL_MOE_DOWN_REDUCE_BATCHED_HIT, experts, dst, views_first ? 0x01u : 0u);
    return count - 1;
}

enum ggml_sycl_fusion_profile_kind {
    GGML_SYCL_FUSION_PROFILE_NONE,
    GGML_SYCL_FUSION_PROFILE_MOE_PIPELINE,
    GGML_SYCL_FUSION_PROFILE_MOE_DUAL_SWIGLU,
    GGML_SYCL_FUSION_PROFILE_MOE_DOWN_REDUCE,
    GGML_SYCL_FUSION_PROFILE_TOPK_MOE,
    GGML_SYCL_FUSION_PROFILE_DELTANET_GLUE,
    GGML_SYCL_FUSION_PROFILE_STATE_IO,
    GGML_SYCL_FUSION_PROFILE_RMS_NORM_MUL_ADD,
    GGML_SYCL_FUSION_PROFILE_RMS_NORM_MUL,
    GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_SWIGLU,
    GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_MMVQ,
    GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_F32,
    GGML_SYCL_FUSION_PROFILE_COUNT,
};

// Partners of non-adjacent dense dual-MMVQ fuses; cleared each graph eval.
// Looked up from try_fuse and the SYCL graph compute loop.
static thread_local std::unordered_set<ggml_tensor *> g_sycl_dual_mmvq_elide;

static int ggml_sycl_try_fuse(
        ggml_backend_sycl_context & ctx,
        ggml_cgraph * cgraph,
        int i,
        ggml_sycl_fusion_profile_kind * profile_kind) {
    GGML_ASSERT(profile_kind != nullptr);
    *profile_kind = GGML_SYCL_FUSION_PROFILE_NONE;
    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID) {
        const int skip = ggml_sycl_try_fuse_moe_pipeline(ctx, cgraph, i);
        if (skip != 0) {
            *profile_kind = GGML_SYCL_FUSION_PROFILE_MOE_PIPELINE;
            return skip;
        }
    }

    // Separate expert gate/up MMIDs share activations and routing ids. Fuse
    // both projections with SwiGLU into one output-producing submission.
    if (!ggml_sycl_moe_dual_swiglu_disabled() &&
        i + 2 < cgraph->n_nodes && node->op == GGML_OP_MUL_MAT_ID &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
        cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
        ggml_tensor * mmid0 = node;
        ggml_tensor * mmid1 = cgraph->nodes[i + 1];
        ggml_tensor * glu   = cgraph->nodes[i + 2];
        const int output = i + 2;
        const bool edges_ok =
            ((glu->src[0] == mmid0 && glu->src[1] == mmid1) ||
             (glu->src[0] == mmid1 && glu->src[1] == mmid0)) &&
            mmid0->src[1] == mmid1->src[1] &&
            mmid0->src[2] == mmid1->src[2];
        if (edges_ok) {
            ggml_sycl_trace_moe_dual_swiglu("candidate", glu->src[0], glu);
        }
        const bool subgraph_ok = ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU },
            { output });
        if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
            subgraph_ok && edges_ok) {
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];
            ggml_tensor * activations = gate->src[1];
            ggml_tensor * ids = gate->src[2];
            const bool activations_overlap =
                ggml_sycl_tensor_ranges_overlap(activations, glu);
            const bool ids_overlap = ggml_sycl_tensor_ranges_overlap(ids, glu);
            if (activations_overlap || ids_overlap) {
                ggml_sycl_trace_moe_dual_swiglu(
                    activations_overlap ? "activations-overlap" : "ids-overlap",
                    gate, glu);
            } else if (ggml_sycl_mul_mat_id_dual_swiglu_fused(
                           ctx, gate, up, glu)) {
                ggml_sycl_trace_moe_dual_swiglu(
                    "dual-integrated-hit", gate, glu);
                *profile_kind = GGML_SYCL_FUSION_PROFILE_MOE_DUAL_SWIGLU;
                return 2;
            } else {
                ggml_sycl_trace_moe_dual_swiglu(
                    "dispatch-reject", gate, glu);
            }
        }
    }

    // Dense shared-expert dual-SwiGLU (MUL_MAT + MUL_MAT + GLU). Default ON.
    // A/B off: GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1.
    static const bool disable_dense_dual = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!disable_dense_dual &&
        i + 2 < cgraph->n_nodes && node->op == GGML_OP_MUL_MAT &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
        cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
        ggml_tensor * mm0 = node;
        ggml_tensor * mm1 = cgraph->nodes[i + 1];
        ggml_tensor * glu = cgraph->nodes[i + 2];
        const int output = i + 2;
        const bool edges_ok =
            ((glu->src[0] == mm0 && glu->src[1] == mm1) ||
             (glu->src[0] == mm1 && glu->src[1] == mm0)) &&
            mm0->src[1] == mm1->src[1];
        const bool subgraph_ok = ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU },
            { output });
        if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
            subgraph_ok && edges_ok) {
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];
            ggml_tensor * activations = gate->src[1];
            if (!ggml_sycl_tensor_ranges_overlap(activations, glu) &&
                ggml_sycl_mul_mat_dense_dual_swiglu_fused(ctx, gate, up, glu)) {
                *profile_kind = GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_SWIGLU;
                return 2;
            }
        }
    }

    // Dense dual MMVQ: two MUL_MATs sharing one activation. Independent
    // projections get graph-sorted apart, so search ahead for a partner and
    // elide it when non-adjacent (see g_sycl_dual_mmvq_elide).
    // Must NOT steal dense dual-SwiGLU pairs (MUL_MAT+MUL_MAT+GLU).
    // Default OFF (parked 2026-07-20: r=3 tg flat / prefill-large regress).
    // Opt-in: GGML_SYCL_ENABLE_DENSE_DUAL_MMVQ=1.
    static const bool enable_dense_dual_mmvq = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_DENSE_DUAL_MMVQ");
        return env != nullptr && std::atoi(env) != 0;
    }();
    static const bool trace_dual_mmvq = []() {
        const char * env = getenv("GGML_SYCL_DENSE_DUAL_MMVQ_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (enable_dense_dual_mmvq && node->op == GGML_OP_MUL_MAT) {
        ggml_tensor * mm0 = node;
        // Already the elided partner of an earlier dual fuse.
        if (g_sycl_dual_mmvq_elide.count(mm0) != 0) {
            return 0;
        }
        const enum ggml_type wtype = mm0->src[0]->type;
        const bool type_ok =
            wtype == GGML_TYPE_Q8_0 || wtype == GGML_TYPE_Q4_K ||
            wtype == GGML_TYPE_Q5_K || wtype == GGML_TYPE_Q6_K;
        int j = -1;
        if (type_ok) {
            // Search remaining graph for the first compatible partner on the
            // same activation. Prefer nearest (keeps full-attn K/V tight).
            for (int k = i + 1; k < cgraph->n_nodes; ++k) {
                ggml_tensor * cand = cgraph->nodes[k];
                if (cand->op != GGML_OP_MUL_MAT) {
                    continue;
                }
                if (g_sycl_dual_mmvq_elide.count(cand) != 0) {
                    continue;
                }
                if (cand->src[1] != mm0->src[1] ||
                    cand->src[0] == mm0->src[0] ||
                    cand->src[0]->type != wtype) {
                    continue;
                }
                if (cand->src[0] == mm0 || cand->src[1] == mm0 ||
                    mm0->src[0] == cand || mm0->src[1] == cand) {
                    continue;
                }
                // Leave gate+up+SwiGLU to dense dual-SwiGLU.
                bool owned_by_swiglu = false;
                if (k + 1 < cgraph->n_nodes &&
                    cgraph->nodes[k + 1]->op == GGML_OP_GLU) {
                    ggml_tensor * glu = cgraph->nodes[k + 1];
                    owned_by_swiglu =
                        (glu->src[0] == mm0 && glu->src[1] == cand) ||
                        (glu->src[0] == cand && glu->src[1] == mm0);
                }
                // Also: if mm0 is immediately followed by GLU with cand as the
                // other src, dense dual-swiglu owns it (adjacent shexp case).
                if (i + 2 < cgraph->n_nodes &&
                    cgraph->nodes[i + 1] == cand &&
                    cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                    ggml_tensor * glu = cgraph->nodes[i + 2];
                    owned_by_swiglu =
                        owned_by_swiglu ||
                        (glu->src[0] == mm0 && glu->src[1] == cand) ||
                        (glu->src[0] == cand && glu->src[1] == mm0);
                }
                if (owned_by_swiglu) {
                    continue;
                }
                j = k;
                break;
            }
        }
        if (j > i) {
            ggml_tensor * mm1 = cgraph->nodes[j];
            if (ggml_sycl_mul_mat_dense_dual_mmvq_fused(ctx, mm0, mm1)) {
                *profile_kind = GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_MMVQ;
                if (j == i + 1) {
                    // Adjacent: classic skip-1 (current + next).
                    return 1;
                }
                // Non-adjacent: elide partner when the loop reaches it; return
                // -1 so the eval loop skips recompute of mm0 only.
                g_sycl_dual_mmvq_elide.insert(mm1);
                if (trace_dual_mmvq) {
                    static std::atomic<int> hits{0};
                    if (hits.fetch_add(1, std::memory_order_relaxed) < 12) {
                        fprintf(stderr,
                                "[treebeard-dense-dual-mmvq] hit-nonadj"
                                " i=%d j=%d wa=%s wb=%s type=%s"
                                " ne1_a=%" PRId64 " ne1_b=%" PRId64
                                " act_ne1=%" PRId64 "\n",
                                i, j, mm0->src[0]->name, mm1->src[0]->name,
                                ggml_type_name(wtype),
                                mm0->src[0]->ne[1], mm1->src[0]->ne[1],
                                mm0->src[1]->ne[1]);
                    }
                }
                return -1;
            }
            if (trace_dual_mmvq) {
                static std::atomic<int> miss{0};
                if (miss.fetch_add(1, std::memory_order_relaxed) < 12) {
                    fprintf(stderr,
                            "[treebeard-dense-dual-mmvq] dispatch-miss"
                            " i=%d j=%d wa=%s wb=%s\n",
                            i, j, mm0->src[0]->name, mm1->src[0]->name);
                }
            }
        }
    }

    // Dense dual F32 GEMV: two F32 MUL_MATs sharing one activation
    // (ssm_alpha+ssm_beta). Prefer equal-shape partners (same nrows) so we do
    // not pair router / sparse F32 mats with the GDN pair by accident.
    // Opt-in: GGML_SYCL_ENABLE_DENSE_DUAL_F32=1 (default OFF until gated).
    static const bool enable_dense_dual_f32 = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_DENSE_DUAL_F32");
        return env != nullptr && std::atoi(env) != 0;
    }();
    static const bool trace_dual_f32 = []() {
        const char * env = getenv("GGML_SYCL_DENSE_DUAL_F32_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (enable_dense_dual_f32 && node->op == GGML_OP_MUL_MAT) {
        ggml_tensor * mm0 = node;
        if (g_sycl_dual_mmvq_elide.count(mm0) != 0) {
            return 0;
        }
        if (mm0->src[0]->type == GGML_TYPE_F32) {
            // Two-pass partner pick: (1) equal nrows (preferred for alpha/beta),
            // (2) any other F32 same-act partner. Retry on dispatch miss.
            auto try_partner = [&](int j) -> int {
                ggml_tensor * mm1 = cgraph->nodes[j];
                if (!ggml_sycl_mul_mat_dense_dual_f32_fused(ctx, mm0, mm1)) {
                    return 0;
                }
                *profile_kind = GGML_SYCL_FUSION_PROFILE_DENSE_DUAL_F32;
                if (j == i + 1) {
                    return 1;
                }
                g_sycl_dual_mmvq_elide.insert(mm1);
                if (trace_dual_f32) {
                    static std::atomic<int> hits{0};
                    if (hits.fetch_add(1, std::memory_order_relaxed) < 12) {
                        fprintf(stderr,
                                "[treebeard-dense-dual-f32] hit-nonadj"
                                " i=%d j=%d wa=%s wb=%s"
                                " ne1_a=%" PRId64 " ne1_b=%" PRId64 "\n",
                                i, j, mm0->src[0]->name, mm1->src[0]->name,
                                mm0->src[0]->ne[1], mm1->src[0]->ne[1]);
                    }
                }
                return -1;
            };
            int equal_j = -1;
            int any_j   = -1;
            for (int k = i + 1; k < cgraph->n_nodes; ++k) {
                ggml_tensor * cand = cgraph->nodes[k];
                if (cand->op != GGML_OP_MUL_MAT) {
                    continue;
                }
                if (g_sycl_dual_mmvq_elide.count(cand) != 0) {
                    continue;
                }
                if (cand->src[1] != mm0->src[1] ||
                    cand->src[0] == mm0->src[0] ||
                    cand->src[0]->type != GGML_TYPE_F32) {
                    continue;
                }
                if (cand->src[0] == mm0 || cand->src[1] == mm0 ||
                    mm0->src[0] == cand || mm0->src[1] == cand) {
                    continue;
                }
                if (any_j < 0) {
                    any_j = k;
                }
                if (equal_j < 0 && cand->src[0]->ne[1] == mm0->src[0]->ne[1] &&
                    cand->src[0]->ne[0] == mm0->src[0]->ne[0]) {
                    equal_j = k;
                    break; // nearest equal-shape is ideal
                }
            }
            if (equal_j > i) {
                if (trace_dual_f32) {
                    static std::atomic<int> cands{0};
                    if (cands.fetch_add(1, std::memory_order_relaxed) < 16) {
                        fprintf(stderr,
                                "[treebeard-dense-dual-f32] equal-cand"
                                " i=%d j=%d wa=%s wb=%s act_ne1=%" PRId64 "\n",
                                i, equal_j, mm0->src[0]->name,
                                cgraph->nodes[equal_j]->src[0]->name,
                                mm0->src[1]->ne[1]);
                    }
                }
                const int skip = try_partner(equal_j);
                if (skip != 0) {
                    return skip;
                }
            }
            // Do not fall through to unequal-shape partners for F32: the only
            // ship wins are equal-shape GDN pairs; unequal often pairs router
            // weight with a 1-row projection and would just shape-reject.
            (void) any_j;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        const int skip = ggml_sycl_try_fuse_moe_down_reduce(ctx, cgraph, i);
        if (skip != 0) {
            *profile_kind = GGML_SYCL_FUSION_PROFILE_MOE_DOWN_REDUCE;
            return skip;
        }
    }

    // topk-moe router fusion: collapses the softmax->top-k->get_rows[->norm][->scale] serial chain that
    // otherwise blocks the expert GEMVs. This chain is submit-bound (not hidden behind a big op), so it
    // is the lever that actually moves decode here. Default ON; A/B with GGML_SYCL_DISABLE_TOPK_MOE=1.
    static const bool disable_topk_moe = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_TOPK_MOE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!disable_topk_moe) {
        const int skip = ggml_sycl_try_fuse_topk_moe(ctx, cgraph, i);
        if (skip != 0) {
            *profile_kind = GGML_SYCL_FUSION_PROFILE_TOPK_MOE;
            return skip;
        }
    }

    // gated-delta-net gate glue: ADD -> SOFTPLUS(unary) -> MUL  =>  softplus(alpha + ssm_dt) * ssm_a.
    // A 3-deep tiny-op serial chain per linear-attn layer (x30); collapse to one elementwise kernel.
    // Default ON; A/B with GGML_SYCL_DISABLE_DELTANET_GLUE=1.
    static const bool disable_dn_glue = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_DELTANET_GLUE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!disable_dn_glue && node->op == GGML_OP_ADD &&
        ggml_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL })) {
        ggml_tensor * add = node;
        ggml_tensor * sp  = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        if (ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS) {
            // a = full-shape operand of the add, b = its broadcast 1-D operand; c = mul's non-softplus operand
            ggml_tensor * a = ggml_are_same_shape(add->src[0], add) ? add->src[0] : add->src[1];
            ggml_tensor * b = (a == add->src[0]) ? add->src[1] : add->src[0];
            ggml_tensor * c = (mul->src[0] == sp) ? mul->src[1] : mul->src[0];
            const bool ok = a && b && c &&
                a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                c->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(a, add) && ggml_is_contiguous(a) && ggml_is_contiguous(mul) &&
                ggml_is_contiguous(b) && ggml_nelements(b) == a->ne[0] &&
                ggml_is_contiguous(c) && ggml_nelements(c) == a->ne[0];
            if (ok) {
                ggml_sycl_op_fused_add_softplus_mul(ctx, a, b, c, mul);
                *profile_kind = GGML_SYCL_FUSION_PROFILE_DELTANET_GLUE;
                return 2;
            }
        }
    }

    // Glue fusion (RMS_NORM+MUL[+ADD]): collapse norm-weight MUL and residual ADD into one kernel.
    // On B70 Qwen3.6-35B-A3B (hybrid MoE + SSM/GDN) with the ship speed path (oneDNN + f16 KV),
    // measured 2026-07-20: tg128 +~5% vs unfused glue (80.87 -> 85.05 t/s, r=3). Prefill flat.
    // Default ON. Disable with GGML_SYCL_DISABLE_FUSION=1, or GGML_SYCL_ENABLE_FUSION=0.
    static const bool enable_glue_fusion = []() {
        const char * disable = getenv("GGML_SYCL_DISABLE_FUSION");
        if (disable != nullptr && std::atoi(disable) != 0) {
            return false;
        }
        const char * enable = getenv("GGML_SYCL_ENABLE_FUSION");
        // Explicit 0 disables; unset or non-zero keeps default ON.
        if (enable != nullptr && std::atoi(enable) == 0) {
            return false;
        }
        return true;
    }();
    if (!enable_glue_fusion) {
        return 0;
    }

    // RMS_NORM (+ MUL weight) (+ ADD residual/bias)  ->  one fused norm kernel
    if (node->op == GGML_OP_RMS_NORM) {
        auto types_ok = [](const ggml_tensor * t) {
            return t->src[0]->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32;
        };
        // longest match first: RMS_NORM + MUL + ADD
        if (ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            ggml_tensor * mul = cgraph->nodes[i + 1];
            ggml_tensor * add = cgraph->nodes[i + 2];
            const bool bcast_ok = !(node == mul->src[1] && !ggml_are_same_shape(mul->src[0], node));
            if (node->src[0]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                types_ok(mul) && types_ok(add) && bcast_ok &&
                ggml_is_contiguous_rows(mul->src[0]) && ggml_is_contiguous_rows(mul->src[1]) &&
                ggml_is_contiguous(add->src[0]) && ggml_is_contiguous_rows(add->src[1])) {
                ggml_sycl_op_rms_norm_fused_add(ctx, node, mul, add);
                *profile_kind = GGML_SYCL_FUSION_PROFILE_RMS_NORM_MUL_ADD;
                return 2;
            }
        }
        // RMS_NORM + MUL
        if (ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_tensor * mul = cgraph->nodes[i + 1];
            const bool bcast_ok = !(node == mul->src[1] && !ggml_are_same_shape(mul->src[0], node));
            if (node->src[0]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                types_ok(mul) && bcast_ok &&
                ggml_is_contiguous_rows(mul->src[0]) && ggml_is_contiguous_rows(mul->src[1])) {
                ggml_sycl_op_rms_norm_fused(ctx, node, mul);
                *profile_kind = GGML_SYCL_FUSION_PROFILE_RMS_NORM_MUL;
                return 1;
            }
        }
    }

    return 0;
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    g_sycl_dual_mmvq_elide.clear();
    ggml_sycl_set_main_device(sycl_ctx->device);

    // Optional Treebeard SYCL per-op profiler (TREEBEARD_SYCL_PROF=1): serializes each op with a
    // queue wait and accumulates host-observed GPU time per op type. Serialization inflates
    // absolutes (removes overlap) but the RELATIVE breakdown + serialized-total-vs-wall ratio
    // reveal where decode time goes. Diagnostic only - not a ship path.
    // TREEBEARD_SYCL_PROF_TRIGGER_FILE: leave execution unmodified until that file exists so a
    // late profile window can open after expensive prompt/state setup without restarting.
    // Legacy aliases (still accepted): SIQ_PROF, SIQ_PROF_TRIGGER_FILE.
    static const bool prof = []() {
        return getenv("TREEBEARD_SYCL_PROF") != nullptr || getenv("SIQ_PROF") != nullptr;
    }();
    static const std::string prof_trigger_file = []() {
        const char * path = getenv("TREEBEARD_SYCL_PROF_TRIGGER_FILE");
        if (path == nullptr) {
            path = getenv("SIQ_PROF_TRIGGER_FILE");
        }
        return path != nullptr ? std::string(path) : std::string();
    }();
    static std::atomic<bool> prof_active { prof_trigger_file.empty() };
    if (prof && !prof_active.load(std::memory_order_relaxed)) {
        FILE * trigger = fopen(prof_trigger_file.c_str(), "rb");
        if (trigger != nullptr) {
            fclose(trigger);
            if (!prof_active.exchange(true, std::memory_order_relaxed)) {
                fprintf(stderr, "[treebeard-sycl-prof] trigger active file=%s\n",
                        prof_trigger_file.c_str());
            }
        }
    }
    const bool prof_now = prof && prof_active.load(std::memory_order_relaxed);
    static double op_us[GGML_OP_COUNT] = { 0 };
    static long   op_n [GGML_OP_COUNT] = { 0 };
    static double unary_us = 0;
    static long unary_n = 0;
    static double fusion_us[GGML_SYCL_FUSION_PROFILE_COUNT] = { 0 };
    static long fusion_n[GGML_SYCL_FUSION_PROFILE_COUNT] = { 0 };
    struct named_profile_stat {
        std::string name;
        double us;
        long n;
    };
    static std::vector<named_profile_stat> mul_mat_stats;
    static std::vector<named_profile_stat> state_op_stats;
    static int    geval = 0;
    auto now_us = []() {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    auto normalize_profile_name = [](const char * raw_name) {
        std::string name = raw_name != nullptr ? raw_name : "unnamed";
        for (size_t pos = 0; pos < name.size();) {
            if (name[pos] < '0' || name[pos] > '9') {
                ++pos;
                continue;
            }
            size_t end = pos;
            while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
                ++end;
            }
            name.replace(pos, end - pos, "*");
            ++pos;
        }
        return name;
    };
    auto mul_mat_profile_name = [&](const ggml_tensor * node) {
        const std::string node_name = ggml_get_name(node);
        const bool generic_name = node_name.empty() || node_name == "unnamed" ||
                                  node_name.rfind("node_", 0) == 0;
        if (!generic_name) {
            return normalize_profile_name(node_name.c_str());
        }

        const ggml_tensor * weights = node->src[0];
        if (weights != nullptr) {
            const std::string weight_name = ggml_get_name(weights);
            if (!weight_name.empty() && weight_name.rfind("node_", 0) != 0) {
                return std::string("weight:") + normalize_profile_name(weight_name.c_str());
            }

            char signature[192];
            snprintf(signature, sizeof(signature),
                     "shape:%s[%" PRId64 "x%" PRId64 "x%" PRId64 "]->%s[%" PRId64 "x%" PRId64 "]",
                     ggml_type_name(weights->type), weights->ne[0], weights->ne[1], weights->ne[2],
                     ggml_type_name(node->type), node->ne[0], node->ne[1]);
            return std::string(signature);
        }

        return normalize_profile_name(node_name.c_str());
    };
    auto state_op_profile_name = [&](const ggml_tensor * node) {
        auto tensor_name = [&](const ggml_tensor * tensor) {
            if (tensor == nullptr) {
                return std::string("none");
            }
            const std::string raw_name = ggml_get_name(tensor);
            if (!raw_name.empty() && raw_name != "unnamed" &&
                raw_name.rfind("node_", 0) != 0) {
                return normalize_profile_name(raw_name.c_str());
            }
            char signature[128];
            snprintf(signature, sizeof(signature),
                     "%s[%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 "]",
                     ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1],
                     tensor->ne[2], tensor->ne[3]);
            return std::string(signature);
        };
        return std::string(ggml_op_name(node->op)) + ":" + tensor_name(node) +
               "<-" + tensor_name(node->src[0]) + "," + tensor_name(node->src[1]);
    };
    auto add_named_profile_stat = [](std::vector<named_profile_stat> & stats,
                                     std::string name, double us) {
        const auto it = std::find_if(stats.begin(), stats.end(), [&](const named_profile_stat & stat) {
            return stat.name == name;
        });
        if (it != stats.end()) {
            it->us += us;
            it->n++;
        } else {
            stats.push_back({ std::move(name), us, 1 });
        }
    };

    static const bool state_io_debug = []() {
        const char * env = getenv("GGML_SYCL_STATE_IO_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    static std::atomic<bool> state_io_dumped { false };
    if (state_io_debug && !state_io_dumped.load(std::memory_order_relaxed)) {
        bool full_width = false;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_GATED_DELTA_NET && node->src[2] != nullptr &&
                node->src[2]->ne[3] == 12) {
                full_width = true;
                break;
            }
        }
        if (full_width && !state_io_dumped.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr, "[treebeard-state-io] event=graph-dump nodes=%d\n", cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (node->op != GGML_OP_GET_ROWS && node->op != GGML_OP_CPY &&
                    node->op != GGML_OP_CONCAT && node->op != GGML_OP_SSM_CONV &&
                    node->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                fprintf(stderr,
                        "[treebeard-state-io] index=%d op=%s name=%s ptr=%p data=%p"
                        " view_src=%p uses=%d",
                        i, ggml_op_name(node->op), ggml_get_name(node), (const void *) node,
                        node->data, (void *) node->view_src,
                        ggml_node_get_use_count(cgraph, i));
                for (int src = 0; src < 6 && node->src[src] != nullptr; ++src) {
                    fprintf(stderr, " src%d=%p/%s/%s", src, (void *) node->src[src],
                            ggml_op_name(node->src[src]->op), ggml_get_name(node->src[src]));
                }
                fputc('\n', stderr);
            }
        }
    }

    struct state_io_plan_entry {
        ggml_tensor * marker;
        ggml_tensor * gather;
        int64_t n_rs;
    };
    std::vector<state_io_plan_entry> state_io_plan;
    std::vector<ggml_tensor *> state_io_direct;
    std::vector<ggml_tensor *> state_io_elide;
    if (ggml_sycl_state_io_enabled()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * marker = cgraph->nodes[i];
            const bool annotated_ssm = marker->op == GGML_OP_SSM_CONV &&
                marker->src[2] != nullptr && marker->src[3] != nullptr &&
                marker->src[4] != nullptr && marker->src[5] != nullptr &&
                marker->src[6] != nullptr;
            const bool annotated_gdn = marker->op == GGML_OP_GATED_DELTA_NET &&
                marker->src[6] != nullptr && marker->src[7] != nullptr &&
                marker->src[8] != nullptr && marker->src[9] != nullptr;
            if (!annotated_ssm && !annotated_gdn) {
                continue;
            }

            ggml_tensor * producer = annotated_ssm ? marker->src[0] : marker;
            ggml_tensor * gather = annotated_ssm
                ? ggml_sycl_view_base(marker->src[0]->src[0])
                : ggml_sycl_view_base(marker->src[5]);
            ggml_tensor * copy = ggml_sycl_find_state_copy(cgraph, producer);
            if (gather == nullptr || gather->op != GGML_OP_GET_ROWS || copy == nullptr) {
                continue;
            }
            ggml_tensor * state_ids = annotated_ssm ? marker->src[3] : marker->src[7];
            const ggml_tensor * ids_all = ggml_sycl_view_base(state_ids);
            const int64_t n_rs = ids_all != nullptr ? ggml_nelements(ids_all) : 0;
            const int64_t n_seqs = annotated_ssm ? marker->ne[2] : marker->src[2]->ne[3];
            if (n_rs < n_seqs) {
                continue;
            }

            state_io_plan.push_back({ marker, gather, n_rs });
            state_io_direct.push_back(marker);
            state_io_elide.push_back(copy);
            if (annotated_ssm) {
                state_io_elide.push_back(marker->src[0]);
            }
        }
        if (state_io_debug && !state_io_direct.empty()) {
            static std::atomic<int> plan_traces { 0 };
            const int trace = plan_traces.fetch_add(1, std::memory_order_relaxed);
            if (trace < 8) {
                fprintf(stderr,
                        "[treebeard-state-io] event=runtime-plan direct=%zu conflict-gathers=%zu elide=%zu\n",
                        state_io_direct.size(), state_io_plan.size(), state_io_elide.size());
            }
        }
    }

    const auto state_io_contains = [](const std::vector<ggml_tensor *> & tensors,
                                      const ggml_tensor * node) {
        return std::find(tensors.begin(), tensors.end(), node) != tensors.end();
    };
    const auto state_io_plan_for_gather = [&](const ggml_tensor * gather) {
        return std::find_if(state_io_plan.begin(), state_io_plan.end(),
                            [&](const state_io_plan_entry & entry) {
                                return entry.gather == gather;
                            });
    };
    const auto state_io_plan_for_marker = [&](const ggml_tensor * marker) {
        return std::find_if(state_io_plan.begin(), state_io_plan.end(),
                            [&](const state_io_plan_entry & entry) {
                                return entry.marker == marker;
                            });
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        const auto gather_plan = state_io_plan_for_gather(node);
        if (gather_plan != state_io_plan.end()) {
            const double tf0 = prof_now ? now_us() : 0.0;
            ggml_tensor * marker = gather_plan->marker;
            if (marker->op == GGML_OP_SSM_CONV) {
                ggml_sycl_state_io_gather_conflicts(
                    *sycl_ctx, marker->src[2], marker->src[3], marker->src[6],
                    marker->src[5], gather_plan->n_rs);
            } else {
                GGML_ASSERT(marker->op == GGML_OP_GATED_DELTA_NET);
                ggml_sycl_state_io_gather_conflicts(
                    *sycl_ctx, marker->src[6], marker->src[7], marker->src[9],
                    marker->src[8], gather_plan->n_rs);
            }
            if (prof_now) {
                sycl_ctx->stream()->wait();
                fusion_us[GGML_SYCL_FUSION_PROFILE_STATE_IO] += now_us() - tf0;
                fusion_n[GGML_SYCL_FUSION_PROFILE_STATE_IO]++;
            }
            continue;
        }
        if (state_io_contains(state_io_elide, node)) {
            continue;
        }
        if (g_sycl_dual_mmvq_elide.count(node) != 0) {
            continue;
        }
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (state_io_contains(state_io_direct, node)) {
            const double tf0 = prof_now ? now_us() : 0.0;
            const auto direct_plan = state_io_plan_for_marker(node);
            GGML_ASSERT(direct_plan != state_io_plan.end());
            if (node->op == GGML_OP_SSM_CONV) {
                ggml_sycl_ssm_conv_state_io(
                    *sycl_ctx, node, node->src[2], node->src[3], node->src[6],
                    node->src[4], node->src[5], direct_plan->n_rs);
            } else {
                GGML_ASSERT(node->op == GGML_OP_GATED_DELTA_NET);
                ggml_sycl_gated_delta_net_state_io(
                    *sycl_ctx, node, node->src[6], node->src[7], node->src[9],
                    node->src[8], direct_plan->n_rs);
            }
            if (prof_now) {
                sycl_ctx->stream()->wait();
                fusion_us[GGML_SYCL_FUSION_PROFILE_STATE_IO] += now_us() - tf0;
                fusion_n[GGML_SYCL_FUSION_PROFILE_STATE_IO]++;
            }
            continue;
        }

        {
            const double tf0 = prof_now ? now_us() : 0.0;
            ggml_sycl_fusion_profile_kind fusion_kind = GGML_SYCL_FUSION_PROFILE_NONE;
            const int nodes_to_skip = ggml_sycl_try_fuse(
                *sycl_ctx, cgraph, i, &fusion_kind);
            if (nodes_to_skip != 0) {
                GGML_ASSERT(fusion_kind > GGML_SYCL_FUSION_PROFILE_NONE &&
                            fusion_kind < GGML_SYCL_FUSION_PROFILE_COUNT);
                if (prof_now) {
                    sycl_ctx->stream()->wait();
                    fusion_us[fusion_kind] += now_us() - tf0;
                    fusion_n[fusion_kind]++;
                }
                // nodes_to_skip > 0: classic skip of subsequent nodes
                // nodes_to_skip < 0: fused current only (partner elided later)
                if (nodes_to_skip > 0) {
                    i += nodes_to_skip;
                }
                continue;
            }
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        const double t0 = prof_now ? now_us() : 0.0;
        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (prof_now) {
            sycl_ctx->stream()->wait();
            const double dt = now_us() - t0;
            if (node->op == GGML_OP_UNARY) { unary_us += dt; unary_n++; }
            else { op_us[node->op] += dt; op_n[node->op]++; }
            if (node->op == GGML_OP_MUL_MAT) {
                add_named_profile_stat(mul_mat_stats, mul_mat_profile_name(node), dt);
            }
            if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY ||
                node->op == GGML_OP_CONCAT || node->op == GGML_OP_GATED_DELTA_NET) {
                add_named_profile_stat(state_op_stats, state_op_profile_name(node), dt);
            }
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    if (prof_now && (++geval % 50 == 0)) {
        struct row { const char * name; double us; long n; };
        std::vector<row> rows;
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            if (op_n[op] > 0) rows.push_back({ ggml_op_name((ggml_op) op), op_us[op], op_n[op] });
        }
        if (unary_n > 0) rows.push_back({ "UNARY", unary_us, unary_n });
        static constexpr const char * fusion_names[GGML_SYCL_FUSION_PROFILE_COUNT] = {
            "FUSED_NONE",
            "FUSED_MOE_PIPE",
            "FUSED_MOE_DUAL",
            "FUSED_MOE_DOWN",
            "FUSED_TOPK_MOE",
            "FUSED_DN_GLUE",
            "FUSED_STATE_IO",
            "FUSED_RMS_ADD",
            "FUSED_RMS",
            "FUSED_DENSE_DUAL",
            "FUSED_DUAL_MMVQ",
            "FUSED_DUAL_F32",
        };
        for (int kind = GGML_SYCL_FUSION_PROFILE_NONE + 1;
             kind < GGML_SYCL_FUSION_PROFILE_COUNT; ++kind) {
            if (fusion_n[kind] > 0) {
                rows.push_back({ fusion_names[kind], fusion_us[kind], fusion_n[kind] });
            }
        }
        std::sort(rows.begin(), rows.end(), [](const row & a, const row & b){ return a.us > b.us; });
        double tot = 0; for (auto & r : rows) tot += r.us;
        fprintf(stderr,
                "[treebeard-sycl-prof] after %d graph evals  serialized-total=%.1f ms  (%.1f us/eval)\n",
                geval, tot / 1000.0, tot / geval);
        for (auto & r : rows) {
            fprintf(stderr, "  %-14s %9.1f ms  %6.1f%%  n=%-8ld %.2f us/op (per-eval n=%.1f)\n",
                    r.name, r.us/1000.0, 100.0*r.us/tot, r.n, r.us/r.n, (double)r.n/geval);
        }
        auto named_rows = mul_mat_stats;
        std::sort(named_rows.begin(), named_rows.end(),
                  [](const named_profile_stat & a, const named_profile_stat & b) {
                      return a.us > b.us;
                  });
        double named_total_us = 0.0;
        for (const auto & row : named_rows) {
            named_total_us += row.us;
        }
        fprintf(stderr,
                "[treebeard-sycl-prof-mul-mat] after %d graph evals window-total=%.1f ms families=%zu\n",
                geval, named_total_us / 1000.0, named_rows.size());
        const size_t named_limit = std::min<size_t>(named_rows.size(), 32);
        for (size_t row_index = 0; row_index < named_limit; ++row_index) {
            const auto & r = named_rows[row_index];
            fprintf(stderr,
                    "  %-54s %9.1f ms  %6.1f%%-mm"
                    "  n=%-8ld %.2f us/op (per-eval n=%.1f)\n",
                    r.name.c_str(), r.us / 1000.0,
                    100.0 * r.us / named_total_us,
                    r.n, r.us / r.n, (double) r.n / 50.0);
        }
        mul_mat_stats.clear();

        auto state_rows = state_op_stats;
        std::sort(state_rows.begin(), state_rows.end(),
                  [](const named_profile_stat & a, const named_profile_stat & b) {
                      return a.us > b.us;
                  });
        double state_total_us = 0.0;
        for (const auto & row : state_rows) {
            state_total_us += row.us;
        }
        fprintf(stderr,
                "[treebeard-sycl-prof-state] after %d graph evals window-total=%.1f ms families=%zu\n",
                geval, state_total_us / 1000.0, state_rows.size());
        const size_t state_limit = std::min<size_t>(state_rows.size(), 32);
        for (size_t row_index = 0; row_index < state_limit; ++row_index) {
            const auto & r = state_rows[row_index];
            fprintf(stderr,
                    "  %-86s %9.1f ms  %6.1f%%-state"
                    "  n=%-8ld %.2f us/op (per-eval n=%.1f)\n",
                    r.name.c_str(), r.us / 1000.0,
                    state_total_us > 0.0 ? 100.0 * r.us / state_total_us : 0.0,
                    r.n, r.us / r.n, (double) r.n / 50.0);
        }
        state_op_stats.clear();
    }
}

#ifdef GGML_SYCL_GRAPH
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }

    // GGML_SYCL_DISABLE_MMID_GRAPH=1 restores the blanket MUL_MAT_ID graph ban (A/B gate).
    static const bool disable_mmid_graph = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMID_GRAPH");
        return env != nullptr && atoi(env) != 0;
    }();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_MUL_MAT_ID:
                // The fused device-routed path reads the routing ids on device — no host sync —
                // so it records into a graph fine (its pool allocs still need the async mem-op
                // extension, same as MUL_MAT). Any node that would fall to the generic path does
                // a blocking ids readback and must keep graphs disabled.
                if (disable_mmid_graph || !g_ggml_sycl_use_async_mem_op ||
                    !ggml_sycl_mul_mat_id_fused_eligible(cgraph->nodes[i])) {
                    GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                                  ggml_op_name(node_op));
                    return false;
                }
                break;
            case GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
        }
    }
    return true;
}

// Initialize only persistent, lazily reordered weight storage before graph capture.
// Running the entire graph as a warm-up is not valid: legal source/destination aliasing
// and stateful writes can make a duplicate evaluation observe different inputs.
static void prepare_graph_reorders(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (ggml_sycl_mul_mat_id_fused_eligible(node)) {
                opt_for_reorder_id(sycl_ctx, node->src[0]);
            }
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT || ggml_backend_buffer_is_sycl_split(node->src[0]->buffer)) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];
        bool use_dmmv = can_use_dequantize_mul_mat_vec(src0, src1, node);
        const bool use_mmvq = can_use_mul_mat_vec_q(src0, src1, node);

        // Keep this precedence rule in lockstep with ggml_sycl_mul_mat().
        if (!g_ggml_sycl_prioritize_dmmv &&
            should_reorder_tensor(*sycl_ctx, node) &&
            ggml_sycl_supports_reorder_mmvq(src0->type)) {
            const bool skip_arc_q4 =
                ggml_sycl_info().devices[sycl_ctx->device].hw_info.arch == gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0;
            if (!skip_arc_q4) {
                use_dmmv = use_dmmv && !use_mmvq;
            }
        }

        if (use_dmmv) {
            opt_for_reorder(sycl_ctx, src0, src1, node, mul_mat_algo::DMMV);
        } else if (use_mmvq) {
            opt_for_reorder(sycl_ctx, src0, src1, node, mul_mat_algo::MMVQ);
        }
    }

    // Reorder uses async allocation and kernels. Capture must not inherit any of their
    // events, and the in-place weight layout must be complete before recording begins.
    sycl_ctx->stream()->wait_and_throw();
}
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    static const bool graph_debug = []() {
        const char * env = getenv("GGML_SYCL_GRAPH_DEBUG");
        return env != nullptr && atoi(env) != 0;
    }();
    static std::atomic<int> graph_debug_count { 0 };
    if (graph_debug && graph_debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        fprintf(stderr, "[SYCL-GRAPH] dispatch disabled=%d uid=%zu nodes=%d\n",
                g_ggml_sycl_disable_graph, cgraph->uid, cgraph->n_nodes);
    }
    bool use_sycl_graph = !g_ggml_sycl_disable_graph && check_graph_compatibility(cgraph);
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        GGML_ASSERT(cgraph->n_nodes > 0);
        ggml_sycl_graph * cached_graph = sycl_ctx->sycl_graph(cgraph->nodes[0]);

        // A nonzero graph UID is the scheduler's promise that topology, tensor addresses,
        // shapes and strides are unchanged. Replaying the finalized executable avoids
        // recording and updating hundreds of kernel submissions on every decode step.
        if (cgraph->uid != 0 && cached_graph->executable && cached_graph->uid == cgraph->uid) {
            sycl_ctx->stream()->ext_oneapi_graph(*(cached_graph->executable));
            if (graph_debug && graph_debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
                fprintf(stderr, "[SYCL-GRAPH] replay uid=%zu nodes=%d\n", cgraph->uid, cgraph->n_nodes);
            }
            return GGML_STATUS_SUCCESS;
        }

        // Lazy quantized-weight reorder uses async allocation events that cannot cross
        // the capture boundary. Prepare only those persistent resources; do not execute
        // graph outputs twice because graphs may contain legal aliases or stateful writes.
        prepare_graph_reorders(sycl_ctx, cgraph);

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        if (!cached_graph->executable || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            cached_graph->executable = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } else {
            try {
                cached_graph->executable->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                cached_graph->executable = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        cached_graph->uid = cgraph->uid;
        sycl_ctx->stream()->ext_oneapi_graph(*(cached_graph->executable));
        if (graph_debug && graph_debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
            fprintf(stderr, "[SYCL-GRAPH] record uid=%zu nodes=%d\n", cgraph->uid, cgraph->n_nodes);
        }
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ggml_backend_sycl_graph_optimize,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_info().device_count;
}


// backend device

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ggml_sycl_set_device(ctx->device);
    SYCL_CHECK(CHECK_TRY_ERROR(
    dpct::dev_mgr::instance().get_device(ctx->device).get_memory_info(*free, *total)));
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
#if defined (GGML_SYCL_F16)
                    return ggml_is_contiguous(op->src[0]) && (op->type == op->src[0]->type);
#else
                    return ggml_is_contiguous(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) && (op->type == op->src[0]->type);
#endif
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                // disable Q1_0 until implementation
                if (a->type == GGML_TYPE_Q1_0 || b->type == GGML_TYPE_Q1_0) {
                    return false;
                }

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;



                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                return ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                         op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q5_0 ||
                         op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_IQ4_NL) &&
                        (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32));
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == src1_type && (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) && src0_type != GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                return false;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
#if defined (GGML_SYCL_F16)
            return ((op->type == GGML_TYPE_F32 || op->type == GGML_SYCL_F16) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_SYCL_F16) && (op->type == op->src[0]->type));
#else
            return (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32) && (op->type == op->src[0]->type);
#endif
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return op->src[0]->ne[0] * sizeof(int) <=
                   ggml_sycl_info().devices[device].smpbo;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= 32;
        }
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(device, op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
