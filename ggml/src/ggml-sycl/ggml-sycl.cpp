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
#include <assert.h>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <limits>
#include <optional>
#include <map>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#include <sycl/half_type.hpp>
// [lx-expert-tile] joint_matrix (XMX) for the fused MoE expert-tile GEMM
#include <sycl/ext/oneapi/matrix/matrix.hpp>

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
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;

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

#if !defined(GGML_SYCL_SUPPORT_VMM)
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
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            info.devices[i].l0_discrete_gpu = r == ZE_RESULT_SUCCESS && !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
        }
#endif
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    // Large buffers can be allocated before ggml_check_sycl() initializes other
    // g_ggml_sycl_enable_* globals, so initialize this one as early as we can.
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
    g_ggml_sycl_use_level_zero_api = 0;
#endif

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

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else {
        return "Unknown";
    }
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);

        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
#endif

#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);

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
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
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

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
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

    if (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
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

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward\n");
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
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

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

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %lu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
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
        case GGML_TYPE_Q1_0:
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
#if defined(GGML_SYCL_SUPPORT_VMM)
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
#endif // defined(GGML_SYCL_SUPPORT_VMM)

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
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
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

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

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

// [lx-router-timer] measurement-only: GGML_SYCL_LX_ROUTER_TIMER=1 times the F32
// GEMM exec with device events + whole-op host time. Env-off = one bool branch.
// Decomposes the DIAG_SKIP_F32 pp512 delta (288ms/pass ~= 7.4ms per router call)
// into GEMM exec vs wrapper overhead.
static bool lx_router_timer_on() {
    static const bool on = []() {
        const char * e = getenv("GGML_SYCL_LX_ROUTER_TIMER");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}
static std::atomic<long> lx_rt_n(0), lx_rt_gemm_us(0), lx_rt_op_us(0);
static bool lx_rt_reg = []() {
    std::atexit([](){
        if (lx_rt_n.load() > 0) {
            fprintf(stderr, "LX_ROUTER_TIMER n=%ld gemm_avg_us=%.1f op_avg_us=%.1f tot_gemm_ms=%.1f\n",
                    lx_rt_n.load(),
                    (double) lx_rt_gemm_us.load() / lx_rt_n.load(),
                    (double) lx_rt_op_us.load() / lx_rt_n.load(),
                    (double) lx_rt_gemm_us.load() / 1000.0);
        }
    });
    return true;
}();

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

    const int64_t row_diff = row_high - row_low;

    // lx diag (diagnostic-only, never promoted): env-gated wall-time decomposition
    // of the scored pp512 path. GGML_SYCL_DIAG_SKIP=0 off; =1 skip GEMM exec (zero
    // dst instead, keeping router ids in-range downstream); =2 skip src conversions
    // AND GEMM. Lets us split pp512 wall into conversions | GEMM | everything-else
    // without ONEDNN_VERBOSE wall inflation.
    static const int lx_diag_skip = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP");
        return e ? atoi(e) : 0;
    }();
    if (lx_diag_skip >= 2) {
        stream->memset(dst_dd_i, 0, row_diff * src1_ncols * sizeof(float));
        return;
    }
    // lx diag ncols-thresholded probe (2026-08-08): splits the GEMM-exec term on the
    // scored (DNN-off oneMKL) path by call size. GGML_SYCL_DIAG_SKIP_TINY_N=N zeroes
    // dst (skip GEMM exec) only when src1_ncols <= N; GGML_SYCL_DIAG_SKIP_LARGE_N=N
    // skips only when src1_ncols > N. Unset envs keep the hook a behavioral no-op.
    // With either env set, per-call size counters are printed to stderr at exit.
    static const int lx_diag_skip_tiny = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP_TINY_N");
        return e ? atoi(e) : 0;
    }();
    static const int lx_diag_skip_large = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP_LARGE_N");
        return e ? atoi(e) : 0;
    }();
    static std::atomic<long> lx_n_tiny(0), lx_n_large(0), lx_n_skip(0);
    static bool lx_diag_count_registered = []() {
        std::atexit([](){
            fprintf(stderr, "LX_DIAG_COUNTS tiny=%ld large=%ld skip=%ld\n",
                    lx_n_tiny.load(), lx_n_large.load(), lx_n_skip.load());
        });
        return true;
    }();
    const bool lx_diag_probe = lx_diag_skip_tiny > 0 || lx_diag_skip_large > 0;
    if (lx_diag_probe) {
        (src1_ncols <= 32 ? lx_n_tiny : lx_n_large)++;
    }
    const bool lx_skip_this =
        (lx_diag_skip_tiny > 0 && src1_ncols <= lx_diag_skip_tiny) ||
        (lx_diag_skip_large > 0 && src1_ncols > lx_diag_skip_large);
    if (lx_skip_this) lx_n_skip++;
    // lx probe debug: one-line first-call/event markers (probe envs only)
    static bool lx_dbg_first_call = true, lx_dbg_first_skip = true;
    if (lx_diag_probe) {
        if (lx_dbg_first_call) {
            lx_dbg_first_call = false;
            fprintf(stderr, "LX_PROBE_CALL first ncols=%lld row_diff=%lld\n",
                    (long long) src1_ncols, (long long) row_diff);
        }
        if (lx_skip_this && lx_dbg_first_skip) {
            lx_dbg_first_skip = false;
            fprintf(stderr, "LX_PROBE_SKIP first ncols=%lld\n", (long long) src1_ncols);
        }
    }
    // [lx-diag-f32] measurement-only: GGML_SYCL_DIAG_SKIP_F32=1 elides the GEMM
    // exec for F32-src0 MUL_MATs (the MoE router GEMV on the prefill/MKL path)
    // — zeroes dst instead. Behavioral no-op when unset. tg/pp wall delta =
    // the F32-GEMM share of the pipeline.
    static const int lx_diag_skip_f32 = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP_F32");
        return e ? atoi(e) : 0;
    }();
    static std::atomic<long> lx_n_f32(0);
    static bool lx_diag_f32_count_registered = []() {
        std::atexit([](){
            if (lx_n_f32.load() > 0) {
                fprintf(stderr, "LX_DIAG_F32_SKIP count=%ld\n", lx_n_f32.load());
            }
        });
        return true;
    }();
    const bool lx_skip_f32 = lx_diag_skip_f32 && src0->type == GGML_TYPE_F32;
    if (lx_skip_f32) lx_n_f32++;
    // [lx-diag-quant] measurement-only: GGML_SYCL_DIAG_SKIP_QUANT=1 elides the
    // GEMM exec (zeroes dst) for QUANTIZED-src0 calls on this F32/oneMKL path —
    // i.e. the IQ4_NL MoE down projections at pp512 that no mmvq/MMQ kernel
    // serves. The F32 router GEMM is NOT skipped, so routing stays intact (the
    // 2026-08-09 decomposition's methodology guard). Zeroing the down-proj dst
    // only drops the expert contribution downstream — no NaN, no collapse.
    // Wall delta vs unset = the quantized-src0 GEMM+conversion share of the
    // scored pp512 pass. Behavioral no-op when unset.
    static const int lx_diag_skip_quant = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP_QUANT");
        return e ? atoi(e) : 0;
    }();
    static std::atomic<long> lx_n_quant(0);
    static bool lx_diag_quant_count_registered = []() {
        std::atexit([](){
            if (lx_n_quant.load() > 0) {
                fprintf(stderr, "LX_DIAG_QUANT_SKIP count=%ld\n", lx_n_quant.load());
            }
        });
        return true;
    }();
    const bool lx_skip_quant = lx_diag_skip_quant && ggml_is_quantized(src0->type);
    if (lx_skip_quant) lx_n_quant++;
    // [lx-diag-name-quant] measurement-only: GGML_SYCL_DIAG_NAME_QUANT=1 prints a
    // per-dst-name histogram of quantized-src0 oneMKL-path calls at exit (no
    // behavior change; pairs with DIAG_SKIP_QUANT to attribute the pp512
    // quantized-GEMM-exec term to specific tensors).
    static const int lx_diag_name_quant = []() -> int {
        const char * e = getenv("GGML_SYCL_DIAG_NAME_QUANT");
        return e ? atoi(e) : 0;
    }();
    if (lx_diag_name_quant && ggml_is_quantized(src0->type)) {
        static std::map<std::string, long> lx_quant_names;
        static bool lx_name_reg = []() {
            std::atexit([](){
                if (!lx_quant_names.empty()) {
                    long tot = 0;
                    for (auto & kv : lx_quant_names) tot += kv.second;
                    fprintf(stderr, "LX_QUANT_NAMES total=%ld distinct=%zu\n",
                            tot, lx_quant_names.size());
                    for (auto & kv : lx_quant_names) {
                        fprintf(stderr, "LX_QUANT_NAME %s count=%ld\n",
                                kv.first.c_str(), kv.second);
                    }
                }
            });
            return true;
        }();
        char lx_nbuf[256];
        snprintf(lx_nbuf, sizeof(lx_nbuf), "%s ty=%s ncols=%lld",
                 dst->name ? dst->name : "?", ggml_type_name(src0->type),
                 (long long) src1_ncols);
        lx_quant_names[std::string(lx_nbuf)]++;
    }
    const auto lx_gemm_or_zero = [&](const auto & gemm_fn) {
        if (lx_diag_skip >= 1 || lx_skip_this || lx_skip_f32 || lx_skip_quant) {
            stream->memset(dst_dd_i, 0, row_diff * src1_ncols * sizeof(float));
        } else {
            gemm_fn();
        }
    };

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

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
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn && ggml_is_contiguous(src0) &&
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
        lx_gemm_or_zero([&] {
            DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                      src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                      src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        });
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
        if (g_ggml_sycl_enable_dnn) {
                lx_gemm_or_zero([&] {
                    DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                         DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
                });
        }
        else
#endif
        {
            lx_gemm_or_zero([&] {
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
            });
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

        lx_gemm_or_zero([&] {
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
#if GGML_SYCL_DNNL
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                auto lx_rt_t0 = std::chrono::steady_clock::now();
                sycl::event lx_rt_ev0;
                if (lx_router_timer_on()) {
                    lx_rt_ev0 = stream->ext_oneapi_submit_barrier();
                }
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
                if (lx_router_timer_on()) {
                    sycl::event lx_rt_ev1 = stream->ext_oneapi_submit_barrier();
                    auto lx_rt_w0 = std::chrono::steady_clock::now();
                    lx_rt_ev1.wait();
                    lx_rt_gemm_us.fetch_add((long)std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - lx_rt_w0).count());
                    lx_rt_op_us.fetch_add((long)std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - lx_rt_t0).count());
                    lx_rt_n.fetch_add(1);
                }
            }
        });
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


// Exported for hybrid topk-moe (Laguna bias): fuse sigmoid/add/norm, stock argsort only.
void ggml_sycl_argsort_f32_i32(ggml_backend_sycl_context & ctx, const float * x, int * dst,
                               int ncols, int nrows, ggml_sort_order order) {
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    argsort_f32_i32_sycl(x, dst, ncols, nrows, order, ctx.stream(), ctx.device, ctx.pool());
}

inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         ctx.stream(), ctx.device, ctx.pool());
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
    char * quant_memo_reuse = nullptr;
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
                // [lx-memo-q81] decode: Q/K/V/gate mul_mats per layer all quantize the
                // same attn_norm src1 into Q8_1; only the first consumer must launch the
                // quantize kernel. Reuse is bit-identical (same src1 bytes -> same q8_1
                // bytes); the arena is a dedicated pool slot never freed for the backend
                // lifetime, and compute_forward invalidates it when any op's dst aliases
                // the memoized src1. gen guards against an intervening requantize of a
                // different src1 overwriting the arena between two consumers.
                const bool memoizable = src1->view_src == nullptr && src1->type == GGML_TYPE_F32 &&
                                        used_devices == 1 && nrows1 == 1;
                quant_memo_reuse = nullptr;
                if (memoizable && ctx.quant_memo_valid &&
                    ctx.quant_memo_gen == ctx.quant_memo_gen_at_store &&
                    ctx.quant_memo_src1 == src1->data &&
                    ctx.quant_memo_type == src1->type &&
                    ctx.quant_memo_ne10 == ne10 &&
                    ctx.quant_memo_nrows1 == nrows1 &&
                    ctx.quant_memo_padded == (size_t) src1_padded_col_size) {
                    // same activation already quantized this graph: skip the kernel
                    quant_memo_reuse = (char *) ctx.quant_memo_buf;
                    ctx.quant_memo_hits++;
                } else {
                    if (memoizable) {
                        // Dedicated USM arena (never a pool slot: the VMM pool is a
                        // strict-LIFO bump allocator, so a held slot would break free()).
                        // 32 KB covers the largest memoizable src1 (ne10=8192 ->
                        // padded 8192 -> 8192*34/32 = 8704 B q8_1). Held for the
                        // backend lifetime; the driver reclaims it at process exit.
                        if (ctx.quant_memo_buf == nullptr) {
                            ctx.quant_memo_buf = sycl::aligned_alloc_device(SYCL_BUFFER_ALIGNMENT, 32768, *stream);
                            GGML_ASSERT(ctx.quant_memo_buf != nullptr);
                            ctx.quant_memo_buf_bytes = 32768;
                        }
                        ctx.quant_memo_src1 = src1->data;
                        ctx.quant_memo_type = src1->type;
                        ctx.quant_memo_ne10 = ne10;
                        ctx.quant_memo_nrows1 = nrows1;
                        ctx.quant_memo_padded = (size_t) src1_padded_col_size;
                        ctx.quant_memo_gen++;
                        ctx.quant_memo_gen_at_store = ctx.quant_memo_gen;
                        ctx.quant_memo_valid = true;
                    }
                    scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                         /*num_src=*/2, " : converting src1 to Q8_1");
                    try {
                        quantize_row_q8_1_sycl<quantize_f>(
                            dev[i].src1_ddf,
                            memoizable ? (char *) ctx.quant_memo_buf : dev[i].src1_ddq,
                            ne10, nrows1, src1_padded_col_size, stream);
                    } catch (sycl::exception const &exc) {
                        std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                                  << ", line:" << __LINE__ << std::endl;
                        std::exit(1);
                    }
                    if (memoizable) {
                        quant_memo_reuse = (char *) ctx.quant_memo_buf;
                    }
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
                char  * src1_ddq_i = quant_memo_reuse != nullptr ? quant_memo_reuse
                                                                  : dev[i].src1_ddq + src1_ddq_i_offset;
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
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
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
    if (g_ggml_sycl_enable_dnn) {
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
    // MMQ (native quantized batched GEMM, in-kernel q4/q6->q8_1 + dp4a) was
    // disabled with a stale "TODO: accuracy issues in MMQ".  The kernels are
    // compiled into the shipped .so (verified: ggml_sycl_op_mul_mat_q export,
    // mul_mat_q4_K_q8_1_sycl strings) and the dp4a integer-dot math is exact.
    // Re-enable for the types mmq.cpp actually implements; the KLD quality
    // gate is the arbiter.  GGML_SYCL_ENABLE_MMQ=0 restores the old gate.
    static const int enable_mmq = ggml_sycl_get_env("GGML_SYCL_ENABLE_MMQ", 0);
    if (!enable_mmq) {
        GGML_UNUSED(type);
        return false;
    }
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
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
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
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
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
        case GGML_TYPE_Q1_0:
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

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q2_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q2_K) == 0);

    const int nblocks = size / sizeof(block_q2_K);

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

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q2_K * x  = (const block_q2_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
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
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
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

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
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
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
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
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
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

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
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

    // Reordered MoE quant weights (SoA from MMVQ bootstrap) must NOT fall through to
    // MMQ/oneDNN multi-row GEMM — that path assumes linear block layout and yields
    // PPL -nan on dual_down expert-loop (confirmed OPT=0 fixes PPL, kills decode).
    // Force chunked reorder-MMVQ for any column count when src0 is already reordered.
    // notes/SHIP_20260731_dual_down_expert_loop_ppl.md
    //
    // [lx-reorder-multicol] GGML_SYCL_LX_REORDER_MULTICOL_MKL=1 (env, default OFF):
    // narrow that guard to the decode / spec-verify width it was written for
    // (ne[1] <= MMVQ_MAX_BATCH_SIZE) and let wider batches fall through to the
    // fp16/oneMKL path. The guard's hazard is raw SoA weights reaching a kernel
    // that assumes linear blocks; ggml_sycl_op_mul_mat_sycl does not read them
    // raw — it dequantizes via ggml_get_to_fp16_sycl, which already dispatches
    // dequantize_row_q{4,5,6}_K_sycl_reorder for reordered banks (convert.cpp),
    // and reorder_qw_q*_k_moe keeps each expert self-contained inside its nb02
    // stride so the per-expert slice pointer stays valid. MMQ is still forbidden
    // on reordered weights (it does read them raw), so the bypass suppresses it.
    // Measured motivation: the 8-column chunking re-reads the whole expert slice
    // ceil(N/8) times at prefill widths. Quality arbitrated by the KLD gate.
    static const bool lx_reorder_multicol_mkl = []() {
        const char * e = getenv("GGML_SYCL_LX_REORDER_MULTICOL_MKL");
        return e != nullptr && std::atoi(e) != 0;
    }();
    const bool lx_reorder_multicol_bypass =
        lx_reorder_multicol_mkl && src1->ne[1] > MMVQ_MAX_BATCH_SIZE;
    {
        ggml_tensor_extra_gpu * extra0 = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        const bool reordered = extra0 && extra0->optimized_feature.reorder;
        if (reordered && lx_reorder_multicol_bypass) {
            use_mul_mat_q = false;
        }
        if (!split && reordered && !lx_reorder_multicol_bypass && ggml_is_quantized(src0->type) &&
            src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            src1->ne[2] == 1 && src1->ne[3] == 1 && src1->ne[1] >= 1 &&
            ggml_sycl_supports_reorder_mmvq(src0->type)) {
            const int64_t ncols = src1->ne[1];
            for (int64_t col0 = 0; col0 < ncols; col0 += MMVQ_MAX_BATCH_SIZE) {
                const int64_t nc = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, ncols - col0);
                ggml_tensor src1_c = *src1;
                ggml_tensor dst_c  = *dst;
                src1_c.ne[1] = nc;
                dst_c.ne[1]  = nc;
                src1_c.data  = (char *) src1->data + col0 * src1->nb[1];
                dst_c.data   = (char *) dst->data  + col0 * dst->nb[1];
                src1_c.nb[2] = src1_c.nb[1] * (size_t) nc;
                src1_c.nb[3] = src1_c.nb[2];
                dst_c.nb[2]  = dst_c.nb[1] * (size_t) nc;
                dst_c.nb[3]  = dst_c.nb[2];
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(
                    ctx, src0, &src1_c, &dst_c, ggml_sycl_op_mul_mat_vec_q);
            }
            return;
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
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        // [lx-diag-mmq] measurement-only: GGML_SYCL_DIAG_NAME_MMQ=1 logs the first
        // MMQ dispatches (src0 type, ncols, dst name) to attribute the pp512 stall.
        static const int lx_diag_name_mmq = []() -> int {
            const char * e = getenv("GGML_SYCL_DIAG_NAME_MMQ");
            return e ? atoi(e) : 0;
        }();
        if (lx_diag_name_mmq) {
            static std::atomic<int> lx_mmq_n(0);
            if (lx_mmq_n.fetch_add(1) < 40) {
                fprintf(stderr, "[lx-diag-mmq] MMQ ty=%s ncols=%lld dst=%s\n",
                        ggml_type_name(src0->type), (long long) src1->ne[1], dst->name);
            }
        }
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}


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

// Dual MoE gate+up + SwiGLU for serial decode (ne12==1). Returns true on hit.
static bool ggml_sycl_mul_mat_id_dual_swiglu_fused(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu) {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (disabled) {
        return false;
    }

    const ggml_tensor * gate_w = gate->src[0];
    const ggml_tensor * up_w   = up->src[0];
    const ggml_tensor * src1   = gate->src[1];
    const ggml_tensor * ids    = gate->src[2];

    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
        gate->src[1] != up->src[1] || gate->src[2] != up->src[2] ||
        gate_w->type != up_w->type ||
        (gate_w->type != GGML_TYPE_Q4_K && gate_w->type != GGML_TYPE_Q5_K &&
         gate_w->type != GGML_TYPE_Q6_K) ||
        src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }

    // Serial decode only (matches mmvq_fused bailouts).
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    if (ne12 != 1) return false;
    if (ids->ne[1] != 1) return false;
    const int64_t n_ids = ids->ne[0];
    if (ne11 != 1 && ne11 != n_ids) return false;
    if (ne10 != gate_w->ne[0] || ne10 != up_w->ne[0] || ne10 % QK8_1 != 0) return false;
    if (gate_w->ne[1] != up_w->ne[1] || gate_w->ne[2] != up_w->ne[2]) return false;
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(glu)) return false;
    if (gate->ne[0] != gate_w->ne[1] || up->ne[0] != up_w->ne[1] ||
        gate->ne[1] != n_ids || up->ne[1] != n_ids ||
        gate->ne[2] != 1 || up->ne[2] != 1 ||
        !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, glu)) {
        return false;
    }
    if (ids->nb[0] != sizeof(int32_t)) return false;

    opt_for_reorder_id(&ctx, gate_w);
    opt_for_reorder_id(&ctx, up_w);
    const ggml_tensor_extra_gpu * gate_extra =
        static_cast<const ggml_tensor_extra_gpu *>(gate_w->extra);
    const ggml_tensor_extra_gpu * up_extra =
        static_cast<const ggml_tensor_extra_gpu *>(up_w->extra);
    if (!gate_extra || !up_extra ||
        !gate_extra->optimized_feature.reorder ||
        !up_extra->optimized_feature.reorder) {
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int n_experts_used   = (int) n_ids;
    const int nrows            = (int) gate_w->ne[1];

    ggml_sycl_pool_alloc<char> src1_q8_alloc(
        ctx.pool(), (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_q8 = src1_q8_alloc.get();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) src1->data, src1_q8, (int) ne10, (int) ne11,
        src1_padded_cols, stream);

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    return ggml_sycl_mul_mat_vec_q_id_dual_swiglu_reorder(
        gate_w->type, gate_w->data, up_w->data, src1_q8,
        (const int32_t *) ids->data, (float *) glu->data,
        (int) ne10, nrows, n_experts_used,
        gate_w->nb[2], up_w->nb[2], glu->nb[1], src1_row_stride, stream);
}

// Graph fuse: MUL_MAT → ADD residual (+ optional second ADD) via reorder MMVQ addend.
// Quality-safe default: decode only (ne11==1). See notes/SHIP_20260731_mmadd_decode_only.md
// Kill: GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
// Research any-batch: GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH=1 (not quality-safe)
int ggml_sycl_fuse_mul_mat_add(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    // Skip non-ADD neighbors but allow VIEW/NONE between mul_mat and add (rare).
    int j_add = i + 1;
    while (j_add < cgraph->n_nodes && j_add <= i + 3) {
        ggml_tensor * n = cgraph->nodes[j_add];
        if (n->op == GGML_OP_ADD) {
            break;
        }
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE || ggml_is_empty(n)) {
            j_add++;
            continue;
        }
        break;
    }
    if (j_add >= cgraph->n_nodes || cgraph->nodes[j_add]->op != GGML_OP_ADD) {
        return 0;
    }
    ggml_tensor * add = cgraph->nodes[j_add];
    if (add->src[0] != mm && add->src[1] != mm) {
        return 0;
    }
    const ggml_tensor * residual = add->src[0] == mm ? add->src[1] : add->src[0];
    if (!residual || !mm->src[0] || !mm->src[1] || !residual->data || !add->data) {
        return 0;
    }
    if (mm->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
        residual->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mm, add) || !ggml_are_same_shape(mm, residual) ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(residual) ||
        !ggml_is_contiguous(mm->src[1])) {
        return 0;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    if (!ggml_is_quantized(w->type) || x->type != GGML_TYPE_F32) {
        return 0;
    }
    // QUALITY-SAFE: decode only (ne11==1). Any-batch prefill fuse wrecks PPL.
    if (x->ne[1] < 1) {
        return 0;
    }
    {
        static const bool any_batch = []() {
            const char * e = getenv("GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH");
            return e != nullptr && std::atoi(e) != 0;
        }();
        if (!any_batch && x->ne[1] != 1) {
            return 0;
        }
    }
    // Only types with reorder MMVQ addend epilogue wired (Q4_K/Q5_K/Q6_K).
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) {
        return 0;
    }
    // Optional second ADD: (mul + r0) + r1  e.g. Laguna shexp+moe then +ffn_inp.
    ggml_tensor *       add2      = nullptr;
    const ggml_tensor * residual2 = nullptr;
    int                 j_add2    = j_add;
    if (x->ne[1] <= 32) {
        int j = j_add + 1;
        while (j < cgraph->n_nodes && j <= j_add + 3) {
            ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_ADD) {
                break;
            }
            if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
                n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE || ggml_is_empty(n)) {
                j++;
                continue;
            }
            break;
        }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_ADD) {
            ggml_tensor * cand = cgraph->nodes[j];
            if ((cand->src[0] == add || cand->src[1] == add) && cand->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(add, cand) && ggml_is_contiguous(cand) && cand->data) {
                const ggml_tensor * r2 = cand->src[0] == add ? cand->src[1] : cand->src[0];
                if (r2 && r2->type == GGML_TYPE_F32 && r2->data && ggml_are_same_shape(add, r2) &&
                    ggml_is_contiguous(r2)) {
                    add2      = cand;
                    residual2 = r2;
                    j_add2    = j;
                }
            }
        }
    }

    const int n_span = j_add2 - i + 1;
    std::vector<ggml_op> ops(n_span, GGML_OP_NONE);
    for (int k = 0; k < n_span; ++k) {
        ops[k] = cgraph->nodes[i + k]->op;
    }
    const int output = j_add2;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_span, ops.data(), &output, 1)) {
        if (add2) {
            add2      = nullptr;
            residual2 = nullptr;
            j_add2    = j_add;
            const int n_span1 = j_add - i + 1;
            std::vector<ggml_op> ops1(n_span1, GGML_OP_NONE);
            for (int k = 0; k < n_span1; ++k) {
                ops1[k] = cgraph->nodes[i + k]->op;
            }
            const int out1 = j_add;
            if (!ggml_can_fuse_subgraph(cgraph, i, n_span1, ops1.data(), &out1, 1)) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    ggml_tensor * dst_add = add2 ? add2 : add;
    void * saved_mm_data = mm->data;
    mm->data = dst_add->data;
    ggml_sycl_mmvq_set_row_addend(static_cast<const float *>(residual->data));
    if (residual2) {
        ggml_sycl_mmvq_set_row_addend2(static_cast<const float *>(residual2->data));
    }
    ggml_sycl_mul_mat(ctx, mm->src[0], mm->src[1], mm);
    ggml_sycl_mmvq_set_row_addend(nullptr);
    ggml_sycl_mmvq_set_row_addend2(nullptr);
    mm->data = saved_mm_data;

    {
        static std::atomic<int> once_single{0};
        static std::atomic<int> once_double{0};
        std::atomic<int> * once = residual2 ? &once_double : &once_single;
        if (once->fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-mm-add] fuse hit (mul_mat+add%s) ne0=%" PRId64 " ne1=%" PRId64
                    " wtype=%s alias_res=%d mm='%s' add='%s' add2='%s'\n",
                    residual2 ? "+add" : "",
                    dst_add->ne[0], dst_add->ne[1], ggml_type_name(w->type),
                    (int) (residual->data == add->data),
                    mm->name ? mm->name : "?", add->name ? add->name : "?",
                    (residual2 && add2 && add2->name) ? add2->name : "-");
        }
    }
    return j_add2 - i; // skip through final ADD (and any elided views)
}



// Graph fuse: RMS_NORM → MUL(weight). Default ON (Laguna pre-norm every layer).
// Kill: GGML_SYCL_DISABLE_RMS_NORM_FUSE=1
int ggml_sycl_fuse_rms_norm_mul(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_RMS_NORM_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rms = cgraph->nodes[i];
    ggml_tensor * mul = cgraph->nodes[i + 1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return 0;
    }
    if (mul->src[0] != rms && mul->src[1] != rms) {
        return 0;
    }
    if (rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !rms->src[0] || rms->src[0]->type != GGML_TYPE_F32 ||
        !rms->data || !mul->data) {
        return 0;
    }
    // Weight operand must be F32.
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (!w || w->type != GGML_TYPE_F32 || !w->data) {
        return 0;
    }
    // Output is MUL (weight-scaled norm). RMS intermediate is elided.
    const int output = i + 1;
    if (!ggml_can_fuse_subgraph(
            cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, { output })) {
        return 0;
    }
    ggml_sycl_op_rms_norm_fused(ctx, rms, mul);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-rms] fuse hit (rms_norm+mul) ne0=%" PRId64 " ne1=%" PRId64 "\n",
                    rms->src[0]->ne[0], rms->src[0]->ne[1]);
        }
    }
    return 1;  // skip following MUL
}

// [lx-norm-rope] Fuse RMS_NORM → MUL → ROPE → VIEW → SET_ROWS (K-cache normed rope)
// into ONE launch, replacing the two current launches (rms+mul fuse, rope+view+
// set_rows fuse) per layer. Env-gated: GGML_SYCL_FUSE_NORM_ROPE=1 (default OFF).
// Kernel is in rope.cpp (rope_neox_normed_sycl); launch gated to NEOX mode and
// ne00 == 128 (the decode K head dim, matching the 64-lane workgroup reduction).
int ggml_sycl_fuse_norm_rope(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * e = getenv("GGML_SYCL_FUSE_NORM_ROPE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!enabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rms      = cgraph->nodes[i];
    ggml_tensor * mul      = cgraph->nodes[i + 1];
    ggml_tensor * rope     = cgraph->nodes[i + 2];
    ggml_tensor * view     = cgraph->nodes[i + 3];
    ggml_tensor * set_rows = cgraph->nodes[i + 4];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
        rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW ||
        set_rows->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    if (mul->src[0] != rms && mul->src[1] != rms) {
        return 0;
    }
    if (rope->src[0] != mul || view->src[0] != rope ||
        set_rows->src[0] != view) {
        return 0;
    }
    if (rms->src[0] == nullptr || rms->src[0]->type != GGML_TYPE_F32 ||
        rms->src[0]->ne[0] != 128) {
        return 0;
    }
    // Serial-decode only: the fused 64-lane workgroup reduction is a measured
    // ~1.6% pp512 prefill tax (same-.so A/B d9150c39, lx-fuse-ab-20260809T153357Z:
    // pp512 1152.4 on vs 1173.1/1170.7 off) vs the shipped 16-lane norm_f32.
    // ne[2] == n_tokens: 1 at decode, >=2 in prefill/ppl. Decode is untouched.
    if (rms->src[0]->ne[2] != 1) {
        return 0;
    }
    // NEOX rope only (normed-neox kernel), matching the K path's mode=2.
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (!(mode & GGML_ROPE_TYPE_NEOX)) {
        return 0;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (w->type != GGML_TYPE_F32 || !w->data) {
        return 0;
    }
    if (set_rows->src[1] == nullptr ||
        set_rows->src[1]->type != GGML_TYPE_I64) {
        return 0;
    }
    const ggml_type dst_type = set_rows->type;
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return 0;
    }
    const int output = i + 4;
    if (!ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW,
              GGML_OP_SET_ROWS },
            { output })) {
        return 0;
    }
    ggml_sycl_rope_norm_fused(ctx, rms, mul, rope, set_rows);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-norm-rope] fuse hit (rms+mul+rope+set_rows) ne0=%" PRId64
                    " ndims=%d mode=%d type=%s x=%s\n",
                    rms->src[0]->ne[0],
                    ((const int32_t *) rope->op_params)[1],
                    ((const int32_t *) rope->op_params)[2],
                    ggml_type_name(dst_type),
                    ggml_type_name(rms->src[0]->type));
        }
    }
    return 4;  // skip MUL, ROPE, VIEW, SET_ROWS
}

// [lx-norm-rope-q] Q-path twin: RMS_NORM + MUL + ROPE with NO trailing
// VIEW+SET_ROWS (the Q rope output feeds flash attention directly). Same fused
// kernel with set_rows==nullptr -> contiguous write to rope's own dst, which is
// exactly the original rope_neox non-scatter indexing. Fires only when the K
// pattern above did not match at this node (guard below) so the K fuse always
// wins the 5-node pattern.
int ggml_sycl_fuse_norm_rope_q(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * e = getenv("GGML_SYCL_FUSE_NORM_ROPE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!enabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rms  = cgraph->nodes[i];
    ggml_tensor * mul  = cgraph->nodes[i + 1];
    ggml_tensor * rope = cgraph->nodes[i + 2];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
        rope->op != GGML_OP_ROPE) {
        return 0;
    }
    if (mul->src[0] != rms && mul->src[1] != rms) {
        return 0;
    }
    if (rope->src[0] != mul) {
        return 0;
    }
    // Never steal the K pattern: if this is really RMS,MUL,ROPE,VIEW,SET_ROWS,
    // let the 5-node fuse above handle it.
    if (i + 4 < cgraph->n_nodes) {
        ggml_tensor * view = cgraph->nodes[i + 3];
        ggml_tensor * sr   = cgraph->nodes[i + 4];
        if (view->op == GGML_OP_VIEW && sr->op == GGML_OP_SET_ROWS &&
            view->src[0] == rope && sr->src[0] == view) {
            return 0;
        }
    }
    if (rms->src[0] == nullptr || rms->src[0]->type != GGML_TYPE_F32 ||
        rms->src[0]->ne[0] != 128) {
        return 0;
    }
    // Serial-decode only, same measured pp512 tax as the K twin (see
    // ggml_sycl_fuse_norm_rope). ne[2] == n_tokens: 1 at decode, 512 at pp512.
    if (rms->src[0]->ne[2] != 1) {
        return 0;
    }
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (!(mode & GGML_ROPE_TYPE_NEOX)) {
        return 0;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (w->type != GGML_TYPE_F32 || !w->data) {
        return 0;
    }
    const ggml_type dst_type = rope->type;
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return 0;
    }
    const int output = i + 2;
    if (!ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE },
            { output })) {
        return 0;
    }
    ggml_sycl_rope_norm_fused(ctx, rms, mul, rope, nullptr);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-norm-rope-q] fuse hit (rms+mul+rope) ne0=%" PRId64
                    " ndims=%d mode=%d type=%s x=%s\n",
                    rms->src[0]->ne[0],
                    ((const int32_t *) rope->op_params)[1],
                    ((const int32_t *) rope->op_params)[2],
                    ggml_type_name(dst_type),
                    ggml_type_name(rms->src[0]->type));
        }
    }
    return 2;  // skip MUL, ROPE
}

// [lx-qk-rope] Merged Q+K normed rope in one launch (serial decode only).
// The Q chain (RMS+MUL+ROPE, contiguous rope dst feeding flash attention) and
// the K chain (RMS+MUL+ROPE+VIEW+SET_ROWS, cache scatter) run back-to-back in
// the graph with the QKV-fuse-satisfied V/K MUL_MAT nodes between them. We
// validate both chains explicitly and fuse the non-contiguous 8-node span via
// ggml_can_fuse_subgraph_ext (both chain outputs listed as outputs, so their
// external consumers — flash attention reading Qcur_rope and the cache — are
// allowed). Saves the second per-layer rope dispatch (~40 launches/token).
// Env-gated by the same GGML_SYCL_FUSE_NORM_ROPE as the two separate fuses;
// GGML_SYCL_DISABLE_QK_ROPE_MERGE=1 falls back to the separate fuses.
int ggml_sycl_fuse_norm_rope_qk(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * e = getenv("GGML_SYCL_FUSE_NORM_ROPE");
        if (e == nullptr || std::atoi(e) == 0) {
            return false;
        }
        const char * dis = getenv("GGML_SYCL_DISABLE_QK_ROPE_MERGE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 5 >= cgraph->n_nodes) {
        return 0;
    }
    // Q chain at i..i+2.
    ggml_tensor * rms_q  = cgraph->nodes[i];
    ggml_tensor * mul_q  = cgraph->nodes[i + 1];
    ggml_tensor * rope_q = cgraph->nodes[i + 2];
    if (rms_q->op != GGML_OP_RMS_NORM || mul_q->op != GGML_OP_MUL ||
        rope_q->op != GGML_OP_ROPE) {
        return 0;
    }
    if ((mul_q->src[0] != rms_q && mul_q->src[1] != rms_q) ||
        rope_q->src[0] != mul_q) {
        return 0;
    }
    // Never steal the K pattern (RMS,MUL,ROPE,VIEW,SET_ROWS at i).
    if (i + 4 < cgraph->n_nodes) {
        ggml_tensor * v0 = cgraph->nodes[i + 3];
        ggml_tensor * sr = cgraph->nodes[i + 4];
        if (v0->op == GGML_OP_VIEW && sr->op == GGML_OP_SET_ROWS &&
            v0->src[0] == rope_q && sr->src[0] == v0) {
            return 0;
        }
    }
    // Scan forward for the K chain (skips the QKV-fuse-satisfied V/K MUL_MATs).
    int j = -1;
    const int scan_max = std::min(i + 8, cgraph->n_nodes - 5);
    for (int k = i + 3; k <= scan_max; ++k) {
        if (cgraph->nodes[k]->op == GGML_OP_RMS_NORM) {
            j = k;
            break;
        }
    }
    if (j < 0) {
        return 0;
    }
    ggml_tensor * rms_k  = cgraph->nodes[j];
    ggml_tensor * mul_k  = cgraph->nodes[j + 1];
    ggml_tensor * rope_k = cgraph->nodes[j + 2];
    ggml_tensor * view_k = cgraph->nodes[j + 3];
    ggml_tensor * sr_k   = cgraph->nodes[j + 4];
    if (mul_k->op != GGML_OP_MUL || rope_k->op != GGML_OP_ROPE ||
        view_k->op != GGML_OP_VIEW || sr_k->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    if ((mul_k->src[0] != rms_k && mul_k->src[1] != rms_k) ||
        rope_k->src[0] != mul_k || view_k->src[0] != rope_k ||
        sr_k->src[0] != view_k) {
        return 0;
    }
    // Shared model params must match between the two chains.
    if (rms_q->src[0] == nullptr || rms_k->src[0] == nullptr ||
        rms_q->src[0]->type != GGML_TYPE_F32 || rms_k->src[0]->type != GGML_TYPE_F32 ||
        rms_q->src[0]->ne[0] != 128 || rms_k->src[0]->ne[0] != 128) {
        return 0;
    }
    // Serial-decode only (the fused 64-lane reduction is a measured pp512 tax).
    if (rms_q->src[0]->ne[2] != 1 || rms_k->src[0]->ne[2] != 1) {
        return 0;
    }
    float eps_q = 0.0f, eps_k = 0.0f;
    memcpy(&eps_q, rms_q->op_params, sizeof(float));
    memcpy(&eps_k, rms_k->op_params, sizeof(float));
    if (eps_q != eps_k) {
        return 0;
    }
    const int mode_q = ((const int32_t *) rope_q->op_params)[2];
    const int mode_k = ((const int32_t *) rope_k->op_params)[2];
    if (!(mode_q & GGML_ROPE_TYPE_NEOX) || !(mode_k & GGML_ROPE_TYPE_NEOX)) {
        return 0;
    }
    const int ndims_q = ((const int32_t *) rope_q->op_params)[1];
    const int ndims_k = ((const int32_t *) rope_k->op_params)[1];
    if (ndims_q != ndims_k) {
        return 0;
    }
    const ggml_tensor * w_q = mul_q->src[0] == rms_q ? mul_q->src[1] : mul_q->src[0];
    const ggml_tensor * w_k = mul_k->src[0] == rms_k ? mul_k->src[1] : mul_k->src[0];
    if (w_q->type != GGML_TYPE_F32 || !w_q->data ||
        w_k->type != GGML_TYPE_F32 || !w_k->data) {
        return 0;
    }
    const ggml_type dst_type_q = rope_q->type;
    const ggml_type dst_type_k = sr_k->type;
    if ((dst_type_q != GGML_TYPE_F32 && dst_type_q != GGML_TYPE_F16) ||
        (dst_type_k != GGML_TYPE_F32 && dst_type_k != GGML_TYPE_F16)) {
        return 0;
    }
    if (sr_k->src[1] == nullptr || sr_k->src[1]->type != GGML_TYPE_I64) {
        return 0;
    }
    // Non-contiguous 8-node span: {i,i+1,i+2} = Q chain, {j..j+4} = K chain.
    const int idxs[8] = { i, i + 1, i + 2, j, j + 1, j + 2, j + 3, j + 4 };
    const ggml_op ops[8] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_VIEW, GGML_OP_SET_ROWS
    };
    const int outputs[2] = { i + 2, j + 4 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 8, ops, outputs, 2)) {
        return 0;
    }
    ggml_sycl_rope_norm_fused_qk(ctx, rms_q, mul_q, rope_q,
                                 rms_k, mul_k, rope_k, view_k, sr_k);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-norm-rope-qk] fuse hit (q+k merged) ndims=%d mode=%d "
                    "qtype=%s ktype=%s qnr=%lld knr=%lld\n",
                    ndims_q, mode_q, ggml_type_name(dst_type_q),
                    ggml_type_name(dst_type_k),
                    (long long) ggml_nrows(rope_q->src[0]),
                    (long long) ggml_nrows(rope_k->src[0]));
        }
    }
    return j + 4 - i;  // skip through K's SET_ROWS (Q chain + gap + K chain)
}

// Graph fuse: ROPE → VIEW → SET_ROWS (KV cache write). Port of CUDA rope_fused.
// Kill: GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE=1
int ggml_sycl_fuse_rope_set_rows(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rope     = cgraph->nodes[i];
    ggml_tensor * view     = cgraph->nodes[i + 1];
    ggml_tensor * set_rows = cgraph->nodes[i + 2];
    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW ||
        set_rows->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    // VIEW of rope; SET_ROWS values = view (src[0]=b values, src[1]=indices)
    if (view->src[0] != rope || set_rows->src[0] != view) {
        return 0;
    }
    if (!rope->src[0] || rope->src[0]->ne[3] != 1) {
        return 0;
    }
    if (set_rows->src[0]->type != GGML_TYPE_F32 && set_rows->src[0]->type != GGML_TYPE_F16) {
        return 0;
    }
    if (!set_rows->src[1] || set_rows->src[1]->type != GGML_TYPE_I64) {
        return 0;
    }
    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return 0;
    }
    // Only norm/neox paths have fusion code in rope kernels
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return 0;
    }
    const int output = i + 2;
    if (!ggml_can_fuse_subgraph(
            cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { output })) {
        return 0;
    }
    ggml_sycl_rope_fused(ctx, rope, set_rows);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-rope] fuse hit (rope+view+set_rows) mode=%d ne0=%" PRId64
                    " ne1=%" PRId64 " type=%s\n",
                    mode, rope->ne[0], rope->ne[1], ggml_type_name(set_rows->type));
        }
    }
    return 2;  // skip VIEW + SET_ROWS
}

// Decode-only Q6_K V MMVQ -> indexed F16 V-cache write.
// Kill: GGML_SYCL_DISABLE_V_MMVQ_SET_ROWS_FUSE=1
bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);

int ggml_sycl_fuse_v_mmvq_set_rows(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_V_MMVQ_SET_ROWS_FUSE");
        return !(dis && std::atoi(dis));
    }();
    if (!enabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }

    ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] ||
        mm->src[0]->type != GGML_TYPE_Q6_K || mm->src[1]->type != GGML_TYPE_F32 ||
        mm->type != GGML_TYPE_F32 || mm->src[0]->ne[0] != 2048 ||
        mm->src[0]->ne[1] != 1024 || mm->src[1]->ne[0] != 2048 ||
        mm->src[1]->ne[1] != 1 || mm->src[1]->ne[2] != 1 || mm->src[1]->ne[3] != 1 ||
        mm->ne[0] != 1024 || ggml_nrows(mm) != 1 || !ggml_is_contiguous(mm)) {
        return 0;
    }

    auto descends_from = [](const ggml_tensor * node, const ggml_tensor * root) {
        for (int depth = 0; node && depth < 4; ++depth) {
            if (node == root) {
                return true;
            }
            if (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE) {
                break;
            }
            node = node->src[0];
        }
        return false;
    };

    ggml_tensor * set_v = nullptr;
    ggml_tensor * foreign_set = nullptr;
    int j_v = -1;
    int j_foreign = -1;
    for (int64_t j = i + 1; j < cgraph->n_nodes && j <= i + 12; ++j) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_NONE || ggml_is_empty(node)) {
            continue;
        }
        if (node->op != GGML_OP_SET_ROWS) {
            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_RMS_NORM ||
                node->op == GGML_OP_ROPE || node->op == GGML_OP_MUL) {
                continue;
            }
            return 0;
        }
        if (descends_from(node->src[0], mm)) {
            set_v = node;
            j_v = j;
            break;
        }
        if (foreign_set != nullptr) {
            return 0;
        }
        foreign_set = node;
        j_foreign = j;
    }
    if (!set_v) {
        return 0;
    }
    if (set_v->type != GGML_TYPE_F16 ||
        !set_v->src[1] || set_v->src[1]->type != GGML_TYPE_I64 ||
        ggml_nelements(set_v->src[1]) != 1 || set_v->ne[0] != 1024 ||
        set_v->nb[1] % sizeof(sycl::half) != 0) {
        return 0;
    }

    const int n_ops = j_v - i + 1;
    std::vector<ggml_op> ops;
    ops.reserve((size_t) n_ops);
    for (int j = i; j <= j_v; ++j) {
        ops.push_back(cgraph->nodes[j]->op);
    }
    int outputs[2] = { j_v, j_foreign };
    const int n_outputs = foreign_set ? 2 : 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), outputs, n_outputs)) {
        return 0;
    }

    opt_for_reorder(&ctx, mm->src[0], mm->src[1], mm, mul_mat_algo::MMVQ);
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(mm->src[0]->extra);
    if (!extra || !extra->optimized_feature.reorder) {
        return 0;
    }

    // Preserve any interleaved independent real ops before V-cache publication.
    for (int j = i + 1; j < j_v; ++j) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_NONE || ggml_is_empty(node) || node == foreign_set) {
            continue;
        }
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, node));
    }
    if (foreign_set) {
        ggml_sycl_op_set_rows(ctx, foreign_set);
    }
    ggml_sycl_mmvq_set_vcache_f16(
        static_cast<sycl::half *>(set_v->data),
        static_cast<const int64_t *>(set_v->src[1]->data),
        (int64_t) (set_v->nb[1] / sizeof(sycl::half)));
    ggml_sycl_mul_mat(ctx, mm->src[0], mm->src[1], mm);
    ggml_sycl_mmvq_set_vcache_f16(nullptr, nullptr, 0);

    static std::atomic<int> once{0};
    if (once.fetch_add(1) == 0) {
        fprintf(stderr, "[lx-vcache-mmvq] fuse hit rows=%" PRId64 " foreign_set=%d\n",
                mm->ne[0], foreign_set != nullptr);
    }
    return j_v - i;
}

// Graph fuse: softplus(gate) × attn (Laguna XS.2 per-head broadcast).
// Kill: GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE=1
int ggml_sycl_fuse_softplus_mul(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * soft = cgraph->nodes[i];
    if (soft->op != GGML_OP_UNARY || ggml_get_unary_op(soft) != GGML_UNARY_OP_SOFTPLUS) {
        return 0;
    }
    if (!soft->src[0] || soft->type != GGML_TYPE_F32 || soft->src[0]->type != GGML_TYPE_F32 ||
        !soft->src[0]->data) {
        return 0;
    }

    // Scan forward over RESHAPE/VIEW only to find MUL that consumes softplus (or reshape of it).
    ggml_tensor * mul = nullptr;
    int           mul_j = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 6; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t) {
            continue;
        }
        if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (t->op == GGML_OP_MUL) {
            mul   = t;
            mul_j = j;
            break;
        }
        // Other real op → not our pattern
        return 0;
    }
    if (!mul || mul_j < 0 || mul->type != GGML_TYPE_F32 || !mul->data) {
        return 0;
    }

    // Trace softplus lineage: soft or reshape/view chain from soft.
    auto is_soft_lineage = [&](const ggml_tensor * t) -> bool {
        for (int depth = 0; t && depth < 4; ++depth) {
            if (t == soft) {
                return true;
            }
            if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
                t = t->src[0];
                continue;
            }
            break;
        }
        return false;
    };

    const ggml_tensor * attn = nullptr;
    if (is_soft_lineage(mul->src[0])) {
        attn = mul->src[1];
    } else if (is_soft_lineage(mul->src[1])) {
        attn = mul->src[0];
    } else {
        return 0;
    }
    if (!attn || attn->type != GGML_TYPE_F32 || !attn->data || !ggml_is_contiguous(attn) ||
        !ggml_is_contiguous(soft->src[0]) || !ggml_is_contiguous(mul)) {
        return 0;
    }

    // Build op list soft .. mul inclusive for can_fuse_subgraph.
    const int n_ops = mul_j - i + 1;
    std::vector<ggml_op> ops;
    ops.reserve((size_t) n_ops);
    for (int j = i; j <= mul_j; ++j) {
        ops.push_back(cgraph->nodes[j]->op);
    }
    const int output = mul_j;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), &output, 1)) {
        return 0;
    }

    // Optional trailing reshape of mul that is a no-op view (same data) — include in skip.
    int skip = n_ops - 1;  // following nodes after soft
    if (mul_j + 1 < cgraph->n_nodes) {
        ggml_tensor * tail = cgraph->nodes[mul_j + 1];
        if (tail && tail->op == GGML_OP_RESHAPE && tail->src[0] == mul &&
            tail->data == mul->data && ggml_nelements(tail) == ggml_nelements(mul)) {
            // Still need can_fuse to allow eliding reshape only if not an output used elsewhere.
            // Safe: view reshape shares storage; compute is no-op. Skip it.
            skip += 1;
        }
    }

    const float * gate_d = (const float *) soft->src[0]->data;
    const float * attn_d = (const float *) attn->data;
    float *       dst_d  = (float *) mul->data;

    const int64_t nelt = ggml_nelements(mul);
    // Layout: softplus src [ne0_g, ne1_g, ...] typically [n_head, T]
    // attn matches mul shape. Broadcast softplus over leading dims of attn/mul.
    const int64_t ne0_a = mul->ne[0];
    const int64_t ne1_a = mul->ne[1];
    const int64_t ne2_a = mul->ne[2];
    const int64_t ne3_a = mul->ne[3];
    const int64_t ne0_g = soft->src[0]->ne[0];
    const int64_t ne1_g = soft->src[0]->ne[1];
    const int64_t ne2_g = soft->src[0]->ne[2];
    const int64_t ne3_g = soft->src[0]->ne[3];

    // Validate broadcast of gate onto mul (ggml rules: dims equal or 1).
    auto dim_ok = [](int64_t a, int64_t g) { return g == a || g == 1 || a == 1; };
    if (!dim_ok(ne0_a, ne0_g) || !dim_ok(ne1_a, ne1_g) || !dim_ok(ne2_a, ne2_g) ||
        !dim_ok(ne3_a, ne3_g)) {
        // XS.2: gate [n_head, T] vs mul [head_dim, n_head, T] — gate lacks leading head_dim.
        // Treat as gate aligned to dims 1.. of mul when soft nelements == ne1_a*ne2_a*ne3_a
        // and ne0_g == ne1_a (n_head) and ne1_g == ne2_a (T).
        const bool xs2 =
            (ne0_g == ne1_a && ne1_g == ne2_a && ne2_g == 1 && ne3_g == 1 && ne3_a == 1 &&
             ggml_nelements(soft->src[0]) == ne1_a * ne2_a);
        if (!xs2) {
            return 0;
        }
        // XS.2 per-head kernel: mul [hd, nh, T], gate [nh, T]
        const int64_t hd = ne0_a;
        const int64_t nh = ne1_a;
        const int64_t nt = ne2_a;
        constexpr int BS = 256;
        const int nblocks = (int) ((nelt + BS - 1) / BS);
        queue_ptr stream = ctx.stream();
        SYCL_CHECK(ggml_sycl_set_device(ctx.device));
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = (int64_t) item.get_global_id(0);
                if (idx >= nelt) {
                    return;
                }
                // idx = d + h*hd + t*hd*nh
                const int64_t d = idx % hd;
                const int64_t tmp = idx / hd;
                const int64_t h = tmp % nh;
                const int64_t t = tmp / nh;
                const float x = gate_d[h + t * nh];
                const float ax = sycl::fabs(x);
                const float m  = sycl::fmax(x, 0.0f);
                const float sp = m + sycl::log1p(sycl::exp(-ax));
                dst_d[idx] = attn_d[idx] * sp;
                (void) d;
            });
        {
            static std::atomic<int> once{0};
            if (once.fetch_add(1) == 0) {
                fprintf(stderr,
                        "[lx-control-softplus-mul] fuse hit (xs2 per-head) hd=%" PRId64
                        " nh=%" PRId64 " T=%" PRId64 "\n",
                        hd, nh, nt);
            }
        }
        return skip;
    }

    // General same-rank broadcast (incl. identical shapes).
    constexpr int BS = 256;
    const int nblocks = (int) ((nelt + BS - 1) / BS);
    queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
        [=](sycl::nd_item<1> item) {
            const int64_t idx = (int64_t) item.get_global_id(0);
            if (idx >= nelt) {
                return;
            }
            // Decode mul linear index into coords (row-major ne0 fastest).
            int64_t rem = idx;
            const int64_t i0 = rem % ne0_a; rem /= ne0_a;
            const int64_t i1 = rem % ne1_a; rem /= ne1_a;
            const int64_t i2 = rem % ne2_a; rem /= ne2_a;
            const int64_t i3 = rem;
            const int64_t g0 = (ne0_g == 1) ? 0 : i0;
            const int64_t g1 = (ne1_g == 1) ? 0 : i1;
            const int64_t g2 = (ne2_g == 1) ? 0 : i2;
            const int64_t g3 = (ne3_g == 1) ? 0 : i3;
            const int64_t gidx =
                g0 + ne0_g * (g1 + ne1_g * (g2 + ne2_g * g3));
            const float x = gate_d[gidx];
            const float ax = sycl::fabs(x);
            const float m  = sycl::fmax(x, 0.0f);
            const float sp = m + sycl::log1p(sycl::exp(-ax));
            dst_d[idx] = attn_d[idx] * sp;
        });
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-softplus-mul] fuse hit (broadcast) nelt=%" PRId64 "\n", nelt);
        }
    }
    return skip;
}

// Graph fuse: attn_gate_proj GEMV + softplus + per-head MUL (XS.2).
// The gate GEMV (q4_K, decode-only) is folded with its UNARY(SOFTPLUS) and the
// attn*softplus per-head MUL into ONE MMVQ launch: the gate GEMV kernel writes
// (a) its own dst, (b) the softplus dst and (c) the MUL dst (XS.2 per-head
// layout mul [hd, nh, nt], gate [nh, nt], row == head) using the exact
// op_softplus formula — bit-identical to the standalone kernels. Saves one
// launch per layer per token (the softplus+mul fuse launch).
// Default ON. Kill: GGML_SYCL_DISABLE_GATE_SOFTPLUS_FUSE=1
int ggml_sycl_fuse_gate_softplus_mul(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_GATE_SOFTPLUS_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const char * mname = mm->name ? mm->name : "";
    if (strncmp(mname, "attn_gate_proj", 14) != 0) {
        return 0;
    }
    // Skip a MUL that scales the GEMV dst by wqkv_gate_s (per-token gating
    // scale, present when the GGUF carries "attn_gate.scale"). It is a
    // cheap 2048-wide elementwise; keep it as a standalone kernel.
    int scan_from = i + 1;
    if (scan_from + 1 < cgraph->n_nodes) {
        ggml_tensor * cand = cgraph->nodes[scan_from];
        if (cand->op == GGML_OP_MUL && (cand->src[0] == mm || cand->src[1] == mm)) {
            scan_from++;
        }
    }
    if (!mm->src[0] || !mm->src[1] || !mm->src[0]->data || !mm->src[1]->data) {
        return 0;
    }
    // Decode-only + MMVQ path prerequisites (mirror can_use_mul_mat_vec_q).
    if (mm->src[1]->ne[1] != 1 || mm->src[1]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
        !ggml_is_quantized(mm->src[0]->type) || mm->src[0]->type != GGML_TYPE_Q4_K) {
        return 0;
    }

    // Scan forward for UNARY(SOFTPLUS) consuming the GEMV dst (views allowed).
    ggml_tensor * soft  = nullptr;
    int           soft_j = -1;
    for (int j = scan_from; j < cgraph->n_nodes && j <= scan_from + 6; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SOFTPLUS) {
            ggml_tensor * s = t->src[0];
            while (s && (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) {
                s = s->src[0];
            }
            if (s == mm) {
                soft   = t;
                soft_j = j;
            }
            break;
        }
        return 0;
    }
    if (!soft || soft_j < 0 || !soft->data || soft->type != GGML_TYPE_F32 ||
        soft->src[0]->type != GGML_TYPE_F32) {
        return 0;
    }

    // Scan forward for the MUL that consumes the softplus lineage.
    ggml_tensor * mul  = nullptr;
    int           mul_j = -1;
    auto is_soft_lineage = [&](const ggml_tensor * t) -> bool {
        for (int depth = 0; t && depth < 4; ++depth) {
            if (t == soft) {
                return true;
            }
            if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
                t = t->src[0];
                continue;
            }
            break;
        }
        return false;
    };
    for (int j = soft_j + 1; j < cgraph->n_nodes && j <= soft_j + 6; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (!t) {
            continue;
        }
        if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (t->op == GGML_OP_MUL) {
            mul   = t;
            mul_j = j;
            break;
        }
        return 0;
    }
    if (!mul || mul_j < 0 || mul->type != GGML_TYPE_F32 || !mul->data ||
        !ggml_is_contiguous(mul) || !ggml_is_contiguous(soft->src[0])) {
        return 0;
    }
    const ggml_tensor * attn = nullptr;
    if (is_soft_lineage(mul->src[0])) {
        attn = mul->src[1];
    } else if (is_soft_lineage(mul->src[1])) {
        attn = mul->src[0];
    } else {
        return 0;
    }
    if (!attn || attn->type != GGML_TYPE_F32 || !attn->data || !ggml_is_contiguous(attn)) {
        return 0;
    }

    // XS.2 per-head layout: gate [nh, nt], mul [hd, nh, nt]. Decode-only nt==1.
    const int64_t ne0_g = soft->src[0]->ne[0];
    const int64_t ne1_g = soft->src[0]->ne[1];
    const int64_t ne0_a = mul->ne[0];
    const int64_t ne1_a = mul->ne[1];
    const int64_t ne2_a = mul->ne[2];
    if (!(ne0_g == ne1_a && ne1_g == ne2_a && ne2_a == 1 && mul->ne[3] == 1 &&
          ggml_nelements(soft->src[0]) == ne1_a * ne2_a)) {
        return 0;
    }

    const int n_ops = mul_j - i + 1;
    std::vector<ggml_op> ops;
    ops.reserve((size_t) n_ops);
    for (int j = i; j <= mul_j; ++j) {
        ops.push_back(cgraph->nodes[j]->op);
    }
    const int output = mul_j;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), &output, 1)) {
        return 0;
    }

    // Exclude the optional gate-scale MUL from the fused span: it stays a
    // standalone launch (executed after this fuse returns).
    const bool gate_scale_skipped = (scan_from > i + 1);

    lx_gate_epilogue ep;
    static const int mode = []() {
        const char * e = getenv("GGML_SYCL_GATE_FUSE_MODE");
        return e != nullptr ? std::atoi(e) : 1;  // 1=full, 2=softplus-only, 3=softplus keep MUL, 4=full+sync, 5=full keep both, 7=mul-only skip=3
    }();
    ep.softplus_dst = (mode == 1 || mode == 4 || mode == 5 || mode == 3) ? (float *) soft->data : nullptr;
    ep.attn         = (mode == 1 || mode == 4 || mode == 5 || mode == 7) ? (const float *) attn->data : nullptr;
    ep.mul_dst      = (mode == 1 || mode == 4 || mode == 5 || mode == 7) ? (float *) mul->data : nullptr;
    ep.hd           = ne0_a;
    ep.nh           = ne1_a;
    ep.nt           = ne2_a;
    ggml_sycl_mmvq_set_gate_epilogue(&ep);
    ggml_sycl_mul_mat(ctx, mm->src[0], mm->src[1], mm);
    ggml_sycl_mmvq_set_gate_epilogue(nullptr);
    if (mode == 4) {
        // full fuse + host sync (race probe)
        ctx.stream()->wait();
    }

    {
        static const bool dump = getenv("GGML_SYCL_GATE_DUMP") != nullptr;
        if (dump && mode != 2 && mode != 4) {
            // host-side sanity dump (blocking): gate dst, softplus dst, attn, mul
            auto copy_f = [&](const void * p, int64_t n) {
                std::vector<float> v((size_t) n);
                SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->memcpy(v.data(), p,
                    (size_t) n * sizeof(float)).wait()));
                float s = 0, mn = 1e30f, mx = -1e30f;
                for (auto x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
                fprintf(stderr, "[lx-gate-dump] n=%lld sum=%.4f min=%.4f max=%.4f first=%.4f\n",
                        (long long) n, (double) s, (double) mn, (double) mx, (double) v[0]);
            };
            fprintf(stderr, "[lx-gate-dump] mm=%s mode=%d hd=%lld nh=%lld nt=%lld soft_data=%p mul_data=%p attn_data=%p alias_soft_mul=%d\n",
                    mname, mode, (long long) ep.hd, (long long) ep.nh, (long long) ep.nt,
                    (void *) soft->data, (void *) mul->data, (void *) attn->data,
                    (int) (soft->data == mul->data));
            copy_f(mm->data, soft->src[0]->ne[0] * soft->src[0]->ne[1]);
            copy_f(soft->data, soft->ne[0] * soft->ne[1]);
            if (mode == 1) {
                copy_f(attn->data, ggml_nelements(attn));
                copy_f(mul->data, ggml_nelements(mul));
            }
            if (dump && mode == 1) {
                // print first 8 attn and mul values for cross-mode comparison
                auto copy_first = [&](const void * p, int64_t n, const char * tag) {
                    std::vector<float> v((size_t) n);
                    SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->memcpy(v.data(), p,
                        (size_t) n * sizeof(float)).wait()));
                    fprintf(stderr, "[lx-gate-first] %s:", tag);
                    for (int64_t k = 0; k < 8 && k < n; ++k) { fprintf(stderr, " %.4f", (double) v[k]); }
                    fprintf(stderr, "\n");
                };
                copy_first(attn->data, ggml_nelements(attn), "attn");
                copy_first(mul->data, ggml_nelements(mul), "mul");
                copy_first(soft->data, soft->ne[0] * soft->ne[1], "sp");
            }
        }
    }

    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-gate-softplus] fuse hit hd=%" PRId64 " nh=%" PRId64
                    " wtype=%s mm='%s'\n",
                    ep.hd, ep.nh, ggml_type_name(mm->src[0]->type), mname);
        }
    }
    if (mode == 3) {
        return soft_j - i;  // softplus written by the launch; MUL stays standalone
    }
    if (mode == 5) {
        return 0;  // fused write happens, but softplus + MUL also run standalone (recompute)
    }
    {
        static const bool dbg_span = getenv("GGML_SYCL_GATE_DBG_SPAN") != nullptr;
        if (dbg_span) {
            fprintf(stderr, "[lx-gate-span] i=%d soft_j=%d mul_j=%d skip=%d nodes:",
                    i, soft_j, mul_j, (int) ((mul_j - i) + (gate_scale_skipped ? 1 : 0)));
            for (int k = i; k <= mul_j && k < cgraph->n_nodes; ++k) {
                fprintf(stderr, " %s(%s)", cgraph->nodes[k]->name ? cgraph->nodes[k]->name : "?",
                        ggml_op_name(cgraph->nodes[k]->op));
            }
            fprintf(stderr, "\n");
        }
    }
    return (mul_j - i) + (gate_scale_skipped ? 1 : 0);  // skip softplus (+views), MUL, and any skipped gate-scale MUL
}

// Graph fuse: ADD → ADD residual chain (Laguna: moe+shexp then +ffn_inp).
// dst = (a + b) + c with left-to-right order matching two stock ADDs.
// Default ON. Kill: GGML_SYCL_DISABLE_ADD_ADD_FUSE=1
int ggml_sycl_fuse_add_add(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_ADD_ADD_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * add0 = cgraph->nodes[i];
    ggml_tensor * add1 = cgraph->nodes[i + 1];
    if (add0->op != GGML_OP_ADD || add1->op != GGML_OP_ADD) {
        return 0;
    }
    if (add1->src[0] != add0 && add1->src[1] != add0) {
        return 0;
    }
    if (!add0->src[0] || !add0->src[1] || !add1->src[0] || !add1->src[1]) {
        return 0;
    }
    const ggml_tensor * a = add0->src[0];
    const ggml_tensor * b = add0->src[1];
    const ggml_tensor * c = add1->src[0] == add0 ? add1->src[1] : add1->src[0];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || c->type != GGML_TYPE_F32 ||
        add0->type != GGML_TYPE_F32 || add1->type != GGML_TYPE_F32) {
        return 0;
    }
    // Contiguous same-shape only (no broadcast surface) — Laguna residual is this case.
    if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, c) ||
        !ggml_are_same_shape(a, add1) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(c) ||
        !ggml_is_contiguous(add1) ||
        !a->data || !b->data || !c->data || !add1->data) {
        return 0;
    }
    const int output = i + 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, { output })) {
        return 0;
    }

    const int64_t nelt = ggml_nelements(add1);
    const float * ad = (const float *) a->data;
    const float * bd = (const float *) b->data;
    const float * cd = (const float *) c->data;
    float *       dd = (float *) add1->data;
    constexpr int BS = 256;
    const int nblocks = (int) ((nelt + BS - 1) / BS);
    queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
        [=](sycl::nd_item<1> item) {
            const int64_t idx = (int64_t) item.get_global_id(0);
            if (idx >= nelt) {
                return;
            }
            // Left-to-right: (a+b)+c matches stock ADD then ADD.
            const float t = ad[idx] + bd[idx];
            dd[idx]       = t + cd[idx];
        });
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-add] fuse hit (add+add residual) nelt=%" PRId64 " ne0=%" PRId64 " ne1=%" PRId64 "\n",
                    nelt, add1->ne[0], add1->ne[1]);
        }
    }
    return 1;  // skip following ADD
}

// dual+down multi-token (expert-loop preferred). Defined after mul_mat_id helpers.
static bool ggml_sycl_mul_mat_id_dual_down_multitoken(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu,
    ggml_tensor * down_mmid, const ggml_tensor * weights, ggml_tensor * dst_final);

// Graph fuse: MUL_MAT_ID + MUL_MAT_ID + GLU(swiglu). Returns following nodes to skip.
int ggml_sycl_fuse_moe_dual_swiglu(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    ggml_tensor * mmid1 = cgraph->nodes[i + 1];
    ggml_tensor * glu   = cgraph->nodes[i + 2];
    if (mmid1->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    const bool edges_ok =
        ((glu->src[0] == node && glu->src[1] == mmid1) ||
         (glu->src[0] == mmid1 && glu->src[1] == node)) &&
        node->src[1] == mmid1->src[1] &&
        node->src[2] == mmid1->src[2];
    if (!edges_ok) {
        return 0;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];

    // dual+down multi-token when weighted-down chain follows GLU.
    // Kill: GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
    // Expert-loop needs reorder-safe multi-col MMVQ (mul_mat chunk fix).
    static const bool enable_dual_down = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MOE_DUAL_DOWN");
        if (dis != nullptr && std::atoi(dis) != 0) {
            return false;
        }
        const char * env = getenv("GGML_SYCL_ENABLE_MOE_DUAL_DOWN");
        if (env == nullptr) {
            return true;
        }
        return std::atoi(env) != 0;
    }();
    if (enable_dual_down && i + 5 < cgraph->n_nodes) {
        ggml_tensor * down_mmid = cgraph->nodes[i + 3];
        ggml_tensor * mul_w     = cgraph->nodes[i + 4];
        if (down_mmid->op == GGML_OP_MUL_MAT_ID && down_mmid->src[1] == glu &&
            down_mmid->src[2] == node->src[2] &&
            mul_w->op == GGML_OP_MUL &&
            (mul_w->src[0] == down_mmid || mul_w->src[1] == down_mmid)) {
            const ggml_tensor * weights =
                mul_w->src[0] == down_mmid ? mul_w->src[1] : mul_w->src[0];
            const int64_t n_embd         = down_mmid->ne[0];
            const int64_t n_experts_used = down_mmid->ne[1];
            const int64_t n_tokens       = down_mmid->ne[2];
            const int n_views = (int) n_experts_used;
            const int n_adds  = (int) n_experts_used - 1;
            const int n_ops = 3 + 2 + n_views + n_adds;
            if (n_experts_used >= 2 && n_experts_used <= 16 &&
                n_tokens >= 2 && n_tokens <= 2048 &&
                weights && weights->type == GGML_TYPE_F32 &&
                weights->ne[0] == 1 && weights->ne[1] == n_experts_used &&
                weights->ne[2] == n_tokens &&
                down_mmid->type == GGML_TYPE_F32 && mul_w->type == GGML_TYPE_F32 &&
                i + n_ops <= cgraph->n_nodes) {
                std::vector<ggml_op> ops;
                ops.reserve((size_t) n_ops);
                ops.push_back(GGML_OP_MUL_MAT_ID);
                ops.push_back(GGML_OP_MUL_MAT_ID);
                ops.push_back(GGML_OP_GLU);
                ops.push_back(GGML_OP_MUL_MAT_ID);
                ops.push_back(GGML_OP_MUL);
                for (int v = 0; v < n_views; ++v) {
                    ops.push_back(GGML_OP_VIEW);
                }
                for (int a = 0; a < n_adds; ++a) {
                    ops.push_back(GGML_OP_ADD);
                }
                const int output = i + n_ops - 1;
                bool views_ok = true;
                for (int v = 0; v < n_views && views_ok; ++v) {
                    ggml_tensor * view = cgraph->nodes[i + 5 + v];
                    views_ok = view && view->op == GGML_OP_VIEW && view->src[0] == mul_w &&
                               view->ne[0] == n_embd && view->ne[1] == n_tokens;
                }
                ggml_tensor * prev = cgraph->nodes[i + 5];
                for (int a = 0; a < n_adds && views_ok; ++a) {
                    ggml_tensor * add = cgraph->nodes[i + 5 + n_views + a];
                    ggml_tensor * v1  = cgraph->nodes[i + 5 + a + 1];
                    views_ok = add && add->op == GGML_OP_ADD &&
                               ((add->src[0] == prev && add->src[1] == v1) ||
                                (add->src[1] == prev && add->src[0] == v1));
                    prev = add;
                }
                ggml_tensor * dst = cgraph->nodes[output];
                const ggml_tensor * down_w = down_mmid->src[0];
                const bool types_ok =
                    down_w &&
                    (down_w->type == GGML_TYPE_Q4_K || down_w->type == GGML_TYPE_Q5_K ||
                     down_w->type == GGML_TYPE_Q6_K) &&
                    down_w->type == gate->src[0]->type &&
                    down_w->ne[0] == gate->src[0]->ne[1] &&
                    down_w->ne[1] == n_embd &&
                    down_w->ne[2] == gate->src[0]->ne[2] &&
                    down_mmid->src[0] == down_w &&
                    down_mmid->data && glu->data && weights->data && dst->data;
                if (views_ok && types_ok && dst && dst->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(dst) && dst->ne[0] == n_embd &&
                    dst->ne[1] == n_tokens &&
                    ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), &output, 1) &&
                    ggml_sycl_mul_mat_id_dual_down_multitoken(
                        ctx, gate, up, glu, down_mmid, weights, dst)) {
                    static std::atomic<int> once{0};
                    if (once.fetch_add(1) == 0) {
                        fprintf(stderr,
                                "[lx-control-moe-dual] fuse hit (dual+down multi-token) "
                                "tokens=%" PRId64 " k=%" PRId64 "\n",
                                n_tokens, n_experts_used);
                    }
                    return n_ops - 1;
                }
            }
        }
    }

    const int output = i + 2;
    if (!ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU },
            { output })) {
        return 0;
    }
    if (ggml_sycl_mul_mat_id_dual_swiglu_fused(ctx, gate, up, glu)) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr, "[lx-control-moe-dual] fuse hit (gate+up+swiglu)\n");
        }
        return 2;
    }
    return 0;
}

// Dense shared-expert dual gate+up + SwiGLU (MUL_MAT + MUL_MAT + GLU).
// Port of package dense dual onto control for Laguna ffn_shexp (Q4_K).
// Kill: GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1
static bool ggml_sycl_mul_mat_dense_dual_swiglu_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * gate,
    const ggml_tensor * up, ggml_tensor * glu) {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (disabled) {
        return false;
    }

    const ggml_tensor * gate_w = gate->src[0];
    const ggml_tensor * up_w   = up->src[0];
    const ggml_tensor * src1   = gate->src[1];

    const bool type_ok =
        gate_w->type == up_w->type &&
        (gate_w->type == GGML_TYPE_Q4_K || gate_w->type == GGML_TYPE_Q5_K ||
         gate_w->type == GGML_TYPE_Q6_K);
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        gate->src[1] != up->src[1] || !type_ok ||
        src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ncols     = src1->ne[0];
    const int64_t ncols_dst = src1->ne[1];
    const int64_t nrows     = gate_w->ne[1];
    if (ncols_dst < 1 || ncols_dst > 32 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        gate_w->ne[0] != ncols || up_w->ne[0] != ncols ||
        up_w->ne[1] != nrows ||
        gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
        up_w->ne[2] != 1 || up_w->ne[3] != 1 ||
        ncols % QK8_1 != 0 ||
        !ggml_is_contiguous(src1) || !ggml_is_contiguous(glu) ||
        !ggml_is_contiguous(gate_w) || !ggml_is_contiguous(up_w) ||
        gate->ne[0] != nrows || up->ne[0] != nrows ||
        gate->ne[1] != ncols_dst || up->ne[1] != ncols_dst ||
        !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, glu) ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(glu, 1) != 0 /* swapped */) {
        return false;
    }

    // Ensure reorder MMVQ layout on both weight matrices (2D dense).
    auto ensure_reorder = [&](const ggml_tensor * w) {
        if (!g_ggml_sycl_enable_optimize || !ctx.opt_feature.reorder) {
            return false;
        }
        if (!ggml_sycl_supports_reorder_mmvq(w->type)) {
            return false;
        }
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(w->extra);
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
    if (!ensure_reorder(gate_w) || !ensure_reorder(up_w)) {
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

    return ggml_sycl_mul_mat_vec_q_dense_dual_swiglu_reorder(
        gate_w->type, gate_w->data, up_w->data, src1_q8, (float *) glu->data,
        (int) ncols, (int) nrows, (int) ncols_dst, bytes_per_qrow, dst_col_stride,
        stream);
}

// Graph fuse: MUL_MAT + MUL_MAT + GLU(swiglu) for dense shared expert.
int ggml_sycl_fuse_dense_dual_swiglu(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    ggml_tensor * mm1 = cgraph->nodes[i + 1];
    ggml_tensor * glu = cgraph->nodes[i + 2];
    if (mm1->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    const bool edges_ok =
        ((glu->src[0] == node && glu->src[1] == mm1) ||
         (glu->src[0] == mm1 && glu->src[1] == node)) &&
        node->src[1] == mm1->src[1];
    if (!edges_ok) {
        return 0;
    }
    const int output = i + 2;
    if (!ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU },
            { output })) {
        return 0;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    if (ggml_sycl_mul_mat_dense_dual_swiglu_fused(ctx, gate, up, glu)) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr, "[lx-control-dense-dual] fuse hit (shared gate+up+swiglu)\n");
        }
        return 2;
    }
    return 0;
}

// Multi-token fused MMID default OFF until golden-proven (single-token path is tip).
// Enable: GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1
// Hard-off: GGML_SYCL_DISABLE_MMID_FUSED_BATCH=1
static bool ggml_sycl_mmid_fused_batch_enabled() {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MMID_FUSED_BATCH");
        if (dis != nullptr && std::atoi(dis) != 0) {
            return false;
        }
        const char * env = getenv("GGML_SYCL_ENABLE_MMID_FUSED_BATCH");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

// Fused MoE MMVQ: device-side ids, no host wait.
// ne12==1 always (decode tip). Multi-token: opt-in ENABLE_MMID_FUSED_BATCH=1, cap 64.
// Multi-token uses per-token single-row launches (bitexact vs ne12==1 path; avoids D2H).
// [lx-diag-decode] Decode wall-decomposition probe (measurement-only, never on by default).
// GGML_SYCL_DIAG_SKIP_DECODE bitmask: 1 = skip fused-QKV launch, 2 = skip MoE
// (mul_mat_id fused) launches at decode, 4 = skip dense vec-q launches. When a bit
// is set the kernel launch is elided (dst stays stale) so llama-bench wall time
// measures the remaining pipeline: dispatch + non-GEMM kernels (rms/rope/fattn/
// softplus/topk/reduce) + host sync. Behavioral no-op when env is unset/0.
int lx_diag_skip_decode(void) {
    static const int skip = []() {
        const char * e = getenv("GGML_SYCL_DIAG_SKIP_DECODE");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return skip;
}

static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    // Single multi-token launch mode (one kernel for all tokens×experts).
    static const bool single_launch = []() {
        const char * e = getenv("GGML_SYCL_MMID_FUSED_SINGLE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    const int64_t ne12_cap = single_launch ? 8192 : 64;
    if (ne12 < 1 || ne12 > ne12_cap) return false;
    if (ne12 > 1 && !ggml_sycl_mmid_fused_batch_enabled() && !single_launch) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;

    const int64_t n_ids_per_group = ids->ne[0];
    // ids layout: [n_experts_used, n_tokens] with ne[1]==ne12
    if (ids->ne[1] != ne12) return false;
    if (ids->nb[0] != sizeof(int32_t)) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    // [lx-diag-decode] measurement-only skip of the whole fused MoE path (quantize
    // + reorder + per-token GEMM launches). ne12 gates above already exclude pp512,
    // so this only fires at decode/ne12<=64.
    if (lx_diag_skip_decode() & 2) {
        return true;
    }

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    // Quantize all token×slot activation rows.
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

    const size_t bytes_per_qrow    = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride   = (ne11 == 1) ? 0 : bytes_per_qrow;
    const size_t src1_token_stride = (size_t) ne11 * bytes_per_qrow;
    const int    ids_row_stride    = (int) (ids->nb[1] / sizeof(int32_t));
    const size_t dst_token_stride  = dst->nb[2];
    const int32_t * ids_base       = (const int32_t *) ids->data;

    // Per-token launches: same kernel as decode (n_tokens=1) with pointer offsets.
    // Avoids multi-token grid bugs while still skipping D2H + counting sort.
    // Single-launch mode (GGML_SYCL_MMID_FUSED_SINGLE=1): one kernel for all tokens.
    if (single_launch) {
        bool ok;
        if (use_reorder) {
            ok = ggml_sycl_mul_mat_vec_q_id_reorder(
                src0->type, src0->data, src1_ddq, ids_base, (float *) dst->data,
                (int) ne10, nrows, n_experts_used,
                src0->nb[2], dst->nb[1], src1_row_stride,
                (int) ne12, ids_row_stride, src1_token_stride, dst_token_stride, stream);
        } else {
            ok = ggml_sycl_mul_mat_vec_q_id(
                src0->type, src0->data, src1_ddq, ids_base, (float *) dst->data,
                (int) ne10, nrows, n_experts_used,
                src0->nb[2], dst->nb[1], src1_row_stride,
                (int) ne12, ids_row_stride, src1_token_stride, dst_token_stride, stream);
        }
        if (!ok) {
            return false;
        }
    } else {
    for (int64_t t = 0; t < ne12; ++t) {
        const int32_t * ids_t = ids_base + t * ids_row_stride;
        const char *    vy_t  = src1_ddq + t * src1_token_stride;
        float *         dst_t = (float *) ((char *) dst->data + t * dst_token_stride);
        bool ok;
        if (use_reorder) {
            ok = ggml_sycl_mul_mat_vec_q_id_reorder(
                src0->type, src0->data, vy_t, ids_t, dst_t, (int) ne10, nrows, n_experts_used,
                src0->nb[2], dst->nb[1], src1_row_stride,
                /*n_tokens=*/1, /*ids_row_stride=*/n_experts_used, /*src1_token_stride=*/0,
                /*dst_token_stride=*/0, stream);
        } else {
            ok = ggml_sycl_mul_mat_vec_q_id(
                src0->type, src0->data, vy_t, ids_t, dst_t, (int) ne10, nrows, n_experts_used,
                src0->nb[2], dst->nb[1], src1_row_stride,
                1, n_experts_used, 0, 0, stream);
        }
        if (!ok) {
            return false;
        }
    }
    }

    {
        static std::atomic<int> once{0};
        if (ne12 > 1 && once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-mmid] multi-token fused (per-token) n_tokens=%" PRId64
                    " k=%d reorder=%d\n",
                    ne12, n_experts_used, (int) use_reorder);
        }
    }
    return true;
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

// [lx-chrono] wait-timer: GGML_SYCL_LX_CHRONO=1 also accumulates the elapsed
// time of every guarded stream->wait() inside the MoE mmid machinery, printed
// at exit as LX_WAIT_TOTAL_NS / LX_WAIT_COUNT. Env-off = one branch, zero cost.
static bool lx_chrono_on() {
    static const bool on = []() {
        const char * e = getenv("GGML_SYCL_LX_CHRONO");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}
static std::atomic<uint64_t> & lx_chrono_wait_ns() {
    static std::atomic<uint64_t> v(0);
    return v;
}
static std::atomic<uint64_t> & lx_chrono_wait_count() {
    static std::atomic<uint64_t> v(0);
    return v;
}
struct lx_chrono_wait_guard {
    const uint64_t t0;
    lx_chrono_wait_guard() : t0((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()) {}
    ~lx_chrono_wait_guard() {
        if (lx_chrono_on()) {
            const uint64_t t1 = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            lx_chrono_wait_ns().fetch_add(t1 - t0, std::memory_order_relaxed);
            lx_chrono_wait_count().fetch_add(1, std::memory_order_relaxed);
        }
    }
};

// [lx-expert-tile] GGML_SYCL_LX_EXPERT_TILE_GEMM=1 (env, default OFF): fused
// XMX dequant-GEMM for the MUL_MAT_ID prefill expert loop, v2. v1 (scalar SIMT
// dequant-dot, results/p2-c1-tile-*/c1-expert-tile.patch) kept the correct
// engagement/skip/metadata plumbing but its kernel was ~2x slower than the
// per-expert oneMKL loop (notes/FINDING_20260811_p2_expert_gemm_sizing.md).
// v2 reuses that plumbing and replaces the kernel with the microbench-proven
// XMX configuration (benchmark/xmx-dequant-gemm, winning cfg at N<=32:
// t16x16x16-wgm64-sgc16-nb32-sk16-kb64-sg16 — 1.33x/1.21x vs MKL at N=16/32):
//   - 16x16x16 fp16*fp16+fp32 joint_matrix (the only hardware-real BMG combo),
//     sub-group 16;
//   - each workgroup dequantizes a KB=64 x WG_M=64 q4_K weight tile into
//     VNNI/ext_intel_packed fp16 SLM (half2 stores) from EITHER serving-state
//     layout (reordered per-expert SoA [qs|scales|dm] or linear block array);
//   - oversubscribed dequant: 16 sub-groups cooperate on the dequant phase,
//     the first 4 do the joint_matrix MADs (WG_M/SG_COLS = 64/16);
//   - split-K across workgroups with fp32 partials + a fixup kernel.
// Scope: src0->type == GGML_TYPE_Q4_K only (gate/up all 38 layers + down in
// 22); q6_K down dispatches stay 100% on the oneMKL loop. Band: experts with
// 1..LX_ET MAX_N=32 routed rows (microbench crossover); larger experts stay on
// the loop, which skips exactly the band, keyed off the SAME counts array.
//
// Grid/enumeration (documented choice): flattened metadata-indexed. The host
// compacts the in-band experts into slots [0..n_slots); workgroup id wg maps
// as slot = wg / (n_m_tiles*split_k), mt = (wg % (n_m_tiles*split_k)) / split_k,
// split = wg % split_k, i.e. grid = n_slots x (ne01/WG_M) m-tiles x split_k
// K-blocks, all launched as ONE 1-D nd_range of 256-thread workgroups. No WG
// is idle (out-of-band experts are never enumerated). Because joint_matrix
// stores cannot mask token rows, each slot's rows are PADDED to a multiple of
// TM=16 in two device-side scratch surfaces indexed by pad-row:
//   act  : fp16 activation staging [n_pad_rows x ne00] (F32 src1_contiguous
//          rows converted on device; pad rows zero-filled) — keeps the A-tile
//          joint_matrix_load identical to the proven microbench (global fp16,
//          stride K);
//   part : fp32 partials [split_k x n_pad_rows x ne01] written by
//          joint_matrix_store.
// A fixup kernel then sums the split_k partials of each REAL row and scatters
// into dst_contiguous. split_k is derived per shape: largest power of two
// <= 16 that divides ne00/KB while keeping >=2 chunks per split (K=2048 ->
// 16, exactly the winning microbench config; K=512 -> 4), then halved until
// the partials fit LX_ET_PART_BUDGET (the microbench measured one expert; the
// multi-expert grid already fills the machine, so shrinking split_k for big
// bands only removes redundancy).
//
// Verify mode: GGML_SYCL_LX_EXPERT_TILE_VERIFY=1 keeps the oneMKL loop
// computing the band too; dst_contiguous is snapshotted after the tile+fixup
// writes and compared row-for-row against the loop's output (max-abs /
// max-rel, rel denominator max(1,|ref|) as in the microbench), printed per
// dispatch. Slow by design; it exists to prove numerics.
static bool lx_expert_tile_gemm_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_SYCL_LX_EXPERT_TILE_GEMM");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

static bool lx_expert_tile_verify_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_SYCL_LX_EXPERT_TILE_VERIFY");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

namespace lx_et {

namespace sem = sycl::ext::oneapi::experimental::matrix;

static constexpr int MAX_N      = 32;                  // band: experts with 1..MAX_N routed rows
static constexpr int TM         = 16;                  // joint_matrix tile (BMG hardware-real combo)
static constexpr int TN         = 16;
static constexpr int TK         = 16;
static constexpr int SG_SIZE    = 16;
static constexpr int WG_M       = 64;                  // output features (of ne01) per workgroup
static constexpr int SG_COLS    = 16;                  // output features per MAD sub-group (RM = SG_COLS/TN = 1)
static constexpr int NSG_MAD    = WG_M / SG_COLS;      // 4 MAD sub-groups
static constexpr int NSG_TOT    = 16;                  // oversubscribed dequant: 16 SGs total
static constexpr int WG_THREADS = NSG_TOT * SG_SIZE;   // 256
static constexpr int KB         = 64;                  // K per chunk: one q4_K super-block quarter
static constexpr int SPLIT_K_MAX = 16;
static constexpr int RN_MAX     = MAX_N / TM;          // 2 token tiles per WG max
static constexpr size_t PART_BUDGET = 128ull * 1024 * 1024;  // split-K partials cap (bytes)

template <typename T> static inline auto gptr(T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space,
                                    sycl::access::decorated::no>(p);
}

// Same math as dequantize_q4_K_common (dequantize.hpp), but the stores go
// through a functor store2(j, v0, v1) with j (even) the in-block index of the
// value pair — needed to write the VNNI/ext_intel_packed SLM layout with
// vectorized half2 stores. Copied from benchmark/xmx-dequant-gemm/main.cpp
// (value math identical; falsified-v1 already proved the mapping vs both bank
// layouts).
template <typename Store2Fn>
static inline void dequantize_q4_K_math(Store2Fn && store2, const uint8_t * __restrict__ qs_ptr, const float dall,
                                        const float dmin, const uint8_t * __restrict__ scales_local, int il, int ir) {
    const int is = 2 * il;
    constexpr int n = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    const int j0 = 64 * il + n * ir;
    store2(j0 + 0,  d1 * (q_vec[0] & 0xF) - m1, d1 * (q_vec[1] & 0xF) - m1);
    store2(j0 + 2,  d1 * (q_vec[2] & 0xF) - m1, d1 * (q_vec[3] & 0xF) - m1);
    store2(j0 + 32, d2 * (q_vec[0] >>  4) - m2, d2 * (q_vec[1] >>  4) - m2);
    store2(j0 + 34, d2 * (q_vec[2] >>  4) - m2, d2 * (q_vec[3] >>  4) - m2);
}

using local_half_ptr = sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space,
                                       sycl::access::decorated::no>;

// One workgroup = (in-band slot, m-tile of WG_M output features, split-K
// block). Phase 1 (all 16 SGs): dequantize the KB x WG_M weight tile into
// packed SLM — 64 tasks of 8 threads (llama's il/ir split), 2 passes over the
// 256 threads. Phase 2 (first 4 SGs): 16x16x16 joint_matrix MADs, A tiles
// straight from the padded fp16 activation staging (row stride ne00), B from
// SLM. fp32 accumulate across all chunks of this split; stores go to the
// padded fp32 partials surface (never to dst rows — masking happens in the
// fixup kernel).
template <bool REORDERED>
static inline void tile_gemm_device(const uint8_t *__restrict__ w_base_all,  // quantized bank (src0->data)
                                    const sycl::half *__restrict__ act,      // [n_pad_rows][ne00] staged fp16
                                    float *__restrict__ part,                // [split_k][n_pad_rows][ne01]
                                    const int32_t *__restrict__ slots,       // [n_slots][3] = {expert, nrows, pad_off}
                                    const int64_t ne00, const int64_t ne01,
                                    const size_t  nb02,                      // bytes per expert slice
                                    const int n_m_tiles, const int split_k,
                                    const int64_t n_pad_rows,
                                    sycl::half * wt, local_half_ptr wt_mp,
                                    const sycl::nd_item<1> & it) {
    const int wg    = (int) it.get_group(0);
    const int slot  = wg / (n_m_tiles * split_k);
    const int rem   = wg % (n_m_tiles * split_k);
    const int mt    = rem / split_k;
    const int split = rem % split_k;

    const int e     = slots[3 * slot + 0];
    const int nrows = slots[3 * slot + 1];
    const int poff  = slots[3 * slot + 2];
    const int RN    = (nrows + TM - 1) / TM;  // 1 or 2 (uniform per WG: barrier-safe)

    const int       m0         = mt * WG_M;
    const int       row_blocks = (int) (ne00 / QK_K);           // super-blocks per weight row
    const int64_t   nbpe       = (int64_t) row_blocks * ne01;   // super-blocks per expert
    const uint8_t * base       = w_base_all + (size_t) e * nb02;
    // reordered per-expert SoA section bases (exactly reorder_qw_q4_k_moe /
    // dequantize_block_q4_K_reorder offset math); unused when !REORDERED
    const uint8_t * sc_base    = base + (size_t) nbpe * (QK_K / 2);
    const uint8_t * dm_base    = sc_base + (size_t) nbpe * K_SCALE_SIZE;

    sycl::sub_group sg = it.get_sub_group();
    const int  lid       = (int) it.get_local_id(0);
    const int  sgid      = (int) sg.get_group_id()[0];
    const bool is_mad_sg = sgid < NSG_MAD;
    const int  sg_m0     = sgid * SG_COLS;  // MAD SGs only

    const int tq = lid / 8;  // dequant task slot within pass
    const int ir = lid % 8;  // llama's ir: 4-value column within quarter

    const int cps = (int) (ne00 / KB) / split_k;  // chunks per split (>= 1, exact)

    sem::joint_matrix<sycl::sub_group, float, sem::use::accumulator, TM, TN> C[RN_MAX];
    for (int rn = 0; rn < RN; ++rn) {
        sem::joint_matrix_fill(sg, C[rn], 0.0f);
    }

    for (int c = 0; c < cps; ++c) {
        const int kb  = split * cps + c;
        const int blk = (kb * KB) / QK_K;        // super-block column
        const int il  = (kb * KB % QK_K) / 64;   // quarter within the super-block
        // cooperative dequant: WG_M tasks (KB==64 -> one quarter per weight row),
        // 8 threads each, WG_THREADS/8 = 32 tasks per pass -> 2 passes
#pragma unroll
        for (int pass = 0; pass < WG_M / (WG_THREADS / 8); ++pass) {
            const int     ml = pass * (WG_THREADS / 8) + tq;    // weight row within the m-tile
            const int64_t ib = (int64_t) (m0 + ml) * row_blocks + blk;
            const uint8_t * qs;
            const uint8_t * scp;
            sycl::half2     dm;
            if constexpr (REORDERED) {
                qs  = base + ib * (QK_K / 2);
                scp = sc_base + ib * K_SCALE_SIZE;
                dm  = *reinterpret_cast<const sycl::half2 *>(dm_base + ib * sizeof(sycl::half2));
            } else {
                const block_q4_K * xb = reinterpret_cast<const block_q4_K *>(base) + ib;
                qs  = xb->qs;
                scp = xb->scales;
                dm  = xb->dm;
            }
            const float dall = dm[0];
            const float dmin = dm[1];
            dequantize_q4_K_math(
                [&](int j, float v0, float v1) {
                    const int kl = j - 64 * il;  // local k within the chunk, even
                    *reinterpret_cast<sycl::vec<sycl::half, 2> *>(&wt[(kl >> 1) * (2 * WG_M) + ml * 2]) =
                        sycl::vec<sycl::half, 2>{(sycl::half) v0, (sycl::half) v1};
                },
                qs, dall, dmin, scp, il, ir);
        }
        sycl::group_barrier(it.get_group());
        if (is_mad_sg) {
#pragma unroll
            for (int kt = 0; kt < KB / TK; ++kt) {
                sem::joint_matrix<sycl::sub_group, sycl::half, sem::use::b, TK, TN, sem::layout::ext_intel_packed> B;
                sem::joint_matrix_load(sg, B, wt_mp + (size_t) (kt * TK / 2) * (2 * WG_M) + sg_m0 * 2,
                                       (size_t) (2 * WG_M));
                for (int rn = 0; rn < RN; ++rn) {
                    sem::joint_matrix<sycl::sub_group, sycl::half, sem::use::a, TM, TK, sem::layout::row_major> A;
                    sem::joint_matrix_load(sg, A,
                        gptr(const_cast<sycl::half *>(act)) +
                            (size_t) (poff + rn * TM) * (size_t) ne00 + (size_t) kb * KB + kt * TK,
                        (size_t) ne00);
                    sem::joint_matrix_mad(sg, C[rn], A, B, C[rn]);
                }
            }
        }
        sycl::group_barrier(it.get_group());
    }

    if (is_mad_sg) {
        float * outp = part + (size_t) split * (size_t) n_pad_rows * (size_t) ne01;
        for (int rn = 0; rn < RN; ++rn) {
            sem::joint_matrix_store(sg, C[rn],
                gptr(outp) + (size_t) (poff + rn * TM) * (size_t) ne01 + m0 + sg_m0,
                (size_t) ne01, sem::layout::row_major);
        }
    }
}

template <bool REORDERED>
static void submit_tile(queue_ptr stream, const uint8_t * w_base_all, const sycl::half * act, float * part,
                        const int32_t * slots, int n_slots, int64_t ne00, int64_t ne01, size_t nb02,
                        int n_m_tiles, int split_k, int64_t n_pad_rows) {
    const size_t n_wg = (size_t) n_slots * n_m_tiles * split_k;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> wtile(sycl::range<1>((size_t) KB * WG_M), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(n_wg * WG_THREADS, WG_THREADS),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                tile_gemm_device<REORDERED>(w_base_all, act, part, slots, ne00, ne01, nb02,
                                            n_m_tiles, split_k, n_pad_rows,
                                            get_pointer(wtile),
                                            wtile.template get_multi_ptr<sycl::access::decorated::no>(), it);
            });
    });
}

}  // namespace lx_et

// launch-shape record for the verify report
struct lx_expert_tile_stats {
    int     n_slots    = 0;
    int64_t n_rows     = 0;
    int64_t n_pad_rows = 0;
    int     split_k    = 1;
    bool    reordered  = false;
};

// Host side: decide engagement for this dispatch and enqueue the staging +
// tile + fixup kernels covering every expert in the [1, MAX_N] band. Returns
// true iff enqueued (the caller must then skip exactly that band in the
// per-expert loop — unless verifying). Returning false leaves the dispatch
// fully on the loop.
static bool lx_expert_tile_gemm_launch(ggml_backend_sycl_context & ctx, queue_ptr stream,
                                       const ggml_tensor * src0, const char * src1_cont, char * dst_cont,
                                       const std::vector<int64_t> & expert_row_counts,
                                       const std::vector<int64_t> & expert_row_offsets,
                                       lx_expert_tile_stats * stats_out) {
    if (src0->type != GGML_TYPE_Q4_K) {
        return false;  // v2 scope: q4_K banks only; q6_K dispatches stay 100% oneMKL
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];  // K (in features)
    const int64_t ne01 = src0->ne[1];  // M (out features)
    const int64_t n_as = src0->ne[2];
    const size_t  nb02 = src0->nb[2];

    if (ne00 % QK_K != 0 || ne01 % lx_et::WG_M != 0) {
        return false;  // whole super-blocks / whole m-tiles only (Laguna: 2048x512, 512x2048 — both pass)
    }
    const int64_t nbpe = ne00 * ne01 / QK_K;  // super-blocks per expert
    if (nb02 != (size_t) nbpe * sizeof(block_q4_K)) {
        return false;  // padded/strided bank: the in-kernel offset math would not hold
    }

    // one-time device capability check: hardware 16x16x16 fp16->fp32
    // joint_matrix + sub-group 16 (true on B70/BMG; keeps the env knob safe on
    // anything else)
    static const int device_ok = [&]() -> int {
        try {
            const sycl::device dev = stream->get_device();
            const auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
            if (std::find(sgs.begin(), sgs.end(), (size_t) lx_et::SG_SIZE) == sgs.end()) {
                return 0;
            }
            namespace sex = sycl::ext::oneapi::experimental;
            for (const auto & c : dev.get_info<sex::info::device::matrix_combinations>()) {
                if ((int) c.msize == lx_et::TM && (int) c.nsize == lx_et::TN && (int) c.ksize == lx_et::TK &&
                    c.atype == lx_et::sem::matrix_type::fp16 && c.btype == lx_et::sem::matrix_type::fp16 &&
                    c.ctype == lx_et::sem::matrix_type::fp32 && c.dtype == lx_et::sem::matrix_type::fp32) {
                    return 1;
                }
            }
        } catch (const std::exception &) {
        }
        return 0;
    }();
    if (!device_ok) {
        static std::atomic<int> warned(0);
        if (warned.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-expert-tile] DISABLED: device lacks %dx%dx%d fp16->fp32 joint_matrix or SG%d\n",
                    lx_et::TM, lx_et::TN, lx_et::TK, lx_et::SG_SIZE);
        }
        return false;
    }

    GGML_ASSERT(expert_row_offsets[n_as] <= INT32_MAX);

    // band census
    int     n_slots    = 0;
    int64_t n_rows     = 0;
    int64_t n_pad_rows = 0;
    for (int64_t e = 0; e < n_as; e++) {
        const int64_t nr = expert_row_counts[e];
        if (nr >= 1 && nr <= lx_et::MAX_N) {
            n_slots++;
            n_rows += nr;
            n_pad_rows += ((nr + lx_et::TM - 1) / lx_et::TM) * lx_et::TM;
        }
    }
    if (n_slots == 0) {
        return false;
    }

    // split-K: winning microbench shape keeps >=2 KB-chunks per split and
    // split_k <= 16 (K=2048 -> 16, the measured config; K=512 -> 4); halve
    // further only if the fp32 partials would blow the scratch budget.
    int           split_k      = 1;
    const int64_t total_chunks = ne00 / lx_et::KB;
    while (split_k * 2 <= lx_et::SPLIT_K_MAX && total_chunks % (int64_t) (split_k * 2) == 0 &&
           (int64_t) (split_k * 2) < total_chunks) {
        split_k *= 2;
    }
    while (split_k > 1 &&
           (size_t) split_k * (size_t) n_pad_rows * (size_t) ne01 * sizeof(float) > lx_et::PART_BUDGET) {
        split_k /= 2;
    }

    // Metadata (one int32 H2D): [slots 3*n_slots: expert, nrows, pad_off]
    // [rows 2*n_rows: dst_row, pad_row] [pad2src n_pad_rows: src row or -1].
    // Bytes are a pure function of the memoized counts/offsets (identical for
    // the layer's gate/up/down ops); host staging persists in the context (see
    // lx_expert_tile_meta_host in common.hpp for the drain contract). Device
    // slots are pool allocs released at return: the in-order queue serializes
    // any later reuse behind the kernels enqueued here.
    std::vector<int32_t> & mh = ctx.lx_expert_tile_meta_host;
    mh.resize((size_t) (3 * n_slots) + (size_t) (2 * n_rows) + (size_t) n_pad_rows);
    int32_t * mh_slots = mh.data();
    int32_t * mh_rows  = mh_slots + 3 * n_slots;
    int32_t * mh_p2s   = mh_rows + 2 * n_rows;
    {
        int32_t s = 0;
        int64_t r = 0, p = 0;
        for (int64_t e = 0; e < n_as; e++) {
            const int64_t nr = expert_row_counts[e];
            if (nr < 1 || nr > lx_et::MAX_N) {
                continue;
            }
            const int64_t roff = expert_row_offsets[e];
            const int64_t npad = ((nr + lx_et::TM - 1) / lx_et::TM) * lx_et::TM;
            mh_slots[3 * s + 0] = (int32_t) e;
            mh_slots[3 * s + 1] = (int32_t) nr;
            mh_slots[3 * s + 2] = (int32_t) p;
            for (int64_t n = 0; n < nr; n++) {
                mh_rows[2 * (r + n) + 0] = (int32_t) (roff + n);
                mh_rows[2 * (r + n) + 1] = (int32_t) (p + n);
            }
            for (int64_t n = 0; n < npad; n++) {
                mh_p2s[p + n] = n < nr ? (int32_t) (roff + n) : -1;
            }
            s++;
            r += nr;
            p += npad;
        }
    }
    ggml_sycl_pool_alloc<int32_t> meta_dev(ctx.pool(), mh.size());
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(meta_dev.get(), mh.data(), mh.size() * sizeof(int32_t))));

    ggml_sycl_pool_alloc<sycl::half> act_dev(ctx.pool(), (size_t) n_pad_rows * (size_t) ne00);
    ggml_sycl_pool_alloc<float>      part_dev(ctx.pool(),
                                              (size_t) split_k * (size_t) n_pad_rows * (size_t) ne01);

    // Serving-state banks are reordered per-expert SoA (reorder_qw with
    // ne[2] > 1 always takes the _moe writers); before warmup / under OPT=0
    // they are linear block arrays. Flag read on host BEFORE the loop can
    // lazily reorder; in-order stream keeps bytes consistent with the flag.
    const ggml_tensor_extra_gpu * extra0    = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool                    reordered = extra0 != nullptr && extra0->optimized_feature.reorder;

    // 1) stage fp16 activations, padded per slot (pad rows zero-filled)
    {
        const int64_t     total  = n_pad_rows * ne00;
        const int64_t     nblk   = (total + 255) / 256;
        const int32_t *   p2s    = meta_dev.get() + 3 * n_slots + 2 * n_rows;
        const float *     srcf   = (const float *) src1_cont;
        sycl::half *      actp   = act_dev.get();
        const int64_t     ne00c  = ne00;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblk * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> item) {
                const int64_t i = (int64_t) item.get_global_linear_id();
                if (i < total) {
                    const int64_t pr   = i / ne00c;
                    const int64_t k    = i % ne00c;
                    const int32_t srow = p2s[pr];
                    actp[i] = srow >= 0 ? (sycl::half) srcf[(size_t) srow * (size_t) ne00c + (size_t) k]
                                        : sycl::half(0.0f);
                }
            });
    }

    // 2) ONE fused XMX tile launch over all in-band experts
    const int n_m_tiles = (int) (ne01 / lx_et::WG_M);
    if (reordered) {
        lx_et::submit_tile<true>(stream, (const uint8_t *) src0->data, act_dev.get(), part_dev.get(),
                                 meta_dev.get(), n_slots, ne00, ne01, nb02, n_m_tiles, split_k, n_pad_rows);
    } else {
        lx_et::submit_tile<false>(stream, (const uint8_t *) src0->data, act_dev.get(), part_dev.get(),
                                  meta_dev.get(), n_slots, ne00, ne01, nb02, n_m_tiles, split_k, n_pad_rows);
    }

    // 3) fixup: sum the split-K partials of each REAL band row into dst_contiguous
    {
        const int64_t   total = n_rows * ne01;
        const int64_t   nblk  = (total + 255) / 256;
        const int32_t * rows  = meta_dev.get() + 3 * n_slots;
        const float *   partp = part_dev.get();
        float *         dstf  = (float *) dst_cont;
        const int64_t   ne01c = ne01;
        const int64_t   npr   = n_pad_rows;
        const int       sk    = split_k;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblk * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> item) {
                const int64_t i = (int64_t) item.get_global_linear_id();
                if (i < total) {
                    const int64_t r = i / ne01c;
                    const int64_t m = i % ne01c;
                    const int32_t dst_row = rows[2 * r + 0];
                    const int32_t pad_row = rows[2 * r + 1];
                    float s = 0.0f;
                    for (int sp = 0; sp < sk; ++sp) {
                        s += partp[((size_t) sp * (size_t) npr + (size_t) pad_row) * (size_t) ne01c + (size_t) m];
                    }
                    dstf[(size_t) dst_row * (size_t) ne01c + (size_t) m] = s;
                }
            });
    }

    static std::atomic<int> announced(0);
    if (announced.fetch_add(1) == 0) {
        fprintf(stderr,
                "[lx-control-expert-tile] engaged(xmx): n_as=%d slots=%d rows=%d pad_rows=%d M=%d K=%d "
                "layout=%s split_k=%d grid=%dx%d cfg=t%dx%dx%d-wgm%d-sgc%d-kb%d-sg%d\n",
                (int) n_as, n_slots, (int) n_rows, (int) n_pad_rows, (int) ne01, (int) ne00,
                reordered ? "reorder-soa" : "linear", split_k,
                (int) ((size_t) n_slots * n_m_tiles * split_k), lx_et::WG_THREADS,
                lx_et::TM, lx_et::TN, lx_et::TK, lx_et::WG_M, lx_et::SG_COLS, lx_et::KB, lx_et::NSG_TOT);
    }

    if (stats_out != nullptr) {
        stats_out->n_slots    = n_slots;
        stats_out->n_rows     = n_rows;
        stats_out->n_pad_rows = n_pad_rows;
        stats_out->split_k    = split_k;
        stats_out->reordered  = reordered;
    }
    return true;
}

// Verify mode: compare the tile kernel's dst_contiguous band rows (snapshotted
// before the oneMKL loop overwrote them) against the loop's own output.
// Prints max-abs-err and max-rel-err (rel denominator max(1,|ref|), matching
// the microbench) per dispatch, to stderr AND to a side log
// (GGML_SYCL_LX_EXPERT_TILE_VERIFY_LOG, default /tmp/lx-expert-tile-verify.log)
// because llama-server swallows stderr into its own log file.
static void lx_expert_tile_verify_report(ggml_backend_sycl_context & ctx, queue_ptr stream,
                                         const ggml_tensor * src0,
                                         const float * mkl_out, const float * tile_out,
                                         const std::vector<int64_t> & expert_row_counts,
                                         const std::vector<int64_t> & expert_row_offsets,
                                         int64_t ne01, const lx_expert_tile_stats & st) {
    const int64_t n_as = src0->ne[2];
    std::vector<int32_t> rows;  // dst_contiguous row index of every band row
    rows.reserve((size_t) st.n_rows);
    for (int64_t e = 0; e < n_as; e++) {
        const int64_t nr = expert_row_counts[e];
        if (nr < 1 || nr > lx_et::MAX_N) {
            continue;
        }
        for (int64_t n = 0; n < nr; n++) {
            rows.push_back((int32_t) (expert_row_offsets[e] + n));
        }
    }
    if (rows.empty()) {
        return;
    }
    const int64_t R = (int64_t) rows.size();

    ggml_sycl_pool_alloc<int32_t> rows_dev(ctx.pool(), (size_t) R);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(rows_dev.get(), rows.data(), (size_t) R * sizeof(int32_t))));
    ggml_sycl_pool_alloc<float> err_dev(ctx.pool(), 2);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(err_dev.get(), 0, 2 * sizeof(float))));

    {
        const int32_t * rd    = rows_dev.get();
        float *         ep    = err_dev.get();
        const int64_t   ne01c = ne01;
        const int64_t   total = R * ne01;
        stream->parallel_for(
            sycl::range<1>((size_t) total),
            sycl::reduction(ep + 0, sycl::maximum<float>()),
            sycl::reduction(ep + 1, sycl::maximum<float>()),
            [=](sycl::id<1> idx, auto & amax, auto & rmax) {
                const int64_t i   = (int64_t) idx[0];
                const int64_t r   = i / ne01c;
                const int64_t m   = i % ne01c;
                const size_t  off = (size_t) rd[r] * (size_t) ne01c + (size_t) m;
                const float   ref = mkl_out[off];
                const float   got = tile_out[off];
                const float   ad  = sycl::fabs(got - ref);
                amax.combine(ad);
                rmax.combine(ad / sycl::fmax(1.0f, sycl::fabs(ref)));
            });
    }

    float err[2] = { 0.0f, 0.0f };
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(err, err_dev.get(), sizeof(err))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));  // also drains the rows H2D before the vector dies

    static std::atomic<int> vcount(0);
    const int d = vcount.fetch_add(1);
    if (d < 400) {
        char line[512];
        snprintf(line, sizeof(line),
                 "[lx-expert-tile-verify] dispatch=%d M=%d K=%d n_as=%d band_experts=%d band_rows=%d "
                 "split_k=%d layout=%s max_abs_err=%.3e max_rel_err=%.3e\n",
                 d, (int) ne01, (int) src0->ne[0], (int) n_as, st.n_slots, (int) st.n_rows,
                 st.split_k, st.reordered ? "reorder-soa" : "linear", (double) err[0], (double) err[1]);
        fputs(line, stderr);
        static FILE * side = []() -> FILE * {
            const char * p = getenv("GGML_SYCL_LX_EXPERT_TILE_VERIFY_LOG");
            FILE * f = fopen(p != nullptr ? p : "/tmp/lx-expert-tile-verify.log", "a");
            if (f != nullptr) {
                fprintf(f, "=== lx-expert-tile verify: new process ===\n");
            }
            return f;
        }();
        if (side != nullptr) {
            fputs(line, side);
            fflush(side);
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

    // Device-routed fused path (decode always; multi-token if ENABLE_MMID_FUSED_BATCH=1).
    if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
        return;
    }

    // [lx-ids-once] The MUL_MAT_ID ops of a MoE layer (gate, up, down) all consume
    // the same selected_experts (ids) tensor. Memoize the D2H ids copy + full
    // stream->wait + counting sort on the first consumer and reuse the identical
    // host bytes for the layer's remaining ops (same ids bytes -> same sort output,
    // bit-identical). Skipping the wait is safe: later ops' async H2D mapping
    // copies read a never-mutated stable vector, and the in-order queue serializes
    // pool-slot reuse. Generation guards recycled tensor pointers across graph
    // executions (each bench rep re-uses the same tensor objects with NEW ids
    // content). Decode (ne12==1) keeps the legacy D2H path.
    const int64_t n_routed_rows = ids->ne[1] * n_ids;
    const bool mmid_memo_hit =
        ne12 > 1 &&
        ctx.mmid_ids_memo_tensor == ids &&
        ctx.mmid_ids_memo_gen_store == ctx.mmid_ids_memo_gen;

    std::vector<char> ids_host;
    std::vector<int64_t> expert_row_counts;
    std::vector<int64_t> expert_row_offsets;
    std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

    if (mmid_memo_hit) {
        // [lx-ids-once] fire counter: proves the D2H+wait+sort skip executes on
        // the scored path (diagnostic only; atexit line LX_IDS_ONCE).
        static std::atomic<long> lx_ids_once_hits(0);
        static bool lx_ids_once_reg = []() {
            std::atexit([](){
                if (lx_ids_once_hits.load() > 0) {
                    fprintf(stderr, "LX_IDS_ONCE hits=%ld\n", lx_ids_once_hits.load());
                }
            });
            return true;
        }();
        lx_ids_once_hits++;
        expert_row_counts  = ctx.mmid_ids_memo_counts;
        expert_row_offsets = ctx.mmid_ids_memo_offsets;
        routed_row_src     = ctx.mmid_ids_memo_mapping;
    } else {
        ids_host.resize(ggml_nbytes(ids));
        const char * ids_dev = (const char *) ids->data;
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
        // also ensures ctx.mmid_row_mapping_host is drained before we use it again
        {
            lx_chrono_wait_guard _w;
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }
    }

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
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // [lx-ids-once] counting sort + memo store happen on the first consumer
        // only (see prelude above); later consumers reuse the stored vectors.
        if (!mmid_memo_hit) {
            mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                    expert_row_counts, expert_row_offsets, routed_row_src);
            ctx.mmid_ids_memo_tensor    = ids;
            ctx.mmid_ids_memo_gen_store = ctx.mmid_ids_memo_gen;
            ctx.mmid_ids_memo_counts    = expert_row_counts;
            ctx.mmid_ids_memo_offsets   = expert_row_offsets;
            ctx.mmid_ids_memo_mapping   = routed_row_src;
        }

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

        // [lx-diag-expcap] measurement-only: GGML_SYCL_LX_DIAG_EXPERT_CAP=N stops the
        // per-expert GEMM loop after N non-empty experts (dst stale for the rest).
        // Behavioral no-op when unset. Wall vs cap = per-expert host submission cost.
        static const int lx_diag_expcap = []() -> int {
            const char * e = getenv("GGML_SYCL_LX_DIAG_EXPERT_CAP");
            return e ? atoi(e) : 0;
        }();
        // [lx-gemm-batch] ONE batched oneMKL fp16 GEMM per MUL_MAT_ID dispatch
        // (prefill, ne12>1, quantized src0) replaces ~256 per-expert
        // submit+convert+gemm+convert-back sequences. Env-gated default OFF
        // (GGML_SYCL_LX_GEMM_BATCH=1). Quality arbitrated by the KLD gate.
        static const bool lx_gemm_batch = []() {
            const char * e = getenv("GGML_SYCL_LX_GEMM_BATCH");
            return e != nullptr && std::atoi(e) != 0;
        }();
        const bool lx_gemm_batch_ok =
            lx_gemm_batch && ggml_is_quantized(src0->type) &&
            src0->type != GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(src0);

        if (lx_gemm_batch_ok) {
            // 1) collect non-empty experts (expert idx, row count, row offset)
            std::vector<int64_t> b_exps, b_erows, b_eoffs;
            b_exps.reserve((size_t) n_as);
            for (int64_t i02 = 0; i02 < n_as; i02++) {
                const int64_t nr = expert_row_counts[i02];
                if (nr > 0) {
                    b_exps.push_back(i02);
                    b_erows.push_back(nr);
                    b_eoffs.push_back(expert_row_offsets[i02]);
                }
            }
            const int64_t n_b = (int64_t) b_exps.size();
            const int64_t M   = src0->ne[1];          // per-expert gemm m (rows of src0)
            const int64_t K   = src0->ne[0];          // per-expert gemm k (== ne10)
            // 2) full quantized src0 weight tensor -> fp16, ONE launch (per-expert
            //    slices stay contiguous: expert e at e*M*K)
            ggml_sycl_pool_alloc<sycl::half> b_src0_f16(ctx.pool(), (size_t) ggml_nelements(src0));
            {
                const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
                GGML_ASSERT(to_fp16_sycl != nullptr);
                to_fp16_sycl((const char *) src0->data, b_src0_f16.get(), (size_t) ggml_nelements(src0), stream);
            }
              // 3) ONE padded-strided dpct::gemm_batch per dispatch: all 256
              //    experts share m=M, k=K and a padded n=n_max (garbage columns
              //    beyond each expert's real row count are never read downstream).
              //    Strided batch kernel only; the grouped oneMKL form aborts with
              //    UR_RESULT_ERROR_OUT_OF_RESOURCES on this device (tested).
              //    Default OFF; quality arbitrated by the KLD gate.
              //    [lx-gb-fix-20260810] the old b_src1_f16 was sized n_routed_rows*K
              //    but both the padding kernel and gemm_batch index it at e*n_max*K
              //    (up to n_as*n_max*K elements) - OOB whenever any expert holds
              //    fewer rows than n_max, i.e. every dispatch. That OOB is exactly
              //    what the 2026-08-10 /tmp/st-* "strided gemm_batch hangs" probe
              //    ran, so that hang does NOT falsify the fixed shape. Fixed here:
              //    b_src1_f16/b_dst_f16 sized n_as*n_max*{K,M} inside the guard; the
              //    standalone src1 conversion (fully overwritten by the padded
              //    conversion below) and the step-5 dst to_fp32 that overwrote the
              //    correct unpack with a linear slice of the padded buffer are
              //    deleted.
              const int64_t n_max = n_b > 0 ? *std::max_element(b_erows.begin(), b_erows.end()) : 0;
              if (n_b > 0 && n_max > 0) {
                  ggml_sycl_pool_alloc<sycl::half> b_src1_f16(ctx.pool(), (size_t) n_as * (size_t) n_max * (size_t) K);
                  ggml_sycl_pool_alloc<sycl::half> b_dst_f16(ctx.pool(), (size_t) n_as * (size_t) n_max * (size_t) M);
                  // padded src1: routed row r goes to pad_pos[r] = expert*r_off + intra
                  ggml_sycl_pool_alloc<int> b_padpos(ctx.pool(), (size_t) n_routed_rows);
                  std::vector<int> pad_pos((size_t) n_routed_rows);
                  for (int64_t e = 0; e < n_as; e++) {
                      const int64_t nr = expert_row_counts[e];
                      for (int64_t j = 0; j < nr; j++) {
                          pad_pos[(size_t) (expert_row_offsets[e] + j)] = (int) (e * n_max + j);
                      }
                  }
                  SYCL_CHECK(CHECK_TRY_ERROR(
                      stream->memcpy(b_padpos.get(), pad_pos.data(), sizeof(int) * (size_t) n_routed_rows)));
                  {
                      // src1_contiguous (F32 packed) -> src1_f16 padded, one launch
                      const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
                      const int64_t total = n_routed_rows * K;
                      const int64_t nblocks = (total + 255) / 256;
                      const int * pad_d = b_padpos.get();
                      sycl::half * dst1 = b_src1_f16.get();
                      const float * src1p = (const float *) src1_contiguous.get();
                      stream->submit([&](sycl::handler & cgh) {
                          cgh.parallel_for(
                              sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * 256), sycl::range<1>(256)),
                              [=](sycl::nd_item<1> item) {
                                  const int64_t id = (int64_t) item.get_global_linear_id();
                                  if (id < total) {
                                      const int64_t row = id / K;
                                      const int64_t col = id % K;
                                      dst1[(size_t) pad_d[row] * (size_t) K + (size_t) col] =
                                          (sycl::half) src1p[(size_t) row * (size_t) K + (size_t) col];
                                  }
                              });
                      });
                  }
                  // ONE strided gemm_batch
                  {
                      const sycl::half alpha_f16 = 1.0f;
                      const sycl::half beta_f16  = 0.0f;
                      const int64_t n_max_i = n_max;
                      const dpct::library_data_t lx_ct = dpct::library_data_t::real_float;
                      SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                          *stream, oneapi::mkl::transpose::trans,
                          oneapi::mkl::transpose::nontrans,
                          (int) M, (int) n_max_i, (int) K, &alpha_f16,
                          b_src0_f16.get(), dpct::library_data_t::real_half, (int) K, (long long) M * (long long) K,
                          b_src1_f16.get(), dpct::library_data_t::real_half, (int) K, (long long) n_max_i * (long long) K,
                          &beta_f16, b_dst_f16.get(), dpct::library_data_t::real_half, (int) M,
                          (long long) n_max_i * (long long) M, (int) n_as, lx_ct)));
                  }
                  // unpack padded fp16 dst -> dst_contiguous F32, one launch
                  {
                      const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
                      const int64_t total = n_routed_rows * M;
                      const int64_t nblocks = (total + 255) / 256;
                      const int * pad_d = b_padpos.get();
                      const sycl::half * csrc = b_dst_f16.get();
                      float * dstd = (float *) dst_contiguous.get();
                      stream->submit([&](sycl::handler & cgh) {
                          cgh.parallel_for(
                              sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * 256), sycl::range<1>(256)),
                              [=](sycl::nd_item<1> item) {
                                  const int64_t id = (int64_t) item.get_global_linear_id();
                                  if (id < total) {
                                      const int64_t row = id / M;
                                      const int64_t col = id % M;
                                      dstd[(size_t) row * (size_t) M + (size_t) col] =
                                          (float) csrc[(size_t) pad_d[row] * (size_t) M + (size_t) col];
                                  }
                              });
                      });
                  }
              }
        } else {
        // [lx-expert-tile] one fused XMX dequant-GEMM launch computes every
        // q4_K expert whose routed row count is in [1, lx_et::MAX_N] BEFORE the
        // per-expert oneMKL loop; the loop then skips exactly that band, keyed
        // off the same expert_row_counts the kernel's metadata was built from.
        // In-order stream ordering: launched after k_copy_src1_to_contiguous,
        // before the loop's GEMMs (which may lazily reorder the bank —
        // serialized behind this kernel) and before the dst scatter. Disabled
        // under the DIAG_EXPERT_CAP measurement hook, whose "first N non-empty
        // experts" semantics the tile band would pollute.
        bool lx_et_engaged = false;
        lx_expert_tile_stats lx_et_stats;
        if (lx_expert_tile_gemm_enabled() && lx_diag_expcap == 0 &&
            src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            dst->ne[0] == src0->ne[1] &&
            nb11 == sizeof(float) * (size_t) ne10 && nb1 == sizeof(float) * (size_t) ne0) {
            lx_et_engaged = lx_expert_tile_gemm_launch(ctx, stream, src0,
                    src1_contiguous.get(), dst_contiguous.get(),
                    expert_row_counts, expert_row_offsets, &lx_et_stats);
        }
        // Verify mode: snapshot the tile+fixup output before the loop (which
        // then recomputes the band via oneMKL and overwrites dst_contiguous);
        // the report kernel after the loop compares the two.
        const bool lx_et_verify = lx_et_engaged && lx_expert_tile_verify_enabled();
        ggml_sycl_pool_alloc<float> lx_et_snap(ctx.pool());
        if (lx_et_verify) {
            lx_et_snap.alloc((size_t) n_routed_rows * (size_t) ne0);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(lx_et_snap.get(), dst_contiguous.get(),
                    sizeof(float) * (size_t) n_routed_rows * (size_t) ne0)));
        }
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }
            // [lx-expert-tile] already computed by the fused tile kernel above;
            // must not be double-computed (same counts array, same band). In
            // verify mode the loop DOES recompute it so dst can be compared.
            if (lx_et_engaged && !lx_et_verify && num_src1_rows <= lx_et::MAX_N) {
                continue;
            }
            if (lx_diag_expcap > 0 && i02 >= lx_diag_expcap) {
                break;
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

            // [lx-mmvq-prefill] GGML_SYCL_LX_MMVQ_PREFILL=1 (env, default OFF):
            // per-expert Q6_K slices at ne12>1 go through the decode-proven
            // q8_1 MMVQ kernel in <=8-column chunks instead of the fp16 oneMKL
            // GEMM. The fp16 path round-trips each expert's weights through
            // DRAM (dequant fp16 write + gemm fp16 read ~= 2x weight traffic per
            // call); MMVQ reads the quantized weights once. IQ4_NL stays on
            // oneMKL (no multi-col switch; ncols=1 MMVQ re-reads the weight
            // slice per column, launch-bound). KLD arbitrates numerics.
            static const bool lx_mmvq_prefill = []() {
                const char * e = getenv("GGML_SYCL_LX_MMVQ_PREFILL");
                return e != nullptr && std::atoi(e) != 0;
            }();
            if (lx_mmvq_prefill && ne12 > 1 && src0->type == GGML_TYPE_Q6_K) {
                for (int64_t col0 = 0; col0 < num_src1_rows; col0 += MMVQ_MAX_BATCH_SIZE) {
                    const int64_t nc = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, num_src1_rows - col0);
                    ggml_tensor src1_c = src1_row;
                    ggml_tensor dst_c  = dst_row;
                    src1_c.ne[1] = nc;
                    dst_c.ne[1]  = nc;
                    src1_c.data  = (char *) src1_row.data + col0 * src1_row.nb[1];
                    dst_c.data   = (char *) dst_row.data  + col0 * dst_row.nb[1];
                    src1_c.nb[2] = src1_c.nb[1] * (size_t) nc;
                    src1_c.nb[3] = src1_c.nb[2];
                    dst_c.nb[2]  = dst_c.nb[1] * (size_t) nc;
                    dst_c.nb[3]  = dst_c.nb[2];
                    ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, &src0_row, &src1_c, &dst_c, ggml_sycl_op_mul_mat_vec_q);
                }
            } else {
                ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }

        if (lx_et_verify) {
            lx_expert_tile_verify_report(ctx, stream, src0,
                                         (const float *) dst_contiguous.get(), lx_et_snap.get(),
                                         expert_row_counts, expert_row_offsets,
                                         (int64_t) ne0, lx_et_stats);
        }
        }  // [lx-expert-tile] end of non-gemm-batch expert-loop else block

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

// ---------------------------------------------------------------------------
// MoE down weighted reduce (Laguna: MUL_MAT_ID → MUL(w) → VIEW×k → ADD×(k-1))
// Surgical port of treebeard-moe-weighted-reduce / down-reduce idea onto control.
// Default ON (measured 2026-07-30: +7.4% formal vs pin with dual+hybrid+dense).
// Kill: GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1  or  ENABLE_MOE_DOWN_WEIGHTED=0
// ---------------------------------------------------------------------------
static void ggml_sycl_moe_weighted_reduce(
    ggml_backend_sycl_context & ctx,
    const float *               expert_values,
    const float *               weight_data,
    float *                     dst_data,
    int64_t                     n_embd,
    int64_t                     n_experts_used,
    int64_t                     n_tokens) {
    const int64_t total      = n_embd * n_tokens;
    constexpr int block_size = 256;
    const int64_t global     = ((total + block_size - 1) / block_size) * block_size;
    const queue_ptr stream   = ctx.stream();
    if (n_tokens == 1 && n_experts_used == 8) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) global), sycl::range<1>(block_size)),
                [=](sycl::nd_item<1> item) {
                    const int64_t row = (int64_t) item.get_global_linear_id();
                    if (row >= n_embd) {
                        return;
                    }
                    float sum = 0.0f;
                    volatile float w0 = expert_values[0 * n_embd + row] * weight_data[0]; sum += w0;
                    volatile float w1 = expert_values[1 * n_embd + row] * weight_data[1]; sum += w1;
                    volatile float w2 = expert_values[2 * n_embd + row] * weight_data[2]; sum += w2;
                    volatile float w3 = expert_values[3 * n_embd + row] * weight_data[3]; sum += w3;
                    volatile float w4 = expert_values[4 * n_embd + row] * weight_data[4]; sum += w4;
                    volatile float w5 = expert_values[5 * n_embd + row] * weight_data[5]; sum += w5;
                    volatile float w6 = expert_values[6 * n_embd + row] * weight_data[6]; sum += w6;
                    volatile float w7 = expert_values[7 * n_embd + row] * weight_data[7]; sum += w7;
                    dst_data[row] = sum;
                });
        });
        return;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) global), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item) {
                const int64_t out = (int64_t) item.get_global_linear_id();
                if (out >= total) {
                    return;
                }
                const int64_t token = out / n_embd;
                const int64_t row   = out - token * n_embd;
                float sum = 0.0f;
                // Laguna always routes k=8 experts. Spell out the same ordered
                // MUL-then-ADD chain so the compiler can remove loop/control and
                // dynamic-stride overhead without changing floating-point order.
                if (n_experts_used == 8) {
                    const float * values = expert_values + (token * 8) * n_embd + row;
                    const float * weights = weight_data + token * 8;
                    volatile float w0 = values[0 * n_embd] * weights[0]; sum += w0;
                    volatile float w1 = values[1 * n_embd] * weights[1]; sum += w1;
                    volatile float w2 = values[2 * n_embd] * weights[2]; sum += w2;
                    volatile float w3 = values[3 * n_embd] * weights[3]; sum += w3;
                    volatile float w4 = values[4 * n_embd] * weights[4]; sum += w4;
                    volatile float w5 = values[5 * n_embd] * weights[5]; sum += w5;
                    volatile float w6 = values[6 * n_embd] * weights[6]; sum += w6;
                    volatile float w7 = values[7 * n_embd] * weights[7]; sum += w7;
                } else {
                    // Expert order 0..k-1 matches the sequential ADD chain.
                    for (int64_t expert = 0; expert < n_experts_used; ++expert) {
                        const float value = expert_values[(token * n_experts_used + expert) * n_embd + row];
                        const float weight = weight_data[token * n_experts_used + expert];
                        // Preserve MUL-then-ADD rounding (no FMA contraction).
                        volatile float weighted = value * weight;
                        sum += weighted;
                    }
                }
                dst_data[out] = sum;
            });
    });
}


// ---- dual+down multi-token expert-loop (from patch 0028 + mul_mat reorder fix) ----
// Kill: GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1
static bool ggml_sycl_mmid_device_sort_enabled() {
    static const bool disabled = []() {
        const char * env = getenv("GGML_SYCL_DISABLE_MMID_DEVICE_SORT");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return !disabled;
}

static void mmid_device_count_experts(
        const int32_t * __restrict__ ids, int * __restrict__ counts,
        const int n_ids, const int n_tokens, const int ids_row_stride,
        const int n_as, const sycl::nd_item<1> & item) {
    const int tid = (int) item.get_global_id(0);
    const int total = n_ids * n_tokens;
    if (tid >= total) {
        return;
    }
    const int slot  = tid % n_ids;
    const int token = tid / n_ids;
    const int32_t expert = ids[(size_t) token * (size_t) ids_row_stride + (size_t) slot];
    if (expert >= 0 && expert < n_as) {
        auto ref = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(counts[expert]);
        ref.fetch_add(1);
    }
}

static void mmid_device_fill_mapping(
        const int32_t * __restrict__ ids, int * __restrict__ next,
        mmid_row_mapping * __restrict__ mapping,
        const int n_ids, const int n_tokens, const int ids_row_stride,
        const int n_as, const sycl::nd_item<1> & item) {
    const int tid = (int) item.get_global_id(0);
    const int total = n_ids * n_tokens;
    if (tid >= total) {
        return;
    }
    const int slot  = tid % n_ids;
    const int token = tid / n_ids;
    const int32_t expert = ids[(size_t) token * (size_t) ids_row_stride + (size_t) slot];
    if (expert < 0 || expert >= n_as) {
        return;
    }
    auto ref = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(next[expert]);
    const int pos = ref.fetch_add(1);
    mapping[pos] = { slot, token };
}

// Exclusive prefix-sum of expert histogram → next[] row offsets (device, no H2D).
// n_as is small (Laguna=256); single-thread sequential scan is free vs a host round-trip.
static void mmid_device_exclusive_scan(
        const int * __restrict__ counts, int * __restrict__ next,
        const int n_as, const sycl::nd_item<1> & item) {
    if (item.get_global_id(0) != 0) {
        return;
    }
    int sum = 0;
    for (int i = 0; i < n_as; ++i) {
        next[i] = sum;
        sum += counts[i];
    }
}


static void moe_dual_swiglu_contiguous(
    const float * __restrict__ gate, const float * __restrict__ up,
    float * __restrict__ dst, const int64_t n,
    const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t) item.get_global_linear_id();
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    // Match dual-MMVQ / stock: silu(g)*up = g/(1+exp(-g))*up
    dst[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
}


static bool ggml_sycl_mul_mat_id_dual_down_multitoken_expert_loop(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu,
    ggml_tensor * down_mmid, const ggml_tensor * weights, ggml_tensor * dst_final) {
    const ggml_tensor * gate_w = gate->src[0];
    const ggml_tensor * up_w   = up->src[0];
    const ggml_tensor * down_w = down_mmid->src[0];
    const ggml_tensor * src1   = gate->src[1];
    const ggml_tensor * ids    = gate->src[2];

    if (!gate_w || !up_w || !down_w || !src1 || !ids) {
        return false;
    }
    if (gate_w->type != up_w->type || gate_w->type != down_w->type ||
        (gate_w->type != GGML_TYPE_Q4_K && gate_w->type != GGML_TYPE_Q5_K &&
         gate_w->type != GGML_TYPE_Q6_K)) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];       // embd
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];       // tokens
    const int64_t n_ids = ids->ne[0];
    const int64_t n_as  = gate_w->ne[2];
    const int64_t n_ff  = gate_w->ne[1];
    const int64_t n_embd = down_mmid->ne[0];
    const int64_t n_experts_used = down_mmid->ne[1];
    const int64_t n_tokens = down_mmid->ne[2];
    const int64_t n_routed_rows = ne12 * n_ids;

    if (ne12 != n_tokens || n_ids != n_experts_used || ne12 < 2 || ne12 > 2048) {
        return false;
    }
    if (ids->ne[1] != ne12 || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (ne11 != 1 && ne11 != n_ids) {
        return false;
    }
    if (!ggml_is_contiguous(src1) || ne10 != gate_w->ne[0] || ne10 % QK8_1 != 0) {
        return false;
    }
    if (down_w->ne[0] != n_ff || down_w->ne[1] != n_embd || down_w->ne[2] != n_as) {
        return false;
    }
    if (up_w->ne[1] != n_ff || up_w->ne[2] != n_as) {
        return false;
    }

    const queue_ptr stream = ctx.stream();
    char * src1_original = (char *) src1->data;
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb_act = sizeof(float) * (size_t) ne10;
    const size_t nb_ff  = sizeof(float) * (size_t) n_ff;
    const size_t nb_em  = sizeof(float) * (size_t) n_embd;
    const size_t nb_wg  = gate_w->nb[2];
    const size_t nb_wu  = up_w->nb[2];
    const size_t nb_wd  = down_w->nb[2];
    GGML_ASSERT(nb11 == nb_act);

    // [lx-gate-up-concat] When enabled, pack [gate; up] weight slices into one
    // contiguous 2*n_ff buffer per expert and issue a single GEMM producing
    // 2*n_ff output rows, then run SwiGLU on the combined output. This halves
    // the number of host GEMM submissions in the prefill expert loop (2 GEMMs
    // → 1 per expert). The packed math is bit-exact: [W_gate; W_up]^T @ x =
    // [gate(x); up(x)]. Default OFF; quality arbitrated by the KLD gate.
    static const bool lx_gate_up_concat = []() {
        const char * e = getenv("GGML_SYCL_LX_GATE_UP_CONCAT");
        return e != nullptr && std::atoi(e) != 0;
    }();
    // For the concat path: packed weights [2*n_ff, n_embd] per expert, packed
    // dst [2*n_ff, N] per expert-batch.
    ggml_sycl_pool_alloc<char> packed_w_scratch;
    ggml_sycl_pool_alloc<char> packed_dst_scratch;
    const size_t nb_2ff = sizeof(float) * (size_t) n_ff * 2;

    ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), n_routed_rows * nb_act);
    ggml_sycl_pool_alloc<char> mid_contiguous(ctx.pool(), n_routed_rows * nb_ff);
    ggml_sycl_pool_alloc<char> up_contiguous(ctx.pool(), n_routed_rows * nb_ff);
    ggml_sycl_pool_alloc<char> down_contiguous(ctx.pool(), n_routed_rows * nb_em);
    ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);

    // Experts scatter target: real down_mmid or scratch if aliases final dst.
    const bool mmid_dst_overlap =
        (down_mmid->data == dst_final->data) ||
        ((uintptr_t) down_mmid->data < (uintptr_t) dst_final->data + ggml_nbytes(dst_final) &&
         (uintptr_t) dst_final->data < (uintptr_t) down_mmid->data + ggml_nbytes(down_mmid));
    std::unique_ptr<ggml_sycl_pool_alloc<float>> expert_scratch;
    float * experts_base = (float *) down_mmid->data;
    if (mmid_dst_overlap) {
        expert_scratch = std::make_unique<ggml_sycl_pool_alloc<float>>(
            ctx.pool(), (size_t) n_embd * (size_t) n_experts_used * (size_t) n_tokens);
        experts_base = expert_scratch->get();
    }

    std::vector<int64_t> expert_row_counts;
    std::vector<int64_t> expert_row_offsets;
    std::vector<int> host_counts_device;
    sycl::event counts_ready_ev;
    bool device_sort_active = false;

    const bool use_device_sort =
        ggml_sycl_mmid_device_sort_enabled() && ids->nb[0] == sizeof(int32_t);

    if (use_device_sort) {
        const int n_tokens_i = (int) ne12;
        const int n_ids_i    = (int) n_ids;
        const int n_as_i     = (int) n_as;
        const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
        const int32_t * ids_d = (const int32_t *) ids->data;
        ggml_sycl_pool_alloc<int> dev_counts(ctx.pool(), n_as);
        ggml_sycl_pool_alloc<int> dev_next(ctx.pool(), n_as);
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dev_counts.get(), 0, sizeof(int) * n_as)));
        constexpr int BS = 256;
        const int total   = n_ids_i * n_tokens_i;
        const int nblocks = (total + BS - 1) / BS;
        int * counts_d = dev_counts.get();
        int * next_d   = dev_next.get();
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                mmid_device_count_experts(ids_d, counts_d, n_ids_i, n_tokens_i, ids_stride, n_as_i, item);
            });
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) {
                mmid_device_exclusive_scan(counts_d, next_d, n_as_i, item);
            });
        mmid_row_mapping * map_d = dev_row_mapping.get();
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                mmid_device_fill_mapping(ids_d, next_d, map_d, n_ids_i, n_tokens_i, ids_stride, n_as_i, item);
            });
        host_counts_device.resize(n_as);
        counts_ready_ev = stream->memcpy(host_counts_device.data(), counts_d, sizeof(int) * n_as);
        device_sort_active = true;
    } else {
        std::vector<char> ids_host(ggml_nbytes(ids));
        {
            lx_chrono_wait_guard _w;
            SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(ids_host.data(), ids->data, ggml_nbytes(ids))));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;
        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(dev_row_mapping.get(), routed_row_src.data(),
                           n_routed_rows * sizeof(mmid_row_mapping))));
    }

    // Pack activations once.
    {
        const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
        sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne10, max_wg));
        sycl::range<3> grid_dims(1, 1, n_routed_rows);
        char * src1_contig = src1_contiguous.get();
        mmid_row_mapping * map_d = dev_row_mapping.get();
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_copy_src1_to_contiguous(
                        src1_original, src1_contig, map_d,
                        ne11, ne10, nb11, nb12, item_ct1);
                });
        });
    }

    if (device_sort_active) {
        // The async D2H counts event alone is not reliable on this L0 stack
        // (observed: host counts read before the 1KB memcpy landed, corrupting
        // the expert histogram). A full queue drain is required; it also gates
        // the in-order scatter below. Cost: 1 host sync/layer, still 3x fewer
        // than the stock per-op D2H+wait path this fuse replaces.
        {
            lx_chrono_wait_guard _w;
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }
        expert_row_counts.assign(n_as, 0);
        expert_row_offsets.assign(n_as + 1, 0);
        for (int64_t e = 0; e < n_as; ++e) {
            expert_row_counts[e] = host_counts_device[e];
            expert_row_offsets[e + 1] = expert_row_offsets[e] + expert_row_counts[e];
        }
        GGML_ASSERT(expert_row_offsets[n_as] == n_routed_rows);
    }

    // Shells templated from live tensors (preserve type/extra/buffer metadata).
    ggml_tensor src0_row = *gate_w;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_mid  = *glu;       // F32 template for mid/swiglu
    ggml_tensor dst_down = *down_mmid; // F32 template for down rows
    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    dst_mid.ne[2]  = 1;
    dst_mid.ne[3]  = 1;
    dst_down.ne[2] = 1;
    dst_down.ne[3] = 1;

    char * gate_w_base = (char *) gate_w->data;
    char * up_w_base   = (char *) up_w->data;
    char * down_w_base = (char *) down_w->data;

    for (int64_t i02 = 0; i02 < n_as; i02++) {
        const int64_t num_src1_rows = expert_row_counts[i02];
        if (num_src1_rows == 0) {
            continue;
        }
        const int64_t expert_row_offset = expert_row_offsets[i02];

        // Packed act [embd, N]
        src1_row.data  = src1_contiguous.get() + expert_row_offset * nb_act;
        src1_row.ne[0] = ne10;
        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[0] = sizeof(float);
        src1_row.nb[1] = nb_act;
        src1_row.nb[2] = num_src1_rows * nb_act;
        src1_row.nb[3] = num_src1_rows * nb_act;

        if (lx_gate_up_concat && gate_w->type == up_w->type &&
            gate_w->ne[0] == up_w->ne[0] && gate_w->ne[1] == up_w->ne[1] &&
            nb_wg == nb_wu) {
            // [lx-gate-up-concat] Pack [gate_w; up_w] per-expert into contiguous
            // [2*n_ff, n_embd] in the packed weight buffer, then one GEMM.
            // Layout: rows [0, n_ff) = gate, rows [n_ff, 2*n_ff) = up.
            // Packed in the same quantized type as the source weights.
            const size_t per_expert_bytes = nb_wg;  // == nb_wu
            // Allocate packed weight scratch once (reused across experts)
            if (packed_w_scratch.get() == nullptr) {
                packed_w_scratch.alloc(ctx.pool(), (size_t) n_as * 2 * per_expert_bytes);
            }
            if (packed_dst_scratch.get() == nullptr) {
                packed_dst_scratch.alloc(ctx.pool(), n_routed_rows * nb_2ff);
            }
            char * packed_w = packed_w_scratch.get() + i02 * 2 * per_expert_bytes;
            // Copy gate rows then up rows into packed_w (device-side memcpy)
            stream->memcpy(packed_w, gate_w_base + i02 * nb_wg, per_expert_bytes);
            stream->memcpy(packed_w + per_expert_bytes, up_w_base + i02 * nb_wu, per_expert_bytes);

            // Single GEMM: [2*n_ff, embd] @ [embd, N] -> [2*n_ff, N]
            src0_row = *gate_w;  // template for type/strides
            src0_row.ne[1] = 2 * n_ff;  // 2x rows
            src0_row.ne[2] = 1;
            src0_row.ne[3] = 1;
            src0_row.nb[1] = gate_w->nb[1];  // row stride unchanged
            src0_row.nb[2] = 2 * n_ff * gate_w->nb[1];
            src0_row.nb[3] = 2 * n_ff * gate_w->nb[1];
            src0_row.data  = packed_w;
            dst_mid.ne[0]  = 2 * n_ff;
            dst_mid.ne[1]  = num_src1_rows;
            dst_mid.nb[0]  = sizeof(float);
            dst_mid.nb[1]  = nb_2ff;
            dst_mid.nb[2]  = num_src1_rows * nb_2ff;
            dst_mid.nb[3]  = num_src1_rows * nb_2ff;
            dst_mid.data   = packed_dst_scratch.get() + expert_row_offset * nb_2ff;
            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_mid);

            // Swiglu: read gate from rows [0, n_ff), up from rows [n_ff, 2*n_ff)
            // Write swiglu result back to mid_contiguous at [n_ff, N]
            {
                const int64_t nelt = num_src1_rows * n_ff;
                const float * gd = (const float *) (packed_dst_scratch.get() + expert_row_offset * nb_2ff);
                const float * ud = gd + n_ff;  // up is second half
                float *       dd = (float *) (mid_contiguous.get() + expert_row_offset * nb_ff);
                constexpr int BS = 256;
                const int nblocks = (int) ((nelt + BS - 1) / BS);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
                    [=](sycl::nd_item<1> item) {
                        moe_dual_swiglu_contiguous(gd, ud, dd, nelt, item);
                    });
            }
        } else {
            // Gate: [ff, embd] @ [embd, N] -> [ff, N]
            src0_row = *gate_w;
            src0_row.ne[2] = 1;
            src0_row.ne[3] = 1;
            src0_row.data  = gate_w_base + i02 * nb_wg;
            dst_mid.ne[0]  = n_ff;
            dst_mid.ne[1]  = num_src1_rows;
            dst_mid.nb[0]  = sizeof(float);
            dst_mid.nb[1]  = nb_ff;
            dst_mid.nb[2]  = num_src1_rows * nb_ff;
            dst_mid.nb[3]  = num_src1_rows * nb_ff;
            dst_mid.data   = mid_contiguous.get() + expert_row_offset * nb_ff;
            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_mid);

            // Up GEMM into up buffer
            src0_row = *up_w;
            src0_row.ne[2] = 1;
            src0_row.ne[3] = 1;
            src0_row.data  = up_w_base + i02 * nb_wu;
            dst_mid.data   = up_contiguous.get() + expert_row_offset * nb_ff;
            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_mid);

            // Swiglu: mid = silu(gate)*up (overwrite mid_contiguous)
            {
                const int64_t nelt = num_src1_rows * n_ff;
                const float * gd = (const float *) (mid_contiguous.get() + expert_row_offset * nb_ff);
                // gate was overwritten by second mul into up_contiguous — restore gate from first write.
                // Fix: gate result is still in mid until we overwrote... wait we wrote gate to mid,
                // then up to up_contiguous. mid still has gate. Good.
                const float * ud = (const float *) (up_contiguous.get() + expert_row_offset * nb_ff);
                float *       dd = (float *) (mid_contiguous.get() + expert_row_offset * nb_ff);
                constexpr int BS = 256;
                const int nblocks = (int) ((nelt + BS - 1) / BS);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
                    [=](sycl::nd_item<1> item) {
                        moe_dual_swiglu_contiguous(gd, ud, dd, nelt, item);
                    });
            }
        }

        // Down: [embd, ff] @ [ff, N] -> [embd, N]
        src0_row = *down_w;
        src0_row.ne[2] = 1;
        src0_row.ne[3] = 1;
        src0_row.data  = down_w_base + i02 * nb_wd;
        src1_row.data  = mid_contiguous.get() + expert_row_offset * nb_ff;
        src1_row.ne[0] = n_ff;
        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[0] = sizeof(float);
        src1_row.nb[1] = nb_ff;
        src1_row.nb[2] = num_src1_rows * nb_ff;
        src1_row.nb[3] = num_src1_rows * nb_ff;
        dst_down.ne[0] = n_embd;
        dst_down.ne[1] = num_src1_rows;
        dst_down.nb[0] = sizeof(float);
        dst_down.nb[1] = nb_em;
        dst_down.nb[2] = num_src1_rows * nb_em;
        dst_down.nb[3] = num_src1_rows * nb_em;
        dst_down.data  = down_contiguous.get() + expert_row_offset * nb_em;
        ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_down);
    }

    const float * weight_ptr = (const float *) weights->data;
    std::unique_ptr<ggml_sycl_pool_alloc<float>> weight_scratch;
    const bool weights_dst_overlap =
        ((uintptr_t) weights->data < (uintptr_t) dst_final->data + ggml_nbytes(dst_final) &&
         (uintptr_t) dst_final->data < (uintptr_t) weights->data + ggml_nbytes(weights));
    if (weights_dst_overlap) {
        const size_t w_nelt = (size_t) n_experts_used * (size_t) n_tokens;
        weight_scratch = std::make_unique<ggml_sycl_pool_alloc<float>>(ctx.pool(), w_nelt);
        ctx.stream()->memcpy(weight_scratch->get(), weights->data, w_nelt * sizeof(float));
        weight_ptr = weight_scratch->get();
    }

    // Packed weighted reduce (skip embd×k×T scatter). Default ON.
    // Kill: GGML_SYCL_DISABLE_MOE_PACKED_REDUCE=1 → scatter + stock reduce.
    // notes/SHIP_20260730_moe_packed_reduce.md
    static const bool enable_packed_reduce = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MOE_PACKED_REDUCE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    int packed_flag = 0;
    if (enable_packed_reduce) {
        // inv[token*k+slot] = packed_row from mapping[pos]={slot,token}
        ggml_sycl_pool_alloc<int> inv_map(ctx.pool(), (size_t) n_routed_rows);
        {
            mmid_row_mapping * map_d = dev_row_mapping.get();
            int * inv_d = inv_map.get();
            const int n_rt = (int) n_routed_rows;
            const int k_i  = (int) n_experts_used;
            constexpr int BS = 256;
            const int nblocks = (n_rt + BS - 1) / BS;
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
                [=](sycl::nd_item<1> item) {
                    const int pos = (int) item.get_global_id(0);
                    if (pos >= n_rt) {
                        return;
                    }
                    const int slot  = map_d[pos].i1;
                    const int token = map_d[pos].i2;
                    inv_d[(size_t) token * (size_t) k_i + (size_t) slot] = pos;
                });
        }
        const float * packed = (const float *) down_contiguous.get();
        float * dst_f = (float *) dst_final->data;
        const int64_t total = n_embd * n_tokens;
        constexpr int block_size = 256;
        const int64_t global = ((total + block_size - 1) / block_size) * block_size;
        const int * inv_d = inv_map.get();
        const int64_t k = n_experts_used;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) global), sycl::range<1>(block_size)),
                [=](sycl::nd_item<1> item) {
                    const int64_t out = (int64_t) item.get_global_linear_id();
                    if (out >= total) {
                        return;
                    }
                    const int64_t token = out / n_embd;
                    const int64_t row   = out - token * n_embd;
                    float sum = 0.0f;
                    for (int64_t slot = 0; slot < k; ++slot) {
                        const int prow = inv_d[token * k + slot];
                        const float value  = packed[(size_t) prow * (size_t) n_embd + (size_t) row];
                        const float weight = weight_ptr[token * k + slot];
                        volatile float weighted = value * weight;
                        sum += weighted;
                    }
                    dst_f[out] = sum;
                });
        });
        packed_flag = 1;
    } else {
        // Scatter expert-batch down → [embd, k, T] then stock reduce
        {
            const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
            sycl::range<3> block_dims(1, 1, std::min((unsigned int) n_embd, max_wg));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            char * dst_orig = (char *) experts_base;
            const char * cont = down_contiguous.get();
            mmid_row_mapping * map_d = dev_row_mapping.get();
            const size_t nb1_sc = nb_em;
            const size_t nb2_sc = nb_em * (size_t) n_experts_used;
            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_orig, cont, map_d, n_embd, nb1_sc, nb2_sc, item_ct1);
                    });
            });
        }
        ggml_sycl_moe_weighted_reduce(
            ctx, experts_base, weight_ptr, (float *) dst_final->data,
            n_embd, n_experts_used, n_tokens);
    }

    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP "
                    "n_tokens=%" PRId64 " k=%" PRId64 " n_experts=%" PRId64 " packed_reduce=%d\n",
                    n_tokens, n_experts_used, n_as, packed_flag);
        }
    }
    return true;
}


// Entry: try expert-loop; fall back to compose (dual fuse + stock mmid + reduce).
static bool ggml_sycl_mul_mat_id_dual_down_multitoken(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu,
    ggml_tensor * down_mmid, const ggml_tensor * weights, ggml_tensor * dst_final) {
    if (!gate || !up || !glu || !down_mmid || !weights || !dst_final) {
        return false;
    }
    if (!glu->data || !down_mmid->data || !weights->data || !dst_final->data) {
        return false;
    }
    // Prefer expert-loop (skip glu write + second sort). Env: DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1
    static const bool enable_eloop = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (enable_eloop &&
        ggml_sycl_mul_mat_id_dual_down_multitoken_expert_loop(
            ctx, gate, up, glu, down_mmid, weights, dst_final)) {
        return true;
    }
    // Compose fallback (graph tensors).
    if (!ggml_sycl_mul_mat_id_dual_swiglu_fused(ctx, gate, up, glu)) {
        return false;
    }
    ggml_sycl_mul_mat_id(ctx, down_mmid);
    const int64_t n_embd         = down_mmid->ne[0];
    const int64_t n_experts_used = down_mmid->ne[1];
    const int64_t n_tokens       = down_mmid->ne[2];
    const bool mmid_dst_overlap =
        (down_mmid->data == dst_final->data) ||
        ((uintptr_t) down_mmid->data < (uintptr_t) dst_final->data + ggml_nbytes(dst_final) &&
         (uintptr_t) dst_final->data < (uintptr_t) down_mmid->data + ggml_nbytes(down_mmid));
    const bool weights_dst_overlap =
        ((uintptr_t) weights->data < (uintptr_t) dst_final->data + ggml_nbytes(dst_final) &&
         (uintptr_t) dst_final->data < (uintptr_t) weights->data + ggml_nbytes(weights));
    const float * weight_ptr = (const float *) weights->data;
    std::unique_ptr<ggml_sycl_pool_alloc<float>> expert_scratch;
    std::unique_ptr<ggml_sycl_pool_alloc<float>> weight_scratch;
    const float * expert_ptr = (const float *) down_mmid->data;
    if (mmid_dst_overlap) {
        const size_t nelt = (size_t) n_embd * (size_t) n_experts_used * (size_t) n_tokens;
        expert_scratch = std::make_unique<ggml_sycl_pool_alloc<float>>(ctx.pool(), nelt);
        ctx.stream()->memcpy(expert_scratch->get(), down_mmid->data, nelt * sizeof(float));
        expert_ptr = expert_scratch->get();
    }
    if (weights_dst_overlap) {
        const size_t w_nelt = (size_t) n_experts_used * (size_t) n_tokens;
        weight_scratch = std::make_unique<ggml_sycl_pool_alloc<float>>(ctx.pool(), w_nelt);
        ctx.stream()->memcpy(weight_scratch->get(), weights->data, w_nelt * sizeof(float));
        weight_ptr = weight_scratch->get();
    }
    ggml_sycl_moe_weighted_reduce(
        ctx, expert_ptr, weight_ptr, (float *) dst_final->data,
        n_embd, n_experts_used, n_tokens);
    {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-moe-dual] multi-token dual+down (compose fallback) "
                    "n_tokens=%" PRId64 " k=%" PRId64 "\n",
                    n_tokens, n_experts_used);
        }
    }
    return true;
}

static bool ggml_sycl_tensor_buf_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t a1 = a0 + (uintptr_t) ggml_nbytes(a);
    const uintptr_t b1 = b0 + (uintptr_t) ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// Fuse Laguna MoE down tail for serial decode (n_tokens small).
// Graph: MUL_MAT_ID → MUL(weights) → VIEW×n_experts → ADD×(n_experts-1)
// Returns following-nodes-to-skip (0 = no fuse).
int ggml_sycl_fuse_moe_down_weighted(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED");
        if (dis != nullptr && std::atoi(dis) != 0) {
            return false;
        }
        const char * env = getenv("GGML_SYCL_ENABLE_MOE_DOWN_WEIGHTED");
        // Default ON when unset; explicit 0 disables.
        if (env == nullptr) {
            return true;
        }
        return std::atoi(env) != 0;
    }();
    if (!enabled) {
        return 0;
    }

    ggml_tensor * mmid = cgraph->nodes[i];
    if (mmid->op != GGML_OP_MUL_MAT_ID || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mul = cgraph->nodes[i + 1];
    if (mul->op != GGML_OP_MUL ||
        (mul->src[0] != mmid && mul->src[1] != mmid)) {
        return 0;
    }
    const ggml_tensor * weights = mul->src[0] == mmid ? mul->src[1] : mul->src[0];

    const int64_t n_embd         = mmid->ne[0];
    const int64_t n_experts_used = mmid->ne[1];
    const int64_t n_tokens       = mmid->ne[2];

    // Laguna serial: k=8, n_embd=2048, n_tokens usually 1 (decode). Cap prefill modestly.
    if (mmid->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        weights == nullptr || weights->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mmid, mul) || mmid->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_experts_used ||
        weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        !ggml_is_contiguous(mmid) || !ggml_is_contiguous(weights) ||
        n_experts_used < 2 || n_experts_used > 16 ||
        n_tokens < 1 || n_tokens > 2048 || n_embd < 1) {
        return 0;
    }

    // Build expected op list: MMID, MUL, VIEW×k, ADD×(k-1)
    const int n_views = (int) n_experts_used;
    const int n_adds  = (int) n_experts_used - 1;
    const int n_ops   = 2 + n_views + n_adds;
    if (i + n_ops > cgraph->n_nodes) {
        return 0;
    }

    std::vector<ggml_op> ops;
    ops.reserve((size_t) n_ops);
    ops.push_back(GGML_OP_MUL_MAT_ID);
    ops.push_back(GGML_OP_MUL);
    for (int v = 0; v < n_views; ++v) {
        ops.push_back(GGML_OP_VIEW);
    }
    for (int a = 0; a < n_adds; ++a) {
        ops.push_back(GGML_OP_ADD);
    }
    const int output = i + n_ops - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), &output, 1)) {
        return 0;
    }

    // Validate view/add wiring: views of mul (or mmid if weights applied in reduce).
    // After expand, views source the weighted experts tensor (mul).
    for (int v = 0; v < n_views; ++v) {
        ggml_tensor * view = cgraph->nodes[i + 2 + v];
        if (view->op != GGML_OP_VIEW || view->src[0] != mul) {
            return 0;
        }
        // view_2d(experts, n_embd, n_tokens, nb2, v*nb1)
        if (view->ne[0] != n_embd || view->ne[1] != n_tokens) {
            return 0;
        }
    }
    // First ADD: view0 + view1; subsequent: prev_add + view[i+1]
    ggml_tensor * prev = cgraph->nodes[i + 2];  // view0
    for (int a = 0; a < n_adds; ++a) {
        ggml_tensor * add = cgraph->nodes[i + 2 + n_views + a];
        ggml_tensor * v1  = cgraph->nodes[i + 2 + a + 1];  // next view
        if (add->op != GGML_OP_ADD) {
            return 0;
        }
        const bool ok = (add->src[0] == prev && add->src[1] == v1) ||
                        (add->src[1] == prev && add->src[0] == v1);
        if (!ok) {
            return 0;
        }
        prev = add;
    }
    ggml_tensor * dst = cgraph->nodes[output];
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens) {
        return 0;
    }

    // [lx-down-scratch] At multi-token (pp512) the graph allocator aliases the
    // route weights [1,k,T] (16 KiB at k=8/T=512) INSIDE the final dst slab
    // (4 MiB). The old code hard-rejected weights/dst overlap, so pp512 ran the
    // unfused MUL + VIEW×k + ADD×(k-1) chain (~8 extra launches/layer) instead of
    // the fused weighted reduce. Copy the weights to pool scratch first — the
    // same no-wait in-order memcpy the dual+down path uses (weights bytes were
    // written by the producer op earlier in the queue; the reduce reads the
    // scratch after the copy, and the dst is only written by the reduce) — and
    // keep the mmid alias a hard reject (mmid is read by the reduce while dst is
    // written, so that alias is a genuine race). Bit-neutral: the fused reduce
    // sums the same products in the same left-to-right expert order as the
    // graph's MUL+ADD chain.
    const bool ov_mmid = ggml_sycl_tensor_buf_overlap(mmid, dst);
    if (ov_mmid) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-moe-down] skip: buffer overlap with dst "
                    "(weights=%d mmid=%d mul=%d) dst=%p nbytes=%zu\n",
                    ggml_sycl_tensor_buf_overlap(weights, dst), ov_mmid,
                    ggml_sycl_tensor_buf_overlap(mul, dst),
                    dst->data, (size_t) ggml_nbytes(dst));
        }
        return 0;
    }
    const float * weight_ptr = (const float *) weights->data;
    std::unique_ptr<ggml_sycl_pool_alloc<float>> weight_scratch;
    if (ggml_sycl_tensor_buf_overlap(weights, dst)) {
        const size_t w_nelt = (size_t) n_experts_used * (size_t) n_tokens;
        weight_scratch = std::make_unique<ggml_sycl_pool_alloc<float>>(ctx.pool(), w_nelt);
        ctx.stream()->memcpy(weight_scratch->get(), weights->data, w_nelt * sizeof(float));
        weight_ptr = weight_scratch->get();
    }

    // Integrated weighted MMVQ (down GEMV * route weight → dst). Default OFF:
    // hits + slightly faster probe but golden FAIL (2026-07-30). Two-step remains tip.
    // Opt-in: GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED=1
    static const bool enable_integrated = []() {
        const char * env = getenv("GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED");
        return env != nullptr && std::atoi(env) != 0;
    }();

    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * ids  = mmid->src[2];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    bool integrated = false;
    if (enable_integrated &&
        ne12 == n_tokens && ne11 == n_experts_used &&
        ne10 == src0->ne[0] && ne10 % QK8_1 == 0 &&
        src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) &&
        ids->ne[0] == n_experts_used && ids->ne[1] == n_tokens &&
        ids->nb[0] == sizeof(int32_t) &&
        n_embd == src0->ne[1] &&
        (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K ||
         src0->type == GGML_TYPE_Q6_K)) {
        opt_for_reorder_id(&ctx, src0);
        const ggml_tensor_extra_gpu * extra =
            static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            const queue_ptr stream = ctx.stream();
            const int src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
            ggml_sycl_pool_alloc<char> src1_q8_alloc(
                ctx.pool(),
                (size_t) ne11 * ne12 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
            char * src1_ddq = src1_q8_alloc.get();
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                (const float *) src1->data, src1_ddq, (int) ne10, (int) (ne11 * ne12),
                src1_padded_cols, stream);

            const size_t bytes_per_qrow    = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
            const size_t src1_row_stride   = bytes_per_qrow;
            const size_t src1_token_stride = (size_t) ne11 * bytes_per_qrow;
            const int    ids_row_stride    = (int) (ids->nb[1] / sizeof(int32_t));
            // weights [1,k,T] contiguous: float stride between tokens = k
            const size_t weights_tok_stride = (size_t) n_experts_used;
            const size_t dst_tok_stride     = dst->nb[1];  // [n_embd, n_tokens]

            integrated = ggml_sycl_mul_mat_vec_q_id_weighted_reorder(
                src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
                (const float *) weight_ptr, (float *) dst->data,
                (int) ne10, (int) n_embd, (int) n_experts_used,
                src0->nb[2], src1_row_stride, (int) n_tokens, ids_row_stride,
                src1_token_stride, weights_tok_stride, dst_tok_stride, stream);
        }
    }

    if (!integrated) {
        // Fallback: Expert down projections into mmid buffer + weighted reduce.
        // weight_ptr may be a pool-scratch copy when the route weights alias dst
        // (pp512 product allocator reuse) — the reduce reads it while dst is
        // written, so the aliased bytes must never be read from the dst slab.
        ggml_sycl_mul_mat_id(ctx, mmid);
        ggml_sycl_moe_weighted_reduce(
            ctx, static_cast<const float *>(mmid->data),
            static_cast<const float *>(weight_ptr),
            static_cast<float *>(dst->data),
            n_embd, n_experts_used, n_tokens);
    }

    // DEBUG: numerical diff integrated weighted-mmvq vs per-expert reference.
    static const bool debug_diff = []() {
        const char * e = getenv("GGML_SYCL_DEBUG_MOE_DOWN_DIFF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (integrated && debug_diff) {
        const queue_ptr dstream = ctx.stream();
        std::vector<float> dst_integrated((size_t) n_embd * n_tokens);
        SYCL_CHECK(CHECK_TRY_ERROR(dstream->memcpy(dst_integrated.data(), dst->data,
            sizeof(float) * (size_t) n_embd * (size_t) n_tokens)));
        SYCL_CHECK(CHECK_TRY_ERROR(dstream->wait()));
        // reference: per-expert down GEMMs into mmid + weighted reduce into scratch
        ggml_sycl_mul_mat_id(ctx, mmid);
        ggml_sycl_pool_alloc<float> dst_ref_alloc(ctx.pool(), (size_t) n_embd * (size_t) n_tokens);
        ggml_sycl_moe_weighted_reduce(ctx,
            static_cast<const float *>(mmid->data),
            static_cast<const float *>(weight_ptr),
            dst_ref_alloc.get(), n_embd, n_experts_used, n_tokens);
        std::vector<float> dst_ref((size_t) n_embd * n_tokens);
        SYCL_CHECK(CHECK_TRY_ERROR(dstream->memcpy(dst_ref.data(), dst_ref_alloc.get(),
            sizeof(float) * (size_t) n_embd * (size_t) n_tokens)));
        SYCL_CHECK(CHECK_TRY_ERROR(dstream->wait()));
        double max_diff = 0.0, max_i = 0.0, max_r = 0.0, sum_i = 0.0, sum_r = 0.0;
        int64_t max_idx = 0;
        for (int64_t idx = 0; idx < n_embd * n_tokens; ++idx) {
            const double a = dst_integrated[idx], b = dst_ref[idx];
            sum_i += std::fabs(a); sum_r += std::fabs(b);
            const double d = std::fabs(a - b);
            if (d > max_diff) { max_diff = d; max_idx = idx; max_i = a; max_r = b; }
        }
        const int64_t dbg_row = max_idx % n_embd, dbg_tok = max_idx / n_embd;
        fprintf(stderr,
                "[lx-debug-moe-down-diff] n_tokens=%" PRId64 " k=%" PRId64 " embd=%" PRId64
                "  max_diff=%.6f @ tok=%" PRId64 " row=%" PRId64
                "  (integ=%.5f ref=%.5f)  mean|integ|=%.4f mean|ref|=%.4f\n",
                n_tokens, n_experts_used, n_embd, max_diff, dbg_tok, dbg_row,
                max_i, max_r, sum_i / (double)(n_embd * n_tokens), sum_r / (double)(n_embd * n_tokens));
    }

    // DEBUG: one-shot dump of the ACTUAL down dst (no interference) to a binary file,
    // so integrated-ON and integrated-OFF runs can be diffed layer-by-layer.
    static const bool dump_dst = []() {
        const char * e = getenv("GGML_SYCL_MOE_DOWN_DUMP");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (dump_dst) {
        static std::atomic<int> dump_idx{0};
        int idx = dump_idx.fetch_add(1);
        if (idx < 128) {
            const queue_ptr dstream = ctx.stream();
            std::vector<float> buf((size_t) n_embd * n_tokens);
            SYCL_CHECK(CHECK_TRY_ERROR(dstream->memcpy(buf.data(), dst->data,
                sizeof(float) * (size_t) n_embd * (size_t) n_tokens)));
            SYCL_CHECK(CHECK_TRY_ERROR(dstream->wait()));
            char path[256];
            snprintf(path, sizeof(path), "/tmp/opencode/moe_down_dst_%s_%03d.bin",
                     integrated ? "ON" : "OFF", idx);
            FILE * f = fopen(path, "wb");
            if (f) {
                fwrite(buf.data(), sizeof(float), (size_t) n_embd * (size_t) n_tokens, f);
                fclose(f);
                fprintf(stderr, "[lx-dump] wrote %s (embd=%" PRId64 " tok=%" PRId64 ")\n",
                        path, n_embd, n_tokens);
            }
        }
    }

    static std::atomic<int> once{0};
    if (once.fetch_add(1) == 0) {
        fprintf(stderr,
                "[lx-control-moe-down] fuse hit (%s) embd=%" PRId64
                " k=%" PRId64 " tokens=%" PRId64 "\n",
                integrated ? "integrated weighted-mmvq" : "weighted reduce",
                n_embd, n_experts_used, n_tokens);
    }
    return n_ops - 1;
}

// [lx-qkv] Fuse the three attention projection GEMVs (Q q4_K, V q6_K, K q4_K —
// all against the SAME attn_norm activation) into ONE reorder-MMVQ launch for
// serial decode (src1->ne[1] == 1). Saves 2 kernel launches + 2 activation
// quantizations per layer per token. Prefill (multi-col src1) is untouched.
// Kill: GGML_SYCL_DISABLE_QKV_FUSE=1
static bool ggml_backend_sycl_satisfied(ggml_tensor * node);
static void ggml_backend_sycl_mark_satisfied(ggml_tensor * node);
int ggml_sycl_fuse_qkv(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_QKV_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled) {
        return 0;
    }

    ggml_tensor * nq = cgraph->nodes[i];
    if (nq->op != GGML_OP_MUL_MAT || nq->src[1] == nullptr || nq->src[0] == nullptr) {
        return 0;
    }
    const char * qname = nq->name ? nq->name : "";
    if (strncmp(qname, "Qcur", 4) != 0) {
        return 0;
    }

    // Scan ahead for the Vcur and Kcur MUL_MATs. Q's own post-processing chain
    // (RMS_NORM -> MUL -> ROPE on Qcur) and pass-through nodes sit between the
    // three projections; they stay real work and are executed after the fused
    // launch (they depend on Qcur, which the fused kernel writes).
    ggml_tensor * nv = nullptr;
    ggml_tensor * nk = nullptr;
    int nk_idx = -1;
    std::vector<int> interleaved;  // graph indices of Q post-processing (RMS/MUL/ROPE)
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 40; ++j) {
        ggml_tensor * nd = cgraph->nodes[j];
        if (ggml_is_empty(nd) || nd->op == GGML_OP_RESHAPE || nd->op == GGML_OP_TRANSPOSE ||
            nd->op == GGML_OP_VIEW || nd->op == GGML_OP_PERMUTE || nd->op == GGML_OP_NONE) {
            continue;
        }
        const char * dname = nd->name ? nd->name : "";
        if (nd->op == GGML_OP_RMS_NORM || nd->op == GGML_OP_MUL || nd->op == GGML_OP_ROPE) {
            interleaved.push_back(j);
            continue;
        }
        if (nd->op != GGML_OP_MUL_MAT || nd->src[1] != nq->src[1] || nd->src[0] == nullptr) {
            static const bool dbg = getenv("GGML_SYCL_DEBUG_QKV_VSET") != nullptr;
            if (dbg) {
                fprintf(stderr, "[lx-dbg-vset] BAIL j=%d op=%s name=%s src1_same=%d\n",
                        j, ggml_op_name(nd->op), dname, nd->src[1] == nq->src[1]);
            }
            return 0;
        }
        if (nv == nullptr && strncmp(dname, "Vcur", 4) == 0) {
            nv = nd;
            continue;
        }
        if (nv != nullptr && strncmp(dname, "Kcur", 4) == 0) {
            nk = nd;
            nk_idx = j;
            break;
        }
        static const bool dbg2 = getenv("GGML_SYCL_DEBUG_QKV_VSET") != nullptr;
        if (dbg2) {
            fprintf(stderr, "[lx-dbg-vset] BAIL2 j=%d name=%s nv=%s\n", j, dname, nv ? "set" : "null");
        }
        return 0;
    }
    if (nv == nullptr || nk == nullptr) {
        return 0;
    }

    // Serial decode only; exact weight types of this model's attention stack.
    if (nq->src[1]->ne[1] != 1 || nq->src[1]->ne[2] != 1 || nq->src[1]->ne[3] != 1) {
        return 0;
    }
    static const bool dbg3 = getenv("GGML_SYCL_DEBUG_QKV_VSET") != nullptr;
    if (dbg3) {
        fprintf(stderr, "[lx-dbg-vset] types q=%s v=%s k=%s\n",
                ggml_type_name(nq->src[0]->type), ggml_type_name(nv->src[0]->type),
                ggml_type_name(nk->src[0]->type));
    }
    if (nq->src[0]->type != GGML_TYPE_Q4_K ||
        (nv->src[0]->type != GGML_TYPE_Q6_K && nv->src[0]->type != GGML_TYPE_Q4_K) ||
        nk->src[0]->type != GGML_TYPE_Q4_K) {
        return 0;
    }
    if (ggml_backend_buffer_is_sycl_split(nq->src[0]->buffer) ||
        ggml_backend_buffer_is_sycl_split(nv->src[0]->buffer) ||
        ggml_backend_buffer_is_sycl_split(nk->src[0]->buffer)) {
        return 0;
    }

    // Same reorder conditions as the MMVQ dispatch path; the reorder itself is
    // idempotent (already-reordered tensors return immediately).
    opt_for_reorder(&ctx, nq->src[0], nq->src[1], nq, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, nv->src[0], nv->src[1], nv, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, nk->src[0], nk->src[1], nk, mul_mat_algo::MMVQ);
    auto qe = static_cast<const ggml_tensor_extra_gpu *>(nq->src[0]->extra);
    auto ve = static_cast<const ggml_tensor_extra_gpu *>(nv->src[0]->extra);
    auto ke = static_cast<const ggml_tensor_extra_gpu *>(nk->src[0]->extra);
    if (!qe || !ve || !ke || !qe->optimized_feature.reorder || !ve->optimized_feature.reorder ||
        !ke->optimized_feature.reorder) {
        return 0;
    }

    // [lx-qkv-gate] Per-head attention gate (attn_gate_proj): same q8_1
    // activation row as Q/V/K, so fold its q4_K GEMV into this launch too and
    // mark its graph node satisfied (the main loop skips it). The segment dot
    // is the same reorder_vec_dot_q_sycl<Q4_K> the standalone MMVQ path runs,
    // against the bit-identical q8_1 activation, so results are bit-identical.
    // [lx-qkv-gate] OPT-IN (default OFF): folding the attn_gate_proj GEMV into
    // the fused Q/V/K launch as a 4th segment is structurally correct (same
    // q8_1 activation, bit-identical dot) but HANGS the B70 GPU at decode
    // (host busy-poll spin, any number of gate workgroups; see
    // results/iter-decode-20260808T184708Z). Keep the standalone gate path the
    // default; enable only for controlled experiments.
    static const bool gate_fold_enabled = []() {
        const char * en = getenv("GGML_SYCL_ENABLE_QKV_GATE");
        return en != nullptr && std::atoi(en) != 0;
    }();
    ggml_tensor * ng = nullptr;
    if (gate_fold_enabled)
    for (int j = nk_idx + 1; j < cgraph->n_nodes && j <= i + 40; ++j) {
        ggml_tensor * nd = cgraph->nodes[j];
        if (ggml_is_empty(nd) || nd->op != GGML_OP_MUL_MAT) {
            continue;  // K's rope/set_rows, the FLASH_ATTN_EXT node and view noise sit between
        }
        static const bool dbg_g = getenv("GGML_SYCL_DEBUG_QKV_VSET") != nullptr;
        if (dbg_g) {
            fprintf(stderr, "[lx-dbg-gate] j=%d op=%s name=%s src1same=%d src0type=%d\n",
                    j, ggml_op_name(nd->op), nd->name ? nd->name : "?",
                    nd->src[1] == nq->src[1], nd->src[0] ? (int) nd->src[0]->type : -1);
        }
        if (nd->src[1] != nq->src[1]) {
            break;  // next layer's Qcur (or another projection) — window over
        }
        const char * dname = nd->name ? nd->name : "";
        if (strncmp(dname, "attn_gate_proj", 14) == 0 && nd->src[0]->type == GGML_TYPE_Q4_K) {
            ng = nd;
            break;
        }
    }
    if (ng != nullptr) {
        opt_for_reorder(&ctx, ng->src[0], ng->src[1], ng, mul_mat_algo::MMVQ);
        auto ge = static_cast<const ggml_tensor_extra_gpu *>(ng->src[0]->extra);
        if (!ge || !ge->optimized_feature.reorder) {
            ng = nullptr;  // fall back to the standalone gate path
        }
    }
    const int nrows_g = ng != nullptr ? (int) ng->src[0]->ne[1] : 0;
    const void * vx_g = ng != nullptr ? ng->src[0]->data : nullptr;
    float * dst_g = ng != nullptr ? (float *) ng->data : nullptr;
    if (ng != nullptr) {
        ggml_backend_sycl_mark_satisfied(ng);
    }

    const int64_t ne10   = nq->src[1]->ne[0];
    const int64_t padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const queue_ptr stream = ctx.stream();

    // Quantize the shared activation once (same q8_1 SoA layout the MMVQ path
    // produces: quants then scales, row padded to MATRIX_ROW_PADDING).
    ggml_sycl_pool_alloc<char> src1_ddq(ctx.pool(),
        (size_t) padded * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        (const float *) nq->src[1]->data, src1_ddq.get(), (int) ne10, 1, (int) padded, stream);

    // Graph allocator aliases Qcur/Vcur (disjoint lifetimes: Q consumed by the
    // RMS/MUL/ROPE chain before V is written). The fused kernel writes both at
    // once, so route V through a scratch and publish it after Q is consumed.
    const bool v_aliases_q = nv->data == nq->data;
    ggml_sycl_pool_alloc<char> v_scratch(ctx.pool(),
        v_aliases_q ? ggml_nbytes(nv) : 0);

    // [lx-qkv-vcache] Find V's SET_ROWS (the f16 KV-cache publish, usually the
    // node right after K's rope+set_rows group). The fused kernel writes V rows
    // straight into the cache, so the SET_ROWS node is satisfied and skipped.
    // If it cannot be found/validated, fall back to the normal publish path.
    auto descends_from_v = [](const ggml_tensor * node, const ggml_tensor * root) {
        for (int depth = 0; node && depth < 4; ++depth) {
            if (node == root) {
                return true;
            }
            if (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE) {
                break;
            }
            node = node->src[0];
        }
        return false;
    };
    ggml_tensor * vset = nullptr;
    static const bool dbg_vset = getenv("GGML_SYCL_DEBUG_QKV_VSET") != nullptr;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 48; ++j) {
        ggml_tensor * nd = cgraph->nodes[j];
        if (ggml_is_empty(nd) || nd->op == GGML_OP_VIEW || nd->op == GGML_OP_RESHAPE ||
            nd->op == GGML_OP_PERMUTE || nd->op == GGML_OP_NONE) {
            continue;
        }
        if (dbg_vset) {
            fprintf(stderr, "[lx-dbg-vset] j=%d op=%s name=%s src0=%s nv=%s\n",
                    j, ggml_op_name(nd->op), nd->name ? nd->name : "?",
                    (nd->src[0] && nd->src[0]->name) ? nd->src[0]->name : "?",
                    nv->name ? nv->name : "?");
        }
        if (nd->op == GGML_OP_SET_ROWS) {
            if (vset == nullptr && descends_from_v(nd->src[0], nv)) {
                vset = nd;
                break;
            }
            continue;  // K's set_rows descends from nk; keep scanning
        }
        if (nd->op == GGML_OP_RMS_NORM || nd->op == GGML_OP_MUL || nd->op == GGML_OP_ROPE ||
            nd->op == GGML_OP_MUL_MAT) {
            continue;  // Q's post chain + the V/K projections this fuse already runs
        }
        break;
    }
    if (dbg_vset) {
        fprintf(stderr, "[lx-dbg-vset] result: vset=%s\n", vset ? vset->name : "NULL");
    }
    const bool vcache_ok = vset != nullptr && vset->type == GGML_TYPE_F16 &&
        vset->src[1] != nullptr && vset->src[1]->type == GGML_TYPE_I64 &&
        ggml_nelements(vset->src[1]) == 1 &&
        vset->ne[0] == nv->src[0]->ne[1] && vset->nb[1] % sizeof(sycl::half) == 0;
    if (dbg_vset) {
        fprintf(stderr, "[lx-dbg-vset] vcache_ok=%d type=%d src1type=%d nel=%lld ne0=%lld want=%lld nb1=%lld\n",
                (int) vcache_ok, vset ? (int) vset->type : -1,
                (vset && vset->src[1]) ? (int) vset->src[1]->type : -1,
                (long long) (vset && vset->src[1] ? ggml_nelements(vset->src[1]) : -1),
                (long long) (vset ? vset->ne[0] : -1),
                (long long) nv->src[0]->ne[1],
                (long long) (vset ? vset->nb[1] : -1));
    }

    if (vcache_ok) {
        ggml_sycl_mmvq_set_vcache_f16((sycl::half *) vset->data,
            (const int64_t *) vset->src[1]->data,
            (int64_t) (vset->nb[1] / sizeof(sycl::half)));
    }

    // [lx-diag-decode] measurement-only skip of the fused-QKV launch (bit 1).
    if (!(lx_diag_skip_decode() & 1)) {
        if (nv->src[0]->type == GGML_TYPE_Q6_K) {
            reorder_mul_mat_vec_q4_q6_q4_k_q8_1_qkv_sycl(
                nq->src[0]->data, nv->src[0]->data, nk->src[0]->data, vx_g,
                src1_ddq.get(), (float *) nq->data,
                (float *) (v_aliases_q ? v_scratch.get() : nv->data), (float *) nk->data, dst_g,
                (int) ne10, (int) nq->src[0]->ne[1], (int) nv->src[0]->ne[1], (int) nk->src[0]->ne[1],
                nrows_g, stream);
        } else {
            reorder_mul_mat_vec_q4_q4_q4_k_q8_1_qkv_sycl(
                nq->src[0]->data, nv->src[0]->data, nk->src[0]->data, vx_g,
                src1_ddq.get(), (float *) nq->data,
                (float *) (v_aliases_q ? v_scratch.get() : nv->data), (float *) nk->data, dst_g,
                (int) ne10, (int) nq->src[0]->ne[1], (int) nv->src[0]->ne[1], (int) nk->src[0]->ne[1],
                nrows_g, stream);
        }
    }

    if (vcache_ok) {
        ggml_sycl_mmvq_set_vcache_f16(nullptr, nullptr, 0);
        ggml_backend_sycl_mark_satisfied(vset);
    }

    // Q's post-processing (RMS_NORM/MUL/ROPE on Qcur) must run after the fused
    // launch since it reads Qcur, exactly as the serial graph would. Run the
    // same fuse chain the main loop applies so fused groups (rms+mul) stay fused
    // and numerics are bit-identical to the unfused path.
    for (size_t k = 0; k < interleaved.size(); ++k) {
        if (interleaved[k] < 0) {
            continue;  // consumed by an earlier sub-fuse
        }
        const int j = interleaved[k];
        const int s = ggml_sycl_fuse(ctx, cgraph, j);
        if (s != 0) {
            // Skip any nodes the sub-fuse consumed (they cannot be in our list).
            for (size_t kk = k + 1; kk < interleaved.size(); ++kk) {
                if (interleaved[kk] <= j + s) {
                    interleaved[kk] = -1;  // consumed by the sub-fuse
                }
            }
            continue;
        }
        GGML_ASSERT(ggml_sycl_compute_forward(ctx, cgraph->nodes[j]));
    }

    // Publish V into its aliased graph slot now that Q has been consumed.
    if (v_aliases_q) {
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(nv->data, v_scratch.get(), ggml_nbytes(nv)).wait()));
    }


    static std::atomic<int> once{0};
    if (once.fetch_add(1) == 0) {
        fprintf(stderr,
                "[lx-control-qkv] fuse hit q=%s[%" PRId64 "] v=%s[%" PRId64 "] k=%s[%" PRId64 "] "
                "gate=%s[%" PRId64 "] ncols=%" PRId64 "\n",
                qe ? "reordered" : "?", nq->src[0]->ne[1], ve ? "reordered" : "?", nv->src[0]->ne[1],
                ke ? "reordered" : "?", nk->src[0]->ne[1],
                ng != nullptr ? "reordered" : "-", ng != nullptr ? nrows_g : 0, ne10);
    }
    // Skip through the Kcur node (Q, V, K all computed above).
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        if (cgraph->nodes[j] == nk) {
            return j - i;
        }
    }
    return 0;
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

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
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

// Non-static so topk-moe hybrid can re-dispatch stock ops (bitexact wiring).
bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;

    // [lx-memo-q81] invalidate the q8_1 src1 memo when this op writes the memoized
    // tensor (its producer runs before its consumers in graph order).
    if (ctx.quant_memo_valid && dst != nullptr) {
        const void * dbase = dst->view_src != nullptr ? dst->view_src->data : dst->data;
        if (dbase == ctx.quant_memo_src1) {
            ctx.quant_memo_valid = false;
        }
    }

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
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
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
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
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
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
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
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
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
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

static void ggml_sycl_debug_checksum_node(ggml_backend_sycl_context & ctx, const ggml_tensor * t, int idx) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) return;
    const size_t n = (size_t) ggml_nelements(t);
    if (n == 0 || n > (1u << 26)) return;  // skip huge (decode activations are small)
    std::vector<float> buf(n);
    queue_ptr stream = ctx.stream();
    try {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(buf.data(), t->data, n * sizeof(float))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    } catch (...) { return; }
    double s = 0.0, mx = 0.0;
    for (size_t k = 0; k < n; ++k) { s += (double) buf[k]; double a = buf[k] < 0 ? -(double)buf[k] : (double)buf[k]; if (a > mx) mx = a; }
    fprintf(stderr, "[cksum] i=%d op=%s name=%s sum=%.6f maxabs=%.6f n=%zu\n",
            idx, ggml_op_name(t->op), t->name ? t->name : "(null)", s, mx, n);
}

// [lx-chrono] zero-behavior host-side per-dispatch chrono: GGML_SYCL_LX_CHRONO=1
// records (op, steady_ns) per dispatched node (fused groups count once) into a
// vector dumped at exit to $LX_CHRONO_OUT (default /tmp/lx-chrono.bin) as
// [i32 op][u64 ns] pairs followed by [i32 -1][u64 end_ns]. Env-off = one branch.
static bool ggml_backend_sycl_graph_compute_impl_stamp(int32_t op, bool fused_away, const char * name) {
    struct Sink {
        bool on = false;
        bool names = false;
        uint64_t t0 = 0;
        std::string path = "/tmp/lx-chrono.bin";
        std::vector<std::pair<int32_t, uint64_t>> ev;
        std::vector<std::string> evn;
        Sink() {
            const char * e = getenv("GGML_SYCL_LX_CHRONO");
            if (e != nullptr && std::atoi(e) != 0) {
                on = true;
                names = getenv("LX_CHRONO_NAMES") != nullptr && std::atoi(getenv("LX_CHRONO_NAMES")) != 0;
                fprintf(stderr, "[lx-chrono] on: out=%s\n",
                        getenv("LX_CHRONO_OUT") != nullptr ? getenv("LX_CHRONO_OUT") : "/tmp/lx-chrono.bin");
                t0 = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
                ev.reserve(1 << 20);
                const char * o = getenv("LX_CHRONO_OUT");
                if (o != nullptr && *o != '\0') path = o;
            }
        }
        ~Sink() {
            if (!on) return;
            fprintf(stderr, "[lx-chrono] WAIT_TOTAL_NS=%llu WAIT_COUNT=%llu\n",
                    (unsigned long long) lx_chrono_wait_ns().load(std::memory_order_relaxed),
                    (unsigned long long) lx_chrono_wait_count().load(std::memory_order_relaxed));
            const uint64_t t1 = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count();
            FILE * f = fopen(path.c_str(), "wb");
            if (f == nullptr) return;
            for (size_t k = 0; k < ev.size(); ++k) {
                fwrite(&ev[k].first, sizeof(ev[k].first), 1, f);
                fwrite(&ev[k].second, sizeof(ev[k].second), 1, f);
                if (names) {
                    const std::string & n = evn[k];
                    const uint32_t len = (uint32_t) n.size();
                    fwrite(&len, sizeof(len), 1, f);
                    if (len) fwrite(n.data(), 1, len, f);
                }
            }
            const int32_t end = -1;
            fwrite(&end, sizeof(end), 1, f);
            fwrite(&t1, sizeof(t1), 1, f);
            fclose(f);
        }
    };
    static Sink sink; // ctor on first stamp, dtor at exit
    if (!sink.on) return false;
    const uint64_t now = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count();
    sink.ev.emplace_back(fused_away ? (op | 0x80000000) : op, now - sink.t0);
    if (sink.names) sink.evn.emplace_back(name != nullptr ? name : "");
    return true;
}

static void ggml_backend_sycl_graph_compute_impl_stamp_start() {
    ggml_backend_sycl_graph_compute_impl_stamp(-3, false, "graph-start"); // graph-start marker
}

// [lx-qkv-vcache] Nodes whose work a fused launch already performed (the V
// SET_ROWS cache publish). The main loop skips them. Cleared per graph compute
// so stale pointers from prior graphs can never match a live node.
static std::vector<ggml_tensor *> g_lx_satisfied;
static bool ggml_backend_sycl_satisfied(ggml_tensor * node) {
    for (size_t k = 0; k < g_lx_satisfied.size(); ++k) {
        if (g_lx_satisfied[k] == node) {
            return true;
        }
    }
    return false;
}
static void ggml_backend_sycl_mark_satisfied(ggml_tensor * node) {
    g_lx_satisfied.push_back(node);
}
static void ggml_backend_sycl_clear_satisfied() {
    g_lx_satisfied.clear();
}

// [lx-norm-qkv] Cross-fuse state: the attn_norm (RMS_NORM+MUL) node is skipped
// by ggml_sycl_fuse_norm_qkv and its work (norm(x)*w) is computed inline by the
// QKV launch's activation quantize (quantize_norm_row_q8_1_sycl). Stash is
// armed per graph compute (gen guard) and consumed by the QKV fuse.
static ggml_tensor * g_lx_norm_qkv_raw = nullptr;  // RMS src0 (raw residual stream)
static ggml_tensor * g_lx_norm_qkv_w   = nullptr;  // MUL weight vector
static ggml_tensor * g_lx_norm_qkv_mul = nullptr;  // the MUL node (== QKV src1)
static float g_lx_norm_qkv_eps  = 0.0f;
static int   g_lx_norm_qkv_wg   = 0;
static uint64_t g_lx_norm_qkv_gen = 0;
static uint64_t g_lx_graph_gen  = 0;
static void ggml_backend_sycl_bump_graph_gen() { g_lx_graph_gen++; }

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);
    ggml_backend_sycl_clear_satisfied();
    ggml_backend_sycl_bump_graph_gen();
    ggml_backend_sycl_graph_compute_impl_stamp_start();
    // [lx-ids-once] one generation per graph execution: ids content is fresh on
    // every compute, even when the same tensor objects are recycled (bench reps).
    // (SYCL graphs would re-record this impl, but MUL_MAT_ID models are hard-
    // rejected from graphs and the harness sets GGML_SYCL_DISABLE_GRAPH=1.)
    sycl_ctx->mmid_ids_memo_gen++;
    static const bool cksum_debug = []() {
        const char * e = getenv("GGML_SYCL_GRAPH_CHECKSUM");
        return e != nullptr && std::atoi(e) != 0;
    }();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_backend_sycl_satisfied(node)) {
            // Work already performed inside an earlier fused launch (V cache
            // publish). Skip the node; its dst was written by the fuse.
            continue;
        }

        ggml_backend_sycl_graph_compute_impl_stamp(node->op, false, node->name);
        static const bool dbg_sr = getenv("GGML_SYCL_DEBUG_SR") != nullptr;
        if (dbg_sr && node->op == GGML_OP_SET_ROWS) {
            fprintf(stderr, "[lx-dbg-sr] LAUNCH SET_ROWS name=%s src0=%s\n",
                    node->name ? node->name : "?", node->src[0] && node->src[0]->name ? node->src[0]->name : "?");
        }

        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            for (int s = 1; s <= nodes_to_skip; ++s) {
                if (i + s < cgraph->n_nodes) {
                    ggml_backend_sycl_graph_compute_impl_stamp(cgraph->nodes[i + s]->op, true, cgraph->nodes[i + s]->name);
                }
            }
            i += nodes_to_skip;
            if (cksum_debug) { ggml_sycl_debug_checksum_node(*sycl_ctx, cgraph->nodes[i], i); }
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
        if (cksum_debug) { ggml_sycl_debug_checksum_node(*sycl_ctx, node, i); }
    }
}

#ifdef GGML_SYCL_GRAPH
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            case GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(node_op));
                return false;
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
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    if (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        if (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } else {
            try {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
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
    /* .graph_optimize          = */ NULL,
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

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
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
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
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
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
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
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
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

                auto res = ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                         op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q5_0 ||
                         op->type == GGML_TYPE_Q1_0 ||
                         op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_IQ4_NL ||
                         op->type == GGML_TYPE_MXFP4 || op->type == GGML_TYPE_NVFP4) &&
                        op->src[0]->type == GGML_TYPE_F32 &&
                        (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32));
                return res;
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M) {
                        return false;
                    }
                }

                return true;
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
            return true;
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
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
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
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
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
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
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

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
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

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;
};

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    if (n_backends != 2) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    if (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        try {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
        } catch (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (n_backends != 2) {
        return false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    if (tensors[0]->type != tensors[1]->type) {
        return false;
    }
    if (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        return false;
    }
    if (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        return false;
    }

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. Cross-device copies
    // go through dev2dev_memcpy because the two devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op).
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        q0->wait();
        q1->wait();
        dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, f16_tmp0, tensors[1]->data, nbytes);
        dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        float * tmp0 = (float *) buf0;
        float * tmp1 = (float *) buf1;

        // COMM-D2D-FIX: the two devices are in SEPARATE SYCL contexts, so a raw
        // q->memcpy of a peer USM pointer is a silent no-op. Route cross-device
        // copies through dev2dev_memcpy (L0 direct copy / host staging). It is
        // synchronous, so wait for the local partials to be produced first.
        q0->wait();
        q1->wait();
        dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, tmp0, tensors[1]->data, nbytes);
        dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    sycl::event c0 = q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    sycl::event c1 = q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: COMM-D2D-FIX-BF16 cross-device copy of compressed bytes via
    // dev2dev_memcpy (separate SYCL contexts; sync copy after compress).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    c0.wait();
    c1.wait();
    dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, inbox0, outbox1, bf16_bytes);
    dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
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
