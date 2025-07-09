/**
  *************************************************
  * @file               :option.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/8/7                
  *************************************************
  */

#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "stdeploy/utils/log_util.h"

namespace stdeploy {

    typedef enum _rknpu2_cpu_name {
        RK356X = 0, /* run on RK356X. */
        RK3588 = 1, /* run on RK3588. */
        RV1106 = 2, /* run on RV1106. */
        RK3576 = 3, /* run on RV1106. */
        RKNPU2_UNDEFINED,
    } RKNPU2CpuName;

    /* The specification of NPU core setting.It has the following choices :
     * RKNN_NPU_CORE_AUTO : Referring to automatic mode, meaning that it will
     * select the idle core inside the NPU.
     * RKNN_NPU_CORE_0 : Running on the NPU0 core.
     * RKNN_NPU_CORE_1: Runing on the NPU1 core.
     * RKNN_NPU_CORE_2: Runing on the NPU2 core.
     * RKNN_NPU_CORE_0_1: Running on both NPU0 and NPU1 core simultaneously.
     * RKNN_NPU_CORE_0_1_2: Running on both NPU0, NPU1 and NPU2 simultaneously.
     */
    typedef enum _rknpu2_core_mask {
        RKNN_NPU_CORE_AUTO = 0,
        RKNN_NPU_CORE_0 = 1,
        RKNN_NPU_CORE_1 = 2,
        RKNN_NPU_CORE_2 = 4,
        RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 | RKNN_NPU_CORE_2,
        RKNN_NPU_CORE_UNDEFINED,
    } RKNPU2CoreMask;


    struct STDEPLOY_DECL RKNPU2BackendOption {
        RKNPU2CpuName cpu_name = RKNPU2CpuName::RV1106;                     /* soc name. */
        RKNPU2CoreMask core_mask = RKNPU2CoreMask::RKNN_NPU_CORE_UNDEFINED; /* core mask. only support 3588 */
        bool shared_weight = false;                                         /* shared weight flag */
        uint64_t rk_context{0};                                             /* rk context using rknn_dup_context */
    };

} //namespace stdeploy




