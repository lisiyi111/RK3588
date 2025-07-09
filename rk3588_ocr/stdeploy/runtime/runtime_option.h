/**
  *************************************************
  * @file               :runtime_option.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */

#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include "stdeploy/runtime/enum_variables.h"

#ifdef ENABLE_TRT_BACKEND
#include "stdeploy/runtime/backends/tensorrt/option.h"
#endif

#ifdef ENABLE_MSTAR_BACKEND
#include "stdeploy/runtime/backends/mstar/option.h"
#endif

#ifdef ENABLE_MNN_BACKEND
#include "stdeploy/runtime/backends/mnn/option.h"
#endif

#ifdef ENABLE_SNPE_BACKEND
#include "stdeploy/runtime/backends/snpe/option.h"
#endif

#ifdef ENABLE_RKNPU_BACKEND
#include "stdeploy/runtime/backends/rknpu/option.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "stdeploy/runtime/backends/rknpu2/option.h"
#endif

#ifdef ENABLE_ORT_BACKEND
#include "stdeploy/runtime/backends/onnxruntime/option.h"
#endif

#ifdef ENABLE_ACL_SVP_BACKEND
#include "stdeploy/runtime/backends/acl_svp/option.h"
#endif

#ifdef ENABLE_NCNN_BACKEND
#include "stdeploy/runtime/backends/ncnn/option.h"
#endif

namespace stdeploy {

    struct STDEPLOY_DECL RuntimeOption {

        /// Option to configure backend
#ifdef ENABLE_TRT_BACKEND
        TrtBackendOption trt_option;
#endif
#ifdef ENABLE_MSTAR_BACKEND
        MstarBackendOption mstar_option;
#endif
#ifdef ENABLE_MNN_BACKEND
        MnnBackendOption mnn_option;
#endif
#ifdef ENABLE_NCNN_BACKEND
        NcnnBackendOption ncnn_option;
#endif
#ifdef ENABLE_RKNPU2_BACKEND
        RKNPU2BackendOption rknpu2_option;
#endif
#ifdef ENABLE_RKNPU_BACKEND
        RKNPUBackendOption rknpu_option;
#endif
#ifdef ENABLE_SNPE_BACKEND
        SnpeBackendOption snpe_option;
#endif
#ifdef ENABLE_ORT_BACKEND
        ORTBackendOption ort_option;
#endif
#ifdef ENABLE_ACL_SVP_BACKEND
        AclSvpBackendOption acl_svp_option;
#endif

        void SetModelPath(const std::string &model_path,
                          const std::string &params_path = "",
                          const ModelFormat &format = ModelFormat::onnx);

        void SetModelBuffer(const std::string &model_buffer,
                            const std::string &params_buffer = "",
                            const ModelFormat &format = ModelFormat::onnx);

        /// Use cpu to inference, the runtime will inference on CPU by default
        void UseCPU();

        void UseMNN();

        void SetCpuThreadNum(int thread_num);

        /// Use Nvidia GPU to inference
        void UseGPU(int gpu_id = 0);

        /// Set TensorRT as inference backend, only support GPU
        void UseTensorRT();

        /// Use Mstar to inference
        void UseMstar();

        /// Use Snpe to inference
        void UseSnpe();

        /// Use rknpu2 to inference
        void UseRKNPU2();

        /// Use rknpu to inference
        void UseRKNPU();

        /// Use trt to inference
        void UseOnnxRuntime();

        /// Use acl svp to inference
        void UseAclSvp();

        void UseNCNN();

        std::string model_file;
        std::string params_file;
        int cpu_thread_num = -1;
        // format of input model
        ModelFormat model_format = ModelFormat::onnx;
        // default will let the backend choose their own default value
        int device_id = 0;
        Backend backend = Backend::UNKNOWN;
        Device device = Device::CPU;

        // enable the check for valid backend, default true.
        bool enable_valid_backend_check = true;
    };

} //name stdeploy
