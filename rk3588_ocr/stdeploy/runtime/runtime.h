/**
  *************************************************
  * @file               :runtime.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */

#pragma once

#include "stdeploy/runtime/runtime_option.h"
#include "stdeploy/runtime/backends/backend.h"
#include "stdeploy/core/sd_tensor.h"
#include "stdeploy/utils/unique_ptr.h"

#ifdef ENABLE_TRT_BACKEND
#include "stdeploy/runtime/backends/tensorrt/trt_backend.h"
#endif

#ifdef ENABLE_MSTAR_BACKEND
#include "stdeploy/runtime/backends/mstar/mstar_backend.h"
#endif

#ifdef ENABLE_MNN_BACKEND
#include "stdeploy/runtime/backends/mnn/mnn_backend.h"
#endif

#ifdef ENABLE_NCNN_BACKEND
#include "stdeploy/runtime/backends/ncnn/ncnn_backend.h"
#endif

#ifdef ENABLE_SNPE_BACKEND
#include "stdeploy/runtime/backends/snpe/snpe_backend.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "stdeploy/runtime/backends/rknpu2/rknpu2_backend.h"
#endif

#ifdef ENABLE_RKNPU_BACKEND
#include "stdeploy/runtime/backends/rknpu/rknpu_backend.h"
#endif

#ifdef ENABLE_ORT_BACKEND
#include "stdeploy/runtime/backends/onnxruntime/onnx_backend.h"
#endif

#ifdef ENABLE_ACL_SVP_BACKEND
#include "stdeploy/runtime/backends/acl_svp/acl_svp_backend.h"
#endif

namespace stdeploy {

    struct STDEPLOY_DECL Runtime {
    public:
        // base function to init by opt
        bool Init(const RuntimeOption &_option);

        // base function to infer by tensor
        bool Infer(std::vector<Tensor> &input_tensors,
                   std::vector<Tensor> &output_tensors);

        // get input tensor num
        int NumInputs() { return backend_->NumInputs(); }

        // get output tensor num
        int NumOutputs() { return backend_->NumOutputs(); }

        TensorDesc GetInputInfo(int index);

        TensorDesc GetOutputInfo(int index);

        // get input tensor desc
        std::vector<TensorDesc> GetInputInfos();

        // get output tensor desc
        std::vector<TensorDesc> GetOutputInfos();

        // base option
        RuntimeOption option;

    private:
        int CreateTrtBackend();

        int CreateMstarBackend();

        int CreateMnnBackend();

        int CreateNcnnBackend();

        int CreateSnpeBackend();

        int CreateRknpu2Backend();

        int CreateRknpuBackend();

        int CreateORTBackend();

        int CreateAclSvpBackend();

        std::unique_ptr<BaseBackend> backend_; // ptr from backend

    };

} //namespace stdeploy