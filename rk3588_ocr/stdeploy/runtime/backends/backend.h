/**
  *************************************************
  * @file               :backend.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/7                
  *************************************************
  */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "stdeploy/runtime/runtime_option.h"
#include "stdeploy/core/sd_tensor.h"
#include "stdeploy/utils/log_util.h"

namespace stdeploy {

    /**
     * @brief Base backend for all the infer backends
     */
    class STDEPLOY_DECL BaseBackend {
    public:
        bool initialized_ = false;

        BaseBackend() = default;

        // virtual 虚函数,派生类通过重写虚函数来实现对基类虚函数的覆盖
        virtual ~BaseBackend() = default;

        // Get init tag
        bool Initialized() const { return initialized_; }

        // Init backend
        // virtual xxx = 0; 纯虚函数，派生类必须实现该纯虚函数 ,派生类中可以加上override 进一步检查
        virtual bool Init(const RuntimeOption &option) = 0;

        // Forward
        virtual bool Infer(std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) = 0;

        // Get number of inputs of the model
        int NumInputs() const { return static_cast<int>(inputs_desc_.size()); }

        // Get number of outputs of the model
        int NumOutputs() const { return static_cast<int>(outputs_desc_.size()); }

        // Get information of input tensor
        TensorDesc GetInputInfo(int index) { return inputs_desc_[index]; };

        // Get information of output tensor
        TensorDesc GetOutputInfo(int index) { return outputs_desc_[index]; };

        // Get information of all the input tensors
        std::vector<TensorDesc> GetInputInfos() { return inputs_desc_; };

        // Get information of all the output tensors
        std::vector<TensorDesc> GetOutputInfos() { return outputs_desc_; };

    protected:
        // input tensor desc
        std::vector<TensorDesc> inputs_desc_;
        // output tensor desc
        std::vector<TensorDesc> outputs_desc_;

    };

} //namespace stdeploy