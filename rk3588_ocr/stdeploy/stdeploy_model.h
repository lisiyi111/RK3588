/**
  *************************************************
  * @file               :stdeploy_model.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/6                
  *************************************************
  */

#pragma once

#include "stdeploy/runtime.h"
#include "stdeploy/utils/perf.h"
#include "stdeploy/utils/file_util.h"

namespace stdeploy {

    /**
     * @brief Base model object for all the vision models
     */
    class STDEPLOY_DECL StDeployModel {
    public:
        StDeployModel() = default;

        //// 虚析构，注意一般基类的析构函数都要用virtual来修饰，这样子在delete基类时，会自底向上取调用即先调用派生类的虚构函数最后到基类
        virtual ~StDeployModel() = default;

        // Get model name
        virtual std::string ModelName() const { return "NameUndefined"; }

        // infer
        virtual bool Infer(std::vector<Tensor> &input_tensors,
                           std::vector<Tensor> &output_tensors);

        // Get number of inputs for this model
        virtual int NumInputsOfRuntime() { return runtime_->NumInputs(); }

        // Get number of outputs for this model
        virtual int NumOutputsOfRuntime() { return runtime_->NumOutputs(); }

        // Get input information for this model by index
        virtual TensorDesc InputInfoOfRuntime(int index) {
            return runtime_->GetInputInfo(index);
        }

        // Get output information for this model by index
        virtual TensorDesc OutputInfoOfRuntime(int index) {
            return runtime_->GetOutputInfo(index);
        }

        // Check if the model is initialized successfully
        virtual bool Initialized() const {
            return runtime_initialized_ && initialized_;
        }

        // enable benchmark timer
        void EnableRecordTime() {
            time_of_runtime_.clear();
            time_of_preprocess_.clear();
            time_of_postprocessor_.clear();
            enable_record_time_ = true;
            std::vector<double>().swap(time_of_preprocess_);
            std::vector<double>().swap(time_of_runtime_);
            std::vector<double>().swap(time_of_postprocessor_);
        }

        // disable benchmark timer
        void DisableRecordTime() {
            enable_record_time_ = false;
        }

        // print benchmark timer
        std::map<std::string, float> PrintBenchmark();

        RuntimeOption runtime_option;
    protected:
        // whether to record inference time, has {preprocess_time,forward_time,postprocessor_time}
        std::vector<double> time_of_runtime_;
        std::vector<double> time_of_preprocess_;
        std::vector<double> time_of_postprocessor_;
        bool enable_record_time_ = false;
        TimeCount tc_;
        int benchmark_number_ = 200;
        // init runtime
        virtual bool InitRuntime();
        // tag for init
        bool initialized_ = false;
        // runtime ptr
        std::shared_ptr<Runtime> runtime_;

    private:
        // runtime status
        bool runtime_initialized_ = false;

    };

} //namespace stdeploy
