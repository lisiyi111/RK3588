/**
  *************************************************
  * @file               :stdeploy_model.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#include "stdeploy/stdeploy_model.h"

namespace stdeploy {

    bool StDeployModel::InitRuntime() {
        if (runtime_initialized_) {
            SDERROR << "The model is already initialized, cannot be initliazed again."
                    << std::endl;
            return false;
        }
        if (runtime_option.backend != Backend::UNKNOWN) {
            if (!IsBackendAvailable(runtime_option.backend)) {
                SDERROR << runtime_option.backend
                        << " is not compiled with current StDeploy library." << std::endl;
                return false;
            }
            runtime_ = std::shared_ptr<Runtime>(new Runtime());
            if (!runtime_->Init(runtime_option)) {
                return false;
            }
            runtime_initialized_ = true;
            return true;
        } else {
            SDERROR << runtime_option.backend
                    << " is not set." << std::endl;
            return false;
        }
    }


    bool StDeployModel::Infer(std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) {
        auto ret = runtime_->Infer(input_tensors, output_tensors);
        return ret;
    }

    std::map<std::string, float> StDeployModel::PrintBenchmark() {
        std::map<std::string, float> benchmark_info_dict;
        if (time_of_runtime_.empty()) {
            STDEPLOY_WARNING("PrintBenchmark require the runtime ran 10 times at least,please check model func");
            return benchmark_info_dict;
        }
        if (time_of_runtime_.size() < 10) {
            SDWARNING << "PrintBenchmark require the runtime ran 10 times at "
                         "least, but now you only ran "
                      << time_of_runtime_.size() << " times." << std::endl;
        }
        double warmup_time_runtime = 0.0;
        double warmup_time_preprocess = 0.0;
        double warmup_time_postprocess = 0.0;
        double remain_runtime_time = 0.0;
        double remain_preprocess_time = 0.0;
        double remain_postprocess_time = 0.0;
        int warmup_iter = 5;
        // runtime
        for (size_t i = 0; i < time_of_runtime_.size(); ++i) {
            if (i < warmup_iter) {
                warmup_time_runtime += time_of_runtime_[i];
            } else {
                remain_runtime_time += time_of_runtime_[i];
            }
        }
        double avg_runtime_time = remain_runtime_time / (time_of_runtime_.size() - warmup_iter);

        // preprocess
        for (size_t i = 0; i < time_of_preprocess_.size(); ++i) {
            if (i < warmup_iter) {
                warmup_time_preprocess += time_of_preprocess_[i];
            } else {
                remain_preprocess_time += time_of_preprocess_[i];
            }
        }
        double avg_preprocess_time = remain_preprocess_time / (time_of_preprocess_.size() - warmup_iter);

        // postprocess
        for (size_t i = 0; i < time_of_postprocessor_.size(); ++i) {
            if (i < warmup_iter) {
                warmup_time_postprocess += time_of_postprocessor_[i];
            } else {
                remain_postprocess_time += time_of_postprocessor_[i];
            }
        }
        double avg_postprocess_time = remain_postprocess_time / (time_of_postprocessor_.size() - warmup_iter);

        double avg_time = avg_postprocess_time + avg_preprocess_time + avg_runtime_time;

        std::cout << "============= Runtime benchmark Info(" << ModelName()
                  << ") =============" << std::endl;
        std::cout << "Total iterations: " << time_of_runtime_.size() << std::endl;
        std::cout << "Total time of preprocess: " << warmup_time_preprocess + remain_preprocess_time << "ms."
                  << std::endl;
        std::cout << "Total time of runtime: " << warmup_time_runtime + remain_runtime_time << "ms."
                  << std::endl;
        std::cout << "Total time of postprocess: " << warmup_time_postprocess + remain_postprocess_time << "ms."
                  << std::endl;
        std::cout << "Warmup iterations: " << warmup_iter << std::endl;
        std::cout << "Average time exclude warmup step: " << avg_time << "ms." << std::endl;
        std::cout << "Average time of preprocess exclude warmup step: "
                  << avg_preprocess_time << "ms." << std::endl;
        std::cout << "Average time of runtime exclude warmup step: "
                  << avg_runtime_time << "ms." << std::endl;
        std::cout << "Average time of postprocess exclude warmup step: "
                  << avg_postprocess_time << "ms." << std::endl;
        std::cout << "Average fps of runtime exclude warmup step: "
                  << float(1000) / avg_runtime_time << "." << std::endl;

        benchmark_info_dict["iterations"] = time_of_runtime_.size();
        benchmark_info_dict["warmup_iter"] = warmup_iter;
        benchmark_info_dict["remain_preprocess_time"] = remain_preprocess_time;
        benchmark_info_dict["remain_runtime_time"] = remain_runtime_time;
        benchmark_info_dict["remain_postprocess_time"] = remain_postprocess_time;
        benchmark_info_dict["avg_preprocess_time"] = avg_preprocess_time;
        benchmark_info_dict["avg_runtime_time"] = avg_runtime_time;
        benchmark_info_dict["avg_postprocess_time"] = avg_postprocess_time;
        return benchmark_info_dict;
    }


} //namespace stdeploy