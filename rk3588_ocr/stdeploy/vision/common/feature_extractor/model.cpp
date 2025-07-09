/**
  *************************************************
  * @file               :model.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/9/23                
  *************************************************
  */

#include "stdeploy/vision/common/feature_extractor/model.h"

namespace stdeploy {
    namespace vision {

        bool FeatureExtractPostprocessor::ReadPostProcessFromConfig(const std::string &config_file) {
            YAML::Node cfg;
            try {
                cfg = YAML::LoadFile(config_file);
            } catch (YAML::BadFile &e) {
                SDERROR << "Failed to load yaml file " << config_file
                        << ", maybe you should check this file." << std::endl;
                return false;
            }
            try {
                if (cfg["POSTPROCESS"].IsDefined()) {
                    for (const auto &op : cfg["POSTPROCESS"]) {
                        std::string op_name = op["type"].as<std::string>();
                        if (op_name == "f1c2") {
                            apply_f1c2_ = true;
                        } else {
                            SDERROR << "Not support function: " << op_name << std::endl;
                            return false;
                        }
                        if (op["feature_map_dim"].IsDefined()) {
                            feature_map_dim_ = op["feature_map_dim"].as<int>();
                        }
                    }
                }
            } catch (YAML::BadConversion &ce) {
                SDERROR << "Failed to convert yaml file " << std::endl;
                return false;
            }
            return true;
        }

        bool FeatureExtractPostprocessor::Run(std::vector<Tensor> &sd_tensors,
                                              PreprocessParams &pre_param,
                                              FeatureMapResult *result) {
            // copy the feature map to result
            auto feature_map_ptr = reinterpret_cast<float *>(sd_tensors[0].data());
            result->embedding.assign(feature_map_ptr, feature_map_ptr + feature_map_dim_);
            return true;
        }


        FeatureExtractor::FeatureExtractor(const std::string &model_file, const std::string &params_file,
                                           const RuntimeOption &custom_option, const ModelFormat &model_format,
                                           const std::string &config_file) {
            runtime_option = custom_option;
            runtime_option.model_format = model_format;
            runtime_option.model_file = model_file;
            runtime_option.params_file = params_file;
            config_file_ = config_file;
            initialized_ = this->Init(); // init model
        }

        bool FeatureExtractor::Init() {
            if (!InitRuntime()) {
                SDERROR << "Failed to initialize stdeploy backend." << std::endl;
                return false;
            }
            preprocessor_ = new BasePreprocessor(config_file_);
            postprocessor_ = new FeatureExtractPostprocessor(config_file_);
            if (!preprocessor_->initialized_ || !postprocessor_->initialized_) {
                SDERROR << "Failed to initialize preprocessor postprocessor from cfg file." << std::endl;
                return false;
            }
            return true;
        }

        bool FeatureExtractor::Predict(sd::Mat &img, FeatureMapResult *result) {
            bool ret;
            if (enable_record_time_) {
                tc_.Start();
            }
            // preprocess
            std::vector<Tensor> input_sd_tensors;
            Tensor input_sd_tensor;
            input_sd_tensor.desc = runtime_->GetInputInfo(0);
            memset(&input_sd_tensor.desc.mem, 0, sizeof(TensorMem));
            ret = preprocessor_->Apply(img, input_sd_tensor);
            input_sd_tensors.emplace_back(input_sd_tensor);
            if (!ret) {
                STDEPLOY_ERROR("preprocessor failed.");
                return ret;
            }
            if (enable_record_time_) {
                tc_.End();
                STDEPLOY_INFO("preprocessor time : %f ms", tc_.Duration());
                tc_.Start();
            }

            // forward
            std::vector<Tensor> output_sd_tensors;
            ret = Infer(input_sd_tensors, output_sd_tensors);
            if (!ret) {
                STDEPLOY_ERROR("forward failed.");
                return ret;
            }
            if (enable_record_time_) {
                tc_.End();
                STDEPLOY_INFO("forward time : %f ms", tc_.Duration());
                tc_.Start();
            }

            // postprocess
            PreprocessParams pre_params = preprocessor_->GetPreprocessParams();
            ret = postprocessor_->Run(output_sd_tensors, pre_params, result);
            if (!ret) {
                STDEPLOY_ERROR("postprocessor failed.");
                return ret;
            }
            if (enable_record_time_) {
                tc_.End();
                STDEPLOY_INFO("postprocessor time : %f ms", tc_.Duration());
            }
            return ret;
        }

        FeatureExtractor::~FeatureExtractor() {
            if (preprocessor_ != nullptr) {
                delete preprocessor_;
                preprocessor_ = nullptr;
            }
            if (postprocessor_ != nullptr) {
                delete postprocessor_;
                postprocessor_ = nullptr;
            }
        }

    } //namespace vision
} //namespace stdeploy
