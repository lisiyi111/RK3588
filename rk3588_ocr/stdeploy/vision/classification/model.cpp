/**
  *************************************************
  * @file               :model.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#include "stdeploy/vision/classification/model.h"

namespace stdeploy {
    namespace vision {
        namespace classification {

            ClsModel::ClsModel(const std::string &model_file,
                               const std::string &params_file,
                               const RuntimeOption &custom_option,
                               const ModelFormat &model_format,
                               const std::string &config_file) {
                runtime_option = custom_option;
                runtime_option.model_format = model_format;
                runtime_option.model_file = model_file;
                runtime_option.params_file = params_file;
                this->config_file_ = config_file;
                initialized_ = this->Init();
            }

            bool ClsModel::Init() {
                if (!InitRuntime()) {
                    SDERROR << "Failed to initialize stdeploy backend." << std::endl;
                    return false;
                }
                preprocessor_ = new BasePreprocessor(this->config_file_);
                postprocessor_ = new ClsPostprocessor(this->config_file_);
                if (!preprocessor_->initialized_ || !postprocessor_->initialized_) {
                    SDERROR << "Failed to initialize preprocessor postprocessor from cfg file." << std::endl;
                    SDERROR << "initialize preprocessor : " << preprocessor_->initialized_ << std::endl;
                    SDERROR << "initialize postprocessor : " << postprocessor_->initialized_ << std::endl;
                    return false;
                }
                return true;
            }

            bool ClsModel::Predict(sd::Mat &img, ClassifyResult *result) {
                bool ret = false;
                if (enable_record_time_) {
                    tc_.Start();
                    STDEPLOY_INFO("***Start preprocess***");
                }

                // preprocess
                std::vector<Tensor> input_sd_tensors;
                Tensor input_tensor;
                input_tensor.desc = runtime_->GetInputInfo(0);
                ret = preprocessor_->Apply(img, input_tensor);
                input_sd_tensors.emplace_back(input_tensor);
                if (!ret) {
                    STDEPLOY_ERROR("preprocessor failed.");
                    return ret;
                }
                if (enable_record_time_) {
                    tc_.End();
                    time_of_preprocess_.push_back(tc_.Duration());
                    STDEPLOY_INFO("Preprocessor time : %f ms", tc_.Duration());
                    STDEPLOY_INFO("***End preprocess***");
                    STDEPLOY_INFO("***Start forward***");
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
                    time_of_runtime_.push_back(tc_.Duration());
                    STDEPLOY_INFO("Forward time : %f ms", tc_.Duration());
                    STDEPLOY_INFO("***End forward***");
                    STDEPLOY_INFO("***Start postprocess***");
                    tc_.Start();
                }

                // postprocess
                ret = postprocessor_->Run(output_sd_tensors, result);
                if (!ret) {
                    STDEPLOY_ERROR("postprocessor failed.");
                    return ret;
                }
                if (enable_record_time_) {
                    tc_.End();
                    time_of_postprocessor_.push_back(tc_.Duration());
                    STDEPLOY_INFO("Postprocess time : %f ms", tc_.Duration());
                    STDEPLOY_INFO("***End postprocess***");
                    if (time_of_runtime_.size() > 3000) {
                        SDWARNING << "There are already 3000 records of runtime, will force to "
                                     "disable record time of runtime now."
                                  << std::endl;
                        enable_record_time_ = false;
                    }
                }
                return ret;
            }

            ClsModel::~ClsModel() {
                if (preprocessor_ != nullptr) {
                    delete preprocessor_;
                    preprocessor_ = nullptr;
                }
                if (postprocessor_ != nullptr) {
                    delete postprocessor_;
                    postprocessor_ = nullptr;
                }
            }

        } //namespace classification
    } //namespace vision
} //namespace stdeploy