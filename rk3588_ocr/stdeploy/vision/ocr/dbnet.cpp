//
// Created by zhuzf on 2023/7/11.
//


#include "stdeploy/vision/ocr/dbnet.h"


namespace stdeploy {
    namespace vision {
        namespace ocr {

            DBNet::DBNet(const std::string &model_file, const std::string &params_file,
                         const RuntimeOption &custom_option,
                         const ModelFormat &model_format, const std::string &config_file) {
                runtime_option = custom_option;
                runtime_option.model_format = model_format;
                runtime_option.model_file = model_file;
                runtime_option.params_file = params_file;
                config_file_ = config_file;
                initialized_ = this->Init();
            }

            bool DBNet::Init() {
                if (!InitRuntime()) {
                    SDERROR << "Failed to initialize stdeploy backend." << std::endl;
                    return false;
                }
                preprocessor_ = new BasePreprocessor(config_file_);
                postprocessor_ = new DBNetPostprocessor(config_file_);
                if (!preprocessor_->initialized_ || !postprocessor_->initialized_) {
                    SDERROR << "Failed to initialize preprocessor postprocessor from cfg file." << std::endl;
                    SDERROR << "initialize preprocessor : " << preprocessor_->initialized_ << std::endl;
                    SDERROR << "initialize postprocessor : " << postprocessor_->initialized_ << std::endl;
                    return false;
                }
                return true;
            }


            bool DBNet::Predict(sd::Mat &img, OCRResult *result) {
                bool ret = false;
                if (enable_record_time_) {
                    tc_.Start();
                }

                // preprocess
                std::vector<Tensor> input_sd_tensors;
                Tensor input_sd_tensor;
                input_sd_tensor.desc = runtime_->GetInputInfo(0);
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

            DBNet::~DBNet() {
                if (preprocessor_ != nullptr) {
                    delete preprocessor_;
                    preprocessor_ = nullptr;
                }
                if (postprocessor_ != nullptr) {
                    delete postprocessor_;
                    postprocessor_ = nullptr;
                }
            }


        } //namespace stdeploy
    } //namespace vision
} //namespace ocr
