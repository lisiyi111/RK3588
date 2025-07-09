/**
  *************************************************
  * @file               :text_direction_cls.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#include "text_angle_cls.h"


namespace stdeploy {
    namespace vision {
        namespace ocr {


            TextAngleCls::TextAngleCls(const std::string &model_file, const std::string &params_file,
                                       const RuntimeOption &custom_option, const ModelFormat &model_format,
                                       const std::string &config_file) {
                runtime_option = custom_option;
                runtime_option.model_format = model_format;
                runtime_option.model_file = model_file;
                runtime_option.params_file = params_file;
                config_file_ = config_file;
                initialized_ = this->Init();
                // 预处理
            }

            bool TextAngleCls::Predict(sd::Mat &img, int32_t *cls_label, float *cls_score) {
                bool ret = false;
                if (enable_record_time_) {
                    tc_.Start();
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
                // the output tensor shape = [batch,2],batch =1
                auto *scores_ptr = reinterpret_cast<float *>(output_sd_tensors[0].data());

                //// 使用 std::max_element 来找到最大值的迭代器（实际上是指向 float 的指针）
                auto maxIt = std::max_element(scores_ptr, scores_ptr + num_classes_);
                *cls_label = std::distance(scores_ptr, maxIt);
                *cls_score = *maxIt;
                if (enable_record_time_) {
                    tc_.End();
                    STDEPLOY_INFO("postprocessor time : %f ms", tc_.Duration());
                }
                return ret;
            }

            bool TextAngleCls::Init() {
                if (!InitRuntime()) {
                    SDERROR << "Failed to initialize stdeploy backend." << std::endl;
                    return false;
                }
                preprocessor_ = new BasePreprocessor(this->config_file_);
                if (!preprocessor_->initialized_) {
                    SDERROR << "Failed to initialize preprocessor from cfg file." << std::endl;
                    SDERROR << "initialize preprocessor : " << preprocessor_->initialized_ << std::endl;
                    return false;
                }
                return true;
            }

            TextAngleCls::~TextAngleCls() {
                if (preprocessor_ != nullptr) {
                    delete preprocessor_;
                    preprocessor_ = nullptr;
                }
            }


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy