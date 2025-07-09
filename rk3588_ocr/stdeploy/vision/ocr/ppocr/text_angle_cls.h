/**
  *************************************************
  * @file               :text_direction_cls.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#pragma once


#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            class STDEPLOY_DECL TextAngleCls : public StDeployModel {

            public:
                explicit TextAngleCls(const std::string &model_file,
                                      const std::string &params_file = "",
                                      const RuntimeOption &custom_option = RuntimeOption(),
                                      const ModelFormat &model_format = ModelFormat::onnx,
                                      const std::string &config_file = "");
                ~TextAngleCls() override;

                std::string ModelName() const override { return "TextAngleCls"; }

                bool Predict(sd::Mat &img, int32_t *cls_label, float *cls_score);

            private:
                bool Init();

                std::string config_file_;
                BasePreprocessor *preprocessor_ = nullptr;
                std::vector<int> cls_image_shape_ = {48, 192};
                int num_classes_ = 2; //0° or 180°
            };


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy

