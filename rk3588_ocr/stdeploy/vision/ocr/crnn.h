/**
  *************************************************
  * @file               :crnn.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/12/27
  *************************************************
  */

#pragma once

#include "stdeploy/utils/log_util.h"
#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/common/result.h"
#include "stdeploy/vision/ocr/crnn_postprocessor.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            class STDEPLOY_DECL CRNN : public StDeployModel {
            public:

                CRNN(const std::string &model_file,
                     const std::string &params_file = "",
                     const RuntimeOption &custom_option = RuntimeOption(),
                     const ModelFormat &model_format = ModelFormat::onnx,
                     const std::string &config_file = "");

                ~CRNN() override;

                std::string ModelName() const override { return "CRNN"; }

                virtual bool Predict(sd::Mat &img, OCRResult *result);

                BasePreprocessor &GetPreprocessor() {
                    return *preprocessor_;
                }

                CRNNPostprocessor &GetPostprocessor() {
                    return *postprocessor_;
                }

            protected:
                std::string config_file_;
                BasePreprocessor *preprocessor_ = nullptr;
                CRNNPostprocessor *postprocessor_ = nullptr;
                bool Init();

            };


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy
