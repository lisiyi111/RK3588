//
// Created by zhuzf on 2023/7/11.
//

#pragma once

#include "stdeploy/utils/log_util.h"
#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/ocr/dbnet_postprocessor.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            class STDEPLOY_DECL DBNet : public StDeployModel {
            public:

                DBNet(const std::string &model_file,
                      const std::string &params_file = "",
                      const RuntimeOption &custom_option = RuntimeOption(),
                      const ModelFormat &model_format = ModelFormat::onnx,
                      const std::string &config_file = "");

                ~DBNet() override;

                std::string ModelName() const override { return "DBNet"; }

                bool Predict(sd::Mat &img, OCRResult *result);

                BasePreprocessor &GetPreprocessor() {
                    return *preprocessor_;
                }

                DBNetPostprocessor &GetPostprocessor() {
                    return *postprocessor_;
                }

            private:
                bool Init();

                std::string config_file_;
                BasePreprocessor *preprocessor_ = nullptr;
                DBNetPostprocessor *postprocessor_ = nullptr;

            };

        } //namespace ocr
    } //namespace vision
} //namespace stdeploy