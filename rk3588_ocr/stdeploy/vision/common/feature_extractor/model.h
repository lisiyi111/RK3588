/**
  *************************************************
  * @file               :model.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/9/23                
  *************************************************
  */

#pragma once

#include "stdeploy/utils/log_util.h"
#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/common/result.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"
#include "yaml-cpp/yaml.h"

namespace stdeploy {
    namespace vision {

        class STDEPLOY_DECL FeatureExtractPostprocessor {
        public:
            explicit FeatureExtractPostprocessor(const std::string &config_file) {
                if (!config_file.empty()) {
                    if (this->ReadPostProcessFromConfig(config_file)) {
                        initialized_ = true;
                    }
                } else {
                    SDWARNING << "config file is empty,please check postprocess func callback" << std::endl;
                    initialized_ = true;
                }
            }

            bool Run(std::vector<Tensor> &sd_tensors,
                     PreprocessParams &pre_param,
                     FeatureMapResult *result);

            bool initialized_ = false;

            void SetDecodeArch(const std::string &decode_arch) {
                if (decode_arch == "f1c2") {
                    apply_f1c2_ = true;
                } else {
                    SDERROR << "Not support function: " << decode_arch << std::endl;
                }
            }

            void SetFeatureMapDim(int dim) {
                feature_map_dim_ = dim;
            }

            int GetFeatureMapDim() const {
                return feature_map_dim_;
            }

        private:

            bool ReadPostProcessFromConfig(const std::string &config_file);

            int feature_map_dim_ = 512;
            bool apply_f1c2_ = false;

        };

        class STDEPLOY_DECL FeatureExtractor : public StDeployModel {
        public:
            explicit FeatureExtractor(const std::string &model_file,
                                      const std::string &params_file = "",
                                      const RuntimeOption &custom_option = RuntimeOption(),
                                      const ModelFormat &model_format = ModelFormat::onnx,
                                      const std::string &config_file = "");

            ~FeatureExtractor() override;

            std::string ModelName() const override { return "FeatureExtractor"; }

            bool Predict(sd::Mat &img, FeatureMapResult *result);

            /// Get preprocessor reference of DBDetectorPreprocessor
            BasePreprocessor &GetPreprocessor() {
                return *preprocessor_;
            }

            /// Get postprocessor reference of DBDetectorPostprocessor
            FeatureExtractPostprocessor &GetPostprocessor() {
                return *postprocessor_;
            }

        protected:
            bool Init();

        private:
            std::string config_file_;
            BasePreprocessor *preprocessor_ = nullptr;
            FeatureExtractPostprocessor *postprocessor_ = nullptr;
        };

    } //namespace vision
} //namespace stdeploy


