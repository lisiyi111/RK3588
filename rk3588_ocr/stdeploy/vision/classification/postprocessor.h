/**
  *************************************************
  * @file               :postprocess.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#pragma once

#include "stdeploy/vision/common/result.h"
#include "stdeploy/vision/utils/sigmoid.h"
#include "stdeploy/vision/utils/argmax.h"
#include "stdeploy/vision/utils/softmax.h"
#include "stdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace stdeploy {
    namespace vision {
        namespace classification {

            class STDEPLOY_DECL ClsPostprocessor {
            public:
                explicit ClsPostprocessor(int topK = 1) {
                    topk_value_ = topK;
                    initialized_ = true;
                }

                explicit ClsPostprocessor(const std::string &config_file);

                bool Run(std::vector <Tensor> &sd_tensors, ClassifyResult *result);

                void SetClsThresh(float cls_thresh) { cls_thresh_ = cls_thresh; }

                float GetClsThresh() const { return cls_thresh_; }

                void SetClsNumClass(int num_class) { num_class_ = num_class; }

                int GetClsNumClass() const { return num_class_; }

                void SetTopK(int topK) { topk_value_ = topK; }

                int GetTopK() const { return topk_value_; }

                void SetDecodeArch(const std::string &decode_arch) {
                    if (decode_arch == "single_label") {
                        apply_single_label_ = true;
                    } else if (decode_arch == "multi_label") {
                        apply_multi_label_ = true;
                    } else {
                        SDERROR << "Not support function: " << decode_arch << std::endl;
                    }
                }

                void SetApplySigmoid() {
                    add_sigmoid_ = true;
                }

                void SetApplySoftmax() {
                    add_softmax_ = true;
                }

            public:
                bool initialized_ = false;

            private:
                bool ReadPostProcessFromConfig(const std::string &config_file);

            private:
                float cls_thresh_ = 0.6;
                int num_class_ = 1000;
                int topk_value_ = 1;
                int add_softmax_ = false;
                int add_sigmoid_ = false;
                bool apply_multi_label_ = false;
                bool apply_single_label_ = false;
            };

        } //namespace classification
    } //namespace vision
} //namespace stdeploy