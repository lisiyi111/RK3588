//
// Created by zhuzf on 2023/7/14.
//

#pragma once

#include "stdeploy/vision/common/image_processes/manager.h"
#include "stdeploy/vision/common/result.h"
#include "stdeploy/utils/log_util.h"
#include "stdeploy/vision/ocr/utils/ocr_postprocess_op.h"
#include "stdeploy/vision/utils/argmax.h"
#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            class STDEPLOY_DECL CRNNPostprocessor {
            public:
                CRNNPostprocessor(const std::string &config_file) {
                    if (!config_file.empty()) {
                        if (ReadPostProcessFromConfig(config_file)) {
                            initialized_ = true;
                        }
                    } else {
                        SDWARNING << "config file is empty,please check postprocess func callback" << std::endl;
                        initialized_ = true;
                    }
                    if (!this->ReadDict(label_path_)) {
                        SDWARNING << "please call SetLabelPath and EnableLabelDict or UpdateOCRDict to update ocr dict"
                                  << std::endl;
                    }
                }

                bool Run(std::vector<Tensor> &sd_tensors,
                         PreprocessParams &pre_params,
                         OCRResult *result);

                bool initialized_ = false;

                void SetStride(int stride) {
                    stride_ = stride;
                }

                int GetStride() const { return stride_; }

                void SetDecodeArch(const std::string &decode_arch) {
                    if (decode_arch == "f1c3") {
                        apply_f1c3_ = true;
                    } else if (decode_arch == "f2c3") {
                        apply_f2c3_ = true;
                    } else {
                        SDERROR << "Not support function: " << decode_arch << std::endl;
                    }
                }

                void SetLabelPath(const std::string &label_path) {
                    label_path_ = label_path;
                }

                void DisableSpaceChar() {
                    add_space_char_ = false;
                }

                void EnableSpaceChar() {
                    add_space_char_ = true;
                    if (add_space_char_) {
                        label_list_.emplace_back(" "); // append space_char
                    }
                }

                bool EnableLabelDict() {
                    return this->ReadDict(label_path_);
                }

                void UpdateOCRDict(std::vector<std::string> &ocr_dict) {
                    label_list_.clear();
                    label_list_.assign(ocr_dict.begin(), ocr_dict.end());
                    STDEPLOY_INFO("update ocr dict:[%zu]", label_list_.size());
                }


            protected:

                bool ReadDict(const std::string &path);

                bool apply_f1c3_ = false;

                bool apply_f2c3_ = false;

                bool add_space_char_ = false;

                std::string label_path_;

                std::vector<std::string> label_list_;

                int stride_ = 8;
            private:
                bool ReadPostProcessFromConfig(const std::string &config_file);

            };


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy