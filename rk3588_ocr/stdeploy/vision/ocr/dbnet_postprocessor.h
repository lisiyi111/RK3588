//
// Created by zhuzf on 2023/7/12.
//

#pragma once

#include "stdeploy/vision/common/image_processes/manager.h"
#include "stdeploy/vision/common/result.h"
#include "stdeploy/utils/log_util.h"
#include "stdeploy/vision/ocr/utils/ocr_postprocess_op.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            class STDEPLOY_DECL DBNetPostprocessor {
            public:
                DBNetPostprocessor(const std::string &config_file) {
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
                         PreprocessParams &pre_params,
                         OCRResult *result);

                /// Set det_db_thresh for the detection postprocess, default is 0.3
                void SetDetDBThresh(float det_db_thresh) { det_db_thresh_ = det_db_thresh; }

                /// Get det_db_thresh of the detection postprocess
                float GetDetDBThresh() const { return det_db_thresh_; }

                /// Set det_db_box_thresh for the detection postprocess, default is 0.6
                void SetDetDBBoxThresh(float det_db_box_thresh) {
                    det_db_box_thresh_ = det_db_box_thresh;
                }

                /// Get det_db_box_thresh of the detection postprocess
                float GetDetDBBoxThresh() const { return det_db_box_thresh_; }

                /// Set det_db_unclip_ratio for the detection postprocess, default is 1.5
                void SetDetDBUnclipRatio(float det_db_unclip_ratio) {
                    det_db_unclip_ratio_ = det_db_unclip_ratio;
                }

                /// Get det_db_unclip_ratio_ of the detection postprocess
                float GetDetDBUnclipRatio() const { return det_db_unclip_ratio_; }

                /// Set det_db_score_mode for the detection postprocess, default is 'slow'
                void SetDetDBScoreMode(const std::string &det_db_score_mode) {
                    det_db_score_mode_ = det_db_score_mode;
                }

                /// Get det_db_score_mode_ of the detection postprocess
                std::string GetDetDBScoreMode() const { return det_db_score_mode_; }

                /// Set use_dilation for the detection postprocess, default is fasle
                void SetUseDilation(int use_dilation) { use_dilation_ = use_dilation; }

                /// Get use_dilation of the detection postprocess
                int GetUseDilation() const { return use_dilation_; }

                bool initialized_ = false;

            private:

                bool ReadPostProcessFromConfig(const std::string &config_file);

                bool apply_f1c4_ = false;
                float det_db_box_thresh_ = 0.6;
                float det_db_thresh_ = 0.3;
                float det_db_unclip_ratio_ = 1.5;
                std::string det_db_score_mode_ = "fast";
                bool use_dilation_ = false;
                PostProcessor util_post_processor_;
            };

        } //namespace ocr
    } //namespace vision
} //namespace stdeploy