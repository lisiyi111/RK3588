//
// Created by zhuzf on 2023/10/23.
//

#pragma once

#include "stdeploy/utils/log_util.h"
#include "stdeploy/core/sd_tensor.h"
#include "stdeploy/vision/common/vision_struct.h"
#include "stdeploy/vision/common/image_processes/manager.h"
#include "stdeploy/vision/common/image_processes/resize.h"
#include "stdeploy/vision/common/image_processes/normalize.h"

namespace stdeploy {
    namespace vision {

        class STDEPLOY_DECL BasePreprocessor : public ProcessorManger {
        public:
            /// init
            explicit BasePreprocessor(const std::string &config_file);

            bool Apply(sd::Mat &mat, sd::Tensor &sd_tensor);

            // defalut :bchw
            void SetInputImageShape(std::vector<int> &shape) {
                m_pre_params.src_batch = shape[0];
                m_pre_params.src_channel = shape[1];
                m_pre_params.src_height = shape[2];
                m_pre_params.src_width = shape[3];
            }

            // defalut :bchw
            void SetOutputImageShape(std::vector<int> &shape) {
                m_pre_params.dst_batch = shape[0];
                m_pre_params.dst_channel = shape[1];
                m_pre_params.dst_height = shape[2];
                m_pre_params.dst_width = shape[3];
            }

            // set the preprocess pipeline:such as resize+normal...
            void SetPreprocessPipeline() {
                base_pre_processors_.clear();
                base_pre_processors_.assign(this->pre_processors_.begin(), this->pre_processors_.end());
            }

            // get preprocess parmas to postprocess
            PreprocessParams GetPreprocessParams() {
                return m_pre_params;
            }

        public:
            bool initialized_ = false;
        private:
            std::vector<std::shared_ptr<Processor>> base_pre_processors_;
            PreprocessParams m_pre_params;
        };


    } //vision
} //stdeploy