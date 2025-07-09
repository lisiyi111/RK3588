//
// Created by zhuzf on 2023/10/23.
//

#include "stdeploy/vision/common/image_processes/preprocessor.h"

namespace stdeploy {
    namespace vision {

        BasePreprocessor::BasePreprocessor(const std::string &config_file) {
            if (!config_file.empty()) {
                bool ret = this->BuildProcessorPipelineFromConfig(config_file);
                if (ret) {
                    base_pre_processors_.clear();
                    base_pre_processors_.assign(this->pre_processors_.begin(), this->pre_processors_.end());
                    initialized_ = true;
                } else {
                    SDERROR << "Init preprocess pipeline from config failed." << std::endl;
                    initialized_ = false;
                }
            } else {
                SDWARNING << "config file is empty,please check preprocess func callback" << std::endl;
                initialized_ = true;
            }
        }

        bool BasePreprocessor::Apply(sd::Mat &mat, sd::Tensor &sd_tensor) {
            // preprocess from yaml pipeline
            bool is_resize = false;
            memset(&m_pre_params, 0, sizeof(PreprocessParams));
            for (auto &pre_processor : base_pre_processors_) {
                std::string pre_process_name = pre_processor->Name();
                if (pre_process_name == "Resize") {
                    if (!(*(pre_processor))(mat, sd_tensor, &m_pre_params)) {
                        SDERROR << "Failed to process image in " << pre_processor->Name() << "."
                                << std::endl;
                        return false;
                    }
                    is_resize = true;
                } else {
                    if (!(*(pre_processor))(mat)) {
                        SDERROR << "Failed to process image in " << pre_processor->Name() << "."
                                << std::endl;
                        return false;
                    }
                }
            }
            if (mat.pro_lib() == ProcLib::OPENCV) {
                /// 目前的设计逻辑，如果预处理使用opencv，则不用物理地址的方式传递数据，TODO: 怎么优化
                memset(&sd_tensor.desc.mem, 0, sizeof(TensorMem));
                sd_tensor.set_data(mat.data());  // last set vir data
                sd_tensor.desc.format = mat.pixel_format();
                /// 针对动态shape,用最后mat的shape
                std::vector<int> shape = {mat.batch_size(), mat.channel(), mat.height(), mat.width()};
                sd_tensor.set_shape(shape);  // last set vir data
                /// 如果没有pipeline,考虑预处理都是在外面做的，主要考虑resize的参数赋值
                if (!is_resize) {
                    m_pre_params.src_height = shape[2];
                    m_pre_params.src_width = shape[3];
                    m_pre_params.dst_height = shape[2];
                    m_pre_params.dst_width = shape[3];
                    m_pre_params.scale_width = 1.0;
                    m_pre_params.scale_height = 1.0;
                }
            }
            return true;
        }


    } //namespace vision
} //namespace stdeploy
