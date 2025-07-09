/**
  *************************************************
  * @file               :base.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#pragma once

#include "stdeploy/utils/log_util.h"
#include "stdeploy/vision/common/mat.h"
#include "stdeploy/core/sd_tensor.h"
#include "stdeploy/vision/common/vision_struct.h"

namespace sd = stdeploy;

namespace stdeploy {
    namespace vision {

        /*! @brief
         * Base class Processor for image process, such as preprocess、postprocess.
         */
        class STDEPLOY_DECL Processor {
        public:
            virtual std::string Name() = 0;

            virtual bool operator()(sd::Mat &mat);

            virtual bool operator()(sd::Mat &mat, sd::Tensor &tensor);

            virtual bool operator()(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params);

            virtual bool ImplByOpenCV(sd::Mat &mat);

            virtual bool ImplByOpenCV(sd::Mat &mat, sd::Tensor &tensor);

            virtual bool ImplByOpenCV(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params);

            virtual bool ImplByRKRga(sd::Mat &mat);

            virtual bool ImplByRKRga(sd::Mat &mat, sd::Tensor &tensor);

            virtual bool ImplByRKRga(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params);

            virtual bool ImplByMstarSCL(sd::Mat &mat);

            virtual bool ImplByMstarSCL(sd::Mat &mat, sd::Tensor &tensor);

            virtual bool ImplByMstarSCL(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params);

        };


    } //namespace vision
} //namespace stdeploy