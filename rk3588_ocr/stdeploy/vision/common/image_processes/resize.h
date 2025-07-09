/**
  *************************************************
  * @file               :resize.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/24                
  *************************************************
  */

#pragma once

#include "stdeploy/vision/common/image_processes/base.h"

namespace stdeploy {
    namespace vision {

        class STDEPLOY_DECL Resize : public Processor {

        public:
            Resize(int width, int height, bool use_pad = false, bool use_rgb = false, bool right_pad = false,
                   int resize_short = -1) {
                width_ = width;
                height_ = height;
                use_pad_ = use_pad; /// letterbox的pad
                use_rgb_ = use_rgb; /// to rgb
                right_pad_ = right_pad; /// ocr的pad,往右边padding
                resize_short_ = resize_short; /// 分类常用的短边缩放再中心裁剪
            }

            bool SetWidthAndHeight(int width, int height) {
                width_ = width;
                height_ = height;
                return true;
            }

            std::tuple<int, int> GetWidthAndHeight() {
                return std::make_tuple(width_, height_);
            }

            std::string Name() { return "Resize"; }

            // only no pad, pad is support ImplByOpenCV(Mat &mat, Tensor &tensor, vision::PreprocessParams *params)
            bool ImplByOpenCV(sd::Mat &mat) override;

            bool ImplByOpenCV(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params) override;

#if  defined(ENABLE_RKNPU2_BACKEND)

            bool ImplByRKRga(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params) override;

#endif

#if  defined(ENABLE_MSTAR_BACKEND)

            bool ImplByMstarSCL(sd::Mat &mat, sd::Tensor &tensor, vision::PreprocessParams *params) override;

#endif
        private:
            int width_;
            int height_;
            bool use_pad_ = false;
            bool use_rgb_ = false;
            bool right_pad_ = false;
            int resize_short_ = -1;
        };


    } //namespace vision
} //namespace stdeploy