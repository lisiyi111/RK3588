/**
  *************************************************
  * @file               :normalize.h
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

        class STDEPLOY_DECL Normalize : public Processor {
        public:
            Normalize(const std::vector<float> &mean, const std::vector<float> &std,
                      float scale = 1 / 255.f, bool use_rgb = false, bool use_chw = false);

            bool ImplByOpenCV(sd::Mat &mat) override;

            std::string Name() { return "Normalize"; }

        private:
            std::vector<float> mean_;
            std::vector<float> std_;
            std::vector<float> alpha_;
            std::vector<float> beta_;
            float scale_;
            bool use_rgb_;
            bool use_chw_;
        };

    } //namespace vision
} //namespace stdeploy