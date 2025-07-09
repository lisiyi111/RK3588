/**
  *************************************************
  * @file               :vis_cls.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/2/23                
  *************************************************
  */
#pragma once

#include "stdeploy/vision/visualize/vis_base.h"

namespace stdeploy {
    namespace vision {

        STDEPLOY_DECL cv::Mat VisClassification(const cv::Mat &im,
                                                const ClassifyResult &result,
                                                int top_k = 5,
                                                float score_threshold = 0.0f,
                                                float font_size = 0.5f);

        STDEPLOY_DECL cv::Mat VisClassification(const cv::Mat &im,
                                                const ClassifyResult &result,
                                                const std::vector<std::string> &labels,
                                                int top_k = 5,
                                                float score_threshold = 0.0f,
                                                float font_size = 0.5f);

    } //namespace vision
} //namespace stdeploy
