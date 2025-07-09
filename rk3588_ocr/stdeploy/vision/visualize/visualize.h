//
// Created by zhuzf on 2023/4/28.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "stdeploy/vision/common/result.h"
#include "stdeploy/vision/visualize/vis_cls.h"

namespace stdeploy {
    namespace vision {


        STDEPLOY_DECL cv::Mat VisDepth(const cv::Mat &im,
                                       const DepthResult &result);



        STDEPLOY_DECL cv::Mat VisOcr(const cv::Mat &im, const OCRResult &result);


        STDEPLOY_DECL cv::Mat VisKeypointDetection(const cv::Mat &im, const KeyPointDetectionResult &results,
                                                   float conf_threshold = 0.5f);


    } //namespace vision
} //namespace stdeploy
