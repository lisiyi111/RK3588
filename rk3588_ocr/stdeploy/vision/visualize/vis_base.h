/**
  *************************************************
  * @file               :vis_base.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/1/15                
  *************************************************
  */

#pragma once

#include <opencv2/opencv.hpp>
#include "stdeploy/vision/common/result.h"

namespace stdeploy {
    namespace vision {

        STDEPLOY_DECL std::vector<int> GenerateColorMap(int num_classes);

    } //namespace vision
} //namespace stdeploy