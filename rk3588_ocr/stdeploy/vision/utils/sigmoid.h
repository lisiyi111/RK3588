/**
  *************************************************
  * @file               :sigmoid.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#pragma once

#include <math.h>
#include <numeric>
#include "stdeploy/utils/log_util.h"

namespace stdeploy {
    namespace vision {


        static inline float sigmoid(float x) {
            return static_cast<float>(1.0f / (1.0f + exp(-x)));
        }

        static inline float unsigmoid(float y) {
            return static_cast<float>(-1.0 * logf((1.0 / y) - 1.0));
        }

        static inline void sigmoids(float *src, float *dst, size_t length) {
            for (size_t i = 0; i < length; i++) {
                dst[i] = sigmoid(src[i]);
            }
        }


    } //namespace vision
} //namespace stdeploy

