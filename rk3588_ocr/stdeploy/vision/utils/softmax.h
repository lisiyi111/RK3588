/**
  *************************************************
  * @file               :softmax.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#pragma once

#include <cmath>

namespace stdeploy {
    namespace vision {

        static inline void softmax(const float *src, float *dst, size_t length) {
            float total = 0.;
            for (size_t i = 0; i < length; i++) {
                total += std::exp(src[i]);
            }
            for (size_t i = 0; i < length; i++) {
                dst[i] = std::exp(src[i]) / total;
            }
        }

        static inline void softmax_v1(float *data, size_t length) {
            // Step 1: 找到最大值以保证数值稳定性
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < length; ++i) {
                if (data[i] > max_val) {
                    max_val = data[i];
                }
            }

            // Step 2: 计算 exp(x - max_val) 并求和
            float sum_exp = 0.0f;
            for (size_t i = 0; i < length; ++i) {
                sum_exp += std::exp(data[i] - max_val);
            }

            // Step 3: 归一化，将结果写回原数组
            for (size_t i = 0; i < length; ++i) {
                data[i] = std::exp(data[i] - max_val) / sum_exp;
            }
        }

    } //namespace vision
} //namespace stdeploy
