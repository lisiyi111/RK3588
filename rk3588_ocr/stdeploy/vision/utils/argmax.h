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
#include "stdeploy/utils/log_util.h"

namespace stdeploy {
    namespace vision {

        /// default input format : bchw
        static inline void seg_argmax_chw(const float *src, uint8_t *dst, int height, int width, int class_num) {
            int step = height * width;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int max_value = -100000000; // logit value
                    int max_id = -1;
                    for (int c = 0; c < class_num; c++) {
                        if (src[c * step + h * width + w] > max_value) {
                            max_value = src[c * step + h * width + w];
                            max_id = c;
                        }
                    }
                    dst[h * width + w] = max_id;
                }
            }
        }


        static inline void seg_argmax_hwc(const float *src, uint8_t *dst, int height, int width, int class_num) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int max_value = -100000000; // logit value
                    int max_id = -1;
                    auto feature_ptr = src; //data_ptr
                    for (int c = 0; c < class_num; c++) {
                        if (src[c] > max_value) {
                            max_value = feature_ptr[c];
                            max_id = c;
                        }
                    }
                    // move ptr addr for align
                    src += class_num;
                    dst[h * width + w] = max_id;
                }
            }
        }

        template<class ForwardIterator>
        inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
            // std::max_element 求范围内最大值元素，std::distance求范围内，迭代器的距离
            return std::distance(first, std::max_element(first, last));
        }

        template<class ForwardIterator>
        inline static size_t argmin(ForwardIterator first, ForwardIterator last) {
            return std::distance(first, std::min_element(first, last));
        }


    } //namespace vision
} //namespace stdeploy
