/**
  *************************************************
  * @file               :l2_normalize.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/1/19                
  *************************************************
  */

#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace utils {

            std::vector<float> L2Normalize(const std::vector<float> &values) {
                size_t num_val = values.size();
                if (num_val == 0) {
                    return {};
                }
                std::vector<float> norm;
                float l2_sum_val = 0.f;
                for (size_t i = 0; i < num_val; ++i) {
                    l2_sum_val += (values[i] * values[i]);
                }
                float l2_sum_sqrt = std::sqrt(l2_sum_val);
                norm.resize(num_val);
                for (size_t i = 0; i < num_val; ++i) {
                    norm[i] = values[i] / l2_sum_sqrt;
                }
                return norm;
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy