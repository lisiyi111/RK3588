/**
  *************************************************
  * @file               :cosine_similarity.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/8/23                
  *************************************************
  */
#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace utils {

            float CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b,
                                   bool normalized) {
                if ((a.size() != b.size()) || (a.size() == 0)) {
                    std::cout << " error The size of a and b must be equal and >= 1." << std::endl;
                    return 0.0;
                }
                size_t num_val = a.size();
                if (normalized) {
                    float mul_a = 0.f, mul_b = 0.f, mul_ab = 0.f;
                    for (size_t i = 0; i < num_val; ++i) {
                        mul_a += (a[i] * a[i]);
                        mul_b += (b[i] * b[i]);
                        mul_ab += (a[i] * b[i]);
                    }
                    return (mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
                }
                auto norm_a = L2Normalize(a);
                auto norm_b = L2Normalize(b);
                float mul_a = 0.f, mul_b = 0.f, mul_ab = 0.f;
                for (size_t i = 0; i < num_val; ++i) {
                    mul_a += (norm_a[i] * norm_a[i]);
                    mul_b += (norm_b[i] * norm_b[i]);
                    mul_ab += (norm_a[i] * norm_b[i]);
                }
                return (mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy