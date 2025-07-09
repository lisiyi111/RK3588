/**
  *************************************************
  * @file               :vis_base.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/1/15                
  *************************************************
  */
#include "stdeploy/vision/visualize/vis_base.h"

namespace stdeploy {
    namespace vision {

        std::vector<int> GenerateColorMap(int num_classes) {
            if (num_classes < 10) {
                num_classes = 10;
            }
            std::vector<int> color_map(num_classes * 3, 0);
            for (int i = 0; i < num_classes; ++i) {
                int j = 0;
                int lab = i;
                while (lab) {
                    color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j));
                    color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
                    color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
                    ++j;
                    lab >>= 3;
                }
            }
            return color_map;
        }

    } //namespace vision
} //namespace stdeploy