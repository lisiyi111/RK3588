/**
  *************************************************
  * @file               :manager.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#include "stdeploy/vision/common/image_processes/manager.h"


namespace stdeploy {
    namespace vision {

        bool ProcessorManger::BuildProcessorPipelineFromConfig(const std::string &config_file) {
            pre_processors_.clear();
            YAML::Node cfg;
            try {
                cfg = YAML::LoadFile(config_file);
            } catch (YAML::BadFile &e) {
                SDERROR << "Failed to load yaml file " << config_file
                        << ", maybe you should check this file." << std::endl;
                return false;
            }
            if (cfg["PREPROCESS"].IsDefined()) {
                for (const auto &op : cfg["PREPROCESS"]) {
                    std::string op_name = op["type"].as<std::string>();
                    if (op_name == "normalize") {
                        if (!op["mean"].IsDefined() || !op["std"].IsDefined()) {
                            SDERROR << " normalize params not set,such as mean/std" << std::endl;
                            return false;
                        }
                        std::vector<float> mean, std;
                        if (op["mean"].IsDefined()) {
                            mean = op["mean"].as<std::vector<float>>();
                        }
                        if (op["std"].IsDefined()) {
                            std = op["std"].as<std::vector<float>>();
                        }
                        bool use_rgb = false;
                        if (op["use_rgb"].IsDefined()) {
                            use_rgb = op["use_rgb"].as<bool>();
                        }
                        bool use_chw = false;
                        if (op["use_chw"].IsDefined()) {
                            use_chw = op["use_chw"].as<bool>();
                        }
                        float scale = 1 / 255.f;
                        if (op["scale"].IsDefined()) {
                            scale = op["scale"].as<float>();
                        }
                        pre_processors_.push_back(
                                std::make_shared<Normalize>(mean, std, scale, use_rgb, use_chw));
                    } else if (op_name == "resize") {
                        if (!op["size"].IsDefined()) {
                            SDERROR << " resize params not set,such as size" << std::endl;
                            return false;
                        }
                        auto target_size = op["size"].as<std::vector<int>>();
                        bool use_pad = false;
                        if (op["use_pad"].IsDefined()) {
                            use_pad = op["use_pad"].as<bool>();
                        }
                        bool use_rgb = false;
                        if (op["use_rgb"].IsDefined()) {
                            use_rgb = op["use_rgb"].as<bool>();
                        }
                        bool right_pad = false;
                        if (op["right_pad"].IsDefined()) {
                            right_pad = op["right_pad"].as<bool>();
                        }
                        int resize_short = -1;
                        if (op["resize_short"].IsDefined()) {
                            resize_short = op["resize_short"].as<int>();
                        }
                        if (resize_short != -1) {
                            if (resize_short <= target_size[0] || resize_short <= target_size[1]) {
                                STDEPLOY_ERROR("resize_short must > target size");
                                return false;
                            }
                        }
                        if ((right_pad && use_pad) || (right_pad && (resize_short != -1)) ||
                            (use_pad && (resize_short != -1))) {
                            STDEPLOY_ERROR(
                                    "Only one of 'use pad', 'right pad', or 'resize short' can be used at a time.");
                            return false;
                        }
                        pre_processors_.push_back(
                                std::make_shared<Resize>(target_size[1], target_size[0],
                                                         use_pad, use_rgb, right_pad,
                                                         resize_short));
                    }
                }
            }
            return true;
        }


    } //namespace vision
} //namespace stdeploy