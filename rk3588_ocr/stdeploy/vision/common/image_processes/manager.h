/**
  *************************************************
  * @file               :manager.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include <memory>
#include "stdeploy/utils/log_util.h"
#include "stdeploy/vision/common/image_processes/base.h"
#include "stdeploy/vision/common/image_processes/resize.h"
#include "stdeploy/vision/common/image_processes/normalize.h"
#include "yaml-cpp/yaml.h"

namespace stdeploy {
    namespace vision {

        class STDEPLOY_DECL ProcessorManger {
        public:
            bool BuildProcessorPipelineFromConfig(const std::string &config_file);
            std::vector<std::shared_ptr<Processor>> pre_processors_;
        };

    } //namespace vision
} //namespace stdeploy