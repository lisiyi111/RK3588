/**
  *************************************************
  * @file               :ppocrv2.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#pragma once

#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"
#include "stdeploy/vision/ocr/dbnet.h"
#include "stdeploy/vision/ocr/crnn.h"
#include "stdeploy/vision/ocr/ppocr/text_angle_cls.h"
#include "stdeploy/vision/ocr/utils/ocr_utils.h"

namespace stdeploy {
    namespace vision {
        namespace pipeline {

            class STDEPLOY_DECL PPOCRv2 : public StDeployModel {

            public:
                PPOCRv2(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::TextAngleCls *cls_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                );

                PPOCRv2(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                );

                PPOCRv2(stdeploy::vision::ocr::CRNN *rec_model);

                ~PPOCRv2() override;

                std::string ModelName() const override { return "PPOCRv2"; }

                bool
                Predict(cv::Mat &img, stdeploy::vision::OCRResult *result, bool use_det = true, bool use_cls = true);

                bool Initialized() const override;


            protected:
                stdeploy::vision::ocr::DBNet *detector_ = nullptr;
                stdeploy::vision::ocr::TextAngleCls *classifier_ = nullptr;
                stdeploy::vision::ocr::CRNN *recognizer_ = nullptr;

            };


        } //namespace pipeline
    } //namespace vision
} //namespace stdeploy