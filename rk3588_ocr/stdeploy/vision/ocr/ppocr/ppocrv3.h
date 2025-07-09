/**
  *************************************************
  * @file               :ppocrv3.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#pragma once

#include "stdeploy/vision/ocr/ppocr/ppocrv2.h"

namespace stdeploy {
    namespace vision {
        namespace pipeline {

            class STDEPLOY_DECL PPOCRv3 : public PPOCRv2 {

            public:
                PPOCRv3(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::TextAngleCls *cls_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                ) : PPOCRv2(det_model, cls_model, rec_model) {
                    // v3 rec model 32->48,crnn->stvr
                }

                PPOCRv3(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                ) : PPOCRv2(det_model, rec_model) {
                    // v3 rec model 32->48,crnn->stvr
                };

                PPOCRv3(stdeploy::vision::ocr::CRNN *rec_model) : PPOCRv2(rec_model) {
                    // v3 rec model 32->48,crnn->stvr
                };

                std::string ModelName() const override { return "PPOCRv3"; }

            };


        } //namespace pipeline
    } //namespace vision
} //namespace stdeploy