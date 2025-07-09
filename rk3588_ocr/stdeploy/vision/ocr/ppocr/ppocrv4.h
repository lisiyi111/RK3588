/**
  *************************************************
  * @file               :ppocrv4.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#pragma once

#include "stdeploy/vision/ocr/ppocr/ppocrv3.h"

namespace stdeploy {
    namespace vision {
        namespace pipeline {

            class STDEPLOY_DECL PPOCRv4 : public PPOCRv3 {
            public:
                std::string ModelName() const override { return "PPOCRv4"; }

                PPOCRv4(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::TextAngleCls *cls_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                ) : PPOCRv3(det_model, cls_model, rec_model) {
                    // v4 rec model 48,stvr
                }

                PPOCRv4(stdeploy::vision::ocr::DBNet *det_model,
                        stdeploy::vision::ocr::CRNN *rec_model
                ) : PPOCRv3(det_model, rec_model) {
                    // v4 rec model 48,stvr
                };

                PPOCRv4(stdeploy::vision::ocr::CRNN *rec_model) : PPOCRv3(rec_model) {
                    // v4 rec model 48,stvr
                };
            };


        } //namespace pipeline
    } //namespace vision
} //namespace stdeploy