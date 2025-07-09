/**
  *************************************************
  * @file               :ppocrv2.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/11/1                
  *************************************************
  */

#include "stdeploy/vision/ocr/ppocr/ppocrv2.h"

namespace stdeploy {
    namespace vision {
        namespace pipeline {


            PPOCRv2::PPOCRv2(stdeploy::vision::ocr::DBNet *det_model, stdeploy::vision::ocr::TextAngleCls *cls_model,
                             stdeploy::vision::ocr::CRNN *rec_model) : detector_(det_model), classifier_(cls_model),
                                                                       recognizer_(rec_model) {
            }

            PPOCRv2::PPOCRv2(stdeploy::vision::ocr::DBNet *det_model, stdeploy::vision::ocr::CRNN *rec_model)
                    : detector_(det_model), recognizer_(rec_model) {
            }

            PPOCRv2::PPOCRv2(stdeploy::vision::ocr::CRNN *rec_model) : recognizer_(rec_model) {
            }

            bool PPOCRv2::Initialized() const {
                if (detector_ != nullptr && !detector_->Initialized()) {
                    return false;
                }
                if (classifier_ != nullptr && !classifier_->Initialized()) {
                    return false;
                }
                if (recognizer_ != nullptr && !recognizer_->Initialized()) {
                    return false;
                }
                return true;
            }

            bool PPOCRv2::Predict(cv::Mat &img, stdeploy::vision::OCRResult *result, bool use_det, bool use_cls) {
                result->Clear();
                std::vector<std::array<int, 8>> det_boxes;
                if (detector_ != nullptr && use_det) {
                    sd::Mat input_data = sd::Mat(img);
                    if (!detector_->Predict(input_data, result)) {
                        STDEPLOY_ERROR("det text error");
                        return false;
                    }
                    det_boxes.assign(result->boxes.begin(), result->boxes.end());
                    stdeploy::vision::ocr::SortBoxes(&det_boxes);
                }
                result->Clear();
                // text recognize
                std::vector<cv::Mat> image_list;
                if (det_boxes.size() == 0) {
                    image_list.emplace_back(img);
                } else {
                    for (size_t i = 0; i < det_boxes.size(); i++) {
                        cv::Mat crop_img = stdeploy::vision::ocr::GetRotateCropImage(img, det_boxes[i]);
                        image_list.emplace_back(crop_img);
                    }
                }
                std::vector<int32_t> cls_labels;
                std::vector<float> cls_scores;
                if (classifier_ != nullptr && use_cls) {
                    for (size_t i = 0; i < image_list.size(); i++) {
                        int32_t cls_label;
                        float cls_score;
                        cv::Mat crop_img = image_list[i].clone();
                        sd::Mat crop_cls_mat = sd::Mat(crop_img);
                        if (!classifier_->Predict(crop_cls_mat, &cls_label, &cls_score)) {
                            STDEPLOY_ERROR("cls text error");
                            return false;
                        }
                        result->cls_labels.emplace_back(cls_label);
                        result->cls_scores.emplace_back(cls_score);
                        if (cls_label == 1 && cls_score > 0.7) {
                            cv::rotate(image_list[i], image_list[i], 1);
                        }
                    }
                }
                for (size_t i = 0; i < image_list.size(); i++) {
                    cv::Mat rec_img = image_list[i].clone();
                    sd::Mat rec_mat = sd::Mat(rec_img);
                    if (!recognizer_->Predict(rec_mat, result)) {
                        STDEPLOY_ERROR("rec text error");
                        return false;
                    }
                }
                if (detector_ != nullptr && det_boxes.size() == result->text.size()) {
                    result->boxes.assign(det_boxes.begin(), det_boxes.end());
                }
                return true;
            }

            PPOCRv2::~PPOCRv2() {
                if (detector_ != nullptr) {
                    delete detector_;
                }
                if (classifier_ != nullptr) {
                    delete classifier_;
                }
                if (recognizer_ != nullptr) {
                    delete recognizer_;
                }
            }


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy