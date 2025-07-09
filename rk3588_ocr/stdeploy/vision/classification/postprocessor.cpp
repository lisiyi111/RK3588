/**
  *************************************************
  * @file               :postprocess.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#include "stdeploy/vision/classification/postprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace classification {

            ClsPostprocessor::ClsPostprocessor(const std::string &config_file) {
                if (!config_file.empty()) {
                    if (this->ReadPostProcessFromConfig(config_file)) {
                        initialized_ = true;
                    }
                } else {
                    SDWARNING << "config file is empty,please check postprocess func callback" << std::endl;
                    initialized_ = true;
                }
            }

            bool ClsPostprocessor::Run(std::vector<Tensor> &sd_tensors, ClassifyResult *result) {
                auto *scores_ptr = reinterpret_cast<float *>(sd_tensors[0].data());
                if (apply_single_label_) {
                    if (add_softmax_) {
                        float *softmax_scores = new float[num_class_];
                        softmax(scores_ptr, softmax_scores, num_class_);
                        result->label_ids = utils::TopKIndices(softmax_scores, num_class_, topk_value_);
                        for (size_t i = 0; i < topk_value_; ++i) {
                            result->scores.push_back(softmax_scores[result->label_ids[i]]);
                        }
                        delete[] softmax_scores;
                    } else {
                        result->label_ids = utils::TopKIndices(scores_ptr, num_class_, topk_value_);
                        for (size_t i = 0; i < topk_value_; ++i) {
                            result->scores.push_back(scores_ptr[result->label_ids[i]]);
                        }
                    }
                } else if (apply_multi_label_) {
                    topk_value_ = num_class_;
                    if (add_sigmoid_) {
                        float *sigmoid_scores = new float[num_class_];
                        sigmoids(scores_ptr, sigmoid_scores, num_class_);
                        for (size_t i = 0; i < topk_value_; ++i) {
                            float score = sigmoid_scores[i];
                            result->scores.push_back(score);
                            result->label_ids.push_back(score > cls_thresh_ ? 1 : 0);
                        }
                        delete[] sigmoid_scores;
                    } else {
                        for (size_t i = 0; i < topk_value_; ++i) {
                            float score = scores_ptr[i];
                            result->scores.push_back(score);
                            result->label_ids.push_back(score > cls_thresh_ ? 1 : 0);
                        }
                    }
                }
                return true;
            }

            bool ClsPostprocessor::ReadPostProcessFromConfig(const std::string &config_file) {
                YAML::Node cfg;
                try {
                    cfg = YAML::LoadFile(config_file);
                } catch (YAML::BadFile &e) {
                    SDERROR << "Failed to load yaml file " << config_file
                            << ", maybe you should check this file." << std::endl;
                    return false;
                }
                try {
                    if (cfg["POSTPROCESS"].IsDefined()) {
                        for (const auto &op : cfg["POSTPROCESS"]) {
                            std::string op_name = op["type"].as<std::string>();
                            if (op_name == "single_label") {
                                apply_single_label_ = true;
                            } else if (op_name == "multi_label") {
                                apply_multi_label_ = true;
                            } else {
                                SDERROR << "Not support function: " << op_name << std::endl;
                                return false;
                            }
                            if (op["add_softmax"].IsDefined()) {
                                add_softmax_ = op["add_softmax"].as<bool>();
                                if (apply_multi_label_) {
                                    SDERROR << "multi_label only support sigmoid " << std::endl;
                                    return false;
                                }
                            }
                            if (op["add_sigmoid"].IsDefined()) {
                                add_sigmoid_ = op["add_sigmoid"].as<bool>();
                                if (apply_single_label_) {
                                    SDERROR << "single_label only support softmax " << std::endl;
                                    return false;
                                }
                            }
                            if (op["topk_value"].IsDefined()) {
                                topk_value_ = op["topk_value"].as<int>();
                            }
                            if (op["cls_thresh"].IsDefined()) {
                                cls_thresh_ = op["cls_thresh"].as<float>();
                            }
                            if (!op["num_class"].IsDefined()) {
                                SDERROR << " cls postprocess params not set,such as num_class" << std::endl;
                                return false;
                            }
                            num_class_ = op["num_class"].as<int>();
                        }
                    }
                } catch (YAML::BadConversion &ce) {
                    SDERROR << "Failed to convert yaml file " << std::endl;
                    return false;
                }
                return true;
            }

        } //namespace classification
    } //namespace vision
} //namespace stdeploy