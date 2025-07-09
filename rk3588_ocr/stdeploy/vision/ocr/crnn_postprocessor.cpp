//
// Created by zhuzf on 2023/7/14.
//

#include "stdeploy/vision/ocr/crnn_postprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            bool CRNNPostprocessor::ReadPostProcessFromConfig(const std::string &config_file) {
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
                            if (op_name == "f1c3") {
                                apply_f1c3_ = true;
                            } else if(op_name == "f2c3"){
                                apply_f2c3_ = true;
                            } else {
                                SDERROR << "Not support function: " << op_name << std::endl;
                                return false;
                            }
                            if (op["label_path"].IsDefined()) {
                                label_path_ = op["label_path"].as<std::string>();
                            }
                            if (op["add_space_char"].IsDefined()) {
                                add_space_char_ = op["add_space_char"].as<bool>();
                            }
                            if (op["stride"].IsDefined()) {
                                stride_ = op["stride"].as<int>();
                            }
                        }
                    }
                } catch (YAML::BadConversion &ce) {
                    SDERROR << "Failed to convert yaml file " << std::endl;
                    return false;
                }
                return true;
            }


            bool CRNNPostprocessor::ReadDict(const std::string &path) {
                std::ifstream in(path);
                if (!in.is_open()) {
                    SDWARNING << "Read char label path not open " << std::endl;
                    return false;
                }
                std::string line;
                std::vector<std::string> m_vec;
                while (getline(in, line)) {
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                        line.pop_back();
                    }
                    label_list_.emplace_back(line);
                }
                label_list_.insert(label_list_.begin(), "black");  // blank char for ctc
                if (add_space_char_) {
                    label_list_.emplace_back(" "); // append space_char
                }
                STDEPLOY_INFO("read ocr dict:[%zu]", label_list_.size());
                return true;
            }

            bool CRNNPostprocessor::Run(std::vector<Tensor> &sd_tensors,
                                        PreprocessParams &pre_params,
                                        OCRResult *result) {

                int model_h = pre_params.dst_height;
                int model_w = pre_params.dst_width;
                int origin_h = pre_params.src_height;
                int origin_w = pre_params.src_width;
                float scale_h = pre_params.scale_height;
                float scale_w = pre_params.scale_width;
                int pad_h = pre_params.pad_height_top;
                int pad_w = pre_params.pad_width_left;
                int lastIndex = 0;
                int count = 0;
                float score = 0.f;
                int shape_w = int(model_w / stride_);
                std::string strRes;
                int shape_h = label_list_.size();
                int align_score = utils::AlignData(shape_h);

                auto cls_score_ptr = reinterpret_cast<float *>(sd_tensors[0].data());
                for (int w = 0; w < shape_w; w++) {
                    int max_index = -1;
                    float max_score = 0.0f;
                    for (int class_idx = 0; class_idx < shape_h; class_idx++) {
                        float box_prob = cls_score_ptr[class_idx];
                        if (box_prob > max_score) {
                            max_score = box_prob;
                            max_index = class_idx;
                        }
                    }
                    if (max_index > 0 && max_index < shape_h && (!(w > 0 && max_index == lastIndex))) {
                        score += max_score;
                        count += 1;
                        strRes.append(label_list_[max_index]);
                    }
                    lastIndex = max_index;
                    // move ptr addr for align
                    cls_score_ptr += align_score;
                }
                score /= (count + 1e-6);
                if (count == 0 || std::isnan(score)) {
                    score = 0.f;
                }
                result->rec_scores.push_back(score);
                result->text.push_back(strRes);
                return true;
            }


        } //namespace ocr
    } //namespace vision
} //namespace stdeploy