//
// Created by zhuzf on 2023/7/12.
//

#include "stdeploy/vision/ocr/dbnet_postprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace ocr {

            bool DBNetPostprocessor::ReadPostProcessFromConfig(const std::string &config_file) {
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
                            if (op_name == "f1c4") {
                                apply_f1c4_ = true;
                            } else {
                                SDERROR << "Not support function: " << op_name << std::endl;
                                return false;
                            }
                            if (op["db_box_thresh"].IsDefined()) {
                                det_db_box_thresh_ = op["db_box_thresh"].as<float>();
                            }
                            if (op["db_thresh"].IsDefined()) {
                                det_db_thresh_ = op["db_thresh"].as<float>();
                            }
                            if (op["db_unclip_ratio"].IsDefined()) {
                                det_db_unclip_ratio_ = op["db_unclip_ratio"].as<float>();
                            }
                            if (op["db_score_mode"].IsDefined()) {
                                det_db_score_mode_ = op["db_score_mode"].as<float>();
                            }
                            if (op["use_dilation"].IsDefined()) {
                                use_dilation_ = op["use_dilation"].as<float>();
                            }
                        }
                    }
                } catch (YAML::BadConversion &ce) {
                    SDERROR << "Failed to convert yaml file " << std::endl;
                    return false;
                }
                return true;
            }


            bool DBNetPostprocessor::Run(std::vector<Tensor> &sd_tensors,
                                         PreprocessParams &pre_params,
                                         OCRResult *result) {

                int output_tensor_num = sd_tensors.size();

                int model_h = pre_params.dst_height;
                int model_w = pre_params.dst_width;
                int origin_h = pre_params.src_height;
                int origin_w = pre_params.src_width;
                float scale_h = pre_params.scale_height;
                float scale_w = pre_params.scale_width;
                int pad_h = pre_params.pad_height_top;
                int pad_w = pre_params.pad_width_left;

                std::array<int, 4> det_img_info = {int(origin_w), int(origin_h), int(model_w), int(model_h)};
                std::array<int, 2> pad_info = {pad_w, pad_h};
                std::array<float, 2> scale_info = {scale_w, scale_h};

                if (!apply_f1c4_) {
                    SDERROR << "db postprocess only support f1c4" << std::endl;
                    return false;
                }

                auto pred_data = (float *) sd_tensors[0].data();

                int n = model_h * model_w;
                std::vector<float> pred(n, 0.0);
                std::vector<unsigned char> cbuf(n, ' ');

                for (int i = 0; i < n; i++) {
                    pred[i] = float(pred_data[i]);
                    cbuf[i] = (unsigned char) ((pred_data[i]) * 255);
                }

                cv::Mat cbuf_map(model_h, model_w, CV_8UC1, (unsigned char *) cbuf.data());
                cv::Mat pred_map(model_h, model_w, CV_32F, (float *) pred.data());
                const double threshold = det_db_thresh_ * 255;
                const double maxvalue = 255;
                cv::Mat bit_map;
                cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
                if (use_dilation_) {
                    cv::Mat dila_ele =
                            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                    cv::dilate(bit_map, bit_map, dila_ele);
                }
                std::vector<std::vector<std::vector<int>>> boxes;
                boxes = util_post_processor_.BoxesFromBitmap(
                        pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
                        det_db_score_mode_);
                // 映射到输入坐标同时转换成矩形行
                boxes = util_post_processor_.FilterTagDetRes(boxes, det_img_info, pad_info, scale_info);
                // boxes to boxes_result
                for (int i = 0; i < boxes.size(); i++) {
                    std::array<int, 8> new_box;
                    int k = 0;
                    for (auto &vec : boxes[i]) {
                        for (auto &e : vec) {
                            new_box[k++] = e;
                        }
                    }
                    result->boxes.emplace_back(new_box);
                }
                return true;
            }



        } //namespace ocr
    } //namespace vision
} //namespace stdeploy