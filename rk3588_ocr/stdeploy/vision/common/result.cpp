/**
  *************************************************
  * @file               :result.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#include "stdeploy/vision/common/result.h"

namespace stdeploy {
    namespace vision {

        void ClassifyResult::Free() {
            // use empty swap data, free data
            std::vector<int32_t>().swap(label_ids);
            std::vector<float>().swap(scores);
        }

        void ClassifyResult::Clear() {
            label_ids.clear();
            scores.clear();
        }

        void ClassifyResult::Resize(int size) {
            label_ids.resize(size);
            scores.resize(size);
        }

        /*
        // 默认移动赋值运算符已经能实现功能，不在手动实现
        ClassifyResult &ClassifyResult::operator=(ClassifyResult &&other) {
            if (&other != this) {
                label_ids = std::move(other.label_ids);
                scores = std::move(other.scores);
            }
            return *this;
        }*/

        std::string ClassifyResult::Str() {
            std::string out;
            out = "ClassifyResult(\nlabel_ids: ";
            for (size_t i = 0; i < label_ids.size(); ++i) {
                out = out + std::to_string(label_ids[i]) + ", ";
            }
            out += "\nscores: ";
            for (size_t i = 0; i < label_ids.size(); ++i) {
                out = out + std::to_string(scores[i]) + ", ";
            }
            out += "\n)";
            return out;
        }


        void Mask::Clear() {
            data.clear();
            shape.clear();
        }

        void Mask::Free() {
            std::vector<uint8_t>().swap(data);
            std::vector<int64_t>().swap(shape);
        }

        void Mask::Reserve(int size) {
            data.reserve(size);
        }

        void Mask::Resize(int size) {
            data.resize(size);
        }

        std::string Mask::Str() {
            std::string out = "Mask(";
            size_t ndim = shape.size();
            for (size_t i = 0; i < ndim; ++i) {
                if (i < ndim - 1) {
                    out += std::to_string(shape[i]) + ",";
                } else {
                    out += std::to_string(shape[i]);
                }
            }
            out += ")\n";
            return out;
        }


        void DetectionResult::Clear() {
            boxes.clear();
            scores.clear();
            label_ids.clear();
            masks.clear();
            contain_masks = false;
            contain_kpts = false;
            contain_rbox = false;
            rboxes.clear();
            angles.clear();
        }

        void DetectionResult::Free() {
            std::vector<std::array<float, 4>>().swap(boxes);
            std::vector<float>().swap(scores);
            std::vector<int32_t>().swap(label_ids);
            std::vector<Mask>().swap(masks);
            contain_masks = false;
            contain_kpts = false;
            contain_rbox = false;
            std::vector<float>().swap(angles);
            std::vector<std::array<float, 8>>().swap(rboxes);
        }

        void DetectionResult::Reserve(int size) {
            boxes.reserve(size);
            scores.reserve(size);
            label_ids.reserve(size);
            if (contain_masks) {
                masks.reserve(size);
            }
            if (contain_kpts) {
                kpts.reserve(size);
            }
            if (contain_rbox) {
                rboxes.reserve(size);
                angles.reserve(size);
            }
        }

        void DetectionResult::Resize(int size) {
            boxes.resize(size);
            scores.resize(size);
            label_ids.resize(size);
            if (contain_masks) {
                masks.resize(size);
            }
            if (contain_kpts) {
                kpts.resize(size);
            }
            if (contain_rbox) {
                rboxes.resize(size);
                angles.resize(size);
            }
        }

        std::string DetectionResult::Str() {
            std::string out;
            if (!contain_masks && !contain_kpts && !contain_rbox) {
                out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
            }
            if (contain_masks && !contain_kpts && !contain_rbox) {
                out =
                        "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id, "
                        "mask_shape]\n";
            }
            if (!contain_masks && contain_kpts && !contain_rbox) {
                out =
                        "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id, "
                        "keypoint]\n";
            }
            if (contain_rbox && !contain_kpts && !contain_masks) {
                out = "DetectionResult: [x1, y1, x2, y2, x2, y3, x4, y4, score, label_id, angle]\n";
            }
            for (size_t i = 0; i < scores.size(); ++i) {
                if (contain_rbox) {
                    out = out + std::to_string(rboxes[i][0]) + ","
                          + std::to_string(rboxes[i][1]) + ", "
                          + std::to_string(rboxes[i][2]) + ", "
                          + std::to_string(rboxes[i][3]) + ", "
                          + std::to_string(rboxes[i][4]) + ", "
                          + std::to_string(rboxes[i][5]) + ", "
                          + std::to_string(rboxes[i][6]) + ", "
                          + std::to_string(rboxes[i][7]) + ", "
                          + std::to_string(scores[i]) + ", "
                          + std::to_string(label_ids[i]) + ", "
                          + std::to_string(angles[i]);
                } else {
                    out = out + std::to_string(boxes[i][0]) + "," +
                          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
                          ", " + std::to_string(boxes[i][3]) + ", " +
                          std::to_string(scores[i]) + ", " + std::to_string(label_ids[i]);
                }
                out += "\n";
                if (!contain_masks && !contain_kpts) {
                    out += "\n";
                }
                if (contain_masks && !contain_kpts) {
                    out += "" + masks[i].Str();
                }
                if (!contain_masks && contain_kpts) {
                    out += "" + kpts[i].Str();
                }
            }
            return out;
        }


        void KeyPointDetectionResult::Clear() {
            keypoints.clear();
            scores.clear();
            num_joints = -1;
        }

        void KeyPointDetectionResult::Free() {
            std::vector<std::array<float, 2>>().swap(keypoints);
            std::vector<float>().swap(scores);
            num_joints = -1;
        }

        void KeyPointDetectionResult::Reserve(int size) {
            keypoints.reserve(size);
        }

        void KeyPointDetectionResult::Resize(int size) {
            keypoints.resize(size);
        }

        std::string KeyPointDetectionResult::Str() {
            std::string out;
            out = "KeyPointDetectionResult: [x, y, conf]\n";
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (scores.size() == keypoints.size()) {
                    out = out + std::to_string(keypoints[i][0]) + "," +
                          std::to_string(keypoints[i][1]) + ", " + std::to_string(scores[i]) +
                          "\n";
                } else {
                    out = out + std::to_string(keypoints[i][0]) + "," +
                          std::to_string(keypoints[i][1]) + "\n";
                }
            }
            return out;
        }


        void OCRResult::Clear() {
            boxes.clear();
            text.clear();
            rec_scores.clear();
            cls_scores.clear();
            cls_labels.clear();
        }

        std::string OCRResult::Str() {
            std::string no_result;
            if (boxes.size() > 0) {
                std::string out;
                for (int n = 0; n < boxes.size(); n++) {
                    out = out + "det boxes: [";
                    for (int i = 0; i < 4; i++) {
                        out = out + "[" + std::to_string(boxes[n][i * 2]) + "," +
                              std::to_string(boxes[n][i * 2 + 1]) + "]";

                        if (i != 3) {
                            out = out + ",";
                        }
                    }
                    out = out + "]";

                    if (rec_scores.size() > 0) {
                        out = out + "rec text: " + text[n] + " rec score:" +
                              std::to_string(rec_scores[n]) + " ";
                    }
                    if (cls_labels.size() > 0) {
                        out = out + "cls label: " + std::to_string(cls_labels[n]) +
                              " cls score: " + std::to_string(cls_scores[n]);
                    }
                    out = out + "\n";
                }
                return out;

            } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
                       cls_scores.size() > 0) {
                std::string out;
                for (int i = 0; i < rec_scores.size(); i++) {
                    out = out + "rec text: " + text[i] + " rec score:" +
                          std::to_string(rec_scores[i]) + " ";
                    out = out + "cls label: " + std::to_string(cls_labels[i]) +
                          " cls score: " + std::to_string(cls_scores[i]);
                    out = out + "\n";
                }
                return out;
            } else if (boxes.size() == 0 && rec_scores.size() == 0 &&
                       cls_scores.size() > 0) {
                std::string out;
                for (int i = 0; i < cls_scores.size(); i++) {
                    out = out + "cls label: " + std::to_string(cls_labels[i]) +
                          " cls score: " + std::to_string(cls_scores[i]);
                    out = out + "\n";
                }
                return out;
            } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
                       cls_scores.size() == 0) {
                std::string out;
                for (int i = 0; i < rec_scores.size(); i++) {
                    out = out + "rec text: " + text[i] + " rec score:" +
                          std::to_string(rec_scores[i]) + " ";
                    out = out + "\n";
                }
                return out;
            }

            no_result = no_result + "No Results!";
            return no_result;
        }

        void SegmentationResult::Clear() {
            label_map.clear();
            score_map.clear();
            shape.clear();
            contain_score_map = false;
        }

        void SegmentationResult::Free() {
            std::vector<uint8_t>().swap(label_map);
            std::vector<float>().swap(score_map);
            std::vector<int64_t>().swap(shape);
            contain_score_map = false;
        }

        void SegmentationResult::Reserve(int size) {
            label_map.reserve(size);
            if (contain_score_map) {
                score_map.reserve(size);
            }
        }

        void SegmentationResult::Resize(int size) {
            label_map.resize(size);
            if (contain_score_map) {
                score_map.resize(size);
            }
        }

        std::string SegmentationResult::Str() {
            std::string out;
            out = "SegmentationResult Image masks 10 rows x 10 cols: \n";
            for (size_t i = 0; i < 10; ++i) {
                out += "[";
                for (size_t j = 0; j < 10; ++j) {
                    out = out + std::to_string(label_map[i * 10 + j]) + ", ";
                }
                out += ".....]\n";
            }
            out += "...........\n";
            if (contain_score_map) {
                out += "SegmentationResult Score map 10 rows x 10 cols: \n";
                for (size_t i = 0; i < 10; ++i) {
                    out += "[";
                    for (size_t j = 0; j < 10; ++j) {
                        out = out + std::to_string(score_map[i * 10 + j]) + ", ";
                    }
                    out += ".....]\n";
                }
                out += "...........\n";
            }
            out += "result shape is: [" + std::to_string(shape[0]) + " " +
                   std::to_string(shape[1]) + "]";
            return out;
        }

        SegmentationResult &SegmentationResult::operator=(SegmentationResult &&other) {
            if (&other != this) {
                label_map = std::move(other.label_map);
                shape = std::move(other.shape);
                contain_score_map = std::move(other.contain_score_map);
                if (contain_score_map) {
                    score_map.clear();
                    score_map = std::move(other.score_map);
                }
            }
            return *this;
        }


        void FaceRecognitionResult::Clear() {
            boxes.clear();
            scores.clear();
            landmarks.clear();
            embeddings.clear();
            names.clear();
        }

        void FaceRecognitionResult::Free() {
            std::vector<std::array<float, 4>>().swap(boxes);
            std::vector<float>().swap(scores);
            std::vector<KeyPointDetectionResult>().swap(landmarks);
            std::vector<FeatureMapResult>().swap(embeddings);
            std::vector<std::string>().swap(names);
        }

        void FaceRecognitionResult::Reserve(int size) {
            boxes.reserve(size);
            scores.reserve(size);
            names.reserve(size);
            if (contain_landmarks) {
                landmarks.reserve(size);
            }
            if (contain_embeddings) {
                embeddings.reserve(size);
            }
        }

        void FaceRecognitionResult::Resize(int size) {
            boxes.resize(size);
            scores.resize(size);
            names.resize(size);
            if (contain_landmarks) {
                landmarks.resize(size);
            }
            if (contain_embeddings) {
                embeddings.resize(size);
            }
        }

        std::string FaceRecognitionResult::Str() {
            std::string out;
            out = "FaceRecognitionResult: [xmin, ymin, xmax, ymax, score]\n";
            if (contain_embeddings && !contain_landmarks) {
                out =
                        "FaceRecognitionResult: [xmin, ymin, xmax, ymax, score, name, "
                        "embedding]\n";
            }
            if (contain_embeddings && contain_landmarks) {
                out =
                        "FaceRecognitionResult: [xmin, ymin, xmax, ymax, score, name, "
                        "keypoint,embedding]\n";
            }
            if (!contain_embeddings && contain_landmarks) {
                out =
                        "FaceRecognitionResult: [xmin, ymin, xmax, ymax, score,keypoint]\n";
            }
            for (size_t i = 0; i < boxes.size(); ++i) {
                out = out + std::to_string(boxes[i][0]) + "," +
                      std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
                      ", " + std::to_string(boxes[i][3]) + ", " +
                      std::to_string(scores[i]);
                if (!names.empty()) {
                    out += ", " + names[i];
                }
                out += "\n";
                if (!contain_embeddings && !contain_landmarks) {
                    out += "\n";
                }
                if (contain_embeddings && !contain_landmarks) {
                    out += embeddings[i].Str();
                }
                if (!contain_embeddings && contain_landmarks) {
                    out += landmarks[i].Str();
                }
                if (contain_embeddings && contain_landmarks) {
                    out += landmarks[i].Str();
                    out += embeddings[i].Str();
                }
            }
            return out;
        }


        FeatureMapResult::FeatureMapResult(const FeatureMapResult &res) {
            embedding.assign(res.embedding.begin(), res.embedding.end());
        }

        void FeatureMapResult::Free() { std::vector<float>().swap(embedding); }

        void FeatureMapResult::Clear() { embedding.clear(); }

        void FeatureMapResult::Reserve(int size) { embedding.reserve(size); }

        void FeatureMapResult::Resize(int size) { embedding.resize(size); }

        std::string FeatureMapResult::Str() {
            std::string out;
            out = "FeatureMapResult: [";
            size_t numel = embedding.size();
            if (numel <= 0) {
                return out + "Empty Result]";
            }
            // max, min, mean
            float min_val = embedding.at(0);
            float max_val = embedding.at(0);
            float total_val = embedding.at(0);
            for (size_t i = 1; i < numel; ++i) {
                float val = embedding.at(i);
                total_val += val;
                if (val < min_val) {
                    min_val = val;
                }
                if (val > max_val) {
                    max_val = val;
                }
            }
            float mean_val = total_val / static_cast<float>(numel);
            out = out + "Dim(" + std::to_string(numel) + "), " + "Min(" +
                  std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
                  "), " + "Mean(" + std::to_string(mean_val) + ")]\n";
            return out;
        }


        void DepthResult::Clear() {
            relative_depth.clear();
            vis_depth.clear();
            min_val = 0.0;
            max_val = 0.0;
            shape.clear();
        }

        void DepthResult::Free() {
            std::vector<uint8_t>().swap(vis_depth);
            std::vector<float>().swap(relative_depth);
            std::vector<int64_t>().swap(shape);
        }

        void DepthResult::Reserve(int size) {
            relative_depth.reserve(size);
            vis_depth.reserve(size);
        }

        void DepthResult::Resize(int size) {
            relative_depth.resize(size);
            vis_depth.resize(size);
        }

        std::string DepthResult::Str() {
            std::string out;
            out = "DepthResult: [";
            size_t ndim = relative_depth.size();
            if (ndim <= 0) {
                return out + "Empty Result]";
            }
            out = out + "Shape(" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + "), " + "Min(" +
                  std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
                  ")]\n";

            out += "DepthResult absolute_depth map centor 10 rows x 10 cols: \n";
            int printWidth = 10;
            int printHeight = 10;
            int height = shape[0];
            int width = shape[1];
            for (int i = height - printHeight; i < height; ++i) {
                out += "[";
                for (int j = 0; j < printWidth; ++j) {
                    out = out + std::to_string(relative_depth[i * width + j]) + ", ";
                }
                out += "]\n";
            }
            return out;
        }


    } //namespace vision
} //namespace stdeploy
