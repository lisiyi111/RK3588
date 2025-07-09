/**
  *************************************************
  * @file               :vis_cls.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/2/23                
  *************************************************
  */

#include "stdeploy/vision/visualize/vis_cls.h"

namespace stdeploy {
    namespace vision {

        cv::Mat VisClassification(const cv::Mat &im,
                                  const ClassifyResult &result,
                                  int top_k,
                                  float score_threshold,
                                  float font_size) {
            int h = im.rows;
            int w = im.cols;
            auto vis_im = im.clone();
            int h_sep = h / 10;
            int w_sep = w / 10;
            if (top_k > result.scores.size()) {
                top_k = result.scores.size();
            }
            for (int i = 0; i < top_k; ++i) {
                if (result.scores[i] < score_threshold) {
                    continue;
                }
                std::string id = std::to_string(result.label_ids[i]);
                std::string score = std::to_string(result.scores[i]);
                if (score.size() > 4) {
                    score = score.substr(0, 4);
                }
                std::string text = id + ":" + score;
                int font = cv::FONT_HERSHEY_SIMPLEX;
                cv::Point origin;
                origin.x = w_sep;
                origin.y = h_sep * (i + 1);
                cv::putText(vis_im, text, origin, font, font_size,
                            cv::Scalar(255, 255, 255), 1);
            }
            return vis_im;
        }

        cv::Mat VisClassification(const cv::Mat &im, const ClassifyResult &result,
                                  const std::vector<std::string> &labels, int top_k,
                                  float score_threshold, float font_size) {
            int h = im.rows;
            int w = im.cols;
            auto vis_im = im.clone();
            int h_sep = h / 10;
            int w_sep = w / 10;
            if (top_k > result.scores.size()) {
                top_k = result.scores.size();
            }
            for (int i = 0; i < top_k; ++i) {
                if (result.scores[i] < score_threshold) {
                    continue;
                }
                std::string id = std::to_string(result.label_ids[i]);
                std::string score = std::to_string(result.scores[i]);
                if (score.size() > 4) {
                    score = score.substr(0, 4);
                }
                std::string text = id + ":" + score;
                if (labels.size() > result.label_ids[i]) {
                    text = labels[result.label_ids[i]] + ":" + text;
                } else {
                    SDWARNING << "The label_id: " << result.label_ids[i]
                              << " in DetectionResult should be less than length of labels:"
                              << labels.size() << "." << std::endl;
                }
                if (text.size() > 16) {
                    text = text.substr(0, 16);
                }
                int font = cv::FONT_HERSHEY_SIMPLEX;
                cv::Point origin;
                origin.x = w_sep;
                origin.y = h_sep * (i + 1);
                cv::putText(vis_im, text, origin, font, font_size,
                            cv::Scalar(255, 255, 255), 1);
            }
            return vis_im;
        }


    } //namespace vision
} //namespace stdeploy