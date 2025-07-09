/**
  *************************************************
  * @file               :normalize.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/27                
  *************************************************
  */

#include "stdeploy/vision/common/image_processes/normalize.h"

namespace stdeploy {
    namespace vision {

        Normalize::Normalize(const std::vector<float> &mean, const std::vector<float> &std, float scale,
                             bool use_rgb, bool use_chw) {
            mean_.assign(mean.begin(), mean.end());
            std_.assign(std.begin(), std.end());
            scale_ = scale;
            use_rgb_ = use_rgb;
            use_chw_ = use_chw;
            for (auto c = 0; c < mean_.size(); ++c) {
                double alpha = scale_;
                double beta = -1.0 * (mean_[c]) / std_[c];
                alpha /= std_[c];
                alpha_.push_back(alpha);
                beta_.push_back(beta);
            }
        }

        bool Normalize::ImplByOpenCV(sd::Mat &mat) {
            int width = mat.width();
            int height = mat.height();
            std::vector<cv::Mat> split_im;
            cv::split(mat.cv_mat, split_im);
            if (use_rgb_) std::swap(split_im[0], split_im[2]);
            for (int c = 0; c < mat.channel(); c++) {
                split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
            }
            if (use_chw_) {
                cv::Mat res(height, width, CV_32FC(mat.channel()));
                for (int i = 0; i < mat.channel(); ++i) {
                    cv::extractChannel(split_im[i],
                                       cv::Mat(height, width, CV_32FC1,
                                               res.ptr() + i * height * width * 4),
                                       0);
                }
                mat.cv_mat = res.clone();
            } else {
                cv::merge(split_im, mat.cv_mat);
            }
            return true;
        }

    } //namespace vision
} //namespace stdeploy