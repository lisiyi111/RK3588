/**
  *************************************************
  * @file               :resize.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#include "stdeploy/vision/common/image_processes/resize.h"
#include <cmath>
#include "stdeploy/vision/utils/utils.h"

#ifdef ENABLE_RKNPU2_BACKEND

#include "stdeploy/runtime/backends/rknpu2/utils.h"

#endif

#ifdef ENABLE_MSTAR_BACKEND

#include "stdeploy/runtime/backends/mstar/utils.h"

#endif

namespace stdeploy {
    namespace vision {

        bool Resize::ImplByOpenCV(sd::Mat &mat) {
            try {
                int origin_w = mat.width();
                int origin_h = mat.height();
                if (use_rgb_) {
                    cv::cvtColor(mat.cv_mat, mat.cv_mat, cv::COLOR_BGR2RGB);
                    mat.set_pixel_format(sd::RGB);
                }
                if (width_ == origin_w && height_ == origin_h) {
                    return true;
                }
                if (width_ > 0 && height_ > 0) {
                    cv::resize(mat.cv_mat, mat.cv_mat, cv::Size(width_, height_), 0, 0);
                    mat.set_height(height_);
                    mat.set_width(width_);
                } else {
                    SDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
                               "or (scale_w > 0 && scale_h > 0)."
                            << std::endl;
                    return false;
                }
                return true;
            } catch (const cv::Exception &e) {
                STDEPLOY_ERROR("resize error,mat:[h=%d,w=%d]", mat.cv_mat.cols, mat.cv_mat.rows);
                return false;
            }
        }

        bool Resize::ImplByOpenCV(Mat &mat, Tensor &tensor, vision::PreprocessParams *params) {
            try {
                int origin_w = mat.width();
                int origin_h = mat.height();
                params->src_width = origin_w;
                params->src_height = origin_h;
                params->dst_width = width_;
                params->dst_height = height_;
                if (use_rgb_) {
                    cv::cvtColor(mat.cv_mat, mat.cv_mat, cv::COLOR_BGR2RGB);
                    mat.set_pixel_format(sd::RGB);
                    tensor.desc.format = sd::RGB;
                }
                if (width_ == origin_w && height_ == origin_h) {
                    params->scale_width = 1.0;
                    params->scale_height = 1.0;
                    params->resize_width = width_;
                    params->resize_height = height_;
                    return true;
                }
                if (width_ > 0 && height_ > 0) {
                    if (right_pad_) {
                        // ocr right padding
                        float ratio_h = float(height_) / float(origin_h);
                        int new_w = int(ratio_h * float(origin_w));
                        if (new_w > width_) {
                            float percent_x = float(width_) / float(origin_w);
                            float percent_y = float(height_) / float(origin_h);
                            cv::resize(mat.cv_mat, mat.cv_mat, cv::Size(width_, height_));
                            params->scale_width = percent_x;
                            params->scale_height = percent_y;
                            params->resize_width = width_;
                            params->resize_height = height_;
                            params->pad_height_top = 0;
                            params->pad_height_bottom = 0;
                            params->pad_width_left = 0;
                            params->pad_width_right = 0;
                        } else {
                            cv::resize(mat.cv_mat, mat.cv_mat, cv::Size(new_w, height_));
                            cv::copyMakeBorder(mat.cv_mat, mat.cv_mat, 0, 0, 0, int(width_ - new_w),
                                               cv::BorderTypes::BORDER_CONSTANT, 0);
                            params->scale_width = 1.0;
                            params->scale_height = ratio_h;
                            params->resize_width = new_w;
                            params->resize_height = height_;
                            params->pad_height_top = 0;
                            params->pad_height_bottom = 0;
                            params->pad_width_left = 0;
                            params->pad_width_right = int(width_ - new_w);
                        }
                    } else if (use_pad_) {
                        // letterbox
                        float percent = std::min(float(width_) / float(origin_w), float(height_) / float(origin_h));
                        int new_width = int(origin_w * percent);
                        int new_height = int(origin_h * percent);
                        cv::resize(mat.cv_mat, mat.cv_mat, cv::Size(new_width, new_height), 0, 0);
                        float dw = width_ - new_width;
                        float dh = height_ - new_height;
                        dw /= 2.0f;
                        dh /= 2.0f;
                        int top = int(std::round(dh - 0.1f));
                        int bottom = int(std::round(dh + 0.1f));
                        int left = int(std::round(dw - 0.1f));
                        int right = int(std::round(dw + 0.1f));
                        /// pad的颜色修改成0，114部分网络如detr会造成网络输出的box偏移
                        cv::copyMakeBorder(mat.cv_mat, mat.cv_mat, top, bottom, left, right, cv::BORDER_CONSTANT,0);
                        params->scale_width = percent;
                        params->scale_height = percent;
                        params->resize_width = new_width;
                        params->resize_height = new_height;
                        params->pad_height_top = top;
                        params->pad_height_bottom = bottom;
                        params->pad_width_left = left;
                        params->pad_width_right = right;
                    } else if (resize_short_ != -1) {
                        float percent = float(resize_short_) / float(std::min(origin_w, origin_h));
                        int new_width = int(origin_w * percent);
                        int new_height = int(origin_h * percent);
                        cv::Mat resize_img;
                        cv::resize(mat.cv_mat, resize_img, cv::Size(new_width, new_height), 0, 0);
                        //// 计算裁剪起始点
                        int w_start = (new_width - width_) / 2;
                        int h_start = (new_height - height_) / 2;
                        //// 计算裁剪结束点
                        int w_end = w_start + width_;
                        int h_end = h_start + height_;
                        //// 检查边界条件
                        if (w_start < 0 || h_start < 0 || w_end > new_width || h_end > new_height) {
                            STDEPLOY_ERROR("crop out of size");
                            return false;
                        }
                        //// 裁剪图像
                        cv::Rect roi(w_start, h_start, width_, height_);
                        mat.cv_mat = resize_img(roi).clone();
                        params->scale_width = percent;
                        params->scale_height = percent;
                        params->resize_width = width_;
                        params->resize_height = height_;
                        params->pad_height_top = 0;
                        params->pad_height_bottom = 0;
                        params->pad_width_left = 0;
                        params->pad_width_right = 0;
                    } else {
                        float percent_x = float(width_) / float(origin_w);
                        float percent_y = float(height_) / float(origin_h);
                        cv::resize(mat.cv_mat, mat.cv_mat, cv::Size(width_, height_), 0, 0);
                        params->scale_width = percent_x;
                        params->scale_height = percent_y;
                        params->resize_width = width_;
                        params->resize_height = height_;
                    }
                    mat.set_height(height_);
                    mat.set_width(width_);
                } else {
                    SDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
                               "or (scale_w > 0 && scale_h > 0)."
                            << std::endl;
                    return false;
                }
                return true;
            } catch (const cv::Exception &e) {
                STDEPLOY_ERROR("resize error,mat:[h=%d,w=%d]", mat.cv_mat.cols, mat.cv_mat.rows);
                return false;
            }
        }


#ifdef ENABLE_RKNPU2_BACKEND

        bool Resize::ImplByRKRga(stdeploy::Mat &mat, stdeploy::Tensor &tensor, vision::PreprocessParams *params) {
            int origin_w = mat.width();
            int origin_h = mat.height();
            params->src_width = origin_w;
            params->src_height = origin_h;
            params->dst_width = width_;
            params->dst_height = height_;
            if (use_rgb_) {
                tensor.desc.format = sd::RGB;
            }
            if (width_ > 0 && height_ > 0) {
                if (!use_pad_) {
                    float percent_x = float(width_) / float(origin_w);
                    float percent_y = float(height_) / float(origin_h);
                    params->scale_width = percent_x;
                    params->scale_height = percent_y;
                    params->resize_width = width_;
                    params->resize_height = height_;
                } else {
                    float percent = std::min(float(width_) / float(origin_w), float(height_) / float(origin_h));
                    int new_width = int(origin_w * percent);
                    int new_height = int(origin_h * percent);
                    float dw = width_ - new_width;
                    float dh = height_ - new_height;
                    dw /= 2.0f;
                    dh /= 2.0f;
                    int top = int(std::round(dh - 0.1f));
                    int bottom = int(std::round(dh + 0.1f));
                    int left = int(std::round(dw - 0.1f));
                    int right = int(std::round(dw + 0.1f));
                    params->scale_width = percent;
                    params->scale_height = percent;
                    params->resize_width = new_width;
                    params->resize_height = new_height;
                    params->pad_height_top = top;
                    params->pad_height_bottom = bottom;
                    params->pad_width_left = left;
                    params->pad_width_right = right;
                }
                int ret = 0;
#ifdef ENABLE_RKNPU2_RV1106
                if(origin_w % 4 == 0) {
#else
                if (origin_w % 16 == 0) {
#endif
                    ret = rga_resize(mat, tensor, params);
                    if (ret != 0) {
                        STDEPLOY_WARNING("try convert image use cpu");
                        params->is_letterbox = true;
                        ret = vision::utils::crop_and_resize_image_cpu(mat, tensor, params);
                    }
                } else {
                    STDEPLOY_WARNING("src width is not 16-aligned, convert image use cpu");
                    params->is_letterbox = true;
                    ret = vision::utils::crop_and_resize_image_cpu(mat, tensor, params);
                }
                if (ret != 0) {
                    STDEPLOY_ERROR("rga resize failed");
                    return false;
                }
            } else {
                SDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
                           "or (scale_w > 0 && scale_h > 0)."
                        << std::endl;
                return false;
            }
            return true;
        }

#endif

#if defined(ENABLE_MSTAR_BACKEND)

        bool Resize::ImplByMstarSCL(stdeploy::Mat &mat, stdeploy::Tensor &tensor, vision::PreprocessParams *params) {
            int origin_w = mat.width();
            int origin_h = mat.height();
            params->src_width = origin_w;
            params->src_height = origin_h;
            params->dst_width = width_;
            params->dst_height = height_;
            tensor.desc.format = sd::NV12;
            if (width_ > 0 && height_ > 0) {
                if (!use_pad_) {
                    float percent_x = float(width_) / float(origin_w);
                    float percent_y = float(height_) / float(origin_h);
                    params->scale_width = percent_x;
                    params->scale_height = percent_y;
                    params->resize_width = width_;
                    params->resize_height = height_;
                } else {
                    float percent = std::min(float(width_) / float(origin_w), float(height_) / float(origin_h));
                    int new_width = int(origin_w * percent);
                    int new_height = int(origin_h * percent);
                    float dw = width_ - new_width;
                    float dh = height_ - new_height;
                    dw /= 2.0f;
                    dh /= 2.0f;
                    int top = int(std::round(dh - 0.1f));
                    int bottom = int(std::round(dh + 0.1f));
                    int left = int(std::round(dw - 0.1f));
                    int right = int(std::round(dw + 0.1f));
                    params->scale_width = percent;
                    params->scale_height = percent;
                    params->resize_width = new_width;
                    params->resize_height = new_height;
                    params->pad_height_top = top;
                    params->pad_height_bottom = bottom;
                    params->pad_width_left = left;
                    params->pad_width_right = right;
                }
                int ret;
                if ((origin_w % 2 == 0) && (origin_h % 2 == 0)) {
                    ret = mi_scl_resize(mat, tensor, params);
                    if (ret != 0) {
                        STDEPLOY_WARNING("try convert image use cpu");
                        params->is_letterbox = true;
                        ret = utils::crop_and_resize_image_cpu(mat, tensor, params);
                    }
                } else {
                    STDEPLOY_WARNING("src width is not 2-aligned, convert image use cpu");
                    params->is_letterbox = true;
                    ret = utils::crop_and_resize_image_cpu(mat, tensor, params);
                }
                if (ret != 0) {
                    STDEPLOY_ERROR("mi scl resize failed");
                    return false;
                }
            } else {
                SDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
                           "or (scale_w > 0 && scale_h > 0)."
                        << std::endl;
                return false;
            }
            return true;
        }

#endif


    } //namespace vision
} //namespace stdeploy
