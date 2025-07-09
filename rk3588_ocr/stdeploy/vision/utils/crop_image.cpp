/**
  *************************************************
  * @file               :crop_image.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/8/20                
  *************************************************
  */

#include "stdeploy/vision/utils/utils.h"
#include "stdeploy/core/sd_tensor.h"

namespace stdeploy {
    namespace vision {
        namespace utils {

            cv::Mat ocr_pad_preprocess(const cv::Mat &img, int width, int height) {
                int rec_img_height_ = height;
                int rec_img_width_ = width;
                int input_width_ = img.cols;
                int input_height_ = img.rows;
                cv::Mat resize_img;
                float ratio_h = float(rec_img_height_) / float(input_height_);
                int new_w = int(ratio_h * float(input_width_));
                if (new_w > rec_img_width_) {
                    cv::resize(img, resize_img, cv::Size(rec_img_width_, rec_img_height_));
                } else {
                    cv::resize(img, resize_img, cv::Size(new_w, rec_img_height_));
                    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(rec_img_width_ - new_w),
                                       cv::BorderTypes::BORDER_CONSTANT, 0);
                }
                return resize_img;
            }

            cv::Mat crop_img_by_rect(const cv::Mat &src_img, cv::Rect &roi) {
                return src_img(roi).clone();
            }

            cv::Mat crop_img_by_min_rect(const cv::Mat &src_img, std::vector<cv::Point> &points) {
                // points is clockwise or counter clockwise
                cv::Rect rect = cv::boundingRect(points);
                return src_img(rect).clone();
            }

            cv::Mat crop_img_by_transform_ocr_preprocess(const cv::Mat &src_img, cv::Rect &rect,
                                                         std::vector<cv::Point2f> &points) {
                // points is clockwise or counter clockwise

                cv::Mat crop_img = crop_img_by_rect(src_img, rect);
                int crop_img_h = crop_img.rows;
                int crop_img_w = crop_img.cols;

                std::vector<cv::Point2f> dst_points;
                dst_points.push_back(cv::Point2f(0, 0));
                dst_points.push_back(cv::Point2f(crop_img_w - 1, 0));
                dst_points.push_back(cv::Point2f(crop_img_w - 1, crop_img_h - 1));
                dst_points.push_back(cv::Point2f(0, crop_img_h - 1));

                cv::Mat perspectiveTransform = cv::getPerspectiveTransform(points, dst_points);
                cv::Mat dst;
                cv::warpPerspective(crop_img, dst, perspectiveTransform, cv::Size(crop_img_w, crop_img_h));

                return dst;
            }


            int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                       int crop_x, int crop_y, int crop_width, int crop_height,
                                       unsigned char *dst, int dst_width, int dst_height,
                                       int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
                if (dst == NULL) {
                    printf("dst buffer is null\n");
                    return -1;
                }

                float x_ratio = (float) crop_width / (float) dst_box_width;
                float y_ratio = (float) crop_height / (float) dst_box_height;

                // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
                //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
                // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
                //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
                // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

                // 从原图指定区域取数据，双线性缩放到目标指定区域
                for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
                    for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
                        int dst_x_offset = dst_x - dst_box_x;
                        int dst_y_offset = dst_y - dst_box_y;

                        int src_x = (int) (dst_x_offset * x_ratio) + crop_x;
                        int src_y = (int) (dst_y_offset * y_ratio) + crop_y;

                        float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
                        float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

                        int index1 = src_y * src_width * channel + src_x * channel;
                        int index2 = index1 + src_width * channel;    // down
                        if (src_y == src_height - 1) {
                            // 如果到图像最下边缘，变成选择上面的像素
                            index2 = index1 - src_width * channel;
                        }
                        int index3 = index1 + 1 * channel;            // right
                        int index4 = index2 + 1 * channel;            // down right
                        if (src_x == src_width - 1) {
                            // 如果到图像最右边缘，变成选择左边的像素
                            index3 = index1 - 1 * channel;
                            index4 = index2 - 1 * channel;
                        }

                        // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
                        //     dst_x, dst_y, dst_x_offset, dst_y_offset,
                        //     src_x, src_y, x_diff, y_diff,
                        //     index1, index2, index3, index4);

                        for (int c = 0; c < channel; c++) {
                            unsigned char A = src[index1 + c];
                            unsigned char B = src[index3 + c];
                            unsigned char C = src[index2 + c];
                            unsigned char D = src[index4 + c];

                            unsigned char pixel = (unsigned char) (
                                    A * (1 - x_diff) * (1 - y_diff) +
                                    B * x_diff * (1 - y_diff) +
                                    C * y_diff * (1 - x_diff) +
                                    D * x_diff * y_diff
                            );

                            dst[(dst_y * dst_width + dst_x) * channel + c] = pixel;
                        }
                    }
                }

                return 0;
            }


            int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                              int crop_x, int crop_y, int crop_width, int crop_height,
                                              unsigned char *dst, int dst_width, int dst_height,
                                              int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {

                unsigned char *src_y = src;
                unsigned char *src_uv = src + src_width * src_height;

                unsigned char *dst_y = dst;
                unsigned char *dst_uv = dst + dst_width * dst_height;

                crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height,
                                       dst_y, dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width,
                                       dst_box_height);

                crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2,
                                       crop_height / 2,
                                       dst_uv, dst_width / 2, dst_height / 2, dst_box_x, dst_box_y, dst_box_width,
                                       dst_box_height);

                return 0;
            }


            int crop_and_resize_image_cpu(stdeploy::Mat &src_mat,
                                          stdeploy::Tensor &dst_tensor,
                                          vision::PreprocessParams *params) {
                /// 特别注意:需要再调用的地方开辟好dst_tensor的内存和设置好size
                if (src_mat.data() == NULL) {
                    STDEPLOY_ERROR("src mem null");
                    return -1;
                }
                if (dst_tensor.data() == NULL) {
                    STDEPLOY_ERROR("dst mem null");
                    return -1;
                }

                if (src_mat.pixel_format() != dst_tensor.desc.format) {
                    STDEPLOY_ERROR("convert_image_cpu only support pixel format same");
                    return -1;
                }
                int src_width = params->src_width;
                int src_height = params->src_height;
                int dst_width = params->dst_width;
                int dst_height = params->dst_height;

                int src_box_x = 0;
                int src_box_y = 0;
                int src_box_w = src_width;
                int src_box_h = src_height;
                /// 如果是crop_resize，注意修改起始点，letter或者resize不需要改动
                if (params->is_crop_resize && !params->is_letterbox) {
                    src_box_x = params->src_box_x;
                    src_box_y = params->src_box_y;
                    src_box_w = params->src_box_w;
                    src_box_h = params->src_box_h;
                }
                int dst_box_x = 0;
                int dst_box_y = 0;
                int dst_box_w = dst_width;
                int dst_box_h = dst_height;
                if (params->is_crop_resize && !params->is_letterbox) {
                    /// 注意做crop_resize修改box
                    dst_box_x = params->dst_box_x;
                    dst_box_y = params->dst_box_y;
                    dst_box_w = params->dst_box_w;
                    dst_box_h = params->dst_box_h;
                }
                if (!params->is_crop_resize && params->is_letterbox) {
                    /// 注意做letterbox修改box
                    dst_box_x = params->pad_width_left;
                    dst_box_y = params->pad_height_top;
                    dst_box_w = params->resize_width;
                    dst_box_h = params->resize_height;
                }

                // fill pad color
                if (dst_box_w != dst_width || dst_box_h != dst_height) {
                    int dst_size = dst_tensor.size();
                    memset(dst_tensor.data(), 0, dst_size);
                }

                int reti = 0;
                if (src_mat.pixel_format() == RGB) {
                    reti = crop_and_scale_image_c(3, (unsigned char *) src_mat.data(), src_width, src_height,
                                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                                  (unsigned char *) dst_tensor.data(), dst_width, dst_height,
                                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
                } else if (src_mat.pixel_format() == RGBA) {
                    reti = crop_and_scale_image_c(4, (unsigned char *) src_mat.data(), src_width, src_height,
                                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                                  (unsigned char *) dst_tensor.data(), dst_width, dst_height,
                                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
                } else if (src_mat.pixel_format() == GRAYSCALE) {
                    reti = crop_and_scale_image_c(1, (unsigned char *) src_mat.data(), src_width, src_height,
                                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                                  (unsigned char *) dst_tensor.data(), dst_width, dst_height,
                                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
                } else if (src_mat.pixel_format() == NV12 || src_mat.pixel_format() == NV21) {
                    reti = crop_and_scale_image_yuv420sp((unsigned char *) src_mat.data(), src_width, src_height,
                                                         src_box_x, src_box_y, src_box_w, src_box_h,
                                                         (unsigned char *) dst_tensor.data(), dst_width, dst_height,
                                                         dst_box_x, dst_box_y, dst_box_w, dst_box_h);
                } else {
                    STDEPLOY_ERROR("no support format %d\n", src_mat.pixel_format());
                }
                if (reti != 0) {
                    STDEPLOY_ERROR("convert_image_cpu fail %d\n", reti);
                    return -1;
                }
                return 0;
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy