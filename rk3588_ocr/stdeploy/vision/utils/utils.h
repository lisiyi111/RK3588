/**
  *************************************************
  * @file               :utils.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/8/31                
  *************************************************
  */

#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "stdeploy/vision/common/mat.h"
#include "stdeploy/vision/common/vision_struct.h"
#include "stdeploy/vision/utils/sigmoid.h"
#include "stdeploy/vision/utils/softmax.h"
#include "stdeploy/vision/utils/argmax.h"
#include "stdeploy/vision/common/result.h"

#define ALIGN_UP(val, alignment) ((( (val)+(alignment)-1)/(alignment))*(alignment))
#define ST_PI 3.1415926535898f

namespace stdeploy {
    namespace vision {
        namespace utils {

            // Compare the contour area
            STDEPLOY_DECL bool cmp_area_contour(const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2);

            // Compare the point y after x
            STDEPLOY_DECL bool comparePoints(const cv::Point2f &a, const cv::Point2f &b);

            // sort the 4 points to Clockwise
            STDEPLOY_DECL std::vector<cv::Point2f> sort_points_clockwise(const std::vector<cv::Point2f> &points);

            static inline float fast_exp(float x) {
                // return exp(x);
                union {
                    uint32_t i;
                    float f;
                } v;
                v.i = (12102203.1616540672 * x + 1064807160.56887296);
                return v.f;
            }

            static inline int AlignData(int value) {
#ifdef ENABLE_MSTAR_339G
                return ALIGN_UP(value,8);
#elif ENABLE_ACL_SVP_BACKEND
#ifdef ENABLE_MNN_BACKEND
                return value;
#else
                return ALIGN_UP(value, 4);
#endif
#else
                return value;
#endif
            }

            template<typename T>
            T clamp(T val, T min, T max) {
                return val > min ? (val < max ? val : max) : min;
            }


            // Top n
            template<typename T>
            std::vector<int32_t> TopKIndices(const T *array, int array_size, int topk) {
                topk = std::min(array_size, topk);
                std::vector<int32_t> res(topk);
                std::set<int32_t> searched;

                for (int32_t i = 0; i < topk; ++i) {
                    T min = static_cast<T>(-99999999);
                    for (int32_t j = 0; j < array_size; ++j) {
                        if (searched.find(j) != searched.end()) {
                            // 如果set中存在跳过当前array值
                            continue;
                        }
                        if (array[j] > min) {
                            res[i] = j;
                            min = *(array + j); // *(array+j) = array[j]
                        }
                    }
                    searched.insert(res[i]);
                }
                return res;
            }

            // L2 Norm  (for face recognition, ...)
            STDEPLOY_DECL std::vector<float> L2Normalize(const std::vector<float> &values);

            // crop image by opencv
            STDEPLOY_DECL cv::Mat crop_img_by_rect(const cv::Mat &src_img, cv::Rect &roi);

            STDEPLOY_DECL cv::Mat crop_img_by_min_rect(const cv::Mat &src_img, std::vector<cv::Point> &points);

            /// 透视变化拉直图形
            STDEPLOY_DECL cv::Mat crop_img_by_transform_ocr_preprocess(const cv::Mat &src_img, cv::Rect &rect,
                                                                       std::vector<cv::Point2f> &points);

            // ocr right padding
            STDEPLOY_DECL cv::Mat ocr_pad_preprocess(const cv::Mat &img, int width, int height);

            // crop and scale rgb image by c
            STDEPLOY_DECL int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                                     int crop_x, int crop_y, int crop_width, int crop_height,
                                                     unsigned char *dst, int dst_width, int dst_height,
                                                     int dst_box_x, int dst_box_y, int dst_box_width,
                                                     int dst_box_height);

            // crop and scale nv12 image by c
            STDEPLOY_DECL int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                                            int crop_x, int crop_y, int crop_width, int crop_height,
                                                            unsigned char *dst, int dst_width, int dst_height,
                                                            int dst_box_x, int dst_box_y, int dst_box_width,
                                                            int dst_box_height);
            // crop and scale mat by c
            STDEPLOY_DECL int crop_and_resize_image_cpu(stdeploy::Mat &src_mat,
                                                        stdeploy::Tensor &dst_tensor,
                                                        stdeploy::vision::PreprocessParams *params);


            // cosine similarity
            STDEPLOY_DECL float CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b,
                                                 bool normalized);

            // nms
            static inline bool cmp_score(const Object &a, const Object &b) {
                return a.score > b.score;
            }

            STDEPLOY_DECL float cal_iou(const Object &a, const Object &b);

            STDEPLOY_DECL int
            nms_sort_boxes(std::vector<Object> &objects, std::vector<int> &keptIndices, float nms_thresh);

            // rotate nms
            STDEPLOY_DECL void GetCovarianceMatrix(const ObbBox &Box, float &A, float &B, float &C);

            STDEPLOY_DECL float cal_probiou(const ObbBox &a, const ObbBox &b);

            STDEPLOY_DECL int
            rotate_nms_sort_boxes(std::vector<ObbBox> &objects, std::vector<int> &keptIndices, float nms_thresh);

            STDEPLOY_DECL std::array<float, 8> rbbox_to_corners(const std::vector<float> &rbbox);

            // face align
            /** \brief Do face align for model with five points.
           *
           * \param[in] image The original image
           * \param[in] result FaceDetectionResult
           * \param[in] std_landmarks Standard face template
           * \param[in] output_size The size of output mat
           */
            STDEPLOY_DECL std::vector<cv::Mat> AlignFaceWithFivePoints(
                    cv::Mat &image, vision::DetectionResult &result,
                    std::vector<std::array<float, 2>> std_landmarks = {{38.2946f, 51.6963f},
                                                                       {73.5318f, 51.5014f},
                                                                       {56.0252f, 71.7366f},
                                                                       {41.5493f, 92.3655f},
                                                                       {70.7299f, 92.2041f}},
                    std::array<int, 2> output_size = {112, 112});

            STDEPLOY_DECL cv::Mat AlignFaceWithFivePoints(
                    cv::Mat &image, std::vector<std::array<float, 2>> &landmarks,
                    std::vector<std::array<float, 2>> std_landmarks = {{38.2946f, 51.6963f},
                                                                       {73.5318f, 51.5014f},
                                                                       {56.0252f, 71.7366f},
                                                                       {41.5493f, 92.3655f},
                                                                       {70.7299f, 92.2041f}},
                    std::array<int, 2> output_size = {112, 112});

            // pixel format transform
            // pixel format to nv12e
            STDEPLOY_DECL void
            rgb_to_nv12(unsigned char *src, int width, int height, unsigned char *dst, bool fill_uv = false);

            STDEPLOY_DECL void
            bgr_to_nv12(unsigned char *src, int width, int height, unsigned char *dst, bool fill_uv = false);

            STDEPLOY_DECL void nv21_to_nv12(unsigned char *src, int width, int height, unsigned char *dst);

            STDEPLOY_DECL void yu12_to_nv12(unsigned char *src, int width, int height, unsigned char *dst);

            STDEPLOY_DECL void
            jpeg_stream_to_nv12(unsigned char *src, int width, int height, int size, unsigned char *dst);


            // pixel format to rgb

            STDEPLOY_DECL void nv12_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst, bool is_rgb = true);

            STDEPLOY_DECL void nv21_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst);

            STDEPLOY_DECL void yu12_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst);

            STDEPLOY_DECL void jpeg_stream_to_rgb(unsigned char *src, int width, int height, int size, cv::Mat &dst);

            // read jpeg file to opencv mat,default bgr
            STDEPLOY_DECL cv::Mat ReadJpegToMat(char *img_path);

            // layout
            STDEPLOY_DECL void chw_to_hwc(const float *nchw, size_t N, size_t C, size_t H, size_t W, float *nhwc);

            // nc1hwc2 int8 data to nchw int8 data
            STDEPLOY_DECL int
            NC1HWC2_int8_to_NCHW_int8(const int8_t *src, int8_t *dst, int *dims, int channel, int h, int w);

            // nc1hwc2 int8 data to nchw float data
            STDEPLOY_DECL int
            NC1HWC2_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int channel, int h, int w,
                                       int zp, float scale);

            // nchw int8 data to nchw float32 data
            STDEPLOY_DECL int NCHW_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int zp, float scale);

            // nchw int8 data to nhwc float32 data
            STDEPLOY_DECL int NHWC_int8_to_NHWC_float(const int8_t *src, float *dst, int *dims, int zp, float scale);

        }//namespace utils
    }//namespace vision
}//namespace stdeploy