/**
  *************************************************
  * @file               :pixel_format_transform.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/9/25                
  *************************************************
  */

#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace utils {


            cv::Mat ReadJpegToMat(char *img_path) {
                std::ifstream file(img_path, std::ifstream::in | std::ifstream::binary);
                if (!file.is_open()) {
                    std::cerr << "ERROR open img path" << std::endl;
                    std::exit(-1);
                }
                file.seekg(0, std::ifstream::end);
                size_t fileSize = file.tellg();
                file.seekg(0, std::ifstream::beg);
                char *buffer = new char[fileSize];
                file.read(buffer, fileSize);
                cv::Mat temp_img = cv::imdecode(cv::Mat(1, fileSize, CV_8UC1, buffer), cv::IMREAD_COLOR);
                // buffer
                delete[] buffer;
                return temp_img;
            }

            void rgb_to_nv12(unsigned char *src, int width, int height, unsigned char *dst, bool fill_uv) {
                cv::Mat img = cv::Mat(height, width, CV_8UC3, src);
                cv::cvtColor(img, img, cv::COLOR_RGB2YUV_I420);
                int usize = width * height / 4;
                char *y = (char *) img.data;
                char *u = y + width * height;
                char *v = u + usize;
                memcpy(dst, y, width * height);
                uchar *uv = dst + width * height;
                for (int i = 0; i < usize; i++) {
                    if (!fill_uv) {
                        uv[i * 2] = u[i];
                        uv[i * 2 + 1] = v[i];
                    } else {
                        uv[i * 2] = v[i];
                        uv[i * 2 + 1] = u[i];
                    }
                }
            }

            void bgr_to_nv12(unsigned char *src, int width, int height, unsigned char *dst, bool fill_uv) {
                cv::Mat img = cv::Mat(height, width, CV_8UC3, src);
                cv::Mat yuvMat;
                cv::cvtColor(img, yuvMat, cv::COLOR_BGR2YUV_I420);
                int usize = width * height / 4;
                char *y = (char *) yuvMat.data;
                char *u = y + width * height;
                char *v = u + usize;
                memcpy(dst, y, width * height);
                uchar *uv = dst + width * height;
                for (int i = 0; i < usize; i++) {
                    if (!fill_uv) {
                        uv[i * 2] = u[i];
                        uv[i * 2 + 1] = v[i];
                    } else {
                        uv[i * 2] = v[i];
                        uv[i * 2 + 1] = u[i];
                    }
                }
            }

            void nv21_to_nv12(unsigned char *src, int width, int height, unsigned char *dst) {
                int uvsize = width * height / 4;
                memcpy(dst, src, width * height);
                uchar *vu = src + width * height;
                uchar *uv = dst + width * height;
                for (int i = 0; i < uvsize; i++) {
                    uv[i * 2] = vu[i * 2 + 1];
                    uv[i * 2 + 1] = vu[i * 2];
                }
            }

            void yu12_to_nv12(unsigned char *src, int width, int height, unsigned char *dst) {
                int usize = width * height / 4;
                uchar *u = src + width * height;
                uchar *v = u + usize;
                memcpy(dst, src, width * height);
                uchar *uv = dst + width * height;
                for (int i = 0; i < usize; i++) {
                    uv[i * 2] = u[i];
                    uv[i * 2 + 1] = v[i];
                }
            }

            void jpeg_stream_to_nv12(unsigned char *src, int width, int height, int size, unsigned char *dst) {
                cv::Mat bgr_img = cv::imdecode(cv::Mat(1, size, CV_8UC1, src), cv::IMREAD_COLOR);
                cv::cvtColor(bgr_img, bgr_img, cv::COLOR_BGR2YUV_I420);
                int usize = width * height / 4;
                char *y = (char *) bgr_img.data;
                char *u = y + width * height;
                char *v = u + usize;
                memcpy(dst, y, width * height);
                uchar *uv = dst + width * height;
                for (int i = 0; i < usize; i++) {
                    uv[i * 2] = u[i];
                    uv[i * 2 + 1] = v[i];
                }
            }

            void nv12_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst, bool is_rgb) {
                cv::Mat nv12_img(height * 1.5, width, CV_8UC1, src);
                if (is_rgb) {
                    cv::cvtColor(nv12_img, dst, cv::COLOR_YUV2RGB_NV12);
                } else {
                    cv::cvtColor(nv12_img, dst, cv::COLOR_YUV2BGR_NV12);
                }
            }

            void nv21_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst) {
                cv::Mat nv21_img(height * 1.5, width, CV_8UC1, src);
                cv::cvtColor(nv21_img, dst, cv::COLOR_YUV2RGB_NV21);
            }

            void yu12_to_rgb(unsigned char *src, int width, int height, cv::Mat &dst) {
                cv::Mat yu12_img(height * 1.5, width, CV_8UC1, src);
                cv::cvtColor(yu12_img, dst, cv::COLOR_YUV2RGB_I420);
            }

            void jpeg_stream_to_rgb(unsigned char *src, int width, int height, int size, cv::Mat &dst) {
                cv::Mat img = cv::imdecode(cv::Mat(1, size, CV_8UC1, src), cv::IMREAD_COLOR);
                cv::cvtColor(img, dst, cv::COLOR_BGR2RGB);
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy