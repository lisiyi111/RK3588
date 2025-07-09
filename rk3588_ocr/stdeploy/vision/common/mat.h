/**
  *************************************************
  * @file               :mat.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/9/25                
  *************************************************
  */

#pragma once

#include "stdeploy/core/sd_type.h"
#include <opencv2/opencv.hpp>

namespace stdeploy {

    class STDEPLOY_DECL Mat {
    public:
        Mat() = default;

        /**
         * @brief construct a Mat for an image
         * @param h height of an image
         * @param w width of an image
         * @param format pixel format of an image, rgb, bgr, gray etc. Note that in
         * case of nv12 or nv21, height is the real height of an image,
         * not height * 3 / 2
         * @param type data type of an pixel in each channel
         * @param device location Mat's buffer stores
         */
        Mat(int h, int w, int channel, PixelFormat format, DataType type, void *data);

        Mat(int h, int w, int channel, PixelFormat format, DataType type, void *virAddr, uint64_t phyAddr);

        Mat(int h, int w, int channel, PixelFormat format, DataType type, void *virAddr, int32_t fdAddr);

        Mat(cv::Mat &mat);

        Mat(const cv::Mat &mat);

        ~Mat();

        void creat(int h, int w, int channel, PixelFormat format, DataType type, void *mem_buffer);

        PixelFormat pixel_format() const { return format_; }

        DataType type() const { return type_; }

        int height() const { return height_; }

        int width() const { return width_; }

        int batch_size() const { return batch_size_; }

        int channel() const { return channel_; }

        int size() const { return size_; }

        int byte_size() const { return bytes_; }

        void set_height(int h) { height_ = h; }

        void set_width(int w) { width_ = w; }

        void set_channel(int c) { channel_ = c; }

        void set_size(int n) { size_ = n; }

        void set_byte_size(int n) { bytes_ = n; }

        void set_pixel_format(PixelFormat format) { format_ = format; }

        void set_data_type(DataType type) { type_ = type; }

        void *data();

        void set_cv_mat(const cv::Mat &mat) {
            cv_mat = mat;
        }

        ProcLib pro_lib() const { return m_pro_lib; }

        void set_pro_lib(ProcLib type) { m_pro_lib = type; }

        cv::Mat cv_mat;
        int32_t fd = 0;
        void *phy_addr = nullptr;
        void *vir_addr = nullptr;

    private:
        PixelFormat format_{PixelFormat::RGB};
        DataType type_{DataType::FP32};
        int width_{0};
        int height_{0};
        int channel_{0};
        int batch_size_{0};
        int size_{0};  // size of elements in mat
        int bytes_{0};
        ProcLib m_pro_lib = ProcLib::OPENCV;
        void *mem_buffer_ = nullptr;
    };

} //namespace stdeploy