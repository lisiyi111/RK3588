/**
  *************************************************
  * @file               :mat.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/9/25                
  *************************************************
  */


#include "stdeploy/vision/common/mat.h"

namespace stdeploy {


    Mat::Mat(int h, int w, int channel, PixelFormat format, DataType type, void *data) {
        height_ = h;
        width_ = w;
        format_ = format;
        type_ = type;
        batch_size_ = 1;
        channel_ = channel;
        if (format_ == NV12 || format_ == NV21) {
            size_ = height_ * width_ * 3 / 2;
            vir_addr = data; // data ptr output malloc
        } else if (format_ == BGR || format_ == RGB) {
            size_ = height_ * width_ * channel;
            cv_mat = cv::Mat(height_, width_, CV_8UC3, data);
        }
        bytes_ = size_ * ReadDataTypeSize(type_);
    }

    Mat::Mat(const cv::Mat &mat) {
        cv_mat = mat.clone();
        format_ = BGR;
        height_ = cv_mat.rows;
        width_ = cv_mat.cols;
        channel_ = cv_mat.channels();
        type_ = FP32;
        size_ = height_ * width_ * channel_;
        bytes_ = size_ * ReadDataTypeSize(type_);
        batch_size_ = 1;
    }

    Mat::Mat(cv::Mat &mat) {
        cv_mat = mat.clone();
        format_ = BGR;
        height_ = cv_mat.rows;
        width_ = cv_mat.cols;
        channel_ = cv_mat.channels();
        type_ = FP32;
        size_ = height_ * width_ * channel_;
        bytes_ = size_ * ReadDataTypeSize(type_);
        batch_size_ = 1;
    }

    void *Mat::data() {
        if (vir_addr != nullptr) {
            return vir_addr;
        } else {
            return static_cast<void *>(cv_mat.data);
        }
    }


    Mat::~Mat() {
        if (mem_buffer_ != nullptr) {
            free(mem_buffer_);
            mem_buffer_ = nullptr;
        }
    }

    Mat::Mat(int h, int w, int channel, PixelFormat format, DataType type, void *virAddr, int32_t fdAddr) {
        height_ = h;
        width_ = w;
        format_ = format;
        type_ = type;
        batch_size_ = 1;
        channel_ = channel;
        if (format_ == NV12 || format_ == NV21) {
            size_ = height_ * width_ * 3 / 2;
        } else if (format_ == BGR || format_ == RGB) {
            size_ = height_ * width_ * channel;
        }
        vir_addr = virAddr; // data ptr output malloc
        bytes_ = size_ * ReadDataTypeSize(type_);
        fd = fdAddr;
    }

    Mat::Mat(int h, int w, int channel, PixelFormat format, DataType type, void *virAddr, uint64_t phyAddr) {
        height_ = h;
        width_ = w;
        format_ = format;
        type_ = type;
        batch_size_ = 1;
        channel_ = channel;
        if (format_ == NV12 || format_ == NV21) {
            size_ = height_ * width_ * 3 / 2;
        } else if (format_ == BGR || format_ == RGB) {
            size_ = height_ * width_ * channel;
        }
        vir_addr = virAddr; // data ptr output malloc
        bytes_ = size_ * ReadDataTypeSize(type_);
        phy_addr = reinterpret_cast<void *>(phyAddr);
    }

    void Mat::creat(int h, int w, int channel, PixelFormat format, DataType type, void *mem_buffer) {
        height_ = h;
        width_ = w;
        format_ = format;
        type_ = type;
        batch_size_ = 1;
        channel_ = channel;
        if (format_ == NV12 || format_ == NV21) {
            size_ = height_ * width_ * 3 / 2;
            mem_buffer_ = (unsigned char *) malloc(size_);
            memcpy(mem_buffer_, mem_buffer, size_);
            vir_addr = mem_buffer_;
        } else if (format_ == BGR || format_ == RGB) {
            size_ = height_ * width_ * channel;
            cv_mat = cv::Mat(height_, width_, CV_8UC3, mem_buffer);
        }
        bytes_ = size_ * ReadDataTypeSize(type_);
    }


} //namespace stdeploy
