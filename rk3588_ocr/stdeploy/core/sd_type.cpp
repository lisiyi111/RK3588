/**
  *************************************************
  * @file               :sd_type.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/9/22                
  *************************************************
  */

#include "stdeploy/core/sd_type.h"
#include <iostream>

namespace stdeploy {

    std::ostream &operator<<(std::ostream &out, const DataType &type) {
        if (type == DataType::BOOL) {
            out << "DataType::BOOL";
        } else if (type == DataType::INT4) {
            out << "DataType::INT4";
        } else if (type == DataType::INT8) {
            out << "DataType::INT8";
        } else if (type == DataType::UINT8) {
            out << "DataType::UINT8";
        } else if (type == DataType::INT16) {
            out << "DataType::INT16";
        } else if (type == DataType::INT32) {
            out << "DataType::INT32";
        } else if (type == DataType::FP16) {
            out << "DataType::FP16";
        } else if (type == DataType::FP32) {
            out << "DataType::FP32";
        } else if (type == DataType::INT64) {
            out << "DataType::INT64";
        } else {
            out << "UNKNOWN-DataType";
        }
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const LayoutType &type) {
        if (type == LayoutType::NCHW) {
            out << "LayoutType::NCHW";
        } else if (type == LayoutType::NHWC) {
            out << "LayoutType::NHWC";
        } else if (type == LayoutType::LAYOUT_OTHER) {
            out << "LayoutType::OTHER";
        } else {
            out << "LayoutType::UNKNOWN";
        }
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const ProcLib &type) {
        if (type == ProcLib::OPENCV) {
            out << "ProcLib::OPENCV";
        } else if (type == ProcLib::RGA) {
            out << "LayoutType::RGA";
        } else if (type == ProcLib::CUDA) {
            out << "ProcLib::CUDA";
        }
        return out;
    }


    int ReadDataTypeSize(const DataType &type) {
        if (type == BOOL) {
            return sizeof(bool);
        } else if (type == INT4) {
            return sizeof(int8_t) / 2;
        } else if (type == INT8) {
            return sizeof(int8_t);
        } else if (type == UINT8) {
            return sizeof(uint8_t);
        } else if (type == INT16) {
            return sizeof(int16_t);
        } else if (type == INT32) {
            return sizeof(int32_t);
        } else if (type == FP16) {
            return sizeof(float) / 2;
        } else if (type == FP32) {
            return sizeof(float);
        } else if (type == INT64) {
            return sizeof(int64_t);
        } else {
            STDEPLOY_ERROR("data type not support %d", type);
            return -1;
        }
    }

    int get_pixel_format_size(int width, int height, PixelFormat format) {
        int size = 0;
        if (format == NV12 || format == NV21) {
            size = height * width * 3 / 2;
        } else if (format == BGR || format == RGB) {
            size = height * width * 3;
        } else if (format == GRAYSCALE) {
            size = height * width * 1;
        } else if (format == FM) {
            size = height * width;
        } else if (format == RGBA) {
            size = height * width * 4;
        }
        return size;
    }

} //namespace stdeploy