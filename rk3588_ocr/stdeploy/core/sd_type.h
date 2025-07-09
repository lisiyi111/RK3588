/**
  *************************************************
  * @file               :sd_type.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/7                
  *************************************************
  */

#pragma once

#include "stdeploy/utils/log_util.h"

namespace stdeploy {

    enum STDEPLOY_DECL PixelFormat {
        BGR,
        RGB,
        GRAYSCALE,
        NV12,
        NV21,
        RGBA,
        FM,
        COUNT,
    };

    enum STDEPLOY_DECL LayoutType {
        NCHW,
        NHWC,
        LAYOUT_OTHER,
        LAYOUT_UNKNOWN,
    };

    enum STDEPLOY_DECL DataType {
        DATATYPE_UNKNOWN,
        FP16,
        FP32,
        INT4,
        INT8,
        UINT8,
        INT16,
        INT32,
        UINT32,
        INT64,
        BOOL,
    };

    enum class STDEPLOY_DECL ProcLib {
        OPENCV, // CPU
        RGA,    // RK
        CUDA,   // GPU
        MI_SCL, // sigMastar
    };

    STDEPLOY_DECL int ReadDataTypeSize(const DataType &type);

    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const DataType &f);

    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const LayoutType &f);

    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const ProcLib &f);

    STDEPLOY_DECL int get_pixel_format_size(int width, int height, PixelFormat pixel_format);

} //namespace stdeploy