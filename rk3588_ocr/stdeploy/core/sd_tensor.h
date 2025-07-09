/**
  *************************************************
  * @file               :sd_tensor.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/11                
  *************************************************
  */

#pragma once

#include <iostream>
#include "stdeploy/core/sd_type.h"

namespace stdeploy {

    using TensorShape = std::vector<int>;

    STDEPLOY_DECL std::string shape_string(const TensorShape &shape);

    struct STDEPLOY_DECL TensorMem {
        void *phyAddr = nullptr;     // zero-copy init phy addr
        void *virAddr = nullptr;    // vir addr
        int32_t fd{0};              // cma fd(rk)
        int32_t zp{0};               // tensor quantization zp(if int8)
        float scale{0.0};              // tensor quantization scale(if int8)
    };

    struct STDEPLOY_DECL TensorDesc {
        std::string name;       // tensor name (images...)
        TensorShape shape;      // tensor shape([1,3,224,224]...)
        DataType data_type;     // tensor data type(float32/float16/int8 ...)
        LayoutType layout;      // tensor layout(nchw/nhwc/other)
        PixelFormat format;     // a temp value to transform data to backend input data format(nv12/rgb)
        TensorMem mem{};        // Tensor mem
        friend std::ostream &operator<<(std::ostream &output,
                                        const TensorDesc &desc) {
            output << "TensorDesc [name: " << desc.name << ", shape: (";
            output << shape_string(desc.shape);
            output << "), data_type: " << desc.data_type << ",layout: ";
            output << desc.layout << "]";
            return output;
        }
    };

    struct STDEPLOY_DECL Tensor {
        TensorDesc desc{};

        // get mem vir addr
        void *data();

        // get const mem vir addr
        const void *data() const;

        // set mem vir addr
        void set_data(void *src);

        // set shape
        void set_shape(std::vector<int> &shape);

        // Total size of tensor memory buffer in bytes
        int bytes() const;

        // Total number of elements in tensor
        int size() const;
    };

} //namespace stdeploy