/**
  *************************************************
  * @file               :sd_tensor.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/11                
  *************************************************
  */

#include "stdeploy/core/sd_tensor.h"
#include <sstream>

namespace stdeploy {

    std::string shape_string(const TensorShape &shape) {
        if (shape.empty()) {
            return "0";
        }
        std::stringstream ss;
        ss << shape[0];
        for (size_t i = 1; i < shape.size(); ++i) ss << "," << shape[i];
        return ss.str();
    }

    void *Tensor::data() {
        return desc.mem.virAddr;
    }

    const void *Tensor::data() const {
        return desc.mem.virAddr;
    }

    void Tensor::set_data(void *src) {
        desc.mem.virAddr = src;
    }

    int Tensor::bytes() const {
        return size() * ReadDataTypeSize(desc.data_type);
    }

    int Tensor::size() const {
        return std::accumulate(desc.shape.begin(), desc.shape.end(), 1, std::multiplies<int>());
    }

    void Tensor::set_shape(std::vector<int> &shape) {
        desc.shape = shape;
    }

} //namespace stdeploy