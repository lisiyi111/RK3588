/**
  *************************************************
  * @file               :enum_variables.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */

#include "stdeploy/runtime/enum_variables.h"

namespace stdeploy {

    /**
     * @brief get live backends
     */
    std::vector<Backend> GetAvailableBackends() {
        std::vector<Backend> backends;
        backends.clear();
#ifdef ENABLE_ORT_BACKEND
        backends.push_back(Backend::ORT);
#endif
#ifdef ENABLE_TRT_BACKEND
        backends.push_back(Backend::TRT);
#endif
#ifdef ENABLE_MSTAR_BACKEND
        backends.push_back(Backend::MSTAR);
#endif
#ifdef ENABLE_MNN_BACKEND
        backends.push_back(Backend::MNN);
#endif
#ifdef ENABLE_NCNN_BACKEND
        backends.push_back(Backend::NCNN);
#endif
#ifdef ENABLE_SNPE_BACKEND
        backends.push_back(Backend::SNPE);
#endif
#ifdef ENABLE_RKNPU2_BACKEND
        backends.push_back(Backend::RKNPU2);
#endif
#ifdef ENABLE_RKNPU_BACKEND
        backends.push_back(Backend::RKNPU);
#endif
#ifdef ENABLE_ACL_SVP_BACKEND
        backends.push_back(Backend::ACL_SVP);
#endif
        return backends;
    }

    /**
     * @brief judge backend live or not
     */
    bool IsBackendAvailable(const Backend &backend) {
        std::vector<Backend> backends = GetAvailableBackends();
        for (size_t i = 0; i < backends.size(); ++i) {
            if (backend == backends[i]) {
                return true;
            }
        }
        return false;
    }

    std::ostream &operator<<(std::ostream &out, const Backend &backend) {
        if (backend == Backend::ORT) {
            out << "Backend::ORT";
        } else if (backend == Backend::TRT) {
            out << "Backend::TRT";
        } else if (backend == Backend::RKNPU2) {
            out << "Backend::RKNPU2";
        } else if (backend == Backend::RKNPU) {
            out << "Backend::RKNPU";
        } else if (backend == Backend::MSTAR) {
            out << "Backend::MSTAR";
        } else if (backend == Backend::NCNN) {
            out << "Backend::NCNN";
        } else if (backend == Backend::MNN) {
            out << "Backend::MNN";
        } else if (backend == Backend::SNPE) {
            out << "Backend::SNPE";
        } else if (backend == Backend::ACL_SVP) {
            out << "Backend::ACL_SVP";
        } else {
            out << "UNKNOWN-Backend";
        }
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const Device &d) {
        switch (d) {
            case Device::CPU:
                out << "Device::CPU";
                break;
            case Device::GPU:
                out << "Device::GPU";
                break;
            case Device::RKNPU2D:
                out << "Device::RKNPU2-NPU";
                break;
            case Device::RKNPUD:
                out << "Device::RKNPU-NPU";
                break;
            case Device::MSTARD:
                out << "Device::MSTAR-IPU";
                break;
            case Device::SNPED:
                out << "Device::SNPE-X";
                break;
            case Device::ACL_SVPD:
                out << "Device::ACL_SVPD";
                break;
            default:
                out << "Device::UNKOWN";
        }
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const ModelFormat &format) {
        if (format == ModelFormat::img) {
            out << "ModelFormat::IMG";
        } else if (format == ModelFormat::onnx) {
            out << "ModelFormat::ONNX";
        } else if (format == ModelFormat::rknn) {
            out << "ModelFormat::RKNN";
        } else if (format == ModelFormat::mnn) {
            out << "ModelFormat::MNN";
        } else if (format == ModelFormat::ncnn) {
            out << "ModelFormat::NCNN";
        } else if (format == ModelFormat::dlc) {
            out << "ModelFormat::DLC";
        } else if (format == ModelFormat::engine) {
            out << "ModelFormat::ENGINE";
        } else if (format == ModelFormat::om) {
            out << "ModelFormat::OM";
        } else {
            out << "UNKNOWN-ModelFormat";
        }
        return out;
    }

} //namespace stdeploy