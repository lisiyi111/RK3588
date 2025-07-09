/**
  *************************************************
  * @file               :enum_variables.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */

#pragma once


#include <ostream>
#include <map>
#include <vector>
#include "stdeploy/utils/log_util.h"

namespace stdeploy {

    enum STDEPLOY_DECL Backend {
        UNKNOWN,  ///< Unknown inference backend
        ORT,  ///< onnx Runtime, support onnx format model, cpu
        TRT,  ///< tensorrt, support onnx/engine format model, Nvidia GPU only
        MSTAR,  ///< mstar, support img format model, Mstar npu only
        MNN,  ///< mnn, support mnn format model, cpu/gpu
        NCNN, ///< ncnn, support bin param format model, cpu/gpu
        SNPE,  ///< snpe, support dlc format model, GPU/CPU/DSP/AIP
        RKNPU2,  ///< rknpu2, support rknn format model, RK npu2 only
        RKNPU,  ///< rknpu2, support rknn format model, RK npu only
        NNIE, ///< hisi, support wk format model,hisi 3559a/3516dv300/3519a
        ACL_SVP, ///< hisi, support om format model,hisi 3519dv500
    };

    /**
     * @brief Get all the available inference backend in FastDeploy
     */
    STDEPLOY_DECL std::vector<Backend> GetAvailableBackends();

    /**
     * @brief Check if the inference backend available
     */
    STDEPLOY_DECL bool IsBackendAvailable(const Backend &backend);

    enum STDEPLOY_DECL Device {
        CPU,
        GPU,
        MSTARD,
        RKNPU2D,
        RKNPUD,
        SNPED,
        NNIED,
        ACL_SVPD,
    };

    /*! Deep learning model format */
    enum STDEPLOY_DECL ModelFormat {
        onnx,         ///< Model with ONNX format
        engine,       ///< Model with ENGINE format
        img,          ///< Model with IMG format
        mnn,          ///< Model with MNN format
        ncnn,          ///< Model with NCNN format
        dlc,          ///< Model with DLC format
        rknn,          ///< Model with RKNN format
        wk,             ///< Model with hisi nnie format
        om,             ///< Model with hisi acl format
    };

    /// Describle all the supported backends for specified model format
    static std::map<ModelFormat, std::vector<Backend>> s_default_backends_by_format = {
            {ModelFormat::onnx,   {Backend::ORT,    Backend::TRT},},
            {ModelFormat::engine, {Backend::TRT}},
            {ModelFormat::img,    {Backend::MSTAR}},
            {ModelFormat::mnn,    {Backend::MNN,    Backend::ACL_SVP}},
            {ModelFormat::ncnn,   {Backend::NCNN}},
            {ModelFormat::dlc,    {Backend::SNPE}},
            {ModelFormat::rknn,   {Backend::RKNPU2, Backend::RKNPU}},
            {ModelFormat::wk,     {Backend::NNIE}},
            {ModelFormat::om,     {Backend::ACL_SVP}},
    };

    /// Describle all the supported backends for specified device
    static std::map<Device, std::vector<Backend>> s_default_backends_by_device = {
            {Device::CPU,      {Backend::ORT, Backend::MNN, Backend::NCNN}},
            {Device::GPU,      {Backend::TRT}},
            {Device::MSTARD,   {Backend::MSTAR}},
            {Device::SNPED,    {Backend::SNPE}},
            {Device::RKNPU2D,  {Backend::RKNPU2}},
            {Device::RKNPUD,   {Backend::RKNPU}},
            {Device::NNIED,    {Backend::NNIE}},
            {Device::ACL_SVPD, {Backend::ACL_SVP}},
    };

    /**
     * @brief check model format match or mot with backend
     */
    inline bool Supported(ModelFormat format, Backend backend) {
        auto iter = s_default_backends_by_format.find(format);
        if (iter == s_default_backends_by_format.end()) {
            SDERROR << "Didn't find format is registered in " <<
                    "s_default_backends_by_format." << std::endl;
            return false;
        }
        for (size_t i = 0; i < iter->second.size(); ++i) {
            if (iter->second[i] == backend) {
                return true;
            }
        }
        std::string msg = Str(iter->second);
        SDERROR << format << " only supports " << msg << ", but now it's "
                << backend << "." << std::endl;
        return false;
    }

    /**
     * @brief check device match or mot with backend
     */
    inline bool Supported(Device device, Backend backend) {
        auto iter = s_default_backends_by_device.find(device);
        if (iter == s_default_backends_by_device.end()) {
            SDERROR << "Didn't find device is registered in " <<
                    "s_default_backends_by_device." << std::endl;
            return false;
        }
        for (size_t i = 0; i < iter->second.size(); ++i) {
            if (iter->second[i] == backend) {
                return true;
            }
        }
        std::string msg = Str(iter->second);
        SDERROR << device << " only supports " << msg << ", but now it's "
                << backend << "." << std::endl;
        return false;
    }

    /// 重载<<，输出Backend/Device/ModelFormat
    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const Backend &b);

    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const Device &d);

    STDEPLOY_DECL std::ostream &operator<<(std::ostream &o, const ModelFormat &f);

}

