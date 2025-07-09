/**
  *************************************************
  * @file               :runtime_option.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */
#include "stdeploy/runtime/runtime_option.h"
#include "stdeploy/runtime/runtime.h"

namespace stdeploy {

    void RuntimeOption::SetModelPath(const std::string &model_path,
                                     const std::string &params_path,
                                     const ModelFormat &format) {
        model_file = model_path;
        params_file = params_path;
        model_format = format;
    }

    void RuntimeOption::SetModelBuffer(const std::string &model_buffer,
                                       const std::string &params_buffer,
                                       const stdeploy::ModelFormat &format) {
        model_file = model_buffer;
        params_file = params_buffer;
        model_format = format;
    }

    void RuntimeOption::UseCPU() {
        device = Device::CPU;
    }

    void RuntimeOption::UseGPU(int gpu_id) {
#ifdef WITH_GPU
        device = Device::GPU;
        device_id = gpu_id;
#else
        SDWARNING << "The StDeploy didn't compile with GPU, will force to use CPU."
                  << std::endl;
        device = Device::CPU;
#endif
    }

    void RuntimeOption::UseTensorRT() {
#ifdef ENABLE_TRT_BACKEND
        backend = Backend::TRT;
#else
        SDERROR << "The StDeploy didn't compile with Trt Backend." << std::endl;
#endif
    }


    void RuntimeOption::UseMNN() {
#ifdef ENABLE_MNN_BACKEND
        backend = Backend::MNN;
#else
        SDERROR << "The StDeploy didn't compile with Mnn Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseNCNN() {
#ifdef ENABLE_NCNN_BACKEND
        backend = Backend::NCNN;
#else
        SDERROR << "The StDeploy didn't compile with ncnn Backend." << std::endl;
#endif
    }

    void RuntimeOption::SetCpuThreadNum(int thread_num) {
        cpu_thread_num = thread_num;
    }

    void RuntimeOption::UseMstar() {
#ifdef ENABLE_MSTAR_BACKEND
        device = Device::MSTARD;
        backend = Backend::MSTAR;
#else
        SDERROR << "The StDeploy didn't compile MSTAR Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseSnpe() {
#ifdef ENABLE_SNPE_BACKEND
        device = Device::SNPED;
        backend = Backend::SNPE;
#else
        SDERROR << "The StDeploy didn't compile SNPED Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseRKNPU2() {
#ifdef ENABLE_RKNPU2_BACKEND
        device = Device::RKNPU2D;
        backend = Backend::RKNPU2;
#else
        SDERROR << "The StDeploy didn't compile RKNPU2 Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseRKNPU() {
#ifdef ENABLE_RKNPU_BACKEND
        device = Device::RKNPUD;
        backend = Backend::RKNPU;
#else
        SDERROR << "The StDeploy didn't compile RKNPU Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseOnnxRuntime() {
#ifdef ENABLE_ORT_BACKEND
        device = Device::CPU;
        backend = Backend::ORT;
#else
        SDERROR << "The StDeploy didn't compile ONNXRUNTIME Backend." << std::endl;
#endif
    }

    void RuntimeOption::UseAclSvp() {
#ifdef ENABLE_ACL_SVP_BACKEND
        device = Device::ACL_SVPD;
        backend = Backend::ACL_SVP;
#else
        SDERROR << "The StDeploy didn't compile ACL_SVP Backend." << std::endl;
#endif
    }


} //namespace stdeploy



