/**
  *************************************************
  * @file               :runtime.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/4                
  *************************************************
  */
#include "stdeploy/runtime/runtime.h"
#include "stdeploy/utils/file_util.h"

namespace stdeploy {

    bool Runtime::Init(const RuntimeOption &_option) {
        int ret;
        option = _option;
        if (!stdeploy::utils::file_exist(option.model_file)) {
            STDEPLOY_ERROR("model file not exist");
            return false;
        }
        if (option.backend == Backend::UNKNOWN) {
            STDEPLOY_ERROR("backend unknown,check backend set or enable xxx backend");
            return false;
        }
        if (option.backend == Backend::TRT) {
            ret = CreateTrtBackend();
        } else if (option.backend == Backend::MSTAR) {
            ret = CreateMstarBackend();
        } else if (option.backend == Backend::MNN) {
            ret = CreateMnnBackend();
        } else if (option.backend == Backend::NCNN) {
            ret = CreateNcnnBackend();
        } else if (option.backend == Backend::SNPE) {
            ret = CreateSnpeBackend();
        } else if (option.backend == Backend::RKNPU2) {
            ret = CreateRknpu2Backend();
        } else if (option.backend == Backend::RKNPU) {
            ret = CreateRknpuBackend();
        } else if (option.backend == Backend::ORT) {
            ret = CreateORTBackend();
        } else if (option.backend == Backend::ACL_SVP) {
            ret = CreateAclSvpBackend();
        } else {
            std::string msg = Str(GetAvailableBackends());
            SDERROR << "The compiled SdDeploy only supports " << msg << ", "
                    << option.backend << " is not supported now." << std::endl;
            return false;
        }
        if (ret != 0) {
            // init backend failed
            return false;
        }
        return true;
    }


    bool Runtime::Infer(std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) {
        return backend_->Infer(input_tensors, output_tensors);
    }

    TensorDesc Runtime::GetInputInfo(int index) {
        return backend_->GetInputInfo(index);
    }

    TensorDesc Runtime::GetOutputInfo(int index) {
        return backend_->GetOutputInfo(index);
    }

    std::vector<TensorDesc> Runtime::GetInputInfos() {
        return backend_->GetInputInfos();
    }

    std::vector<TensorDesc> Runtime::GetOutputInfos() {
        return backend_->GetOutputInfos();
    }

    int Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
        backend_ = utils::make_unique<TrtBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized trt inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "TrtBackend is not available, please compiled with ENABLE_TRT_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::TRT in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateMstarBackend() {
#ifdef ENABLE_MSTAR_BACKEND
        backend_ = utils::make_unique<MstarBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized mstar inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "IpuBackend is not available, please compiled with ENABLE_MSTAR_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::IPU in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateMnnBackend() {
#ifdef ENABLE_MNN_BACKEND
        backend_ = utils::make_unique<MnnBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized mnn inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "MnnBackend is not available, please compiled with ENABLE_MNN_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::MNN in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateNcnnBackend() {
#ifdef ENABLE_NCNN_BACKEND
        backend_ = utils::make_unique<NcnnBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized ncnn inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "NcnnBackend is not available, please compiled with ENABLE_NCNN_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::NCNN in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateSnpeBackend() {
#ifdef ENABLE_SNPE_BACKEND
        backend_ = utils::make_unique<SNPEBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized snpe inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "SnpeBackend is not available, please compiled with ENABLE_SNPE_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::SNPE in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateRknpu2Backend() {
#ifdef ENABLE_RKNPU2_BACKEND
        backend_ = utils::make_unique<RKNPU2Backend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized rkppu2 inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "RKNPU2Backend is not available, please compiled with ENABLE_RKNPU2_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::RKNPU2 in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateRknpuBackend() {
#ifdef ENABLE_RKNPU_BACKEND
        backend_ = utils::make_unique<RKNPUBackend>();
        if (!backend_->Init(option)) {
            SDERROR << "Failed to initialized rkppu inference backend" << std::endl;
            return -1;
        }
#else
            SDERROR << "RKNPUBackend is not available, please compiled with ENABLE_RKNPU_BACKEND=ON." << std::endl;
            return -1;
#endif
        SDINFO << "Runtime initialized with Backend::RKNPU in " << option.device << "."
               << std::endl;
        return 0;
    }

    int Runtime::CreateORTBackend() {
#ifdef ENABLE_ORT_BACKEND
        backend_ = utils::make_unique<ORTBackend>();
        if (!backend_->Init(option)) {
            SDERROR << "Failed to initialized onnxruntime inference backend" << std::endl;
            return -1;
        }
#else
        SDERROR << "ONNXBackend is not available, please compiled with ENABLE_ONNX_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::ORT in " << option.device << "."
               << std::endl;
        return 0;
    }


    int Runtime::CreateAclSvpBackend() {
#ifdef ENABLE_ACL_SVP_BACKEND
        backend_ = utils::make_unique<AclSvpBackend>();
        if(!backend_->Init(option)){
            SDERROR<<"Failed to initialized hisi acl svp inference backend"<<std::endl;
            return -1;
        }
#else
        SDERROR << "AclSvpBackend is not available, please compiled with ENABLE_ACL_SVP_BACKEND=ON." << std::endl;
        return -1;
#endif
        SDINFO << "Runtime initialized with Backend::ACL in " << option.device << "."
               << std::endl;
        return 0;
    }

} //namespace stdeploy