/**
  *************************************************
  * @file               :base.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */

#include "stdeploy/vision/common/image_processes/base.h"

namespace stdeploy {
    namespace vision {

        bool Processor::operator()(sd::Mat &mat) {
            if (mat.pro_lib() == ProcLib::RGA) {
#ifdef ENABLE_RKNPU2_BACKEND
                return ImplByRKRga(mat);
#else
                SDASSERT(false, "Stdeploy didn't compile with RGA.");
#endif
            } else if (mat.pro_lib() == ProcLib::MI_SCL) {
#ifdef ENABLE_MSTAR_BACKEND
                return ImplByMstarSCL(mat);
#else
                SDASSERT(false, "Stdeploy didn't compile with mi scl.");
#endif
            } else {
                return ImplByOpenCV(mat);
            }
        }


        bool Processor::operator()(sd::Mat &mat, sd::Tensor &tensor) {
            if (mat.pro_lib() == ProcLib::RGA) {
#ifdef ENABLE_RKNPU2_BACKEND
                return ImplByRKRga(mat, tensor);
#else
                SDASSERT(false, "Stdeploy didn't compile with RGA.");
#endif
            } else if (mat.pro_lib() == ProcLib::MI_SCL) {
#ifdef ENABLE_MSTAR_BACKEND
                return ImplByMstarSCL(mat, tensor);
#else
                SDASSERT(false, "Stdeploy didn't compile with mi scl.");
#endif
            } else {
                return ImplByOpenCV(mat, tensor);
            }
        }

        bool Processor::operator()(sd::Mat &mat, Tensor &tensor, vision::PreprocessParams *params) {
            if (mat.pro_lib() == ProcLib::RGA) {
#ifdef ENABLE_RKNPU2_BACKEND
#ifdef ENABLE_RKNPU2_RV1106
                if (mat.fd <= 0 && mat.phy_addr == nullptr) {
                    STDEPLOY_WARNING(
                            "rknpu2 rga must set input mat fd or phy addr,but now is vir addr,default use opencv");
                    mat.set_pro_lib(ProcLib::OPENCV);
                    return ImplByOpenCV(mat, tensor, params);
                }
#endif
                return ImplByRKRga(mat, tensor, params);
#else
                SDASSERT(false, "Stdeploy didn't compile with RGA.");
#endif
            } else if (mat.pro_lib() == ProcLib::MI_SCL) {
#ifdef ENABLE_MSTAR_BACKEND
                if (mat.phy_addr == nullptr) {
                    STDEPLOY_WARNING(
                            "mi scl must set input mat phy addr,but not set,default use opencv");
                    mat.set_pro_lib(ProcLib::OPENCV);
                    return ImplByOpenCV(mat, tensor, params);
                }
                return ImplByMstarSCL(mat, tensor, params);
#else
                SDASSERT(false, "Stdeploy didn't compile with mi scl.");
#endif
            } else {
                return ImplByOpenCV(mat, tensor, params);
            }
        }

        bool Processor::ImplByOpenCV(sd::Mat &mat) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByOpenCV(sd::Mat &mat, Tensor &tensor) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByOpenCV(sd::Mat &mat, Tensor &tensor, vision::PreprocessParams *params) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByRKRga(sd::Mat &mat) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByRKRga(Mat &mat, Tensor &tensor) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByRKRga(Mat &mat, Tensor &tensor, vision::PreprocessParams *params) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByMstarSCL(sd::Mat &mat) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByMstarSCL(Mat &mat, Tensor &tensor) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }

        bool Processor::ImplByMstarSCL(Mat &mat, Tensor &tensor, vision::PreprocessParams *params) {
            SDERROR << Name() << " Not Implement Yet." << std::endl;
            return false;
        }


    } //namespace vision
} //namespace stdeploy