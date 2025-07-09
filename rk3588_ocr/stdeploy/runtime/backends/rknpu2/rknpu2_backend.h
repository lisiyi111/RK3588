/**
  *************************************************
  * @file               :rknpu2_backend.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/8/7
  *************************************************
  */
#pragma once

#include "rknn_api.h"
#include "stdeploy/runtime/backends/backend.h"
#include "stdeploy/runtime/backends/rknpu2/option.h"
#include "stdeploy/runtime/backends/rknpu2/ops/topk.h"

namespace stdeploy {

    class STDEPLOY_DECL RKNPU2Backend : public BaseBackend {
    public:
        /***************************** BaseBackend API *****************************/
        RKNPU2Backend() = default;

        ~RKNPU2Backend() override;

        bool Init(const RuntimeOption &runtime_option) override;

        bool Infer(std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) override;
        /***************************** BaseBackend API *****************************/

    private:
        /*
         *  @name       LoadModel
         *  @brief      Read the model and initialize rknn context.
         *  @param      model_path: the path of RKNN model.
         *  @return     bool
         *  @note       None
         */
        bool LoadModel(const std::string &model_path);

#if defined(ENABLE_RKNPU2_RK3588) || defined(ENABLE_RKNPU2_RK356X) || defined(ENABLE_RKNPU2_RK3576)

        /*
         *  @name       LoadModel
         *  @brief      Read the model and initialize rknn context.
         *  @param      ctx_in: the rknn context.
         *  @return     bool
         *  @note       ctx_in must be init success and net is same
         */
        bool LoadModel(rknn_context *ctx_in);

#endif

        /*
         *  @name       InitInputAndOutputNumber
         *  @brief      Initialize io_num_.
         *  @param
         *  @return     bool
         *  @note       The private variable ctx must be initialized to use this function.
         */
        bool InitInputAndOutputNumber();

        /*
         *  @name       InitRKNNTensorAddress
         *  @brief      Allocate memory for input_attrs_ and output_attrs_.
         *  @param
         *  @return     bool
         *  @note
         */
        bool InitRKNNTensorAddress();

        /*
         *  @name       InitRKNNTensorMemory
         *  @brief      Allocate memory for input and output tensors.
         *  @param
         *  @return     bool
         *  @note
         */
        bool InitRKNNTensorMemory();

        /*
         *  @name       DumpTensorAttr
         *  @brief      Get the model's detailed inputs and outputs
         *  @param      rknn_tensor_attr
         *  @return     None
         *  @note
         */
        void DumpTensorAttr(rknn_tensor_attr &attr);

        /*
         *  @name       GetSDKAndDeviceVersion
         *  @brief      Get RKNPU2 sdk and device version.
         *  @param
         *  @return     bool
         *  @note       The private variable ctx must be initialized to use this function.
         */
        bool GetSDKAndDeviceVersion();

        /*
         *  @name       SetCoreMask
         *  @brief      Set NPU core for model
         *  @param      core_mask: The specification of NPU core setting.
         *  @return     bool
         *  @note       Only support RK3588
         */
        bool SetCoreMask(const RKNPU2CoreMask &core_mask) const;

        /*
         *  @name       RuntimeOptionIsApplicable
         *  @brief      This function is used to determine whether the RuntimeOption
         *              meets the operating conditions of RKNPU2.
         *  @param
         *  @return     bool
         *  @note
         */
        bool RuntimeOptionIsApplicable(const RuntimeOption &runtime_option);

        /*
         *  @name       GetInputOutputTensorDesc
         *  @brief      Get input output tensor desc
         *  @param
         *  @return     bool
         *  @note
         */
        bool GetInputOutputTensorDesc();

        /*
         *  @name       ConvertToDataType
         *  @brief      convert rknn tensor type to stdeploy datatype
         *  @param      rknn_tensor_type
         *  @return     DataType
         *  @note
         */
        DataType ConvertToDataType(rknn_tensor_type &type);

        /*
         *  @name       ConvertToRKNNTensorType
         *  @brief      convert stdeploy datatype  to  rknn tensor type
         *  @param      DataType
         *  @return     rknn_tensor_type
         *  @note
         */
        rknn_tensor_type ConvertToRKNNTensorType(DataType &type);


    private:

        RKNPU2BackendOption rknpu2_option_;                         // option
        rknn_context ctx_{};                                        // context
        rknn_sdk_version sdk_ver_{};                                // rknpu2 sdk version
        rknn_input_output_num io_num_{0, 0};        // model input output number
        rknn_tensor_attr *input_attrs_ = nullptr;                   // input model desc
        rknn_tensor_attr *output_attrs_ = nullptr;                  // output model desc
        std::vector<rknn_tensor_mem *> input_mems_;                 // zero-copy input mem
        std::vector<rknn_tensor_mem *> output_mems_;                // zero-copy output mem
        bool io_num_init_ = false;                                  // flag statement init io_num_
        bool tensor_attrs_init_ = false;                            // flag statement init input_attrs_/output_attrs_
        bool tensor_memory_init_ = false;                           // flag statement init input_mems_/output_mems_
    };

}// namespace stdeploy

