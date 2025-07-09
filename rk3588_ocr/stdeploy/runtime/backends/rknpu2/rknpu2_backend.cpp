//
// Created by zhuzf on 2023/8/01.
//

#include "stdeploy/runtime/backends/rknpu2/rknpu2_backend.h"

namespace stdeploy {

    bool RKNPU2Backend::Infer(std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        int ret;
        for (int i = 0; i < NumInputs(); ++i) {
            PixelFormat format = inputs[i].desc.format;
            if (format != RGB && format != BGR && format != FM) {
                STDEPLOY_ERROR("rknpu2 only support input tensor format RGB/BGR/FM,but now is %d", format);
                return false;
            }
            // if not fd,using input_mems[i] virt_addr, else using input_mems[i] fd(rga)
            if (inputs[i].desc.mem.fd == 0 || inputs[i].desc.mem.phyAddr == nullptr) {
                int width = input_attrs_[i].dims[2];
                int stride = input_attrs_[i].w_stride;
                if (width != stride) {
                    /// rk nhwc
                    STDEPLOY_INFO("input data align");
                    int height = input_attrs_[i].dims[1];
                    int channel = input_attrs_[i].dims[3];
                    // copy from src to dst with stride
                    auto *src_ptr = (uint8_t *) inputs[i].data();
                    auto *dst_ptr = (uint8_t *) input_mems_[i]->virt_addr;
                    // width-channel elements
                    int src_wc_elems = width * channel;
                    int dst_wc_elems = stride * channel;
                    for (int h = 0; h < height; ++h) {
                        memcpy(dst_ptr, src_ptr, src_wc_elems);
                        src_ptr += src_wc_elems;
                        dst_ptr += dst_wc_elems;
                    }
                } else {
                    memcpy(input_mems_[i]->virt_addr, inputs[i].data(), inputs[i].bytes());
                }
            } else {
                // fd or phyaddr
            }
        }
        // run rknn
        ret = rknn_run(ctx_, nullptr);
        if (ret != RKNN_SUCC) {
            SDERROR << "rknn run error! ret=" << ret << std::endl;
            return false;
        }
        outputs.resize(NumOutputs()); // resize 初始化 outputs
        for (uint32_t i = 0; i < NumOutputs(); ++i) {
            outputs[i].desc = outputs_desc_[i];
            outputs[i].set_data(const_cast<void *> (output_mems_[i]->virt_addr));
#ifdef ENABLE_RKNPU2_RV1106
            /// rknpu2 default is nc1hwc2,set ro nhwc layout
            outputs[i].desc.layout = NHWC;
#else
            outputs[i].desc.layout = NCHW;
#endif
        }
        return true;
    }

    bool RKNPU2Backend::Init(const RuntimeOption &option) {
        if (!this->RuntimeOptionIsApplicable(option)) {
            SDERROR << "Runtime option is not applicable." << std::endl;
            return false;
        }
        if (rknpu2_option_.shared_weight) {
#if defined(ENABLE_RKNPU2_RK3588) || defined(ENABLE_RKNPU2_RK356X) || defined(ENABLE_RKNPU2_RK3576)
            if (rknpu2_option_.rk_context == 0) {
                SDERROR << "First ctx init failed,load model failed" << std::endl;
                return false;
            }
            if (!this->LoadModel(&rknpu2_option_.rk_context)) {
                SDERROR << "Load model shared weights failed" << std::endl;
                return false;
            }
#else
            SDERROR << "rknn_dup_context only support when soc is RK35xx,please check cpu_name and soc" << std::endl;
            return false;
#endif
        } else {
            if (!this->LoadModel(option.model_file)) {
                SDERROR << "Load model failed" << std::endl;
                return false;
            }
            rknpu2_option_.rk_context = ctx_;
        }
        if (!this->InitInputAndOutputNumber()) {
            SDERROR << "Init input and output number failed" << std::endl;
            return false;
        }
        if (!GetSDKAndDeviceVersion()) {
            SDERROR << "Get SDK and device version failed" << std::endl;
            return false;
        }
        // set core mask only support rk3588
        if (rknpu2_option_.cpu_name == RKNPU2CpuName::RK3588) {
            if (!this->SetCoreMask(rknpu2_option_.core_mask)) {
                SDERROR << "rk3588 set core mask failed" << std::endl;
            } else {
                SDINFO << "rk3588 set core mask success" << std::endl;
            }
        }
        if (!this->InitRKNNTensorAddress()) {
            SDERROR << "Init rknn tensor address failed" << std::endl;
            return false;
        }
        if (!this->InitRKNNTensorMemory()) {
            SDERROR << "Init tensor memory failed." << std::endl;
            return false;
        }
        if (!this->GetInputOutputTensorDesc()) {
            SDERROR << "Get tensor desc failed." << std::endl;
            return false;
        }
        return true;
    }

    RKNPU2Backend::~RKNPU2Backend() {
        SDINFO << "release rknpu2 mem" << std::endl;
        if (tensor_attrs_init_) {
            // free rk tensor attrs
            if (input_attrs_ != nullptr) {
                free(input_attrs_);
            }
            if (output_attrs_ != nullptr) {
                free(output_attrs_);
            }
        }
        if (tensor_memory_init_) {
            // free rk tensor mem
            for (uint32_t i = 0; i < io_num_.n_input; i++) {
                rknn_destroy_mem(ctx_, input_mems_[i]);
            }
            for (uint32_t i = 0; i < io_num_.n_output; i++) {
                rknn_destroy_mem(ctx_, output_mems_[i]);
            }
        }
        if (ctx_ != 0) {
            // free rk context
            rknn_destroy(ctx_);
            ctx_ = 0;
        }

    }

    void RKNPU2Backend::DumpTensorAttr(rknn_tensor_attr &attr) {
        if (attr.n_dims > 1) {
            std::string shape_str;
            for (int i = 0; i < attr.n_dims; i++) {
                shape_str += " " + std::to_string(attr.dims[i]);
            }
            SDINFO << "Shape: " << shape_str << std::endl;
        }
        STDEPLOY_INFO("name=%s,n_dims=%d,n_elems=%d,size=%d,fmt=%s,type=%s,qnt_type=%s,zp=%d,scale=%f", attr.name,
                      attr.n_dims, attr.n_elems, attr.size,
                      get_format_string(attr.fmt), get_type_string(attr.type), get_qnt_type_string(attr.qnt_type),
                      attr.zp, attr.scale);
    }


    bool RKNPU2Backend::GetSDKAndDeviceVersion() {
        int ret;
        ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver_, sizeof(sdk_ver_));
        if (ret != RKNN_SUCC) {
            SDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
            return false;
        }
        SDINFO << "rknpu2 runtime version: " << sdk_ver_.api_version << std::endl;
        SDINFO << "rknpu2 driver version: " << sdk_ver_.drv_version << std::endl;
        return true;
    }

    bool RKNPU2Backend::SetCoreMask(const RKNPU2CoreMask &core_mask) const {
        if (rknpu2_option_.cpu_name != RKNPU2CpuName::RK3588) {
            SDERROR << "SetCoreMask only support when soc is RK3588." << std::endl;
            return false;
        }
#ifdef ENABLE_RKNPU2_RK3588
        int ret = rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(core_mask));
        if (ret != RKNN_SUCC) {
            SDERROR << "The function(rknn_set_core_mask) failed! ret=" << ret
                    << std::endl;
            return false;
        }
        return true;
#else
        SDERROR << "SetCoreMask only support when soc is RK3588,please check cpu_name and soc" << std::endl;
        return false;
#endif
    }


    bool RKNPU2Backend::LoadModel(const std::string &model_path) {
        int ret = rknn_init(&ctx_, (void *) model_path.c_str(), 0, 0, nullptr);
        if (ret != RKNN_SUCC) {
            SDERROR << "The function(rknn_init) failed! ret=" << ret << std::endl;
            return false;
        }
#if defined(ENABLE_RKNPU2_RK3588) || defined(ENABLE_RKNPU2_RK356X) || defined(ENABLE_RKNPU2_RK3576)
        // register a custom op 注册自定义算子
        // 1.topk
        rknn_custom_op user_op[1];
        memset(user_op, 0, sizeof(rknn_custom_op));
        strncpy(user_op[0].op_type, "TopK", RKNN_MAX_NAME_LEN - 1);
        user_op[0].version = 1;
        user_op[0].target = RKNN_TARGET_TYPE_CPU;
        user_op[0].compute = stdeploy::ops::rknpu2::compute_custom_topk_fp;
        ret = rknn_register_custom_ops(ctx_, user_op, 1);
        if (ret < 0) {
            printf("rknn_register_custom_op fail! ret = %d\n", ret);
            return false;
        }
#endif
        return true;
    }

#if defined(ENABLE_RKNPU2_RK3588) || defined(ENABLE_RKNPU2_RK356X) || defined(ENABLE_RKNPU2_RK3576)
    bool RKNPU2Backend::LoadModel(rknn_context *ctx_in) {
        // rknn_dup_context not support rv1106/rv1103
        int ret = rknn_dup_context(ctx_in, &ctx_);
        printf("Loading model for share_weight\n");
        if (ret != RKNN_SUCC) {
            std::cout << "The function(rknn_dup_context) failed! ret=" << ret << std::endl;
            return false;
        }
        return true;
    }
#endif

    bool RKNPU2Backend::InitInputAndOutputNumber() {
        if (io_num_init_) {
            SDINFO << "The private variable io_num_ has been initialized."
                   << std::endl;
            return false;
        }
        int ret;
        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret != RKNN_SUCC) {
            SDERROR << "The function rknn_query(RKNN_QUERY_IN_OUT_NUM) failed! ret=" << ret << std::endl;
            return false;
        }
        io_num_init_ = true;
        return true;
    }


    bool RKNPU2Backend::InitRKNNTensorAddress() {
        if (tensor_attrs_init_) {
            SDINFO << "Private variable input_attrs_ and output_attrs_ memory has "
                      "been allocated. Please do not allocate memory repeatedly or "
                      "memory leak may occur."
                   << std::endl;
            return false;
        }
        if (!io_num_init_) {
            this->InitInputAndOutputNumber();
        }
        if (io_num_.n_input == 0) {
            SDERROR << "The number of input tensors is 0." << std::endl;
            return false;
        }

        if (io_num_.n_output == 0) {
            SDERROR << "The number of output tensors is 0." << std::endl;
            return false;
        }

        // Allocate memory for privae variable input_attrs_.
        input_attrs_ = (rknn_tensor_attr *) malloc(sizeof(rknn_tensor_attr) * io_num_.n_input);
        memset(input_attrs_, 0, sizeof(rknn_tensor_attr) * io_num_.n_input);
        for (uint32_t i = 0; i < io_num_.n_input; i++) {
            int ret;
            input_attrs_[i].index = i;
#ifdef ENABLE_RKNPU2_RV1106
            ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs_[i]),
                             sizeof(rknn_tensor_attr));
#else
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                             sizeof(rknn_tensor_attr));
#endif
            if (ret != RKNN_SUCC) {
                SDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
                return false;
            }
            if ((input_attrs_[i].fmt != RKNN_TENSOR_NHWC) &&
                (input_attrs_[i].fmt != RKNN_TENSOR_UNDEFINED)) {
                SDERROR << "rknpu2_backend only support input format is NHWC or UNDEFINED"
                        << std::endl;
                return false;
            }
//            DumpTensorAttr(input_attrs_[i]);
        }
        // Allocate memory for private variable output_attrs_.
        output_attrs_ =
                (rknn_tensor_attr *) malloc(sizeof(rknn_tensor_attr) * io_num_.n_output);
        memset(output_attrs_, 0, io_num_.n_output * sizeof(rknn_tensor_attr));
        for (uint32_t i = 0; i < io_num_.n_output; i++) {
            int ret = RKNN_SUCC;
            output_attrs_[i].index = i;
#ifdef ENABLE_RKNPU2_RV1106
            ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_attrs_[i]),
                             sizeof(rknn_tensor_attr));
            if (output_attrs_[i].n_dims == 5) {
                // rknpu2 output tensor layout default is nc1hwc2
                // only support RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, not support RKNN_QUERY_OUTPUT_ATTR
                ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs_[i]),
                                 sizeof(rknn_tensor_attr));
            }
            // set output float32 while output tensor layout int8 or uint8 or float 16
            if ((output_attrs_[i].type == RKNN_TENSOR_INT8) || (output_attrs_[i].type == RKNN_TENSOR_UINT8)
                || (output_attrs_[i].type == RKNN_TENSOR_FLOAT16)) {
                output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
            }
#else
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                             sizeof(rknn_tensor_attr));
            // set output float32 while output tensor layout int8 or uint8 or float 16
            if ((output_attrs_[i].type == RKNN_TENSOR_INT8) || (output_attrs_[i].type == RKNN_TENSOR_UINT8)
                || (output_attrs_[i].type == RKNN_TENSOR_FLOAT16)) {
                output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
            }
#endif
            if (ret != RKNN_SUCC) {
                SDERROR << "The function(rknn_query) failed! ret=" << ret << std::endl;
                return false;
            }
//            DumpTensorAttr(output_attrs_[i]);
        }
        tensor_attrs_init_ = true;
        return true;
    }


    bool RKNPU2Backend::InitRKNNTensorMemory() {
        if (tensor_memory_init_) {
            SDINFO << "Private variable input_mems_ and output_mems_ memory has "
                      "been allocated. Please do not allocate memory repeatedly or "
                      "memory leak may occur."
                   << std::endl;
            return false;
        }
        int ret;
        input_mems_.resize(io_num_.n_input);
        output_mems_.resize(io_num_.n_output);

        for (uint32_t i = 0; i < io_num_.n_input; i++) {
            if (input_attrs_[i].qnt_type == RKNN_TENSOR_QNT_NONE) {
                // TODO:临时将非图片的输入默认为fp32，怎么根据不同的输入类型开辟不同的内存，非图片类的FM要* sizeof(float)，图片类还是UINT8
                rknn_tensor_type input_type;
                rknn_tensor_format input_layout;
                input_type = RKNN_TENSOR_FLOAT32;
                input_layout = RKNN_TENSOR_NHWC;
                input_attrs_[i].type = input_type;
                input_attrs_[i].fmt = input_layout;
                input_mems_[i] = rknn_create_mem(ctx_, input_attrs_[i].n_elems * sizeof(float));
            } else {
                rknn_tensor_type input_type;
                rknn_tensor_format input_layout;
                input_type = RKNN_TENSOR_UINT8;
                input_layout = RKNN_TENSOR_NHWC;
                input_attrs_[i].type = input_type;
                input_attrs_[i].fmt = input_layout;
                input_mems_[i] = rknn_create_mem(ctx_, input_attrs_[i].size_with_stride);
            }
            if (input_mems_[i] == nullptr) {
                SDERROR << "The function(rknn_create_mem) failed! ret=" << ret
                        << std::endl;
                return false;
            }
            // Set input tensor memory
            ret = rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
            if (ret != RKNN_SUCC) {
                SDERROR << "The function(rknn_set_io_mem) failed! ret=" << ret
                        << std::endl;
                return false;
            }
            DumpTensorAttr(input_attrs_[i]);
        }
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            // set output tensor size, TODO: output size cal function
            uint32_t output_size;
#ifdef ENABLE_RKNPU2_RV1106
            output_size = output_attrs_[i].size_with_stride * sizeof(float);
#else
            if (output_attrs_[i].type == RKNN_TENSOR_INT64) {
                output_size = output_attrs_[i].n_elems * sizeof(int64_t);
            } else {
                output_size = output_attrs_[i].n_elems * sizeof(float);
            }
#endif
            output_mems_[i] = rknn_create_mem(ctx_, output_size);
            if (output_mems_[i] == nullptr) {
                SDERROR << "The function(rknn_create_mem) failed! ret=" << ret
                        << std::endl;
                return false;
            }
            ret = rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
            if (ret != RKNN_SUCC) {
                SDERROR << "The function(rknn_set_io_mem) failed! ret=" << ret
                        << std::endl;
                return false;
            }
            DumpTensorAttr(output_attrs_[i]);
        }
        tensor_memory_init_ = true;
        return true;
    }

    bool RKNPU2Backend::RuntimeOptionIsApplicable(const RuntimeOption &runtime_option) {
        if (!Supported(runtime_option.model_format, Backend::RKNPU2)) {
            SDERROR << "The model format is not supported for RKNPU2." << std::endl;
            return false;
        }
        if (!Supported(runtime_option.device, Backend::RKNPU2)) {
            SDERROR << "The device is not supported for RKNPU2." << std::endl;
            return false;
        }
        rknpu2_option_ = runtime_option.rknpu2_option;
        return true;
    }

    bool RKNPU2Backend::GetInputOutputTensorDesc() {
        // set input output tensor desc
        inputs_desc_.resize(io_num_.n_input);
        outputs_desc_.resize(io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_input; i++) {
            std::string temp_name = input_attrs_[i].name;
            std::vector<int> temp_shape{};
            temp_shape.resize(input_attrs_[i].n_dims);
            for (uint32_t j = 0; j < input_attrs_[i].n_dims; j++) {
                temp_shape[j] = (int) input_attrs_[i].dims[j];
            }
            DataType temp_type = ConvertToDataType(input_attrs_[i].type);
            inputs_desc_[i].shape = temp_shape;
            inputs_desc_[i].name = temp_name;
            inputs_desc_[i].data_type = temp_type;
            inputs_desc_[i].layout = stdeploy::LayoutType::NHWC;
            // set fd to sd tensor desc
            inputs_desc_[i].mem.fd = input_mems_[i]->fd;
            inputs_desc_[i].mem.phyAddr = (void *) input_mems_[i]->phys_addr;
            inputs_desc_[i].mem.virAddr = input_mems_[i]->virt_addr;
        }
        for (uint32_t i = 0; i < io_num_.n_output; i++) {
            int n_dims = static_cast<int>(output_attrs_[i].n_dims);
            // Copy output_attrs_ to output tensor
            std::string temp_name = output_attrs_[i].name;
            std::vector<int> temp_shape{};
            temp_shape.resize(n_dims);
            for (uint32_t j = 0; j < n_dims; j++) {
                temp_shape[j] = (int) output_attrs_[i].dims[j];
            }
            DataType temp_type = ConvertToDataType(output_attrs_[i].type);
            outputs_desc_[i].shape = temp_shape;
            outputs_desc_[i].name = temp_name;
            outputs_desc_[i].data_type = temp_type;
        }
        return true;
    }

    DataType RKNPU2Backend::ConvertToDataType(rknn_tensor_type &type) {
        if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT16) {
            return DataType::FP16;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT32) {
            return DataType::FP32;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_INT8) {
            return DataType::INT8;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_INT16) {
            return DataType::INT16;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_INT32) {
            return DataType::INT32;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_UINT8) {
            return DataType::UINT8;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_INT64) {
            return DataType::INT64;
        }
        if (type == rknn_tensor_type::RKNN_TENSOR_BOOL) {
            return DataType::BOOL;
        }
        SDERROR << "DataType don't support this type" << std::endl;
        return DataType::DATATYPE_UNKNOWN;
    }

    rknn_tensor_type RKNPU2Backend::ConvertToRKNNTensorType(DataType &type) {
        if (type == DataType::FP16) {
            return rknn_tensor_type::RKNN_TENSOR_FLOAT16;
        }
        if (type == DataType::FP32) {
            return rknn_tensor_type::RKNN_TENSOR_FLOAT32;
        }
        if (type == DataType::INT8) {
            return rknn_tensor_type::RKNN_TENSOR_INT8;
        }
        if (type == DataType::INT16) {
            return rknn_tensor_type::RKNN_TENSOR_INT16;
        }
        if (type == DataType::INT32) {
            return rknn_tensor_type::RKNN_TENSOR_INT32;
        }
        if (type == DataType::INT64) {
            return rknn_tensor_type::RKNN_TENSOR_INT64;
        }
        if (type == DataType::UINT8) {
            return rknn_tensor_type::RKNN_TENSOR_UINT8;
        }
        if (type == DataType::BOOL) {
            return rknn_tensor_type::RKNN_TENSOR_BOOL;
        }
        if (type == DataType::INT4) {
            return rknn_tensor_type::RKNN_TENSOR_INT4;
        }
        SDERROR << "rknn_tensor_type don't support this type" << std::endl;
        return RKNN_TENSOR_TYPE_MAX;
    }

} //namespace stdeploy