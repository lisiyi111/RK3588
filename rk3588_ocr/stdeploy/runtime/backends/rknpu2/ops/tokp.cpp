/**
  *************************************************
  * @file               :tokp.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/3/27                
  *************************************************
  */

#include "stdeploy/runtime/backends/rknpu2/ops/topk.h"

namespace stdeploy {
    namespace ops {
        namespace rknpu2{

            bool comp_func(std::pair<float, int> a, std::pair<float, int> b) {
                return (a.first > b.first);
            }

            //  1*8400*1*1
            void topk(const float* in_data, float* out_ind, int n, int k)
            {
                const float* in_tmp = in_data;
                float* out_ind_tmp = out_ind;
                std::vector<std::pair<float, int>> vec;
                for (int j = 0; j < n; j++) {
                    vec.push_back(std::make_pair(in_tmp[j], j));
                }
                std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp_func);
                for (int q = 0; q < k; q++) {
                    out_ind_tmp[q] = vec[q].second;
                }
            }

            /**
             * compute_custom_topk_fp
             * */
            int compute_custom_topk_fp(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                                       rknn_custom_op_tensor* outputs, uint32_t n_outputs)
            {
                unsigned char*      in_ptr   = (unsigned char*)inputs[0].mem.virt_addr + inputs[0].mem.offset;
                unsigned char*      out_ptr1  = (unsigned char*)outputs[1].mem.virt_addr + outputs[1].mem.offset;

                const float*        in_data  = (const float*)in_ptr;
                float *           out_data1 = (float*)out_ptr1;

                int N = inputs[0].attr.dims[1];
                int K = outputs[1].attr.dims[1];
                topk(in_data, out_data1, N, K);
                return 0;
            }

        } //namespace rknpu2
    } //namespace ops
} //namespace stdeploy