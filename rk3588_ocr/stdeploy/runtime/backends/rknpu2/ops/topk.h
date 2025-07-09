/**
  *************************************************
  * @file               :topk.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/3/27                
  *************************************************
  */

#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <rknn_custom_op.h>

namespace stdeploy {
    namespace ops {
        namespace rknpu2 {

            bool comp_func(std::pair<float, int> a, std::pair<float, int> b);

            //  1*8400*1*1
            void topk(const float *in_data, float *out_ind, int n, int k);

            /**
             * compute_custom_topk_fp
             * */
            int compute_custom_topk_fp(rknn_custom_op_context *op_ctx, rknn_custom_op_tensor *inputs, uint32_t n_inputs,
                                       rknn_custom_op_tensor *outputs, uint32_t n_outputs);

        } //namespace rknpu2
    } //namespace ops
} //namespace stdeploy