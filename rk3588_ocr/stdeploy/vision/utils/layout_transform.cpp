/**
  *************************************************
  * @file               :chw2hwc.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2024/10/26                
  *************************************************
  */

#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace utils {

            void chw_to_hwc(const float *nchw, size_t N, size_t C, size_t H, size_t W, float *nhwc) {
                for (int ni = 0; ni < N; ni++) {
                    for (int hi = 0; hi < H; hi++) {
                        for (int wi = 0; wi < W; wi++) {
                            for (int ci = 0; ci < C; ci++) {
                                memcpy(nhwc + ni * H * W * C + hi * W * C + wi * C + ci,
                                       nchw + ni * C * H * W + ci * H * W + hi * W + wi, sizeof(float));
                            }
                        }
                    }
                }
            }

            // 量化模型的npu输出结果为int8数据类型，后处理要按照int8数据类型处理
            // 如下提供了int8排布的NC1HWC2转换成int8的nchw转换代码
            int NC1HWC2_int8_to_NCHW_int8(const int8_t *src, int8_t *dst, int *dims, int channel, int h, int w) {
                int batch = dims[0];
                int C1 = dims[1];
                int C2 = dims[4];
                int hw_src = dims[2] * dims[3];
                int hw_dst = h * w;
                for (int i = 0; i < batch; i++) {
                    src = src + i * C1 * hw_src * C2;
                    dst = dst + i * channel * hw_dst;
                    for (int c = 0; c < channel; ++c) {
                        int plane = c / C2;
                        const int8_t *src_c = plane * hw_src * C2 + src;
                        int offset = c % C2;
                        for (int cur_h = 0; cur_h < h; ++cur_h)
                            for (int cur_w = 0; cur_w < w; ++cur_w) {
                                int cur_hw = cur_h * w + cur_w;
                                dst[c * hw_dst + cur_h * w + cur_w] = src_c[C2 * cur_hw + offset];
                            }
                    }
                }
                return 0;
            }


            // 量化模型的npu输出结果为int8数据类型，后处理要按照int8数据类型处理
            // 如下提供了int8排布的NC1HWC2转换成float的nchw转换代码
            int
            NC1HWC2_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int channel, int h, int w, int zp,
                                       float scale) {
                int batch = dims[0];
                int C1 = dims[1];
                int C2 = dims[4];
                int hw_src = dims[2] * dims[3];
                int hw_dst = h * w;
                for (int i = 0; i < batch; i++) {
                    src = src + i * C1 * hw_src * C2;
                    dst = dst + i * channel * hw_dst;
                    for (int c = 0; c < channel; ++c) {
                        int plane = c / C2;
                        const int8_t *src_c = plane * hw_src * C2 + src;
                        int offset = c % C2;
                        for (int cur_h = 0; cur_h < h; ++cur_h)
                            for (int cur_w = 0; cur_w < w; ++cur_w) {
                                int cur_hw = cur_h * w + cur_w;
                                dst[c * hw_dst + cur_h * w + cur_w] =
                                        (src_c[C2 * cur_hw + offset] - zp) * scale; // int8-->float
                            }
                    }
                }
                return 0;
            }


            // 量化模型的npu输出结果为int8数据类型，后处理要按照int8数据类型处理
            // 如下提供了int8排布的NCHW转换成float的nchw转换代码
            int
            NCHW_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int zp, float scale) {
                int batch = dims[0];
                int channel = dims[1];
                int height = dims[2];
                int width = dims[3];
                for (int i = 0; i < batch * channel * height * width; i++) {
                    dst[i] = (float) (src[i] - zp) * scale; // int8-->float
                }
                return 0;
            }

            int
            NHWC_int8_to_NHWC_float(const int8_t *src, float *dst, int *dims, int zp, float scale) {
                int batch = dims[0];
                int height = dims[1];
                int width = dims[2];
                int channel = dims[3];
                for (int i = 0; i < batch * channel * height * width; i++) {
                    dst[i] = (float) (src[i] - zp) * scale; // int8-->float
                }
                return 0;
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy