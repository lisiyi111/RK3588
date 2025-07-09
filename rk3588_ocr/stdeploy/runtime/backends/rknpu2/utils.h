//
// Created by zhuzf on 2023/8/27.
//

#pragma once

#include <iostream>
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include <opencv2/opencv.hpp>
#include "dma_alloc.h"
#include "stdeploy/vision/common/mat.h"
#include "stdeploy/core/sd_tensor.h"
#include "stdeploy/vision/common/vision_struct.h"


namespace stdeploy {

    // impl rga resize
    int impl_rga_resize(rga_buffer_handle_t src_handle,
                        rga_buffer_handle_t dst_handle,
                        int src_format,
                        int dst_format,
                        vision::PreprocessParams *params);
    // rga resize
    int rga_resize(stdeploy::Mat &src_mat, stdeploy::Tensor &dst_tensor, vision::PreprocessParams *params);

    // rv1106/3 free phy addr
    STDEPLOY_DECL void rv110x_free_fd(int src_buf_size, void *src_buf, int *src_dma_fd);

    // rv1106/3 malloc phy addr
    STDEPLOY_DECL int rv110x_malloc_fd(int src_buf_size, void **src_buf, int *src_dma_fd);

    // rk35x8 free phy addr
    STDEPLOY_DECL void rk35x8_free_dma_fd(int src_buf_size, void *src_buf, int *src_dma_fd);

    // rk35x8 malloc phy addr
    STDEPLOY_DECL int rv35x8_malloc_dma_fd(int src_buf_size, void **src_buf, int *src_dma_fd);

    // rk flush flush
    STDEPLOY_DECL void rk_flush_cache(int src_dma_fd);

} //namespace stdeploy