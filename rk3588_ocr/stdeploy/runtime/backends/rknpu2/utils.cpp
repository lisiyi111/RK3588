//
// Created by zhuzf on 2023/4/28.
//

#include "stdeploy/runtime/backends/rknpu2/utils.h"

namespace stdeploy {

    void rk_flush_cache(int src_dma_fd) {
        dma_sync_cpu_to_device(src_dma_fd); /// 注意别忘记flush
    }

    int impl_rga_resize(rga_buffer_handle_t src_handle, rga_buffer_handle_t dst_handle, int src_format, int dst_format,
                        vision::PreprocessParams *params) {
        int ret = 0;

        rga_buffer_t src_img, dst_img;
        src_img = wrapbuffer_handle(src_handle, params->src_width, params->src_height, src_format);
        dst_img = wrapbuffer_handle(dst_handle, params->dst_width, params->dst_height, dst_format);

        im_rect src_rect, dst_rect;
        src_rect.x = 0;
        src_rect.y = 0;
        src_rect.width = params->src_width;
        src_rect.height = params->src_height;

        dst_rect.x = 0 + params->pad_width_left;
        dst_rect.y = 0 + params->pad_height_top;
        dst_rect.width = params->resize_width;
        dst_rect.height = params->resize_height;

        ret = imcheck(src_img, dst_img, {}, dst_rect);
        if (IM_STATUS_NOERROR != ret) {
            STDEPLOY_INFO("rga letter box resize check error!");
            return -1;
        }

        ret = improcess(src_img, dst_img, {}, src_rect, dst_rect, {}, IM_SYNC);
        if (ret != IM_STATUS_SUCCESS) {
            STDEPLOY_ERROR("%s running failed, %s\n", "rga box resize", imStrError((IM_STATUS) ret));
            return -1;
        }

        return 0;
    }


    int rga_resize(stdeploy::Mat &src_mat, stdeploy::Tensor &dst_tensor, vision::PreprocessParams *params) {

        int ret = 0;
        int src_size = src_mat.size();
        int dst_size = dst_tensor.size();
        int src_format = src_mat.pixel_format();
        int dst_format = dst_tensor.desc.format;
        if (src_format != stdeploy::RGB && src_format != stdeploy::BGR && src_format != stdeploy::NV12) {
            STDEPLOY_ERROR("rga_resize src format only support rgr/bgr/nv12,but now is %d", src_format)
            return -1;
        }
        if (dst_format != stdeploy::RGB && dst_format != stdeploy::BGR) {
            STDEPLOY_ERROR("rga_resize dst format only support rgr/bgr,but now is %d", dst_format)
            return -1;
        }
        int rk_src_format = RK_FORMAT_RGB_888;
        if (src_format == stdeploy::BGR) {
            rk_src_format = RK_FORMAT_BGR_888;
        } else if (src_format == stdeploy::NV12) {
            rk_src_format = RK_FORMAT_YCbCr_420_SP;
        }
        int rk_dst_format = RK_FORMAT_RGB_888;
        if (dst_format == stdeploy::BGR) {
            rk_dst_format = RK_FORMAT_BGR_888;
        }
        rga_buffer_handle_t src_handle, dst_handle;
        if (src_mat.phy_addr != nullptr) {
            src_handle = importbuffer_physicaladdr((uint64_t) src_mat.phy_addr, src_size);
            dst_handle = importbuffer_physicaladdr((uint64_t) dst_tensor.desc.mem.phyAddr, dst_size);
        } else if (src_mat.fd > 0) {
            src_handle = importbuffer_fd(src_mat.fd, src_size);
            dst_handle = importbuffer_fd(dst_tensor.desc.mem.fd, dst_size);
        } else {
            src_handle = importbuffer_virtualaddr(src_mat.data(), src_size);
            dst_handle = importbuffer_virtualaddr(dst_tensor.data(), dst_size);
        }
        if (src_handle <= 0) {
            STDEPLOY_ERROR("src handle error %d\n", src_handle);
            ret = -1;
            if (src_handle > 0) {
                releasebuffer_handle(src_handle);
            }
            if (dst_handle > 0) {
                releasebuffer_handle(dst_handle);
            }
            return ret;
        }

        if (dst_handle <= 0) {
            STDEPLOY_ERROR("dst handle error %d\n", dst_handle);
            ret = -1;
            if (src_handle > 0) {
                releasebuffer_handle(src_handle);
            }
            if (dst_handle > 0) {
                releasebuffer_handle(dst_handle);
            }
            return ret;
        }

        ret = impl_rga_resize(src_handle, dst_handle, rk_src_format, rk_dst_format, params);
        if (ret != 0) {
            STDEPLOY_ERROR("impl_rga_resize error\n");
            ret = -1;
            if (src_handle > 0) {
                releasebuffer_handle(src_handle);
            }
            if (dst_handle > 0) {
                releasebuffer_handle(dst_handle);
            }
            return ret;
        }

        if (src_handle > 0) {
            releasebuffer_handle(src_handle);
        }
        if (dst_handle > 0) {
            releasebuffer_handle(dst_handle);
        }

        return ret;
    }

    int rv110x_malloc_fd(int src_buf_size, void **src_buf, int *src_dma_fd) {
        /* Allocate dma_buf from CMA, return dma_fd and virtual address */
        int ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_buf_size, src_dma_fd, src_buf);
        if (ret < 0) {
            STDEPLOY_ERROR("alloc src CMA buffer failed!");
            return -1;
        }
        return 0;
    }

    void rv110x_free_fd(int src_buf_size, void *src_buf, int *src_dma_fd) {
        /// 释放开辟的fd地址
        dma_buf_free(src_buf_size, src_dma_fd, src_buf);
    }

    int rv35x8_malloc_dma_fd(int src_buf_size, void **src_buf, int *src_dma_fd) {
        /*
         * Allocate dma_buf within 4G from dma32_heap,
         * return dma_fd and virtual address.
         */
        int ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, src_buf_size, src_dma_fd, src_buf);
        if (ret < 0) {
            STDEPLOY_ERROR("alloc src CMA buffer failed!");
            return -1;
        }
        return 0;
    }

    void rk35x8_free_dma_fd(int src_buf_size, void *src_buf, int *src_dma_fd) {
        dma_buf_free(src_buf_size, src_dma_fd, src_buf);
    }

} //namespace stdeploy