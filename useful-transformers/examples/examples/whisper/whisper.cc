#include "whisper.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <fstream>

#include "bias.h"
#include "softmax.h"
//#include <pybind11/numpy.h>
//#include <pybind11/pybind11.h>

//namespace py = pybind11;


AudioEncoder::AudioEncoder(int num_layers, int n_ctx, int n_state, int n_head)
    : num_layers(num_layers),
      n_ctx(n_ctx),
      n_state(n_state),
      n_head(n_head),
      conv0(n_ctx * 2, 80, n_state, 1),
      conv1(n_ctx * 2, n_state, n_state, 2),
      conv0_bias(n_state),
      conv1_bias(n_state),
      positional_embedding(n_ctx * n_state),
      blocks(num_layers, n_ctx, n_state, n_head),
      ln_post_gamma(n_state),
      ln_post_beta(n_state) {}

template <typename T>
void layernorm(std::vector<T> &x, int rows, int cols,
               const std::vector<T> &gamma, const std::vector<T> &beta,
               float eps = 1e-5) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    float mean = x[i * cols + 0];
    for (int j = 1; j < cols; ++j) {
      mean += x[i * cols + j];
    }
    mean /= cols;
    float var = 0.0;
    for (int j = 0; j < cols; ++j) {
      float elem = x[i * cols + j] - mean;
      elem *= elem;
      var += elem;
    }
    var /= cols;
    float denom = std::sqrt(var + eps);
    for (int j = 0; j < cols; ++j) {
      float elem = x[i * cols + j];
      x[i * cols + j] = (elem - mean) / denom * gamma[j] + beta[j];
    }
  }
}

void AudioEncoder::call() {
  conv0.call();
  std::vector<__fp16> conv1_input(n_ctx * 2 * n_state);
  bias_and_gelu(conv0.output.data(), conv1_input.data(), conv0_bias, n_ctx * 2,
                n_state);

  conv1.copy_A(conv1_input.data());
  conv1.call();
  bias_and_gelu(conv1.output.data(), blocks.blocks[0]->attn.Q.get_A_ptr(),
                conv1_bias, n_ctx, n_state);
  //将blocks.blocks[0]->attn.Q.A 转成std::vector<float>类型
  //std::vector<float> attn_Q_A(n_ctx * n_state);
  //blocks.blocks[0]->attn.Q.get_A(attn_Q_A.data(), n_ctx, n_state);
  //py::buffer_info buf_info(
  //    attn_Q_A.data(),                           /* Pointer to data */
  //    sizeof(float),                             /* Size of one scalar */
  //    py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //    2,                                         /* Number of dimensions */
  //    { n_ctx, n_state },                        /* Shape of the array */
  //    { sizeof(float) * n_state, sizeof(float) } /* Strides (in bytes) for each index */
  //);
  //
  //py::array_t<float> A_np(buf_info);
  //py::module np = py::module::import("numpy");
  //np.attr("save")("conv_outA_wrong.npy", A_np);

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      blocks.blocks[0]->attn.Q.A_at(i, j) +=
          positional_embedding[i * n_state + j];
    }
  }
  blocks.call();
  
  // int i_temp = 0;
  // for(auto block : blocks.blocks) 
  // {
  //   i_temp++;
  //   std::vector<float> y(n_ctx * n_state);
  //   //copy block->y to y
  //   for(int i = 0; i < n_ctx; i++) {
  //     for(int j = 0; j < n_state; j++) {
  //       y[i * n_state + j] = block->y[i * n_state + j];
  //     }
  //   }
  //   py::buffer_info buf_info(
  //     y.data(),                           /* Pointer to data */
  //     sizeof(float),                             /* Size of one scalar */
  //     py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //     2,                                         /* Number of dimensions */
  //     { n_ctx, n_state },                        /* Shape of the array */
  //     { sizeof(float) * n_state, sizeof(float) } /* Strides (in bytes) for each index */
  //   );

  //   py::array_t<float> block_y(buf_info);
  //   np.attr("save")("block" + std::to_string(i_temp) + "_wrong.npy", block_y);

  //   if(i_temp == 6)
  //   {
  //     //save block->attn.Q.A to block_6_lx_1_wrong.npy file
  //     std::vector<float> attn_Q_A2(n_ctx * n_state);
  //     //copy block->attn.Q.A to attn_Q_A2
  //     for(int i = 0; i < n_ctx; i++) {
  //       for(int j = 0; j < n_state; j++) {
  //         attn_Q_A2[i * n_state + j] = block->attn.Q.A_at(i, j);
  //       }
  //     }
  //     py::buffer_info buf_info(
  //       attn_Q_A2.data(),                           /* Pointer to data */
  //       sizeof(float),                             /* Size of one scalar */
  //       py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //       2,                                         /* Number of dimensions */
  //       { n_ctx, n_state },                        /* Shape of the array */
  //       { sizeof(float) * n_state, sizeof(float) } /* Strides (in bytes) for each index */
  //     );

  //     py::array_t<float> block_attn_Q_A(buf_info);
  //     np.attr("save")("block_6_lx_1_wrong.npy", block_attn_Q_A);

  //     //save block->x_copy to block_6_x_2_wrong.npy file
  //     std::vector<float> x_copy(n_ctx * n_state);
  //     //copy block->x_copy to x_copy
  //     for(int i = 0; i < n_ctx; i++) {
  //       for(int j = 0; j < n_state; j++) {
  //         x_copy[i * n_state + j] = block->x_copy[((j / 8) * n_ctx + i) * 8 + (j % 8)];
  //       }
  //     }
  //     py::buffer_info buf_info2(
  //       x_copy.data(),                           /* Pointer to data */
  //       sizeof(float),                             /* Size of one scalar */
  //       py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //       2,                                         /* Number of dimensions */
  //       { n_ctx, n_state },                        /* Shape of the array */
  //       { sizeof(float) * n_state, sizeof(float) } /* Strides (in bytes) for each index */
  //     );

  //     py::array_t<float> block_x_copy(buf_info2);
  //     np.attr("save")("block_6_x_2_wrong.npy", block_x_copy);

  //     //save block->fc1.A to block_6_lx_2_wrong.npy file
  //     std::vector<float> fc1_A(n_ctx * n_state);
  //     //copy block->fc1.A to fc1_A
  //     for(int i = 0; i < n_ctx; i++) {
  //       for(int j = 0; j < n_state; j++) {
  //         fc1_A[i * n_state + j] = block->fc1.A_at(i, j);
  //       }
  //     }
  //     py::buffer_info buf_info3(
  //       fc1_A.data(),                           /* Pointer to data */
  //       sizeof(float),                             /* Size of one scalar */
  //       py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //       2,                                         /* Number of dimensions */
  //       { n_ctx, n_state },                        /* Shape of the array */
  //       { sizeof(float) * n_state, sizeof(float) } /* Strides (in bytes) for each index */
  //     );

  //     py::array_t<float> block_fc1_A(buf_info3);
  //     np.attr("save")("block_6_lx_2_wrong.npy", block_fc1_A);
  //   }
  // }

  int last_layer_idx = blocks.blocks.size() - 1;
  layernorm(blocks.blocks[last_layer_idx]->y, n_ctx, n_state, ln_post_gamma,
            ln_post_beta);
}

TextDecoder::TextDecoder(int num_layers, int n_text_max_ctx, int n_state,
                         int n_head, int n_audio_ctx, int n_vocab)
    : num_layers(num_layers),
      n_text_max_ctx(n_text_max_ctx),
      n_state(n_state),
      n_head(n_head),
      n_audio_ctx(n_audio_ctx),
      n_vocab(n_vocab),
      positional_embedding(n_text_max_ctx * n_state),
      blocks(num_layers, 1, n_state, n_head, n_audio_ctx),
      ln_gamma(n_state),
      ln_beta(n_state),
      detokenizer0(1, n_state, n_vocab / 3, 0),
      detokenizer1(1, n_state, n_vocab / 3, 1),
      detokenizer2(1, n_state, n_vocab / 3, 2) {
  assert(n_vocab % 3 == 0);
}

void TextDecoder::call(int prompt) {
  int offset = blocks.blocks[0]->attn.cur_kv_len;
  int slice_len = n_vocab / 3;
  Matmul *detokenizer = prompt < slice_len       ? &detokenizer0
                        : prompt < 2 * slice_len ? &detokenizer1
                                                 : &detokenizer2;
  for (int j = 0; j < n_state; ++j) {
    blocks.blocks[0]->attn.Q.A_at(0, j) =
        detokenizer->B_at(j, prompt % (n_vocab / 3)) +
        positional_embedding[(0 + offset) * n_state + j];
  }
  blocks.call(1);
  int last_layer_idx = blocks.blocks.size() - 1;
  layernorm(blocks.blocks[last_layer_idx]->y, 1, n_state, ln_gamma, ln_beta);

#pragma omp parallel sections
  {
#pragma omp section
    {
      detokenizer0.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer0.call();
    }
#pragma omp section
    {
      detokenizer1.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer1.call();
    }
#pragma omp section
    {
      detokenizer2.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer2.call();
    }
  }
}

inline void suppress(__fp16 *src, int begin, int end,
                     const std::vector<int> &tokens) {
  // 0xfc00 is minus infinity for __fp16.
  uint16_t neg_inf_u16 = 0xfc00;
  __fp16 minus_inf = *(__fp16 *)&neg_inf_u16;
  for (int token : tokens) {
    if (token < begin || token >= end) continue;
    src[token] = minus_inf;
  }
}

inline void suppress16and32(__fp16 *src, float32_t *src2, int begin, int end,
                     const std::vector<int> &tokens) {
  // 0xfc00 is minus infinity for __fp16.
  uint16_t neg_inf_u16 = 0xfc00;
  __fp16 minus_inf = *(__fp16 *)&neg_inf_u16;
  uint32_t neg_inf_u32 = 0xff800000;
  float32_t minus_inf32 = *(float32_t *)&neg_inf_u32;
  for (int token : tokens) {
    if (token < begin || token >= end) continue;
    src[token] = minus_inf;
    src2[token] = minus_inf32;
  }
}

inline void suppress32(float32_t *src, int begin, int end,
                     const std::vector<int> &tokens) {
  // 0xff800000 is minus infinity for float32_t.
  uint32_t neg_inf_u32 = 0xff800000;
  float32_t minus_inf32 = *(float32_t *)&neg_inf_u32;
  for (int token : tokens) {
    if (token < begin || token >= end) continue;
    src[token] = minus_inf32;
  }
}

void TextDecoder::get_logits(__fp16 *logits) {
#pragma omp parallel sections
  {
#pragma omp section
    { copy_C_to_fp16(&detokenizer0, logits, 1, n_vocab / 3); }
#pragma omp section
    { copy_C_to_fp16(&detokenizer1, logits + n_vocab / 3, 1, n_vocab / 3); }
#pragma omp section
    { copy_C_to_fp16(&detokenizer2, logits + 2 * n_vocab / 3, 1, n_vocab / 3); }
  }
}

void TextDecoder::get_logits32(float *logits, 
                               const std::vector<int> &suppress_tokens) {
#pragma omp parallel sections
  {
#pragma omp section
    { copy_C_to_fp32(&detokenizer0, logits, 1, n_vocab / 3); 
      suppress32(logits, 0, n_vocab / 3, suppress_tokens);
    }
#pragma omp section
    { copy_C_to_fp32(&detokenizer1, logits + n_vocab / 3, 1, n_vocab / 3); 
      suppress32(logits, n_vocab / 3, 2 * n_vocab / 3, suppress_tokens);
    }
#pragma omp section
    { copy_C_to_fp32(&detokenizer2, logits + 2 * n_vocab / 3, 1, n_vocab / 3); 
      suppress32(logits, 2 * n_vocab / 3, n_vocab, suppress_tokens);
    }
  }
}

void TextDecoder::log_softmax(__fp16 *logits,
                              const std::vector<int> &suppress_tokens) {
  __fp16 max0, max1, max2;
#pragma omp parallel sections
  {
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer0, logits, 1, n_vocab / 3);
      suppress(logits, 0, n_vocab / 3, suppress_tokens);
      max0 = compute_max(logits, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer1, logits + n_vocab / 3, 1, n_vocab / 3);
      suppress(logits, n_vocab / 3, 2 * n_vocab / 3, suppress_tokens);
      max1 = compute_max(logits + n_vocab / 3, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer2, logits + 2 * n_vocab / 3, 1, n_vocab / 3);
      suppress(logits, 2 * n_vocab / 3, n_vocab, suppress_tokens);
      max2 = compute_max(logits + 2 * n_vocab / 3, n_vocab / 3);
    }
  }
  __fp16 max = std::max(std::max(max0, max1), max2);
  ::log_softmax(logits, n_vocab, max);
  // ::log_softmax32(logits, logits32, n_vocab, max);
}

void TextDecoder::log_softmax32(__fp16 *logits, float32_t *logits32,
                                const std::vector<int> &suppress_tokens) {
  __fp16 max0, max1, max2;
#pragma omp parallel sections
  {
#pragma omp section
    {
      copy_C_to_fp16and32(&detokenizer0, logits, logits32, 1, n_vocab / 3);
      suppress16and32(logits, logits32, 0, n_vocab / 3, suppress_tokens);
      max0 = compute_max(logits, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16and32(&detokenizer1, logits + n_vocab / 3,  logits32 + n_vocab / 3, 1, n_vocab / 3);
      suppress16and32(logits, logits32, n_vocab / 3, 2 * n_vocab / 3, suppress_tokens);
      max1 = compute_max(logits + n_vocab / 3, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16and32(&detokenizer2, logits + 2 * n_vocab / 3,  logits32 + 2 * n_vocab / 3, 1, n_vocab / 3);
      suppress16and32(logits, logits32, 2 * n_vocab / 3, n_vocab, suppress_tokens);
      max2 = compute_max(logits + 2 * n_vocab / 3, n_vocab / 3);
    }
  }
  __fp16 max = std::max(std::max(max0, max1), max2);
  ::log_softmax32(logits, logits32, n_vocab, max);
}

WhisperModel::WhisperModel(int n_mels, int n_audio_ctx, int n_audio_state,
                           int n_audio_head, int n_audio_layer, int n_text_ctx,
                           int n_text_state, int n_text_head, int n_text_layer,
                           int n_vocab)
    : n_mels(n_mels),
      n_audio_ctx(n_audio_ctx),
      n_audio_state(n_audio_state),
      n_audio_head(n_audio_head),
      n_audio_layer(n_audio_layer),
      n_text_ctx(n_text_ctx),
      n_text_state(n_text_state),
      n_text_head(n_text_head),
      n_text_layer(n_text_layer),
      n_vocab(n_vocab),
      encoder(n_audio_layer, n_audio_ctx, n_audio_state, n_audio_head),
      decoder(n_text_layer, n_text_ctx, n_text_state, n_text_head, n_audio_ctx,
              n_vocab) {}
