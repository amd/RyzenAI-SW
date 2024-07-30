#include <torch/extension.h>

#include "../include/softmax_torch.hpp"

// for Llama2
#define NUM_HEADS 32    // query_states.sizes()[1];
#define EMBED_SIZE 2048 // query_states.sizes()[2];
#define HEAD_DIM 128    // query_states.sizes()[3];

// for win24
// #define NUM_HEADS 1     // query_states.sizes()[1];
// #define EMBED_SIZE 4096 // query_states.sizes()[2];
// #define HEAD_DIM 512    // query_states.sizes()[3];

#define THRESHOLD 513
#define NUM_THREADS 8
#define BLOCK_SIZE EMBED_SIZE / NUM_THREADS

namespace cpu {
class mha {
public:
  mha();
  ~mha();
  torch::Tensor bmm_scale;
  aie::softmax_torch npu_softmax;

  void attention_head(torch::Tensor query_states, torch::Tensor key_states,
                      torch::Tensor value_states, torch::Tensor attention_mask,
                      torch::Tensor attn_output);

  torch::Tensor mha_multithread(torch::Tensor query_states,
                                torch::Tensor key_states,
                                torch::Tensor value_states,
                                torch::Tensor attention_mask);

  torch::Tensor mha_tensorized(torch::Tensor query_states,
                               torch::Tensor key_states,
                               torch::Tensor value_states,
                               torch::Tensor attention_mask);

  torch::Tensor mha_flat(torch::Tensor query_states, torch::Tensor key_states,
                         torch::Tensor value_states,
                         torch::Tensor attention_mask);

  void cpu::mha::attention_head_flat(torch::Tensor qh, torch::Tensor kh,
                                     torch::Tensor vh,
                                     torch::Tensor attention_mask,
                                     torch::Tensor oh);

  torch::Tensor mha_flat_multithread(torch::Tensor query_states,
                                     torch::Tensor key_states,
                                     torch::Tensor value_states,
                                     torch::Tensor attention_mask);

  torch::Tensor mha_top(torch::Tensor query_states, torch::Tensor key_states,
                        torch::Tensor value_states,
                        torch::Tensor attention_mask);
};
} // namespace cpu
