#include "../include/mha.hpp"
#include <math.h>
#include <thread>

cpu::mha::mha() {
  const float sc[] = {(float)HEAD_DIM};
  bmm_scale = torch::from_blob((void *)sc, {1, 1});
  bmm_scale = 1 / torch::sqrt(bmm_scale);
}

cpu::mha::~mha() {}

void cpu::mha::attention_head(torch::Tensor qh, torch::Tensor kh,
                              torch::Tensor vh, torch::Tensor attention_mask,
                              torch::Tensor oh) {

  auto res = torch::matmul(qh, kh);
  res = res * bmm_scale;
  res = res + attention_mask;
  res = torch::nn::functional::softmax(res, -1).to(torch::kBFloat16);
  torch::matmul_out(oh, res, vh);
}

torch::Tensor cpu::mha::mha_multithread(torch::Tensor query_states,
                                        torch::Tensor key_states,
                                        torch::Tensor value_states,
                                        torch::Tensor attention_mask) {
  key_states = key_states.transpose(2, 3);
  auto attn_output =
      torch::empty({1, NUM_HEADS, EMBED_SIZE, HEAD_DIM}).to(torch::kBFloat16);

  std::thread mha_threads[NUM_HEADS];
  for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
    mha_threads[head_idx] =
        std::thread(&cpu::mha::attention_head, this, query_states[0][head_idx],
                    key_states[0][head_idx], value_states[0][head_idx],
                    attention_mask[0][0], attn_output[0][head_idx]);
  }
  for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
    mha_threads[head_idx].join();
  }

  return attn_output;
}

torch::Tensor cpu::mha::mha_tensorized(torch::Tensor query_states,
                                       torch::Tensor key_states,
                                       torch::Tensor value_states,
                                       torch::Tensor attention_mask) {

  auto res =
      torch::matmul(query_states, key_states.transpose(2, 3)) * bmm_scale;
  res = res + attention_mask;
  // res.mul_(bmm_scale); // in place is slower
  // res.add_(attention_mask);
  res = torch::nn::functional::softmax(res, -1).to(torch::kBFloat16);
  /*
    { // dummy code - example
      torch::Tensor a =
          torch::rand({32, 2048, 2048}).to(torch::kBFloat16).contiguous();
      torch::Tensor b =
          torch::rand({1, 2048, 2048}).to(torch::kBFloat16).contiguous();
      auto xCasted = static_cast<int16_t *>(a.data_ptr());
      auto yCasted = static_cast<int16_t *>(b.data_ptr());
      int B = a.sizes()[0];
      int M = a.sizes()[1];
      int K = a.sizes()[2];
      std::tuple<int, int, int> a_shape = {B, M, K};
      int16_t *c;
      c = cpu::mha::npu_softmax.run_softmax<int16_t>(
          xCasted, yCasted, B, M, K, false, "bfloat16", "bfloat16", "bfloat16");
      torch::Tensor out = torch::from_blob(c, {B, M, K}, torch::kBFloat16);
    }
  */
  res = torch::matmul(res, value_states);
  return res;
}

torch::Tensor cpu::mha::mha_flat(torch::Tensor query_states,
                                 torch::Tensor key_states,
                                 torch::Tensor value_states,
                                 torch::Tensor attention_mask) {
  key_states = key_states.transpose(2, 3);
  torch::Tensor attn_output = torch::zeros({1, NUM_HEADS, EMBED_SIZE, HEAD_DIM})
                                  .to(torch::kBFloat16)
                                  .contiguous();

  query_states = query_states.contiguous();
  key_states = key_states.contiguous();
  value_states = value_states.contiguous();
  attention_mask = attention_mask.contiguous();

  using namespace torch::indexing;

  for (int h = 0; h < NUM_HEADS; h += 1) {
    torch::Tensor vv = value_states.index({0, h, Slice(), Slice()});
    torch::Tensor kk = key_states.index({0, h, Slice(), Slice()});
    for (int i = 0; i < EMBED_SIZE; i += 128) {
      // a = ixm
      torch::Tensor qq = query_states.index({0, h, Slice(i, i + 128), Slice()});
      torch::Tensor bca = torch::matmul(qq, kk) * bmm_scale;
      bca = bca + attention_mask.index({0, 0, Slice(i, i + 128), Slice()});
      bca = torch::nn::functional::softmax(bca, -1).to(torch::kBFloat16);
      bca = torch::matmul(bca, vv);
      attn_output.index_put_({0, h, Slice(i, i + 128), Slice()}, bca);
    }
  }

  return attn_output;
}

void cpu::mha::attention_head_flat(torch::Tensor qh, torch::Tensor kh,
                                   torch::Tensor vh,
                                   torch::Tensor attention_mask,
                                   torch::Tensor oh) {

  using namespace torch::indexing;

  for (int thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
    const int block_idx = thread_idx * BLOCK_SIZE;

    cpu::mha::attention_head(
        qh.index({Slice(block_idx, block_idx + BLOCK_SIZE), Slice()}), kh, vh,
        attention_mask.index(
            {Slice(block_idx, block_idx + BLOCK_SIZE), Slice()}),
        oh.index({Slice(block_idx, block_idx + BLOCK_SIZE), Slice()}));
    /*
    attn_threads[thread_idx] =
      std::thread(&cpu::mha::attention_head_fused_grp, this,
                            qh.index({Slice(block_idx, block_idx +  block_size),
    Slice()}), kh, vh, attention_mask.index({Slice(block_idx, block_idx +
    block_size), Slice()}), oh.index({Slice(block_idx, block_idx +  block_size),
    Slice()})
                            );
    */
  }
  // for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
  //   attn_threads[thread_idx].join();
  // }
}

torch::Tensor cpu::mha::mha_flat_multithread(torch::Tensor query_states,
                                             torch::Tensor key_states,
                                             torch::Tensor value_states,
                                             torch::Tensor attention_mask) {
  key_states = key_states.transpose(2, 3);
  torch::Tensor attn_output = torch::empty({1, NUM_HEADS, EMBED_SIZE, HEAD_DIM})
                                  .to(torch::kBFloat16)
                                  .contiguous();

  query_states = query_states.contiguous();
  key_states = key_states.contiguous();
  value_states = value_states.contiguous();
  attention_mask = attention_mask.contiguous();

  using namespace torch::indexing;

  std::thread mha_threads[NUM_HEADS];

  if ((EMBED_SIZE % 8) == 0) {
#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
      torch::Tensor qh = query_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor vh = value_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor kh = key_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor oh = attn_output.index({0, head_idx, Slice(), Slice()});
      mha_threads[head_idx] = std::thread(&cpu::mha::attention_head_flat, this,
                                          qh, kh, vh, attention_mask[0][0], oh);
    }
  } else {
#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
      torch::Tensor qh = query_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor vh = value_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor kh = key_states.index({0, head_idx, Slice(), Slice()});
      torch::Tensor oh = attn_output.index({0, head_idx, Slice(), Slice()});
      mha_threads[head_idx] = std::thread(&cpu::mha::attention_head, this, qh,
                                          kh, vh, attention_mask[0][0], oh);
    }
  }

  for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
    mha_threads[head_idx].join();
  }

  return attn_output;
}

torch::Tensor cpu::mha::mha_top(torch::Tensor query_states,
                                torch::Tensor key_states,
                                torch::Tensor value_states,
                                torch::Tensor attention_mask) {

  int seq_len = query_states.sizes()[2];
  if (seq_len < THRESHOLD) {
    return cpu::mha::mha_tensorized(query_states, key_states, value_states,
                                    attention_mask);
  } else {
    return cpu::mha::mha_flat_multithread(query_states, key_states,
                                          value_states, attention_mask);
  }
}
