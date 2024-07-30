#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

static std::vector<char> load_bin(const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Couldn't open file : "s + filename);
  }

  std::istreambuf_iterator<char> begin_it{ifs}, end_it;
  std::vector<char> data(begin_it, end_it);
  return data;
}

static int max_abs_error(const int16_t *vec1, const int16_t *vec2,
                         size_t size) {
  int err = 0;
  for (size_t i = 0; i < size; ++i) {
    err = std::max(err, std::abs(vec1[i] - vec2[i]));
  }
  return err;
}

using MatrixShape = std::array<size_t, 2>;
static void matmul_ref(const int16_t *A, const int8_t *B, int16_t *C,
                       MatrixShape A_shape, MatrixShape B_shape) {
  auto M = A_shape[0];
  auto K = A_shape[1];
  auto N = B_shape[1];
  if (K != B_shape[0]) {
    throw std::runtime_error("Matmul : Shape mismatch in inner dimension");
  }

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc{0};
      for (size_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[m * K + k]) *
               static_cast<int32_t>(B[k * N + n]);
      }
      C[m * N + n] = static_cast<int16_t>(
          std::clamp<int32_t>(acc, std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()));
    }
  }
}

static float vec_sum(const std::vector<uint8_t> &out2) {
  int16_t *ptr = (int16_t *)out2.data();
  size_t size = out2.size() / sizeof(int16_t);
  return std::accumulate(ptr, ptr + size, 0.0f,
                         [](float a, int16_t b) { return a + b; });
}

static void cpu_ref(const OpsFusion::Metadata &meta,
                    const std::vector<void *> &ins,
                    const std::vector<void *> &outs, size_t M, size_t K,
                    size_t N) {
  // Weights
  std::vector<std::vector<char>> weights;
  for (auto &[tname, tinfo] : meta.tensor_map) {
    if (tinfo.parent_name != "const")
      continue;
    std::vector<char> data = load_bin(tinfo.file_name);

    std::cout << "Load const : " << tinfo.file_name << ", " << data.size()
              << ", " << tinfo.file_size << std::endl;
    if (data.size() != tinfo.file_size) {
      throw std::runtime_error("Size of file:"s + tinfo.file_name +
                               " doesn't match with metadata info.");
    }
    weights.push_back(std::move(data));
  }
  int16_t *X = reinterpret_cast<int16_t *>(ins.front());
  int16_t *Y = reinterpret_cast<int16_t *>(outs.front());

  MatrixShape act_shape{M, K};
  MatrixShape wts_shape{K, N};

  matmul_ref(X, reinterpret_cast<int8_t *>(weights.at(0).data()), Y, act_shape,
             wts_shape);
}

static void test_model(const std::string &meta_json, size_t niters,
                       const std::string &xclbin) {
  std::string xclbin_fname = xclbin;
  auto meta = OpsFusion::load_meta_json(meta_json);

  auto context =
      ryzenai::dynamic_dispatch::xrt_context::get_instance(xclbin_fname)
          ->get_context();

  OpsFusion::FusionRuntime rt(&context, "DPU");
  rt.init(meta);
  auto fops2 = rt.get_txns();

  // Prepare inputs
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<Tensor> in_tensors =
      OpsFusion::MetaUtils::get_input_tensors(meta);
  for (auto &tensor : in_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);
    std::vector<uint8_t> in(sz, 1);
    rand_init_int((int8_t *)in.data(), in.size() / sizeof(int8_t));
    inputs.push_back(std::move(in));
    tensor.data = inputs.back().data();
  }

  // Prepare outputs
  std::vector<Tensor> out_tensors =
      OpsFusion::MetaUtils::get_output_tensors(meta);
  std::vector<std::vector<uint8_t>> outputs;
  for (auto &tensor : out_tensors) {
    size_t sz = std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                Utils::get_size_of_type(tensor.dtype);

    outputs.emplace_back(sz, 1);
    tensor.data = outputs.back().data();
  }

  std::cout << OpsFusion::MetaUtils::get_summary(rt.get_meta()) << std::endl;

  std::cout << "Executing for iterations:" << niters << std::endl;
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < niters; ++i) {
    rt.execute(in_tensors, out_tensors);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "Avg. Time (ms) : "
            << std::chrono::duration<float, std::milli>(t2 - t1).count() /
                   niters
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json> [niters=1]" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    size_t niters = (argc > 2) ? std::atoll(argv[2]) : 1;
    std::string xclbin =
        (argc > 3) ? std::string(argv[3]) : PSF_A8W8_QDQ_XCLBIN_REL_PATH;
    test_model(meta_json, niters, xclbin);

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
