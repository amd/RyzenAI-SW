/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <assert.h>
#include <glog/logging.h>

#include <algorithm>  // std::generate
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "onnx/onnx_task.hpp"
#include "util/env_config.hpp"

using namespace std;
using namespace cv;

struct SegmentationResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Segmentation result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
};

class SegmentationOnnx : public OnnxTask {
 public:
  static std::unique_ptr<SegmentationOnnx> create(
      const std::string& model_name, const OnnxConfig& onnx_config) {
    return std::unique_ptr<SegmentationOnnx>(
        new SegmentationOnnx(model_name, onnx_config));
  }

 protected:
  SegmentationOnnx(const std::string& model_name,
                   const OnnxConfig& onnx_config);
  SegmentationOnnx(const SegmentationOnnx&) = delete;

 public:
  virtual ~SegmentationOnnx() {}
  virtual std::vector<SegmentationResult> run(
      const std::vector<cv::Mat>& image);
  virtual SegmentationResult run(const cv::Mat& image) {
    return run(std::vector<cv::Mat>{image})[0];
  }

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  // int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;
};
namespace onnx_segmentation {
template <class T>
void max_index_c(T* d, int c, int g, uint8_t* results) {
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it-d;
    d += c;
  }
}
template <typename T>
std::vector<T> permute(const T* input, size_t C, size_t H, size_t W) {
  std::vector<T> output(C * H * W);
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
      }
    }
  }
  return output;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}
}  // namespace onnx_segmentation

SegmentationOnnx::SegmentationOnnx(const std::string& model_name,
                                   const OnnxConfig& onnx_config)
    : OnnxTask(model_name, onnx_config) {
  using namespace onnx_segmentation;
  auto input_shape = input_shapes_[0];
  int total_number_elements = onnx_segmentation::calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = channel * height * width;
  output_tensor_ptr.resize(1);
}

std::vector<SegmentationResult> SegmentationOnnx::run(
    const std::vector<cv::Mat>& image) {
  using namespace onnx_segmentation;
  cv::Mat resize_image;
  auto input_shape = input_shapes_[0];
  auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto size = cv::Size((int)width, (int)height);
  auto real_batch = std::min((uint32_t)image.size(), (uint32_t)batch);
  auto batch_size = channel * height * width;

  for (auto i = 0u; i < real_batch; ++i) {
    cv::resize(image[i], resize_image, size);
    set_input_image_rgb(resize_image,
                        input_tensor_values.data() + i * batch_size,
                        std::vector<float>{103.53f, 116.28f, 123.675f},
                        std::vector<float>{0.017429f, 0.017507f, 0.01712475f});
  }
  input_tensors = convert_input(input_tensor_values, input_tensor_values.size(),
                                input_shape);

  run_task(input_tensors, output_tensors);
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  auto output_batch_size =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount() / batch;
  auto oc = output_shapes_[0][1];
  auto oh = output_shapes_[0][2];
  auto ow = output_shapes_[0][3];
  std::vector<SegmentationResult> results;
  for (auto i = 0u; i < real_batch; ++i) {
    auto hwc =
        permute(output_tensor_ptr[0] + i * output_batch_size, oc, oh, ow);
    cv::Mat result(oh, ow, CV_8UC1);
    max_index_c(hwc.data(), oc, oh * ow, result.data);
    results.emplace_back(SegmentationResult{(int)ow, (int)oh, result});
  }
  return results;
}
