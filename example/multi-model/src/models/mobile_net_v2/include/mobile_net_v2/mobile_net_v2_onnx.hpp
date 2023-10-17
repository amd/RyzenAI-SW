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

namespace onnx_mobile_net_v2 {

static cv::Mat croppedImage(const cv::Mat& image, int height, int width);
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}
std::pair<int, int> find_black_border(const cv::Mat& image) {
  int height_pad = 0, width_pad = 0;
  int height = image.rows;
  int width = image.cols;
  int mid_height = image.rows / 2;
  int mid_width = image.cols / 2;
  for (int i = 0; i < height; i++) {
    auto pixel = image.at<cv::Vec3b>(i, mid_width);
    if (pixel.val[0] > 1 || pixel.val[1] > 1 || pixel.val[2] > 1) {
      height_pad = i;
      break;
    }
  }
  for (int i = 0; i < width; i++) {
    auto pixel = image.at<cv::Vec3b>(mid_height, i);
    if (pixel.val[0] > 1 || pixel.val[1] > 1 || pixel.val[2] > 1) {
      width_pad = i;
      break;
    }
  }
  return std::make_pair(height_pad, width_pad);
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  auto [height_pad, width_pad] = find_black_border(image);
  auto real_image = image(cv::Range(height_pad, image.rows - height_pad),
                          cv::Range(width_pad, image.cols - width_pad));
  cv::Mat resized_image;
  cv::resize(real_image, resized_image, size);
  return resized_image;
}

static void set_input_image(const cv::Mat& image, float* data) {
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = c;  // abs(c - 2);  // BRG to RGB
        auto image_data = float(image.at<cv::Vec3b>(h, w)[c_t]) / 128.0f - 1.0f;
        data[h * image.cols * 3 + w * 3 + c] = (float)image_data;
      }
    }
  }
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static std::vector<std::pair<int, float>> topk(float* score, size_t size,
                                               int K) {
  auto indices = std::vector<int>(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}
}  // namespace onnx_mobile_net_v2

struct MobileNetV2OnnxResult {
  struct Score {
    ///  The index of the result in the ImageNet.
    int index;
    ///  Confidence of this category.
    float score;
  };
  /**
   *A vector of object width confidence in the first k; k defaults to 5 and
   *can be modified through the model configuration file.
   */
  std::vector<Score> scores;
};

class MobileNetV2Onnx : public OnnxTask {
 public:
  static std::unique_ptr<MobileNetV2Onnx> create(
      const std::string& model_name, const OnnxConfig& onnx_config) {
    return std::unique_ptr<MobileNetV2Onnx>(
        new MobileNetV2Onnx(model_name, onnx_config));
  }
  virtual ~MobileNetV2Onnx() {}
  MobileNetV2Onnx(const std::string& model_name, const OnnxConfig& onnx_config)
      : OnnxTask(model_name, onnx_config) {}

  MobileNetV2Onnx(const MobileNetV2Onnx&) = delete;

  std::vector<MobileNetV2OnnxResult> run(
      const std::vector<cv::Mat> batch_images) {
    std::vector<std::vector<int64_t>> input_shapes = get_input_shapes();
    std::vector<std::vector<int64_t>> output_shapes = get_output_shapes();

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes[0];
    int total_number_elements =
        onnx_mobile_net_v2::calculate_product(input_shape);
    std::vector<float> input_tensor_values(total_number_elements);
    auto hw_batch = input_shape[0];
    auto valid_batch = std::min((int)hw_batch, (int)batch_images.size());

    preprocess(batch_images, input_tensor_values, input_shape);

    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()));
    std::vector<Ort::Value> output_tensors;
    run_task(input_tensors, output_tensors);

    auto results = postprocess(output_tensors[0], valid_batch);
    return results;
  }

  MobileNetV2OnnxResult run(const cv::Mat image) {
    return run(vector<cv::Mat>(1, image))[0];
  }

 protected:
  void preprocess(const std::vector<cv::Mat>& images,
                  std::vector<float>& input_tensor_values,
                  std::vector<int64_t>& input_shape) {
    auto batch = input_shape[0];
    auto channel = input_shape[3];
    auto height = input_shape[1];
    auto width = input_shape[2];
    auto batch_size = channel * height * width;

    auto size = cv::Size((int)width, (int)height);
    for (auto index = 0; index < batch; ++index) {
      auto resize_image =
          onnx_mobile_net_v2::preprocess_image(images[index], size);
      onnx_mobile_net_v2::set_input_image(
          resize_image, input_tensor_values.data() + batch_size * index);
    }
  }
  std::vector<MobileNetV2OnnxResult> postprocess(Ort::Value& output_tensor,
                                                 int valid_batch) {
    std::vector<MobileNetV2OnnxResult> results;
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto channel = output_shape[1];
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
    for (auto index = 0; index < valid_batch; ++index) {
      auto tb_top5 = onnx_mobile_net_v2::topk(
          output_tensor_ptr + channel * index + 1, channel - 1, 5);
      MobileNetV2OnnxResult r;
      for (const auto& v : tb_top5) {
        r.scores.push_back(MobileNetV2OnnxResult::Score{v.first, v.second});
      }
      results.emplace_back(r);
    }
    return results;
  }
};
