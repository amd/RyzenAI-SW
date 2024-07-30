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
struct RetinafaceOnnxResult {
  struct Face {
    std::vector<float> bbox;  // x, y, width, height
    float score;
  };
  std::vector<Face> faces;
};

struct SelectedOutput {
  float score;
  int index;
  std::vector<float> box;
  std::vector<float> box_decoded;
  friend inline bool operator<(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return lhs.score < rhs.score;
  }

  friend inline bool operator>(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return lhs.score > rhs.score;
  }
};
namespace onnx_retinaface {
std::vector<std::vector<float>> generate_anchors() {
  int width = 320;
  int height = 320;
  std::vector<std::vector<int>> min_sizes{{16, 32}, {64, 128}, {256, 512}};
  std::vector<int> steps{8, 16, 32};
  std::vector<std::vector<int>> feat_maps;
  int anchor_cnt = 0;
  for (auto k = 0u; k < min_sizes.size(); ++k) {
    feat_maps.emplace_back(
        std::vector<int>{int(std::ceil(((float)height) / steps[k])),
                         int(std::ceil(((float)width) / steps[k]))});
    anchor_cnt += feat_maps[k][0] * feat_maps[k][1] *
                  static_cast<int>(min_sizes[k].size());
  }
  // std::cout << "anchor_cnt:" << anchor_cnt;
  std::vector<std::vector<float>> anchors(anchor_cnt,
                                          std::vector<float>(4, 0.f));
  auto index = 0;
  for (auto k = 0u; k < min_sizes.size(); ++k) {
    auto x_step = ((float)steps[k]) / width;
    auto y_step = ((float)steps[k]) / height;
    auto x_start = 0.5f * x_step;
    auto y_start = 0.5f * y_step;
    for (auto i = 0; i < feat_maps[k][0]; ++i) {
      for (auto j = 0; j < feat_maps[k][1]; ++j) {
        for (auto min_size : min_sizes[k]) {
          auto s_kx = (float)min_size / width;
          auto s_ky = (float)min_size / height;
          anchors[index][0] = x_start + j * x_step;
          anchors[index][1] = y_start + i * y_step;
          anchors[index][2] = s_kx;
          anchors[index][3] = s_ky;
          index++;
        }
      }
    }
  }
  return anchors;
}

void decode(const float* src, const float* anchor, float* dst) {
  std::vector<float> variance{0.1f, 0.2f};
  dst[0] = anchor[0] + variance[0] * anchor[2] * src[0];
  dst[1] = anchor[1] + variance[0] * anchor[3] * src[1];
  dst[2] = anchor[2] * std::exp(src[2] * variance[1]);
  dst[3] = anchor[3] * std::exp(src[3] * variance[1]);
  dst[0] -= dst[2] / 2;
  dst[1] -= dst[3] / 2;
  dst[2] += dst[0];
  dst[3] += dst[1];
}

std::vector<std::vector<SelectedOutput>> select(Ort::Value& loc,
                                                Ort::Value& conf,
                                                float score_thresh) {
  auto output_shape = conf.GetTensorTypeAndShapeInfo().GetShape();
  auto batch = output_shape[0];
  auto feat_map_size = output_shape[1];

  auto conf_last_dim = output_shape[2];
  auto loc_last_dim = 4;

  auto conf_ptr = conf.GetTensorMutableData<float>();
  auto loc_ptr = loc.GetTensorMutableData<float>();

  std::vector<std::vector<SelectedOutput>> batch_result(batch);
  for (auto b = 0; b < batch; ++b) {
    auto& result = batch_result[b];
    result.reserve(200);
    auto cur_conf_ptr = conf_ptr + b * feat_map_size * conf_last_dim;
    auto cur_loc_ptr = loc_ptr + b * feat_map_size * loc_last_dim;
    for (auto i = 0; i < feat_map_size; ++i) {
      if (cur_conf_ptr[i * conf_last_dim + 1] > score_thresh) {
        auto index = i;
        auto score = cur_conf_ptr[i * conf_last_dim + 1];
        auto box = std::vector<float>(cur_loc_ptr + i * loc_last_dim,
                                      cur_loc_ptr + (i + 1) * loc_last_dim);

        auto select = SelectedOutput{score, index, box, box};
        result.emplace_back(select);
      }
    }
  }
  return batch_result;
}

vector<SelectedOutput> topK(const vector<SelectedOutput>& input, int k) {
  // assert(k >= 0);
  int size = (int)input.size();
  int num = std::min(size, k);
  std::vector<SelectedOutput> result(input.begin(), input.begin() + num);
  std::make_heap(result.begin(), result.begin() + num, std::greater<>());
  for (auto i = num; i < size; ++i) {
    if (input[i] > result[0]) {
      std::pop_heap(result.begin(), result.end(), std::greater<>());
      result[num - 1] = input[i];
    }
  }

  for (auto i = 0; i < num; ++i) {
    std::pop_heap(result.begin(), result.begin() + num - i, std::greater<>());
  }
  // std::stable_sort(result.begin(), result.end(), compare);
  return result;
}

static float overlap(float x1, float w1, float x2, float w2) {
  float left = std::max(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
  float right = std::min(x1 + w1 / 2.0f, x2 + w2 / 2.0f);
  return right - left;
}

float cal_iou_xywh(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0f / union_area;
}

float cal_iou_xyxy(vector<float> box, vector<float> truth) {
  float box_w = box[2] - box[0];
  float box_h = box[3] - box[1];
  float truth_w = truth[2] - truth[0];
  float truth_h = truth[3] - truth[1];
  float w = overlap(box[0], box_w, truth[0], truth_w);
  float h = overlap(box[1], box_h, truth[1], truth_h);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0f / union_area;
}

float cal_iou_yxyx(vector<float> box, vector<float> truth) {
  float box_h = box[2] - box[0];
  float box_w = box[3] - box[1];
  float truth_h = truth[2] - truth[0];
  float truth_w = truth[3] - truth[1];
  float h = overlap(box[0], box_h, truth[0], truth_h);
  float w = overlap(box[1], box_w, truth[1], truth_w);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0f / union_area;
}

vector<SelectedOutput> nms(vector<SelectedOutput>& candidates, float nms_thresh,
                           float score_thresh, int max_output_num,
                           bool need_sort) {
  vector<SelectedOutput> result;
  auto compare = [](const SelectedOutput& l, const SelectedOutput& r) {
    return l.score >= r.score;
  };

  // Todo: sort
  if (need_sort) {
    std::stable_sort(candidates.begin(), candidates.end(), compare);
  }

  // nms;
  auto size = candidates.size();
  vector<bool> exist_box(size, true);
  for (size_t i = 0; i < size; ++i) {
    if (!exist_box[i]) {
      continue;
    }
    if (candidates[i].score < score_thresh) {
      exist_box[i] = false;
      continue;
    }
    result.push_back(candidates[i]);
    for (size_t j = i + 1; j < size; ++j) {
      if (!exist_box[j]) {
        continue;
      }
      if (candidates[j].score < score_thresh) {
        exist_box[j] = false;
        continue;
      }
      float overlap = 0.0;
      overlap =
          cal_iou_xyxy(candidates[i].box_decoded, candidates[j].box_decoded);
      if (overlap >= nms_thresh) {
        exist_box[j] = false;
      }
    }
  }

  if (result.size() > (unsigned int)max_output_num) {
    result.resize(max_output_num);
  }
  return result;
}
}  // namespace onnx_retinaface
// model class
class RetinafaceOnnx : public OnnxTask {
 public:
  static std::unique_ptr<RetinafaceOnnx> create(const std::string& model_name,
                                                const OnnxConfig& onnx_config) {
    return std::unique_ptr<RetinafaceOnnx>(
        new RetinafaceOnnx(model_name, onnx_config));
  }

 protected:
  explicit RetinafaceOnnx(const std::string& model_name,
                          const OnnxConfig& onnx_config);
  RetinafaceOnnx(const RetinafaceOnnx&) = delete;

 public:
  virtual ~RetinafaceOnnx() {}
  virtual std::vector<RetinafaceOnnxResult> run(
      const std::vector<cv::Mat>& mats);
  virtual RetinafaceOnnxResult run(const cv::Mat& mats);

 private:
  std::vector<RetinafaceOnnxResult> postprocess();
  // RetinafaceOnnxResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> input_tensor_ptr;
  std::vector<float*> output_tensor_ptr;
  int output_tensor_size = 3;
  int channel = 0;
  int sHeight = 0;
  int sWidth = 0;
  std::vector<std::vector<float>> anchors_;
};
void RetinafaceOnnx::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  auto size = cv::Size((int)sWidth, (int)sHeight);
  if (image.size() != size) {
    float x_scale = ((float)sWidth) / image.cols;
    float y_scale = ((float)sHeight) / image.rows;
    float scale = std::min(x_scale, y_scale);
    auto resize_size =
        cv::Size((int)(image.cols * scale), (int)(image.rows * scale));
    cv::resize(image, resized_image, resize_size);
  } else {
    resized_image = image;
  }
  set_input_image_bgr(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{104.0f, 117.0f, 123.0f}, std::vector<float>{1, 1, 1});
  return;
}

// preprocess
void RetinafaceOnnx::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
  return;
}

// postprocess
std::vector<RetinafaceOnnxResult> RetinafaceOnnx::postprocess() {
  using namespace onnx_retinaface;
  auto& loc = output_tensors[0];
  auto& conf = output_tensors[1];
  // auto& landm = output_tensors[2];

  int pre_nms_num = 1000;
  auto nms_thresh = 0.2f;
  int max_output_num = 200;

  auto score_thresh = 0.02f;

  auto output_shape = conf.GetTensorTypeAndShapeInfo().GetShape();
  auto batch = output_shape[0];

  std::vector<RetinafaceOnnxResult> batch_results(batch);
  // 1. select all scores over score_thresh
  auto batch_selected = select(loc, conf, score_thresh);

  for (auto b = 0; b < batch; ++b) {
    // 2. topk
    auto topk_selected = topK(batch_selected[b], pre_nms_num);
    // 3. decode
    for (auto i = 0u; i < topk_selected.size(); ++i) {
      auto index = topk_selected[i].index;
      // 3.1 decode box
      decode(topk_selected[i].box.data(), anchors_[index].data(),
             topk_selected[i].box_decoded.data());
    }
    // 4. nms
    auto nms_result =
        nms(topk_selected, nms_thresh, score_thresh, max_output_num, false);
    //  5. make result
    batch_results[b].faces.resize(nms_result.size());
    for (auto i = 0u; i < nms_result.size(); ++i) {
      batch_results[b].faces[i].score = nms_result[i].score;
      batch_results[b].faces[i].bbox = nms_result[i].box_decoded;
    }
  }

  return batch_results;
}

// std::vector<RetinafaceOnnxResult> RetinafaceOnnx::postprocess() {
//   std::vector<RetinafaceOnnxResult> ret;
//   for (auto index = 0; index < (int)real_batch; ++index) {
//     ret.emplace_back(postprocess(index));
//   }
//   return ret;
// }

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

RetinafaceOnnx::RetinafaceOnnx(const std::string& model_name,
                               const OnnxConfig& onnx_config)
    : OnnxTask(model_name, onnx_config) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
  batch_size = channel * sHeight * sWidth;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  anchors_ = onnx_retinaface::generate_anchors();
}

RetinafaceOnnxResult RetinafaceOnnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<RetinafaceOnnxResult> RetinafaceOnnx::run(
    const std::vector<cv::Mat>& mats) {
  preprocess(mats);
  Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Value::CreateTensor<float>(
        info, input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0].data(), input_shapes_[0].size());

  } else {
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        info, input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0].data(), input_shapes_[0].size()));
  }

  run_task(input_tensors, output_tensors);
  for (int i = 0; i < output_tensor_size; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }

  std::vector<RetinafaceOnnxResult> ret = postprocess();
  return ret;
}