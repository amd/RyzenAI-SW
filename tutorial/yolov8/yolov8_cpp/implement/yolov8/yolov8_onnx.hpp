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
#include <opencv2/imgproc/imgproc_c.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
<<<<<<< HEAD
<<<<<<< HEAD
#include <numeric>  //accumulate
=======
#include <numeric> //accumulate
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
#include <numeric> //accumulate
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(ENABLE_YOLO_DEBUG, "0");

using namespace std;
using namespace cv;

<<<<<<< HEAD
<<<<<<< HEAD
static float overlap(float x1, float w1, float x2, float w2) {
=======
static float overlap(float x1, float w1, float x2, float w2)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
static float overlap(float x1, float w1, float x2, float w2)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

<<<<<<< HEAD
<<<<<<< HEAD
static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
static float cal_iou(vector<float> box, vector<float> truth)
{
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0)
    return 0;
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

<<<<<<< HEAD
<<<<<<< HEAD
static void applyNMS(const vector<vector<float>>& boxes,
                     const vector<float>& scores, const float nms,
                     const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
static void applyNMS(const vector<vector<float>> &boxes,
                     const vector<float> &scores, const float nms,
                     const float conf, vector<size_t> &res)
{
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i)
  {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t> &ls, const pair<float, size_t> &rs)
              {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
<<<<<<< HEAD
<<<<<<< HEAD
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
            [](auto &km)
            { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i)
  {
    size_t i = ordered[_i];
    if (!exist_box[i])
      continue;
    if (scores[i] < conf)
    {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
<<<<<<< HEAD
<<<<<<< HEAD
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    for (size_t _j = _i + 1; _j < count; ++_j)
    {
      size_t j = ordered[_j];
      if (!exist_box[j])
        continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms)
        exist_box[j] = false;
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    }
  }
}

<<<<<<< HEAD
<<<<<<< HEAD
static void letterbox(const cv::Mat input_image, cv::Mat& output_image,
                      const int height, const int width, float& scale,
                      int& left, int& top) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
static void letterbox(const cv::Mat input_image, cv::Mat &output_image,
                      const int height, const int width, float &scale,
                      int &left, int &top)
{
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  cv::Mat image_tmp;

  scale = std::min(float(width) / input_image.cols,
                   float(height) / input_image.rows);
  scale = std::min(scale, 1.0f);
  int unpad_w = round(input_image.cols * scale);
  int unpad_h = round(input_image.rows * scale);
  image_tmp = input_image.clone();

<<<<<<< HEAD
<<<<<<< HEAD
  if (input_image.size() != cv::Size(unpad_w, unpad_h)) {
=======
  if (input_image.size() != cv::Size(unpad_w, unpad_h))
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  if (input_image.size() != cv::Size(unpad_w, unpad_h))
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    cv::resize(input_image, image_tmp, cv::Size(unpad_w, unpad_h),
               cv::INTER_LINEAR);
  }

  float dw = (width - unpad_w) / 2.0f;
  float dh = (height - unpad_h) / 2.0f;

  top = round(dh - 0.1);
  int bottom = round(dh + 0.1);
  left = round(dw - 0.1);
  int right = round(dw + 0.1);

  cv::copyMakeBorder(image_tmp, output_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  return;
}

<<<<<<< HEAD
<<<<<<< HEAD
static vector<float> softmax(const std::vector<float>& input) {
=======
static vector<float> softmax(const std::vector<float> &input)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
static vector<float> softmax(const std::vector<float> &input)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
<<<<<<< HEAD
<<<<<<< HEAD
                 [sum](float v) { return v / sum; });
  return output;
}

static vector<float> conv(const vector<vector<float>>& input) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
                 [sum](float v)
                 { return v / sum; });
  return output;
}

static vector<float> conv(const vector<vector<float>> &input)
{
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  // input size is 4 x 16
  // kernel is 16 x 1, value is 0,1,...,15
  vector<float> output(4, 0.0f);

<<<<<<< HEAD
<<<<<<< HEAD
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 16; col++) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  for (int row = 0; row < 4; row++)
  {
    for (int col = 0; col < 16; col++)
    {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
      output[row] += input[row][col] * col;
    }
  }
  return output;
}

<<<<<<< HEAD
<<<<<<< HEAD
static vector<vector<float>> make_anchors(int w, int h) {
  vector<vector<float>> anchor_points;
  anchor_points.reserve(w * h);
  for (int i = 0; i < w; ++i) {
    float sy = i + 0.5f;
    for (int j = 0; j < h; ++j) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
static vector<vector<float>> make_anchors(int w, int h)
{
  vector<vector<float>> anchor_points;
  anchor_points.reserve(w * h);
  for (int i = 0; i < w; ++i)
  {
    float sy = i + 0.5f;
    for (int j = 0; j < h; ++j)
    {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
      float sx = j + 0.5f;
      vector<float> anchor(2);
      anchor[0] = sx;
      anchor[1] = sy;
      anchor_points.emplace_back(anchor);
    }
  }
  return anchor_points;
}

<<<<<<< HEAD
<<<<<<< HEAD
static vector<float> dist2bbox(const vector<float>& distance,
                               const vector<float>& point, const float stride) {
=======
static vector<float> dist2bbox(const vector<float> &distance,
                               const vector<float> &point, const float stride)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
static vector<float> dist2bbox(const vector<float> &distance,
                               const vector<float> &point, const float stride)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  vector<float> box;
  box.resize(4);
  float x1 = point[0] - distance[0];
  float y1 = point[1] - distance[1];
  float x2 = point[0] + distance[2];
  float y2 = point[1] + distance[3];
<<<<<<< HEAD
<<<<<<< HEAD
  box[0] = (x1 + x2) / 2.0f * stride;  // x_c
  box[1] = (y1 + y2) / 2.0f * stride;  // y_c
  box[2] = (x2 - x1) * stride;         // width
  box[3] = (y2 - y1) * stride;         // height
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  box[0] = (x1 + x2) / 2.0f * stride; // x_c
  box[1] = (y1 + y2) / 2.0f * stride; // y_c
  box[2] = (x2 - x1) * stride;        // width
  box[3] = (y2 - y1) * stride;        // height
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  return box;
}

// return value
<<<<<<< HEAD
<<<<<<< HEAD
struct Yolov8OnnxResult {
=======
struct Yolov8OnnxResult
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
struct Yolov8OnnxResult
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  /**
   *@struct BoundingBox
   *@brief Struct of detection result with an object.
   */
<<<<<<< HEAD
<<<<<<< HEAD
  struct BoundingBox {
=======
  struct BoundingBox
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  struct BoundingBox
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    /// Classification.
    int label;
    /// Confidence. The value ranges from 0 to 1.
    float score;
    /// (x0,y0,x1,y1). x0, x1 Range from 0 to the input image columns.
    /// y0,y1. Range from 0 to the input image rows.
    std::vector<float> box;
  };
  /// All objects, The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};

// model class
<<<<<<< HEAD
<<<<<<< HEAD
class Yolov8Onnx : public OnnxTask {
 public:
  static std::unique_ptr<Yolov8Onnx> create(const std::string& model_name,
                                            const float conf_thresh_) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
class Yolov8Onnx : public OnnxTask
{
public:
  static std::unique_ptr<Yolov8Onnx> create(const std::string &model_name,
                                            const float conf_thresh_)
  {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    // cout << "create" << endl;
    return std::unique_ptr<Yolov8Onnx>(
        new Yolov8Onnx(model_name, conf_thresh_));
  }

<<<<<<< HEAD
<<<<<<< HEAD
 protected:
  explicit Yolov8Onnx(const std::string& model_name, const float conf_thresh_);
  Yolov8Onnx(const Yolov8Onnx&) = delete;

 public:
  virtual ~Yolov8Onnx() {}
  virtual std::vector<Yolov8OnnxResult> run(const std::vector<cv::Mat>& mats);
  virtual Yolov8OnnxResult run(const cv::Mat& mats);

 private:
  std::vector<Yolov8OnnxResult> postprocess();
  Yolov8OnnxResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx, float& scale, int& left,
                  int& top);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
protected:
  explicit Yolov8Onnx(const std::string &model_name, const float conf_thresh_);
  Yolov8Onnx(const Yolov8Onnx &) = delete;

public:
  virtual ~Yolov8Onnx() {}
  virtual std::vector<Yolov8OnnxResult> run(const std::vector<cv::Mat> &mats);
  virtual Yolov8OnnxResult run(const cv::Mat &mats);

private:
  std::vector<Yolov8OnnxResult> postprocess();
  Yolov8OnnxResult postprocess(int idx);
  void preprocess(const cv::Mat &image, int idx, float &scale, int &left,
                  int &top);
  void preprocess(const std::vector<cv::Mat> &mats);

private:
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
<<<<<<< HEAD
<<<<<<< HEAD
  std::vector<float*> input_tensor_ptr;
  std::vector<float*> output_tensor_ptr;
=======
  std::vector<float *> input_tensor_ptr;
  std::vector<float *> output_tensor_ptr;
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  std::vector<float *> input_tensor_ptr;
  std::vector<float *> output_tensor_ptr;
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  int output_tensor_size = 4;
  int channel = 0;
  int sHeight = 0;
  int sWidth = 0;
  float stride[4] = {0, 8, 16, 32};
  float conf_thresh = 0.f;
  float nms_thresh = 0.7f;
  int num_classes = 80;
  int max_nms_num = 300;
  int max_boxes_num = 30000;

  vector<float> scales;
  vector<int> left;
  vector<int> top;
};

<<<<<<< HEAD
<<<<<<< HEAD
void Yolov8Onnx::preprocess(const cv::Mat& image, int idx, float& scale,
                            int& left, int& top) {
=======
void Yolov8Onnx::preprocess(const cv::Mat &image, int idx, float &scale,
                            int &left, int &top)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
void Yolov8Onnx::preprocess(const cv::Mat &image, int idx, float &scale,
                            int &left, int &top)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  cv::Mat resized_image;
  letterbox(image, resized_image, sHeight, sWidth, scale, left, top);
  set_input_image_rgb(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{0, 0, 0},
                      std::vector<float>{0.00392157, 0.00392157, 0.00392157});
  return;
}

// preprocess
<<<<<<< HEAD
<<<<<<< HEAD
void Yolov8Onnx::preprocess(const std::vector<cv::Mat>& mats) {
=======
void Yolov8Onnx::preprocess(const std::vector<cv::Mat> &mats)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
void Yolov8Onnx::preprocess(const std::vector<cv::Mat> &mats)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  scales.resize(real_batch);
  left.resize(real_batch);
  top.resize(real_batch);

<<<<<<< HEAD
<<<<<<< HEAD
  for (auto i = 0; i < real_batch; ++i) {
=======
  for (auto i = 0; i < real_batch; ++i)
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  for (auto i = 0; i < real_batch; ++i)
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    preprocess(mats[i], i, scales[i], left[i], top[i]);
  }
  return;
}

inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }

// postprocess
<<<<<<< HEAD
<<<<<<< HEAD
Yolov8OnnxResult Yolov8Onnx::postprocess(int idx) {
=======
Yolov8OnnxResult Yolov8Onnx::postprocess(int idx)
{
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
Yolov8OnnxResult Yolov8Onnx::postprocess(int idx)
{
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  vector<vector<float>> boxes;
  int count = 0;
  vector<vector<vector<float>>> pre_output;
  __TIC__(DECODE)
<<<<<<< HEAD
<<<<<<< HEAD
  for (int i = 1; i < output_tensor_size; i++) {
    int ca = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  for (int i = 1; i < output_tensor_size; i++)
  {
    int ca = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG))
    {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
      LOG(INFO) << "channel=" << ca << ", height=" << ha << ", width=" << wa
                << ", stride=" << stride[i] << ", conf=" << conf_thresh
                << ", idx=" << idx << endl;
    }
    auto anchor_points = make_anchors(wa, ha);
    int sizeOut = wa * ha;

    boxes.reserve(boxes.size() + sizeOut);
    auto conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
    pre_output.reserve(pre_output.size() + sizeOut);
<<<<<<< HEAD
<<<<<<< HEAD
#define POS(C) ((C)*ha * wa + h * wa + w)
    for (int h = 0; h < ha; ++h) {
      for (int w = 0; w < wa; ++w) {
        vector<vector<float>> pre_output_unit;
        pre_output_unit.resize(4);

        for (auto t = 0; t < 4; t++) {
          vector<float> softmax_;
          softmax_.reserve(16);
          for (auto m = 0; m < 16; m++) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
#define POS(C) ((C) * ha * wa + h * wa + w)
    for (int h = 0; h < ha; ++h)
    {
      for (int w = 0; w < wa; ++w)
      {
        vector<vector<float>> pre_output_unit;
        pre_output_unit.resize(4);

        for (auto t = 0; t < 4; t++)
        {
          vector<float> softmax_;
          softmax_.reserve(16);
          for (auto m = 0; m < 16; m++)
          {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
            float value =
                output_tensor_ptr[i][POS(t * 16 + m) + idx * ca * wa * ha];
            softmax_.emplace_back(value);
          }
          pre_output_unit[t] = softmax(softmax_);
        }
        auto distance = conv(pre_output_unit);
        auto dbox = dist2bbox(distance, anchor_points[h * wa + w], stride[i]);
<<<<<<< HEAD
<<<<<<< HEAD
        for (auto m = 0; m < num_classes; ++m) {
          auto score = output_tensor_ptr[i][POS(64 + m) + idx * ca * wa * ha];
          if (score > conf_desigmoid) {
            count++;
            vector<float> box(6);
            for (int j = 0; j < 4; j++) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
        for (auto m = 0; m < num_classes; ++m)
        {
          auto score = output_tensor_ptr[i][POS(64 + m) + idx * ca * wa * ha];
          if (score > conf_desigmoid)
          {
            count++;
            vector<float> box(6);
            for (int j = 0; j < 4; j++)
            {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
              box[j] = dbox[j];
            }
            float cls_score = 1.0 / (1 + exp(-1.0f * score));
            box[4] = m;
            box[5] = cls_score;
            boxes.emplace_back(box);
          }
        }
      }
    }
  }
  __TOC__(DECODE)
<<<<<<< HEAD
<<<<<<< HEAD
  auto compare = [=](vector<float>& lhs, vector<float>& rhs) {
    return lhs[5] > rhs[5];
  };
  if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
    LOG(INFO) << "boxes_total_size=" << boxes.size();
  }
  if (static_cast<int>(boxes.size()) > max_boxes_num) {
    std::partial_sort(boxes.begin(), boxes.begin() + max_boxes_num, boxes.end(),
                      compare);
    boxes.resize(max_boxes_num);
  } else {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  auto compare = [=](vector<float> &lhs, vector<float> &rhs)
  {
    return lhs[5] > rhs[5];
  };
  if (ENV_PARAM(ENABLE_YOLO_DEBUG))
  {
    LOG(INFO) << "boxes_total_size=" << boxes.size();
  }
  if (static_cast<int>(boxes.size()) > max_boxes_num)
  {
    std::partial_sort(boxes.begin(), boxes.begin() + max_boxes_num, boxes.end(),
                      compare);
    boxes.resize(max_boxes_num);
  }
  else
  {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    std::sort(boxes.begin(), boxes.end(), compare);
  }

  /* Apply the computation for NMS */
  vector<vector<vector<float>>> boxes_for_nms(num_classes);
  vector<vector<float>> scores(num_classes);

<<<<<<< HEAD
<<<<<<< HEAD
  for (const auto& box : boxes) {
=======
  for (const auto &box : boxes)
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  for (const auto &box : boxes)
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    boxes_for_nms[box[4]].push_back(box);
    scores[box[4]].push_back(box[5]);
  }

  __TIC__(NMS)
  vector<vector<float>> res;
<<<<<<< HEAD
<<<<<<< HEAD
  for (auto i = 0; i < num_classes; i++) {
=======
  for (auto i = 0; i < num_classes; i++)
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  for (auto i = 0; i < num_classes; i++)
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    vector<size_t> result_k;
    applyNMS(boxes_for_nms[i], scores[i], nms_thresh, 0, result_k);
    res.reserve(res.size() + result_k.size());
    transform(result_k.begin(), result_k.end(), back_inserter(res),
<<<<<<< HEAD
<<<<<<< HEAD
              [&](auto& k) { return boxes_for_nms[i][k]; });
  }
  __TOC__(NMS)

  if (static_cast<int>(res.size()) > max_nms_num) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
              [&](auto &k)
              { return boxes_for_nms[i][k]; });
  }
  __TOC__(NMS)

  if (static_cast<int>(res.size()) > max_nms_num)
  {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    __TIC__(PSORT)
    std::partial_sort(res.begin(), res.begin() + max_nms_num, res.end(),
                      compare);
    __TOC__(PSORT)
    res.resize(max_nms_num);
<<<<<<< HEAD
<<<<<<< HEAD
  } else {
=======
  }
  else
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  }
  else
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    __TIC__(SORT)
    std::sort(res.begin(), res.end(), compare);
    __TOC__(SORT)
  }

  __TIC__(BBOX)
  vector<Yolov8OnnxResult::BoundingBox> results;
<<<<<<< HEAD
<<<<<<< HEAD
  for (const auto& r : res) {
=======
  for (const auto &r : res)
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  for (const auto &r : res)
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    Yolov8OnnxResult::BoundingBox result;
    result.score = r[5];
    result.label = r[4];
    result.box.resize(4);
    result.box[0] = (r[0] - r[2] / 2.0f - left[idx]) / scales[idx];
    result.box[1] = (r[1] - r[3] / 2.0f - top[idx]) / scales[idx];
    result.box[2] = result.box[0] + r[2] / scales[idx];
    result.box[3] = result.box[1] + r[3] / scales[idx];
    results.push_back(result);
  }
  __TOC__(BBOX)

  return Yolov8OnnxResult{results};
}

<<<<<<< HEAD
<<<<<<< HEAD
std::vector<Yolov8OnnxResult> Yolov8Onnx::postprocess() {
  std::vector<Yolov8OnnxResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
std::vector<Yolov8OnnxResult> Yolov8Onnx::postprocess()
{
  std::vector<Yolov8OnnxResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index)
  {
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    ret.emplace_back(postprocess(index));
  }
  return ret;
}

<<<<<<< HEAD
<<<<<<< HEAD
static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

Yolov8Onnx::Yolov8Onnx(const std::string& model_name, const float conf_thresh_)
    : OnnxTask(model_name) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  // cout << total_number_elements << endl; 
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
static int calculate_product(const std::vector<int64_t> &v)
{
  int total = 1;
  for (auto &i : v)
    total *= (int)i;
  return total;
}

Yolov8Onnx::Yolov8Onnx(const std::string &model_name, const float conf_thresh_)
    : OnnxTask(model_name)
{
  int total_number_elements = calculate_product(input_shapes_[0]);
  // cout << total_number_elements << endl;
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
<<<<<<< HEAD
<<<<<<< HEAD
  if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
=======
  if (ENV_PARAM(ENABLE_YOLO_DEBUG))
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  if (ENV_PARAM(ENABLE_YOLO_DEBUG))
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    LOG(INFO) << "channel=" << channel << ", height=" << sHeight
              << ", width=" << sWidth << endl;
  }
  batch_size = channel * sHeight * sWidth;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  conf_thresh = conf_thresh_;
}

<<<<<<< HEAD
<<<<<<< HEAD
Yolov8OnnxResult Yolov8Onnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<Yolov8OnnxResult> Yolov8Onnx::run(
    const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);
  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
=======
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
Yolov8OnnxResult Yolov8Onnx::run(const cv::Mat &mats)
{
  return run(vector<cv::Mat>(1, mats))[0];
}
std::vector<Yolov8OnnxResult> Yolov8Onnx::run(
    const std::vector<cv::Mat> &mats)
{
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);
  if (input_tensors.size())
  {
    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors[0] = Ort::Value::CreateTensor<float>(info,
                                                       input_tensor_values.data(), input_tensor_values.size(),
                                                       input_shapes_[0].data(), input_shapes_[0].size());
  }
  else
  {
    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(info,
                                                            input_tensor_values.data(), input_tensor_values.size(),
                                                            input_shapes_[0].data(), input_shapes_[0].size()));
<<<<<<< HEAD
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
  }

  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
<<<<<<< HEAD
<<<<<<< HEAD
  for (int i = 1; i < output_tensor_size; i++) {
=======
  for (int i = 1; i < output_tensor_size; i++)
  {
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
  for (int i = 1; i < output_tensor_size; i++)
  {
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<Yolov8OnnxResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> d78b7488 (Merge branch 'dev' into unified_public)
=======
>>>>>>> f83a0188 (Merge pull request #94 from VitisAI/dev)
