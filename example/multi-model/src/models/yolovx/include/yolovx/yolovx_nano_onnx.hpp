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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "onnx/onnx_task.hpp"

DEF_ENV_PARAM(ENABLE_YOLO_DEBUG, "0");

using namespace std;
using namespace cv;

namespace onnx_yolovx {

static float overlap(float x1, float w1, float x2, float w2) {
  float left = (float)max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = (float)min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return (float)(inter_area * 1.0 / union_area);
}

static void applyNMS(const vector<vector<float>>& boxes,
                     const vector<float>& scores, const float nms,
                     const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  sort(order.begin(), order.end(),
       [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
         return ls.first > rs.first;
       });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

static void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om,
                      float& scale) {
  scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}
}  // namespace onnx_yolovx

// return value
struct YolovxnanoOnnxResult {
  /**
   *@struct BoundingBox
   *@brief Struct of detection result with an object.
   */
  struct BoundingBox {
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
class YolovxnanoOnnx : public OnnxTask {
 public:
  static std::unique_ptr<YolovxnanoOnnx> create(const std::string& model_name,
                                                const float conf_thresh_,const OnnxConfig& onnx_config) {
    return std::unique_ptr<YolovxnanoOnnx>(
        new YolovxnanoOnnx(model_name, conf_thresh_, onnx_config));
  }

 protected:
  explicit YolovxnanoOnnx(const std::string& model_name,
                          const float conf_thresh_,const OnnxConfig& onnx_config);
  YolovxnanoOnnx(const YolovxnanoOnnx&) = delete;

 public:
  virtual ~YolovxnanoOnnx() {}
  virtual std::vector<YolovxnanoOnnxResult> run(
      const std::vector<cv::Mat>& mats);
  virtual YolovxnanoOnnxResult run(const cv::Mat& mats);

 private:
  std::vector<YolovxnanoOnnxResult> postprocess();
  YolovxnanoOnnxResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx, float& scale);
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
  float stride[3] = {8, 16, 32};
  float conf_thresh = 0.f;
  float conf_desigmoid = 0.f;
  float nms_thresh = 0.65f;
  int num_classes = 80;
  int anchor_cnt = 1;
  vector<float> scales;
};