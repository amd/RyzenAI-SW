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
#include "yolov8_onnx.hpp"
#include "demo_nx1x4.hpp"
#include <iostream>
#include "color.hpp"

static cv::Mat process_result(cv::Mat& image, const Yolov8OnnxResult& result, bool is_jpeg=false) {
  __TIC__(process_result)
  float width_ratio = 1;
  float height_ratio = 1;
  if(cap_height != display_height || cap_width != display_width) {
    width_ratio = (float)display_width / (float)image.cols;
    height_ratio = (float)display_height / (float)image.rows;
    cv::resize(image, image, cv::Size(display_width, display_height));
  }
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto box = res.box;
    box[0] *= width_ratio;
    box[2] *= width_ratio;
    box[1] *= height_ratio;
    box[3] *= height_ratio;
    
    if (enable_result_print) {
      std::cout << "result: " << label << "\t"  << classes[label] << "\t" << std::fixed << std::setprecision(2)
          << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
          << std::setprecision(4) << res.score << "\n";
    }
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
              cv::Scalar(b[label], g[label], r[label]), 3, 1, 0);
    cv::putText(image, classes[label] + " " + std::to_string(res.score),
                    cv::Point(box[0] + 5, box[1] + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(b[label], g[label], r[label]), 2, 4);
  }
  __TOC__(process_result)
  return image;
}

int main(int argc, char* argv[]) {
  return vitis::ai::main_for_video_demo(
      argc, argv, [&] {return  Yolov8Onnx::create("DetectionModel_int.onnx", 0.3);}, process_result);

  return 0;
}
