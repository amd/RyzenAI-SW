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
#include <iostream>
#include "color.hpp"

static void process_result(cv::Mat& image, const Yolov8OnnxResult& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;
    std::cout << "result: " << label << "\t"  << classes[label] << "\t" << std::fixed << std::setprecision(5)
         << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
         << std::setprecision(6) << res.score << "\n";
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
              cv::Scalar(b[label], g[label], r[label]), 3, 1, 0);
    cv::putText(image, classes[label] + " " + std::to_string(res.score),
                    cv::Point(box[0] + 5, box[1] + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(b[label], g[label], r[label]), 2, 4);
  }
  return;
}


int main(int argc, char* argv[]) {
  printf("YOLOV8 JPEG SAMPLE\n");

  if (argc < 3) {
    std::cout << "usage: " << argv[0] << "<model_name> <image>" << std::endl;
    return 0;
  }


  std::cout << "load model " << argv[1] << endl;
  auto model = Yolov8Onnx::create(std::string(argv[1]), 0.3);
  if (!model) {  // supress coverity complain
    std::cout << "failed to create model\n";
    return 0;
  }

  cv::Mat image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "cannot load " << argv[2] << std::endl;
    return -1;
  }


  __TIC__(ONNX_RUN)
  auto results = model->run(image);
  __TOC__(ONNX_RUN)

  __TIC__(SHOW)
  process_result(image, results);
  auto out_file = "result.jpg";
  cv::imwrite(out_file, image);
  __TOC__(SHOW)

  return 0;
}
