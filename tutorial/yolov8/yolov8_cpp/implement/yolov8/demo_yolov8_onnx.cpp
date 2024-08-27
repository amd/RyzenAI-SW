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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "color.hpp"


using namespace cv;


static void process_result(cv::Mat& image, const Yolov8OnnxResult& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;

    std::cout << "result: " << label << "\t"  << classes[label] << "\t" << std::fixed << std::setprecision(2)
         << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
         << std::setprecision(4) << res.score << "\n";
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
              cv::Scalar(b[label], g[label], r[label]), 3, 1, 0);
    cv::putText(image, classes[label] + " " + std::to_string(res.score),
                    cv::Point(box[0] + 5, box[1] + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(b[label], g[label], r[label]), 2, 4);
                    // cv::Scalar(230, 216, 173), 2, 4);
  }
  return;
}

int main(int argc, char* argv[]) {
  printf("YOLOV8 CAMERA SAMPLE\n");
  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  if (!cap.isOpened()) { cout << "Error: Cannot open the video camera." << endl; return -1; }

  if (argc < 2) {
    std::cout << "usage: " << argv[0] << "<model_name>" << std::endl;
    return 0;
  }

  std::cout << "load model " << argv[1] << endl;
  auto model = Yolov8Onnx::create(std::string(argv[1]), 0.3);
  if (!model) {  // supress coverity complain
    std::cout << "failed to create model\n";
    return 0;
  }

  std::vector<cv::Mat> images(1);
  while(1) {
    __TIC__(CAPTURE)
    cap.read(images[0]);
    __TOC__(CAPTURE)

    __TIC__(ONNX_RUN)
    auto results = model->run(images);
    __TOC__(ONNX_RUN)

    __TIC__(SHOW)
    process_result(images[0], results[0]);
    cv::imshow("yolov8-camera", images[0]);
    
    if (char(cv::waitKey(1)) == 27) 
      exit(1);
    __TOC__(SHOW)
  }

  return 0;
}
