#pragma once
#include <vector>

#include "retinaface.hpp"
cv::Mat process_result_retinaface(cv::Mat& image,
                                  const RetinafaceOnnxResult& result,
                                  bool is_jpeg) {
  for (auto i = 0u; i < result.faces.size(); ++i) {
    auto& face = result.faces[i];
    int x = static_cast<int>(face.bbox[0] * image.cols);
    int y = static_cast<int>(face.bbox[1] * image.rows);
    int w = static_cast<int>((face.bbox[2] - face.bbox[0]) * image.cols);
    int h = static_cast<int>((face.bbox[3] - face.bbox[1]) * image.rows);
    cv::rectangle(image, cv::Point(x, y), cv::Point(x + w, y + h),
                  cv::Scalar(0, 255, 255), 2, 1, 0);
  }
  cv::putText(image, std::string("RETINAFACE"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}