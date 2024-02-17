#pragma once
#include "resnet50_pt_onnx.hpp"
namespace resnet50_helper {
const char* lookup(int index) {
  static const char* table[] = {
#include "word_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
}
}  // namespace resnet50_helper
// for Resnet50PtOnnx
static cv::Mat process_result_resnet50(cv::Mat& image,
                                       const Resnet50PtOnnxResult& result,
                                       bool is_jpeg) {
  if(!result.scores.empty()){
    auto r = result.scores[0];
    auto cls = std::string("") + resnet50_helper::lookup(r.index);
    cv::putText(image, cls, cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(255, 0, 255), 2, 1);
    auto prob =std::string("prob. ") + std::to_string(r.score);
    cv::putText(image, prob, cv::Point(50, 120), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0), 2, 1);
  }
  cv::putText(image, std::string("RESNET50"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}