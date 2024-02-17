#pragma once
#include "mobile_net_v2_onnx.hpp"
namespace mobilenetv2_helper {
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
}  // namespace mobilenetv2_helper
static cv::Mat process_result_mobilenet_v2(cv::Mat& image,
                                           const MobileNetV2OnnxResult& result,
                                           bool is_jpeg) {
  if (!result.scores.empty()) {
    auto r = result.scores[0];
    auto cls = std::string("") + mobilenetv2_helper::lookup(r.index);
    cv::putText(image, cls, cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 0, 255), 2, 1);
    auto prob = std::string("prob. ") + std::to_string(r.score);
    cv::putText(image, prob, cv::Point(50, 120), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2, 1);
  }
  cv::putText(image, std::string("MOBILENETV2"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}