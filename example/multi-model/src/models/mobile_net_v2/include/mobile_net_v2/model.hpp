#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace onnx_mobile_net_v2 {
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
cv::Mat preprocess_one(const cv::Mat& image, cv::Size size) {
  auto [height_pad, width_pad] = onnx_mobile_net_v2::find_black_border(image);
  auto real_image = image(cv::Range(height_pad, image.rows - height_pad),
                          cv::Range(width_pad, image.cols - width_pad));
  cv::Mat resized_image;
  cv::resize(real_image, resized_image, size);
  return resized_image;
}
struct Result {
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
Image show_reusult(Image& image, const Result& result) {
  if (!result.scores.empty()) {
    auto r = result.scores[0];
    auto cls = std::string("") + onnx_mobile_net_v2::lookup(r.index);
    cv::putText(image, cls, cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 255), 2, 1);
    auto prob = std::string("prob. ") + std::to_string(r.score);
    cv::putText(image, prob, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2, 1);
  }
  cv::putText(image, std::string("MOBILENETV2"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}
}  // namespace onnx_mobile_net_v2
class MobileNetV2 : public Model {
 public:
  MobileNetV2() {}
  virtual ~MobileNetV2() {}
  void preprocess(const std::vector<Image>& images) override {
    std::vector<float>& input_data_0 = session_->get_input(0);
    std::vector<int64_t>& input_shape_0 = session_->get_input_shape(0);
    int batch_size = images.size();
    input_shape_0[0] = batch_size;
    int64_t total_number_elements =
        std::accumulate(input_shape_0.begin(), input_shape_0.end(), int64_t{1},
                        std::multiplies<int64_t>());
    input_data_0.resize(size_t(total_number_elements));
    auto channel = input_shape_0[3];
    auto height = input_shape_0[1];
    auto width = input_shape_0[2];
    auto batch_element_size = channel * height * width;
    auto size = cv::Size((int)width, (int)height);
    for (auto index = 0; index < batch_size; ++index) {
      auto resize_image =
          onnx_mobile_net_v2::preprocess_one(images[index], size);
      onnx_mobile_net_v2::set_input_image(
          resize_image, input_data_0.data() + batch_element_size * index);
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    const std::vector<int64_t>& output_shape_0 =
        session_->get_output_shape_from_tensor(0);
    auto batch_size = images.size();
    auto channel = output_shape_0[1];
    float* output_0_ptr = session_->get_output(0);
    std::vector<onnx_mobile_net_v2::Result> results;
    for (auto index = 0; index < batch_size; ++index) {
      auto tb_top5 = onnx_mobile_net_v2::topk(
          output_0_ptr + channel * index + 1, channel - 1, 5);
      onnx_mobile_net_v2::Result r;
      for (const auto& v : tb_top5) {
        r.scores.push_back(
            onnx_mobile_net_v2::Result::Score{v.first, v.second});
      }
      results.emplace_back(r);
    }
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = results[index];
      auto image = images[index];
      image_results.push_back(onnx_mobile_net_v2::show_reusult(image, result));
    }
    return image_results;
  }
};
REGISTER_MODEL(mobile_net_v2, MobileNetV2)