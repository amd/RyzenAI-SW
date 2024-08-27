#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace onnx_resnet50 {

cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}
//(image_data - mean) * scale, BRG2RGB and hwc2chw
static void set_input_image(const cv::Mat& image, float* data) {
  static float mean[3] = {103.53f, 116.28f, 123.675f};
  static float scales[3] = {0.017429f, 0.017507f, 0.01712475f};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2);  // BRG to RGB
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scales[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}

static std::vector<float> softmax(float* data, int64_t size) {
  auto output = std::vector<float>(size);
  std::transform(data, data + size, output.begin(), expf);
  auto sum =
      std::accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                        int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}
cv::Mat preprocess_one(const cv::Mat& image, cv::Size size) {
  float smallest_side = 256;
  float scale = smallest_side / ((image.rows > image.cols) ? (float)image.cols
                                                           : (float)image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size((int)(image.cols * scale), (int)(image.rows * scale)));
  return croppedImage(resized_image, size.height, size.width);
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
    auto cls = std::string("") + lookup(r.index);
    cv::putText(image, cls, cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 255), 2, 1);
    auto prob = std::string("prob. ") + std::to_string(r.score);
    cv::putText(image, prob, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2, 1);
  }
  cv::putText(image, std::string("RESNET50"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}
}  // namespace onnx_resnet50
class Resnet50 : public Model {
 public:
  Resnet50() {}
  virtual ~Resnet50() {}
  void preprocess(const std::vector<Image>& images) override {
    std::vector<float>& input_data_0 = session_->get_input(0);
    std::vector<int64_t>& input_shape_0 = session_->get_input_shape(0);
    int batch_size = images.size();
    input_shape_0[0] = batch_size;
    int64_t total_number_elements =
        std::accumulate(input_shape_0.begin(), input_shape_0.end(), int64_t{1},
                        std::multiplies<int64_t>());
    input_data_0.resize(size_t(total_number_elements));
    auto channel = input_shape_0[1];
    auto height = input_shape_0[2];
    auto width = input_shape_0[3];
    auto batch_element_size = channel * height * width;
    auto size = cv::Size((int)width, (int)height);
    for (auto index = 0; index < batch_size; ++index) {
      auto resize_image = onnx_resnet50::preprocess_one(images[index], size);
      onnx_resnet50::set_input_image(
          resize_image, input_data_0.data() + batch_element_size * index);
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    const std::vector<int64_t>& output_shape_0 =
        session_->get_output_shape_from_tensor(0);
    auto batch_size = images.size();
    auto channel = output_shape_0[1];
    float* output_0_ptr = session_->get_output(0);
    std::vector<onnx_resnet50::Result> results;
    for (auto index = 0; index < batch_size; ++index) {
      auto softmax_output =
          onnx_resnet50::softmax(output_0_ptr + channel * index, channel);
      auto tb_top5 = onnx_resnet50::topk(softmax_output, 5);
      onnx_resnet50::Result r;
      for (const auto& v : tb_top5) {
        r.scores.push_back(onnx_resnet50::Result::Score{v.first, v.second});
      }
      results.emplace_back(r);
    }
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = results[index];
      auto image = images[index];
      image_results.push_back(onnx_resnet50::show_reusult(image, result));
    }
    return image_results;
  }
};
REGISTER_MODEL(resnet50, Resnet50)