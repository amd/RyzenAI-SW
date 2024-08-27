#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace segmentation {
template <class T>
void max_index_c(T* d, int c, int g, uint8_t* results) {
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it - d;
    d += c;
  }
}
template <typename T>
std::vector<T> permute(const T* input, size_t C, size_t H, size_t W) {
  std::vector<T> output(C * H * W);
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
      }
    }
  }
  return output;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}
void set_input_image_internal(const cv::Mat& image, float* data,
                              const std::vector<float>& mean,
                              const std::vector<float>& scale, bool btrans) {
  // BGR->RGB (maybe) and HWC->CHW
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = btrans ? abs(c - 2) : c;
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scale[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}
void set_input_image_rgb(const cv::Mat& image, float* data,
                         const std::vector<float>& mean,
                         const std::vector<float>& scale) {
  return set_input_image_internal(image, data, mean, scale, true);
}
struct Result {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Segmentation result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
};
Image show_reusult(Image& image, const Result& result) {
  cv::Mat resized_seg;
  cv::resize(result.segmentation, resized_seg, image.size());
  cv::Mat thresholded_seg;
  cv::threshold(resized_seg, thresholded_seg, 1, 255, cv::THRESH_BINARY);
  cv::Mat colored_seg;
  cv::applyColorMap(thresholded_seg, colored_seg, cv::COLORMAP_JET);
  cv::Mat mixed_image;
  cv::addWeighted(image, 0.5, colored_seg, 0.5, 0.0, mixed_image);
  cv::putText(mixed_image, std::string("SEGMENTATION"),
              cv::Point(20, image.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 255), 1, 1);
  return mixed_image;
}
}  // namespace segmentation
class Segmentation : public Model {
 public:
  Segmentation() {
    means_ = std::vector<float>{103.53f, 116.28f, 123.675f};
    scales_ = std::vector<float>{0.017429f, 0.017507f, 0.01712475f};
  }
  virtual ~Segmentation() {}
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
    cv::Mat resize_image;
    for (auto index = 0; index < batch_size; ++index) {
      cv::resize(images[index], resize_image, size);
      segmentation::set_input_image_rgb(
          resize_image, input_data_0.data() + index * batch_element_size,
          means_, scales_);
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    auto output_shape_0 = session_->get_output_shape_from_tensor(0);
    auto batch_size = images.size();
    auto output_0_ptr = session_->get_output(0);
    int64_t output_0_batch_number_elements =
        std::accumulate(output_shape_0.begin() + 1, output_shape_0.end(),
                        int64_t{1}, std::multiplies<int64_t>());
    auto oc = output_shape_0[1];
    auto oh = output_shape_0[2];
    auto ow = output_shape_0[3];
    std::vector<segmentation::Result> results;
    for (auto i = 0u; i < batch_size; ++i) {
      auto hwc = segmentation::permute(
          output_0_ptr + i * output_0_batch_number_elements, oc, oh, ow);
      cv::Mat result(oh, ow, CV_8UC1);
      segmentation::max_index_c(hwc.data(), oc, oh * ow, result.data);
      results.emplace_back(segmentation::Result{(int)ow, (int)oh, result});
    }
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = results[index];
      auto image = images[index];
      image_results.push_back(segmentation::show_reusult(image, result));
    }
    return image_results;
  }

 private:
  std::vector<float> means_;
  std::vector<float> scales_;
};
REGISTER_MODEL(segmentation, Segmentation)