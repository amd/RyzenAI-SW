/*
 *  Copyright 2022 Xilinx Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **/
#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm> // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#if _WIN32
extern "C" {
#  include "util/getopt.h"
}
#  include <codecvt>
#  include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#else
#  include <getopt.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        int64_t batch);
static cv::Mat croppedImage(const cv::Mat& image, int height, int width);
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size);
static void set_input_image(const cv::Mat& image, float* data);
static std::vector<float> softmax(float* data, int64_t size);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
static const char* lookup(int index);
#define CHECK(expr, msg)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << (msg) << std::endl;          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
// resnet50 preprocess
static void preprocess_resnet50(const std::vector<std::string>& files,
                                std::vector<float>& input_tensor_values,
                                std::vector<int64_t>& input_shape) {
  auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto batch_size = channel * height * width;

  auto size = cv::Size((int)width, (int)height);
  auto images = read_images(files, batch);
  CHECK(images.size() == (long unsigned int)batch,
        "images number be read into input buffer must be equal to batch");

  for (auto index = 0; index < batch; ++index) {
    auto resize_image = preprocess_image(images[index], size);
    set_input_image(resize_image,
                    input_tensor_values.data() + batch_size * index);
  }
}

// resnet50 postprocess
static void postprocess_resnet50(const std::vector<std::string>& files,
                                 Ort::Value& output_tensor) {
  auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto batch = output_shape[0];
  auto channel = output_shape[1];
  auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
  auto images = read_images(files, batch);
  CHECK(images.size() == (long unsigned int)batch,
        "images number be read into input buffer must be equal to batch");
  for (auto index = 0; index < batch; ++index) {
    auto softmax_output = softmax(output_tensor_ptr + channel * index, channel);
    auto tb_top5 = topk(softmax_output, 5);
    std::cout << "batch_index: " << index << std::endl;
    print_topk(tb_top5);
    auto top1 = tb_top5[0];
    auto cls = std::string("") + lookup(top1.first) + " prob. " +
               std::to_string(top1.second);
    auto image = images[index];
    cv::putText(image, cls, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(20, 20, 180), 1, 1);
    cv::imshow("resnet50_result_batch_" + std::to_string(index), image);
    cv::waitKey(0);
    cv::imwrite("result_batch_" + std::to_string(index) + "_" +
                    files[index % files.size()],
                image);
  }
}

// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

static void usage() {
  std::cout << "usage: resnet50_pt <onnx model> [-b <batch_number>] [-c "
               "'json_config'] [-n: disable vitisAI_ep]"
               " [-p: enable onnx profiler] <img_url> [img_url]... \n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  int opt = 0;
  int64_t batch_number = 1;
  bool enable_profiler = false;
  string json_config = "";
  bool enable_ep = true;
  bool dump = false;
  while ((opt = getopt(argc, argv, "b:c:p:d:n")) != -1) {
    switch (opt) {
    case 'b':
      batch_number = std::stoi(optarg);
      break;
    case 'p':
      enable_profiler = true;
      break;
    case 'c':
      json_config = std::string(optarg);
      break;
    case 'd':
      dump = true;
      break;
    case 'n':
      enable_ep = false;
      cout << "enable_ep = false" << endl;
      break;
    default:
      usage();
      exit(1);
    }
  }
  if (optind >= argc - 1) {
    usage();
    exit(1);
  }
#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[optind]));
  auto profile_str_name = std::string("profile_test_onnx_runner");
  auto profile_name =
      std::wstring(profile_str_name.begin(), profile_str_name.end());
#else
  auto model_name = std::string(argv[optind]);
#endif
  std::vector<std::string> g_image_files;
  for (auto i = optind + 1; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_pt");
  auto session_options = Ort::SessionOptions();
  if (enable_profiler) {
#if _WIN32
    session_options.EnableProfiling(profile_name.c_str());
#else
    session_options.EnableProfiling("profile_resnet50_pt");
#endif
  }
  if (enable_ep) {
    // assume running at root directory and json is inside bin folder
#ifdef _WIN32
    auto config_file = std::string("../vaip_config.json");
#else
    auto config_file = std::string("/etc/vaip_config.json");
#endif
    if (json_config.size()) {
      config_file = json_config;
    }
    auto config_key = std::string{"config_file"};
    auto options =
        std::unordered_map<std::string, std::string>{{config_key, config_file}};
    session_options.AppendExecutionProvider("VitisAI", options);
  }
  auto session = Ort::Session(env, model_name.data(), session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_count = session.GetInputCount();
  auto input_shapes = std::vector<std::vector<int64_t>>();
  auto input_names_ptr = std::vector<Ort::AllocatedStringPtr>();
  auto input_names = std::vector<const char*>();
  input_shapes.reserve(input_count);
  input_names_ptr.reserve(input_count);
  input_names.reserve(input_count);
  cout << "Input Node Name/Shape (" << input_count << "):" << endl;
  for (size_t i = 0; i < input_count; i++) {
    input_shapes.push_back(
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    auto name = session.GetInputNameAllocated(i, allocator);
    input_names.push_back(name.get());
    input_names_ptr.push_back(std::move(name));
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i])
         << endl;
  }

  // print name/shape of outputs
  auto output_count = session.GetOutputCount();
  auto output_shapes = std::vector<std::vector<int64_t>>();
  auto output_names_ptr = std::vector<Ort::AllocatedStringPtr>();
  auto output_names = std::vector<const char*>();
  output_shapes.reserve(output_count);
  output_names_ptr.reserve(output_count);
  output_names.reserve(output_count);
  cout << "Output Node Name/Shape (" << output_count << "):" << endl;
  for (size_t i = 0; i < output_count; i++) {
    auto shape =
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    output_shapes.push_back(shape);
    auto name = session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name.get());
    output_names_ptr.push_back(std::move(name));
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i])
         << endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  // Create a single Ort tensor of random numbers.
  auto input_shape = input_shapes[0];
  if (input_shape[0] == -1) {
    input_shape[0] = batch_number;
  }
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values(total_number_elements);
  preprocess_resnet50(g_image_files, input_tensor_values, input_shape);

  std::vector<Ort::Value> input_tensors;
  Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size()));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() ==
             input_shape);
  cout << "\ninput_tensor shape: "
       << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape())
       << endl;
  if (dump) {
    auto input_tensor_ptr = input_tensors[0].GetTensorData<float>();
    auto batch = input_shape[0];
    auto total_number_elements =
        input_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    auto size = total_number_elements / batch * sizeof(float);
    for (int index = 0; index < batch; ++index) {
      auto filename = "onnx_input_chw_float_" + std::to_string(0) + "_batch_" +
                      std::to_string(index) + "_float.bin";
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      CHECK(std::ofstream(filename, mode)
                .write((char*)input_tensor_ptr + size * index, size)
                .good(),
            std::string(" faild to write to ") + filename);
    }
  }

  // pass data through model
  cout << "Running model...";
  try {
    auto output_tensors =
        session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == output_count &&
           output_tensors[0].IsTensor());
    auto output_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    cout << "output_tensor_shape: " << print_shape(output_shape) << endl;
    postprocess_resnet50(g_image_files, output_tensors[0]);
  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }
  return 0;
}

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        int64_t batch) {
  std::vector<cv::Mat> images(batch);
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty(),
          std::string("cannot read image from ") + file);
  }
  return images;
}

static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  float smallest_side = 256;
  float scale = smallest_side / ((image.rows > image.cols) ? (float)image.cols
                                                           : (float)image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * (int)scale, image.rows * (int)scale));
  return croppedImage(resized_image, size.height, size.width);
}

//(image_data - mean) * scale, BRG2RGB and hwc2chw
static void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {103.53f, 116.28f, 123.675f};
  float scales[3] = {0.017429f, 0.017507f, 0.01712475f};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2); // BRG to RGB
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

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
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

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
  for (const auto& v : topk) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first)
              << std::resetiosflags(std::ios::left) << std::endl;
  }
}

static const char* lookup(int index) {
  static const char* table[] = {
#include "word_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
}
