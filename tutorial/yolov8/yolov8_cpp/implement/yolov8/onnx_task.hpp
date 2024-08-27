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
#pragma once
#include <onnxruntime_cxx_api.h>

#include <sstream>

#include <onnxruntime_session_options_config_keys.h>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"
#if _WIN32
extern "C"
{
#include "util/getopt.h"
}
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif

DEF_ENV_PARAM(DEBUG_ONNX_TASK, "0")

extern int onnx_x = -1;
extern int onnx_y = -1;
extern bool onnx_disable_spinning = false;
extern bool enable_result_print = false;
extern bool onnx_disable_spinning_between_run = false;
extern std::string intra_op_thread_affinities = "";

#if 0
static void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    Ort::GetApi().ReleaseStatus(status);
    exit(1);
  }
}
#endif
// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t> &v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

std::vector<std::vector<int64_t>> get_input_shapes(Ort::Session *session)
{
  std::vector<std::vector<int64_t>> input_shapes;
  const size_t num_input_nodes = session->GetInputCount();
  input_shapes.resize(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; i++)
  {
    auto type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_shapes[i] = tensor_info.GetShape();
  }
  return input_shapes;
}
std::vector<std::vector<int64_t>> get_output_shapes(Ort::Session *session)
{
  std::vector<std::vector<int64_t>> output_shapes;
  const size_t num_output_nodes = session->GetOutputCount();
  output_shapes.resize(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++)
  {
    auto type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_shapes[i] = tensor_info.GetShape();
  }
  return output_shapes;
}
void get_input_names(Ort::Session *session,
                     std::vector<Ort::AllocatedStringPtr> &input_names_ptr,
                     std::vector<const char *> &input_node_names)
{
  const size_t num_input_nodes = session->GetInputCount();
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < num_input_nodes; i++)
  {
    auto input_name = session->GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));
  }
}
void get_output_names(Ort::Session *session,
                      std::vector<Ort::AllocatedStringPtr> &output_names_ptr,
                      std::vector<const char *> &output_node_names)
{
  const size_t num_output_nodes = session->GetOutputCount();
  output_names_ptr.reserve(num_output_nodes);
  output_node_names.reserve(num_output_nodes);
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < num_output_nodes; i++)
  {
    auto output_name = session->GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(output_name.get());
    output_names_ptr.push_back(std::move(output_name));
  }
}

class OnnxTask
{
public:
  explicit OnnxTask(const std::string &model_name)
      : model_name_(model_name),
        env_(ORT_LOGGING_LEVEL_WARNING, model_name_.c_str()),
        session_options_(Ort::SessionOptions())
  {
    auto options = std::unordered_map<std::string, std::string>({});
    options["config_file"] = "../bin/vaip_config.json";
    // optional, eg: cache path and cache key: /tmp/my_cache/abcdefg
    // options["CacheDir"] = "/tmp/my_cache";
    // options["CacheKey"] = "abcdefg";

    session_options_.AppendExecutionProvider_VitisAI(options);

    if (onnx_x > 0)
    {
      fprintf(stdout, "Setting intra_op_num_threads to %d\n", onnx_x);
      session_options_.SetIntraOpNumThreads(onnx_x);
    }

    if (onnx_y > 0)
    {
      fprintf(stdout, "Setting inter_op_num_threads to %d\n", onnx_y);
      session_options_.SetInterOpNumThreads(onnx_y);
    }

    if (onnx_disable_spinning)
    {
      fprintf(stdout, "Disabling intra-op thread spinning entirely\n");
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    }

    if (onnx_disable_spinning_between_run)
    {
      fprintf(stdout, "Disabling intra-op thread spinning between runs\n");
      session_options_.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop,
                                      "1");
    }

    if (!intra_op_thread_affinities.empty())
    {
      fprintf(stdout, "Setting intra op thread affinity as %s\n",
              intra_op_thread_affinities.c_str());
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigIntraOpThreadAffinities,
          intra_op_thread_affinities.c_str());
    }

    auto model_name_basic = strconverter.from_bytes(model_name_);

    session_.reset(new Ort::Session(env_, model_name_basic.c_str(), session_options_));
    input_shapes_ = ::get_input_shapes(session_.get());
    output_shapes_ = ::get_output_shapes(session_.get());
    if (input_shapes_[0][0] == -1)
    {
      input_shapes_[0][0] = 1;
      output_shapes_[0][0] = 1;
    }
    ::get_input_names(session_.get(), input_names_ptr_, input_node_names_);
    ::get_output_names(session_.get(), output_names_ptr_, output_node_names_);
    for (auto name : input_node_names_)
    {
      input_names_.push_back(name);
    }
    for (auto name : output_node_names_)
    {
      output_names_.push_back(name);
    }
  }

  OnnxTask(const OnnxTask &) = delete;

  virtual ~OnnxTask() {}

  size_t getInputWidth() const { return input_shapes_[0][3]; };
  size_t getInputHeight() const { return input_shapes_[0][2]; };
  size_t get_input_batch() const { return input_shapes_[0][0]; }

  std::vector<std::vector<int64_t>> get_input_shapes() { return input_shapes_; }

  std::vector<std::string> get_input_names(){return input_names_;}

  std::vector<std::string> get_output_names(){return output_names_;}

  std::vector<std::vector<int64_t>> get_output_shapes()
  {
    return output_shapes_;
  }

  void set_input_image_rgb(const cv::Mat &image, float *data,
                           const std::vector<float> &mean,
                           const std::vector<float> &scale)
  {
    return set_input_image_internal(image, data, mean, scale, true);
  }
  void set_input_image_bgr(const cv::Mat &image, float *data,
                           const std::vector<float> &mean,
                           const std::vector<float> &scale)
  {
    return set_input_image_internal(image, data, mean, scale, false);
  }
  void set_input_image_internal(const cv::Mat &image, float *data,
                                const std::vector<float> &mean,
                                const std::vector<float> &scale, bool btrans)
  {
    // BGR->RGB (maybe) and HWC->CHW
    for (int c = 0; c < 3; c++)
    {
      for (int h = 0; h < image.rows; h++)
      {
        for (int w = 0; w < image.cols; w++)
        {
          auto c_t = btrans ? abs(c - 2) : c;
          auto image_data =
              (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scale[c_t];
          data[c * image.rows * image.cols + h * image.cols + w] =
              (float)image_data;
        }
      }
    }
  }
  std::vector<Ort::Value> convert_input(
      std::vector<float> &input_values, size_t size,
      const std::vector<int64_t> input_tensor_shape)
  {
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        info, input_values.data(), size, input_tensor_shape.data(),
        input_tensor_shape.size()));
    return input_tensors;
  }

  void run_task(const std::vector<Ort::Value> &input_tensors,
                std::vector<Ort::Value> &output_tensors)
  {
    auto &input_names = input_names_;
    if (ENV_PARAM(DEBUG_ONNX_TASK))
    {
      auto input_shapes = get_input_shapes();
      std::cout << "Input Node Name/Shape (" << input_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < input_names.size(); i++)
      {
        std::cout << "\t" << input_names[i] << " : "
                  << print_shape(input_shapes[i]) << std::endl;
      }
    }
    auto &output_names = output_names_;
    if (ENV_PARAM(DEBUG_ONNX_TASK))
    {
      auto output_shapes = get_output_shapes();
      std::cout << "Output Node Name/Shape (" << output_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < output_names.size(); i++)
      {
        std::cout << "\t" << output_names[i] << " : "
                  << print_shape(output_shapes[i]) << std::endl;
      }
    }
    output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(),
                      input_tensors.data(), input_tensors.size(),
                      output_node_names_.data(), output_node_names_.size());
  }

protected:
  std::string model_name_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  // std::vector<Ort::Value> input_tensors_;
  // std::vector<Ort::Value> output_tensors_;
  std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
  std::vector<const char *> input_node_names_;
  std::vector<Ort::AllocatedStringPtr> output_names_ptr_;
  std::vector<const char *> output_node_names_;
};
