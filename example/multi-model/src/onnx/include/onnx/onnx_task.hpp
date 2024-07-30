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
#include <core/session/experimental_onnxruntime_cxx_api.h>

#include <sstream>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "util/env_config.hpp"
#if _WIN32
extern "C" {
#include "util/getopt.h"
}
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
inline std::wstring_convert<convert_t, wchar_t> strconverter;
#endif

#include <iostream>
#include <map>

DEF_ENV_PARAM(DEBUG_ONNX_TASK, "0")

struct OnnxConfig {
  int onnx_x{-1};
  int onnx_y{-1};
  bool onnx_disable_spinning{false};
  bool onnx_disable_spinning_between_run{false};
  std::string intra_op_thread_affinities{""};
  bool using_onnx_ep{false};
  static OnnxConfig default_value() { return OnnxConfig{}; }
  std::string to_string() {
    std::stringstream ss;
    ss << "onnx_x:" << onnx_x << " "
       << "onnx_y:" << onnx_y << " "
       << "onnx_disable_spinning:" << onnx_disable_spinning << " "
       << "onnx_disable_spinning_between_run:"
       << onnx_disable_spinning_between_run << " "
       << "intra_op_thread_affinities:" << intra_op_thread_affinities << " "
       << "using_onnx_ep:" << using_onnx_ep << " ";
    return ss.str();
  }
};

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
static std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

class SessionManager {
 public:
  static SessionManager& get_instance() {
    static SessionManager instance_;
    return instance_;
  }
  struct SessionInfo {
    std::string model_name_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::shared_ptr<Ort::Experimental::Session> session_;
  };
  Ort::Experimental::Session* get(const std::string& model_name,
                                  const OnnxConfig& config) {
    if (is_singleton_) {
      auto iter = sessions_.find(model_name);
      if (iter == sessions_.end()) {
        sessions_[model_name] = build_session(model_name, config);
      }
      return sessions_[model_name].session_.get();
    } else {
      std::string model_key = model_name + "@" + std::to_string(model_counter);
      sessions_[model_key] = build_session(model_name, config);
      model_counter++;
      return sessions_[model_key].session_.get();
    }
  }
  SessionInfo build_session(const std::string& model_name,
                            const OnnxConfig& config) {
    SessionInfo session_info;
    session_info.env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());
    session_info.session_options_ = Ort::SessionOptions();
    auto& session_options_ = session_info.session_options_;
    auto options = std::unordered_map<std::string, std::string>({});
    options["config_file"] = "../vaip_config.json";
    if (!config.using_onnx_ep) {
      LOG(INFO) << "using VitisAI";
      session_options_.AppendExecutionProvider("VitisAI", options);
    }
    if (config.onnx_x >= 0) {
      fprintf(stdout, "Setting intra_op_num_threads to %d\n", config.onnx_x);
      session_options_.SetIntraOpNumThreads(config.onnx_x);
    }

    if (config.onnx_y >= 0) {
      fprintf(stdout, "Setting inter_op_num_threads to %d\n", config.onnx_y);
      session_options_.SetInterOpNumThreads(config.onnx_y);
    }

    if (config.onnx_disable_spinning) {
      fprintf(stdout, "Disabling intra-op thread spinning entirely\n");
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    }

    if (config.onnx_disable_spinning_between_run) {
      fprintf(stdout, "Disabling intra-op thread spinning between runs\n");
      session_options_.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop,
                                      "1");
    }

    if (!config.intra_op_thread_affinities.empty()) {
      fprintf(stdout, "Setting intra op thread affinity as %s\n",
              config.intra_op_thread_affinities.c_str());
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigIntraOpThreadAffinities,
          config.intra_op_thread_affinities.c_str());
    }

    auto model_name_basic = strconverter.from_bytes(model_name);
    session_info.session_.reset(new Ort::Experimental::Session(
        session_info.env_, model_name_basic, session_options_));
    return session_info;
  }

  void set_singleton(bool flag) { is_singleton_ = flag; }

 private:
  SessionManager() {}
  bool is_singleton_{false};
  int model_counter{0};
  std::map<std::string, SessionInfo> sessions_;
};

class OnnxTask {
 public:
  explicit OnnxTask(const std::string& model_name,
                    const OnnxConfig config = OnnxConfig::default_value())
      : model_name_(model_name) {
    session_ = SessionManager::get_instance().get(model_name, config);
    input_shapes_ = session_->GetInputShapes();
    output_shapes_ = session_->GetOutputShapes();
    if (input_shapes_[0][0] == -1) {
      input_shapes_[0][0] = 1;
      output_shapes_[0][0] = 1;
    }
  }

  OnnxTask(const OnnxTask&) = delete;

  virtual ~OnnxTask() {}

  size_t getInputWidth() const { return input_shapes_[0][3]; };
  size_t getInputHeight() const { return input_shapes_[0][2]; };
  size_t get_input_batch() const { return input_shapes_[0][0]; }

  std::vector<std::vector<int64_t>> get_input_shapes() { return input_shapes_; }

  std::vector<std::string> get_input_names() {
    return session_->GetInputNames();
  }

  std::vector<std::string> get_output_names() {
    return session_->GetOutputNames();
  }

  std::vector<std::vector<int64_t>> get_output_shapes() {
    return output_shapes_;
  }

  void set_input_image_rgb(const cv::Mat& image, float* data,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale) {
    return set_input_image_internal(image, data, mean, scale, true);
  }
  void set_input_image_bgr(const cv::Mat& image, float* data,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale) {
    return set_input_image_internal(image, data, mean, scale, false);
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

  std::vector<Ort::Value> convert_input(
      std::vector<float>& input_values, size_t size,
      const std::vector<int64_t> input_tensor_shape) {
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_values.data(), size, input_tensor_shape));
    return input_tensors;
  }

  void run_task(const std::vector<Ort::Value>& input_tensors,
                std::vector<Ort::Value>& output_tensors) {
    std::vector<std::string> input_names = session_->GetInputNames();
    if (ENV_PARAM(DEBUG_ONNX_TASK)) {
      auto input_shapes = get_input_shapes();
      std::cout << "Input Node Name/Shape (" << input_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < input_names.size(); i++) {
        std::cout << "\t" << input_names[i] << " : "
                  << print_shape(input_shapes[i]) << std::endl;
      }
    }
    std::vector<std::string> output_names = session_->GetOutputNames();
    if (ENV_PARAM(DEBUG_ONNX_TASK)) {
      auto output_shapes = get_output_shapes();
      std::cout << "Output Node Name/Shape (" << output_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < output_names.size(); i++) {
        std::cout << "\t" << output_names[i] << " : "
                  << print_shape(output_shapes[i]) << std::endl;
      }
    }

    output_tensors = session_->Run(session_->GetInputNames(), input_tensors,
                                   session_->GetOutputNames());
  }

 protected:
  std::string model_name_;
  Ort::Experimental::Session* session_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
};
