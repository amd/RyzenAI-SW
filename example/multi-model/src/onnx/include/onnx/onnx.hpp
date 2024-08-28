#pragma once
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <map>
#include <sstream>
#include <string>
#if _WIN32
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
inline std::wstring_convert<convert_t, wchar_t> strconverter;
#endif

#include <functional>
#include <iostream>

#include "processing/sync_image_to_image_model.hpp"
#include "util/check.hpp"
#include "util/config.hpp"
#include "util/fs.hpp"

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
    std::shared_ptr<Ort::Session> session_;
  };
  Ort::Session* get(const std::string& model_name, const Config& config) {
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
                            const Config& config) {
    SessionInfo session_info;
    session_info.env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());
    session_info.session_options_ = Ort::SessionOptions();
    auto& session_options_ = session_info.session_options_;
    auto options = std::unordered_map<std::string, std::string>({});

    if (!config.contains("using_onnx_ep")) {
      // CONFIG_GET(config, bool, using_onnx_ep, "using_onnx_ep")
      // if (!using_onnx_ep) {
      PRINT("Using VitisAI")
      if (config.contains("vaip_config")) {
        CONFIG_GET(config, std::string, vaip_config_path, "vaip_config")
        CHECK_WITH_INFO(is_file(vaip_config_path), vaip_config_path)
        CHECK_WITH_INFO(check_extension(vaip_config_path, ".json"),
                        vaip_config_path)
        options["config_file"] = vaip_config_path;
      } else {
        options["config_file"] = "../bin/vaip_config.json";
        CHECK(is_file("../bin/vaip_config.json"))
      }
      session_options_.AppendExecutionProvider_VitisAI(options);
    }
    {
      CONFIG_GET(config, int, onnx_x, "onnx_x")
      CHECK(onnx_x >= 0)
      PRINT("Setting intra_op_num_threads to " << onnx_x);
      session_options_.SetIntraOpNumThreads(onnx_x);
    }
    {
      CONFIG_GET(config, int, onnx_y, "onnx_y")
      CHECK(onnx_y >= 0)
      PRINT("Setting inter_op_num_threads to " << onnx_y);
      session_options_.SetInterOpNumThreads(onnx_y);
    }
    if (config.contains("onnx_disable_spinning")) {
      PRINT("Disabling intra-op thread spinning entirely");
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    }
    if (config.contains("onnx_disable_spinning_between_run")) {
      PRINT("Disabling intra-op thread spinning between runs");
      session_options_.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop,
                                      "1");
    }
    if (config.contains("intra_op_thread_affinities")) {
      CONFIG_GET(config, std::string, intra_op_thread_affinities,
                 "intra_op_thread_affinities")
      PRINT("Setting intra op thread affinity as "
            << intra_op_thread_affinities);
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigIntraOpThreadAffinities,
          intra_op_thread_affinities.c_str());
    }
    auto model_name_basic = strconverter.from_bytes(model_name);
    session_info.session_.reset(new Ort::Session(
        session_info.env_, model_name_basic.c_str(), session_options_));
    return session_info;
  }

  void set_singleton(bool flag) { is_singleton_ = flag; }

 private:
  SessionManager() {}
  bool is_singleton_{false};
  int model_counter{0};
  std::map<std::string, SessionInfo> sessions_;
};
std::vector<std::vector<int64_t>> get_input_shapes(Ort::Session* session) {
  std::vector<std::vector<int64_t>> input_shapes;
  const size_t num_input_nodes = session->GetInputCount();
  input_shapes.resize(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_shapes[i] = tensor_info.GetShape();
  }
  return input_shapes;
}
std::vector<std::vector<int64_t>> get_output_shapes(Ort::Session* session) {
  std::vector<std::vector<int64_t>> output_shapes;
  const size_t num_output_nodes = session->GetOutputCount();
  output_shapes.resize(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_shapes[i] = tensor_info.GetShape();
  }
  return output_shapes;
}
void get_input_names(Ort::Session* session,
                     std::vector<Ort::AllocatedStringPtr>& input_names_ptr,
                     std::vector<const char*>& input_node_names) {
  const size_t num_input_nodes = session->GetInputCount();
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session->GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));
  }
}
void get_output_names(Ort::Session* session,
                      std::vector<Ort::AllocatedStringPtr>& output_names_ptr,
                      std::vector<const char*>& output_node_names) {
  const size_t num_output_nodes = session->GetOutputCount();
  output_names_ptr.reserve(num_output_nodes);
  output_node_names.reserve(num_output_nodes);
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session->GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(output_name.get());
    output_names_ptr.push_back(std::move(output_name));
  }
}
template <typename T>
std::string to_string(const std::vector<T>& vs) {
  std::stringstream ss;
  ss << "(";
  for (auto v : vs) {
    ss << v << ",";
  }
  ss << ")";
  return ss.str();
}
class SessionHelper {
 public:
  SessionHelper() {}
  void init(const Config& config) {
    CONFIG_GET(config, std::string, onnx_model_path, "onnx_model_path")
    CHECK_WITH_INFO(is_file(onnx_model_path), onnx_model_path)
    onnx_model_path = absolute(onnx_model_path);
    CHECK_WITH_INFO(check_extension(onnx_model_path, ".onnx"), onnx_model_path);
    CONFIG_GET(config, Config, session_config, "onnx_config")
    session_ =
        SessionManager::get_instance().get(onnx_model_path, session_config);
    input_shapes_ = get_input_shapes(session_);
    output_shapes_ = get_output_shapes(session_);
    input_tensor_values_.resize(input_shapes_.size());
    get_input_names(session_, input_names_ptr_, input_node_names_);
    get_output_names(session_, output_names_ptr_, output_node_names_);
    PRINT("ONNX file: " << onnx_model_path)
    std::cout << this->summary_to_string();
  }
  SessionHelper(const SessionHelper&) = delete;
  std::vector<int64_t>& get_input_shape(int index) {
    return input_shapes_[index];
  }
  std::vector<float>& get_input(int index) {
    return input_tensor_values_[index];
  }
  // const std::vector<std::vector<int64_t>>& get_output_shapes_from_model() {
  //   return output_shapes_;
  // }
  const std::vector<int64_t>& get_output_shape_from_model(int index) {
    return output_shapes_[index];
  }
  const std::vector<int64_t> get_output_shape_from_tensor(int index) {
    return output_tensors_[index].GetTensorTypeAndShapeInfo().GetShape();
  }
  void run() {
    output_tensors_ =
        session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(),
                      input_tensors_.data(), input_tensors_.size(),
                      output_node_names_.data(), output_node_names_.size());
  }
  void convert_inputs() {
    size_t input_arg_num = input_shapes_.size();
    if (!input_tensors_.empty()) {
      for (size_t index = 0; index < input_arg_num; index++) {
        Ort::MemoryInfo info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = input_shapes_[index];
        std::vector<float>& data = input_tensor_values_[index];
        input_tensors_[index] = Ort::Value::CreateTensor<float>(
            info, data.data(), data.size(), input_shape.data(),
            input_shape.size());
      }
    } else {
      for (size_t index = 0; index < input_arg_num; index++) {
        Ort::MemoryInfo info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = input_shapes_[index];
        std::vector<float>& data = input_tensor_values_[index];
        input_tensors_.push_back(Ort::Value::CreateTensor<float>(
            info, data.data(), data.size(), input_shape.data(),
            input_shape.size()));
      }
    }
  }
  std::vector<float*> get_outputs() {
    CHECK(output_tensors_.size() == output_shapes_.size())
    std::vector<float*> outputs;
    outputs.reserve(3);
    for (size_t i = 0; i < output_shapes_.size(); i++) {
      CHECK(output_tensors_[i].IsTensor())
      outputs.push_back(output_tensors_[i].GetTensorMutableData<float>());
    }
    return outputs;
  }
  float* get_output(int index) {
    // PRINT("size " << output_tensors_.size() << " " << output_shapes_.size())
    return output_tensors_[index].GetTensorMutableData<float>();
  }
  std::string summary_to_string() {
    std::stringstream ss;
    CHECK(input_node_names_.size() == input_shapes_.size());
    ss << "Inputs: \n";
    for (int i = 0; i < input_node_names_.size(); i++) {
      ss << "  " << input_node_names_[i] << " -> "
         << ::to_string(input_shapes_[i]) << "\n";
    }
    CHECK(output_node_names_.size() == output_shapes_.size())
    ss << "Outputs: \n";
    for (int i = 0; i < output_node_names_.size(); i++) {
      ss << "  " << output_node_names_[i] << " -> "
         << ::to_string(output_shapes_[i]) << "\n";
    }
    return ss.str();
  }

 private:
  // std::string model_name_;
  Ort::Session* session_;
  std::vector<std::vector<float>> input_tensor_values_;
  std::vector<Ort::Value> input_tensors_;
  std::vector<Ort::Value> output_tensors_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
  std::vector<const char*> input_node_names_;
  std::vector<Ort::AllocatedStringPtr> output_names_ptr_;
  std::vector<const char*> output_node_names_;
};
class Model : public SyncImageToImageModel {
 public:
  Model() {}
  virtual ~Model() {}
  void init(const Config& config) override {
    session_ = std::unique_ptr<SessionHelper>(new SessionHelper{});
    session_->init(config);
  }
  Image run(const Image& image) override {
    auto images = std::vector<Image>{image};
    preprocess(images);
    session_->convert_inputs();
    session_->run();
    auto result = postprocess(images);
    return result[0];
  }

 protected:
  virtual void preprocess(const std::vector<Image>& input) = 0;
  virtual std::vector<Image> postprocess(const std::vector<Image>& input) = 0;
  std::unique_ptr<SessionHelper> session_{nullptr};
};
