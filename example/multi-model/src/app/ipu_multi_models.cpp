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
#include <assert.h>
#include <core/session/onnxruntime_cxx_api.h>

#include <algorithm>  // std::generate
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>
#if _WIN32
extern "C" {
#include "util/getopt.h"
}
#else
#include <getopt.h>
#endif
#include <glog/logging.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "processing/processing.hpp"
#include "util/nlohmann_json.hpp"
// models
#include "mobile_net_v2/run_model.hpp"
#include "resnet50/run_model.hpp"
#include "retinaface/run_model.hpp"
#include "segmentation/run_model.hpp"
#include "yolovx/run_model.hpp"
using json = nlohmann::json;
using namespace vitis::ai;
struct ChannelConfig {
  size_t model_filter_id;
  size_t thread_num;
  std::string onnx_model_path;
  float confidence_threshold;
  std::string video_file_path;
  OnnxConfig onnx_config;
  void parse(const std::string& name, const json& data) {
    try {
      if (!data.contains("onnx_model_path")) {
        LOG(INFO) << name << " config not set onnx_model_path, abort";
        abort();
      } else {
        onnx_model_path = data["onnx_model_path"].get<std::string>();
      }
      if (!data.contains("video_file_path")) {
        LOG(INFO) << name << " config not set video_file_path, abort";
        abort();
      } else {
        video_file_path = data["video_file_path"].get<std::string>();
      }
      if (!data.contains("model_filter_id")) {
        LOG(INFO) << name << " config not set model_filter_id, abort";
        abort();
      } else {
        model_filter_id = data["model_filter_id"].get<size_t>();
      }
      if (!data.contains("thread_num")) {
        LOG(INFO) << name << " config not set thread_num, using default: 1";
        thread_num = 1;
      } else {
        thread_num = data["thread_num"].get<size_t>();
      }
      if (!data.contains("confidence_threshold")) {
        LOG(INFO) << name
                  << " config not set confidence_threshold, using default: 0.3";
        confidence_threshold = 0.3;
      } else {
        confidence_threshold = data["confidence_threshold"].get<float>();
      }
      if (!data.contains("onnx_config")) {
        auto default_value = OnnxConfig::default_value();
        LOG(INFO) << name << " config not set onnx_config, using default: "
                  << default_value.to_string();
        onnx_config = OnnxConfig::default_value();
      } else {
        onnx_config = parse_onnx_config(data["onnx_config"]);
      }
    } catch (std::exception& e) {
      LOG(INFO) << e.what();
      abort();
    }
  }
  OnnxConfig parse_onnx_config(const json& data) {
    auto parsed_value = OnnxConfig::default_value();
    if (!data.contains("onnx_x")) {
      LOG(INFO) << " config not set onnx_x, using default: -1";
    } else {
      parsed_value.onnx_x = data["onnx_x"].get<int>();
    }
    if (!data.contains("onnx_y")) {
      LOG(INFO) << " config not set onnx_y, using default: -1";
    } else {
      parsed_value.onnx_y = data["onnx_y"].get<int>();
    }
    if (!data.contains("onnx_disable_spinning")) {
      LOG(INFO)
          << " config not set onnx_disable_spinning, using default: false";
    } else {
      parsed_value.onnx_disable_spinning =
          data["onnx_disable_spinning"].get<bool>();
    }
    if (!data.contains("onnx_disable_spinning_between_run")) {
      LOG(INFO) << " config not set onnx_disable_spinning_between_run, using "
                   "default: false";
    } else {
      parsed_value.onnx_disable_spinning_between_run =
          data["onnx_disable_spinning_between_run"].get<bool>();
    }
    if (!data.contains("intra_op_thread_affinities")) {
      LOG(INFO) << " config not set intra_op_thread_affinities, using default: "
                   "";
    } else {
      parsed_value.intra_op_thread_affinities =
          data["intra_op_thread_affinities"].get<std::string>();
    }
    if (!data.contains("using_onnx_ep")) {
      LOG(INFO) << " config not set using_onnx_ep, using default: false";
    } else {
      parsed_value.using_onnx_ep = data["using_onnx_ep"].get<bool>();
    }
    return parsed_value;
  }
  std::string to_string() {
    std::stringstream ss;
    ss << "model_filter_id:" << model_filter_id << " "
       << "thread_num:" << thread_num << " "
       << "confidence_threshold:" << confidence_threshold << " "
       << "onnx_model_path:" << onnx_model_path << " "
       << "video_file_path:" << video_file_path << " "
       << onnx_config.to_string();
    return ss.str();
  }
};

struct Config {
  size_t split_matrix_size;
  size_t screen_height;
  size_t screen_width;
  std::vector<std::pair<std::string, ChannelConfig>> channel_configs;
  void parse(const std::string& config_path) {
    try {
      std::ifstream f(config_path);
      json config = json::parse(f);
      if (!config.contains("split_matrix_size")) {
        LOG(INFO) << " config not set split_matrix_size, using default: 1";
        split_matrix_size = 1;
      } else {
        split_matrix_size = config["split_matrix_size"].get<size_t>();
      }
      assert(split_matrix_size >= 1 && split_matrix_size <= 4);
      if (!config.contains("screen_height")) {
        LOG(INFO) << " config not set screen_height, using default: 576";
        screen_height = 576;
      } else {
        screen_height = config["screen_height"].get<size_t>();
      }
      if (!config.contains("screen_width")) {
        LOG(INFO) << " config not set screen_width, using default: 1024";
        screen_width = 1024;
      } else {
        screen_width = config["screen_width"].get<size_t>();
      }
      for (auto& e : config["models"].items()) {
        ChannelConfig channel_config;
        channel_config.parse(e.key(), e.value());
        channel_configs.push_back(std::make_pair(e.key(), channel_config));
      }
    } catch (std::exception& e) {
      LOG(INFO) << e.what();
      abort();
    }
  }
  std::vector<std::string> collect_video_file_path() {
    std::vector<std::string> result;
    for (auto e : channel_configs) {
      result.push_back(e.second.video_file_path);
    }
    return result;
  }
  std::vector<size_t> collect_thread_num() {
    std::vector<size_t> result;
    for (auto e : channel_configs) {
      result.push_back(e.second.thread_num);
    }
    return result;
  }
};
// TODO: use static register pattern
std::function<std::unique_ptr<Filter>()> create_filter(
    size_t index, ChannelConfig channel_config) {
  if (0 == index) {
    return [=] {
      return vitis::ai::create_dpu_filter(
          [=] {
            return YolovxnanoOnnx::create(channel_config.onnx_model_path,
                                          channel_config.confidence_threshold,
                                          channel_config.onnx_config);
          },
          process_result_yolovx);
    };
  }
  if (2 == index) {
    return [=] {
      return vitis::ai::create_dpu_filter(
          [channel_config] {
            return Resnet50PtOnnx::create(channel_config.onnx_model_path,
                                          channel_config.onnx_config);
          },
          process_result_resnet50);
    };
  }
  // if (3 == index) {
  //   return [=] {
  //     return vitis::ai::create_dpu_filter(
  //         [=] {
  //           return Yolov8Onnx::create(channel_config.onnx_model_path,
  //                                     channel_config.confidence_threshold,
  //                                     channel_config.onnx_config);
  //         },
  //         process_result_yolov8);
  //   };
  // }
  // if (4 == index) {
  //   return [=] {
  //     return vitis::ai::create_dpu_filter(
  //         [=] {
  //           return
  //           DepthEfficientNetOnnx::create(channel_config.onnx_model_path,
  //                                                channel_config.onnx_config);
  //         },
  //         process_result_depth_efficient_net);
  //   };
  // }
  if (5 == index) {
    return [=] {
      return vitis::ai::create_dpu_filter(
          [=] {
            return MobileNetV2Onnx::create(channel_config.onnx_model_path,
                                           channel_config.onnx_config);
          },
          process_result_mobilenet_v2);
    };
  }
  if (6 == index) {
    return [=] {
      return vitis::ai::create_dpu_filter(
          [=] {
            return RetinafaceOnnx::create(channel_config.onnx_model_path,
                                          channel_config.onnx_config);
          },
          process_result_retinaface);
    };
  }
  if (7 == index) {
    return [=] {
      return vitis::ai::create_dpu_filter(
          [=] {
            return SegmentationOnnx::create(channel_config.onnx_model_path,
                                            channel_config.onnx_config);
          },
          process_result_segmentation);
    };
  }
  // if (8 == index) {
  //   return [=] {
  //     return vitis::ai::create_dpu_filter(
  //         [=] {
  //           return Yolov5Onnx::create(channel_config.onnx_model_path,
  //                                     channel_config.confidence_threshold,
  //                                     channel_config.onnx_config);
  //         },
  //         process_result_yolov5);
  //   };
  // }
  return nullptr;
}

std::vector<cv::Rect> cal_gui_layout(size_t height, size_t width,
                                     size_t split_matrix_size) {
  std::vector<cv::Rect> rects;
  assert(split_matrix_size != 0);
  if (split_matrix_size == 1) {
    rects.push_back(cv::Rect{0, 0, int(width), int(height)});
    return rects;
  }
  size_t part_height = height / split_matrix_size;
  size_t part_width = width / split_matrix_size;
  for (size_t i = 0; i < split_matrix_size; i++) {
    for (size_t j = 0; j < split_matrix_size; j++) {
      rects.push_back(cv::Rect{int(j * part_width), int(i * part_height),
                               int(part_width), int(part_height)});
    }
  }
  return rects;
}
int run(Config& config) {
  signal(SIGINT, MyThread::signal_handler);
  SessionManager::get_instance().set_singleton(true);
  std::vector<std::function<std::unique_ptr<Filter>()>> filters;
  for (size_t i = 0; i < 4 && i < config.channel_configs.size(); i++) {
    auto model_filter_id = config.channel_configs[i].second.model_filter_id;
    // assert(model_filter_id >= 0 && model_filter_id < 4);
    filters.push_back(
        create_filter(model_filter_id, config.channel_configs[i].second));
  }
  auto avi_files = config.collect_video_file_path();
  auto num_of_threads = config.collect_thread_num();

  g_show_width = int(config.screen_width);
  g_show_height = int(config.screen_height);
  size_t matrix_split_num = int(config.split_matrix_size);
  gui_layout() = cal_gui_layout(g_show_height, g_show_width, matrix_split_num);
  LOG(INFO) << "g_show_width: " << g_show_width
            << "g_show_height: " << g_show_height
            << "matrix_split_num: " << matrix_split_num;

  std::shared_ptr<GuiThread> gui_thread;

  if (ENV_PARAM(DEMO_USE_GLOBAL_GUI)) {
    gui_thread = GuiThread::instance();
    LOG(INFO) << "use global gui thread";
  }
  std::vector<Channel> channels;
  channels.reserve(filters.size());
  for (auto ch = 0u; ch < filters.size(); ++ch) {
    channels.emplace_back(ch, avi_files[ch], filters[ch],
                          int(num_of_threads[ch]));
  }
  // start everything
  MyThread::start_all();
  if (gui_thread) {
    gui_thread->wait();
  }
  LOG(INFO) << "press Ctrl-C to exit....";
  while (!exiting) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG(INFO) << "waiting all thread to shutdown....";

  MyThread::stop_all();
  MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}
int main(int argc, char* argv[]) {
  if (argc < 1) {
    LOG(INFO) << "need config!!!";
    return 0;
  }
  std::string config_path = argv[1];
  Config config;
  config.parse(config_path);
  if (config.channel_configs.empty()) {
    LOG(INFO) << "channel config size zero!!!";
  }
  for (size_t i = 0; i < 4 && i < config.channel_configs.size(); i++) {
    LOG(INFO) << "config " << config.channel_configs[i].first << " -> "
              << config.channel_configs[i].second.to_string();
  }
  try {
    run(config);
  } catch (std::exception& e) {
    LOG(INFO) << e.what();
  }
  return 0;
}
