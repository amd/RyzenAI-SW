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
#include <glog/logging.h>
#include <opencv2/imgproc/types_c.h>
#include <signal.h>
// #include <unistd.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <type_traits>

#include "util/getopt.h"
// #include <vitis/ai/bounded_queue.hpp>
#include "util/bounded_queue.hpp"
// #include <vitis/ai/env_config.hpp>
#include "util/env_config.hpp"

using namespace std::chrono_literals;

DEF_ENV_PARAM(DEBUG_DEMO, "1")
DEF_ENV_PARAM(DEMO_USE_X, "0")
DEF_ENV_PARAM(DEMO_USE_GLOBAL_GUI, "1")
DEF_ENV_PARAM(DEBUG_CHANNEL_SPLIT, "0")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM(DEBUG_VEK280_HDMI, "0")
DEF_ENV_PARAM(DEBUG_SHOW_FPS, "0")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")

static bool exiting = false;

// set the layout
static int g_show_width = 1920;
static int g_show_height = 1080;

inline std::vector<cv::Rect>& gui_layout() {
  static std::vector<cv::Rect> rects;
  return rects;
}

namespace vitis {
namespace ai {

// Read a video without doing anything
struct VideoByPass {
 public:
  int run(const cv::Mat& input_image) { return 0; }
};

// Do nothing after after excuting
inline cv::Mat process_none(cv::Mat image, int fake_result, bool is_jpeg) {
  return image;
}

// A struct that can storage data and info for each frame
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  float max_fps;
  float fps;
  std::string channel_name;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;

struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) {
    exiting = true;
    stop_all();
  }
  static void stop_all() {
    for (auto& th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto& th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Thread num " << all_threads().size();
    for (auto& th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread* me) { return me->main(); }
  void main() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is started";
    while (!stop_) {
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;

  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT A Thread";
    all_threads().push_back(this);
  }

  virtual ~MyThread() {  //
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is starting";
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is stopping.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "waiting for [" << name() << "] ended";
      thread_->join();
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "thread [" << name() << "] is stopped";
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};

// std::vector<MyThread *> MyThread::all_threads_;
struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string& video_file, queue_t* queue)
      : MyThread{},
        channel_id_{channel_id},
        video_file_{video_file},
        frame_id_{0},
        video_stream_{},
        queue_{queue} {
    open_stream();
    auto& cap = *video_stream_.get();
    if (is_camera_) {
      cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    }
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    FrameInfo frameinfo{channel_id_, ++frame_id_};

    auto& cap = *video_stream_.get();

    cv::Mat image;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << name() << "begin to decode frame id:" << frame_id_;
    cap >> image;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << name() << "image shape:" << image.size();
    // image = cv::Mat(288, 512, CV_8UC3, cv::Scalar(255, 0, 0));
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << name() << "end to decode frame id:" << frame_id_;
    auto video_ended = image.empty();
    if (video_ended) {
      // loop the video
      open_stream();
      return 0;
    }
    frameinfo.mat = image;  // no need copy

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << name() << " decode queue size " << queue_->size();
    // if (queue_->size() > 0 && is_camera_ == true) {
    //  return 0;
    //}
    // while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image},

    while (!queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << name() << " push queue fail, queue_ size:" << queue_->size();
    }
    // usleep(1000);
    std::this_thread::sleep_for(10ms);
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DecodeThread-"} + std::to_string(channel_id_);
  }

  void open_stream() {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
                   : new cv::VideoCapture(video_file_));
    if (!video_stream_->isOpened()) {
      LOG(FATAL)
          << "[UNILOG][FATAL][VAILIB_DEMO_VIDEO_OPEN_ERROR][Can not open "
             "video stream!]  video name: "
          << video_file_;
      stop();
    }
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  std::unique_ptr<cv::VideoCapture> video_stream_;
  queue_t* queue_;
  bool is_camera_;
};

static std::string create_pipeline(int x, int y, int width, int height,
                                   int id) {
  std::ostringstream str;
  if (ENV_PARAM(DEMO_USE_X)) {
    str << "appsrc"            //
        << " ! videoconvert "  //
        << " ! ximagesink ";
    return str.str();
  }

  /*
  str << "appsrc"            //
      << " ! videoconvert "  //
      << " ! video/x-raw, width=" << width << ","
      << " height=" << height  //
      << "! kmssink driver-name=xlnx plane-id=" << 36 + id
      << " render-rectangle=\"<" << x << "," << y << "," << width << ","
      << height << ">\" sync=false";
      */
  if (id == 0)
    str << "appsrc"
        << " ! videoconvert "
        << " ! vvas_xabrscaler "
           "xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" "
           "kernel-name=v_multi_scaler:{v_multi_scaler_1}"
        << " ! video/x-raw, width=" << 960 << ","
        << " height=" << 540  //
        << " ! kmssink driver-name=xlnx plane-id=" << 34 + id
        << " render-rectangle=\"<" << 0 << "," << 0 << "," << 960 << "," << 540
        << ">\" sync=false";
  else if (id == 1)
    str << "appsrc"
        << " ! videoconvert "
        << " ! vvas_xabrscaler "
           "xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" "
           "kernel-name=v_multi_scaler:{v_multi_scaler_2}"
        << " ! video/x-raw, width=" << 960 << ","
        << " height=" << 540  //
        << "! kmssink driver-name=xlnx plane-id=" << 34 + id
        << " render-rectangle=\"<" << 960 << "," << 0 << "," << 960 << ","
        << 540 << ">\" sync=false";
  else if (id == 2)
    str << "appsrc"
        << " ! videoconvert "
        << " ! vvas_xabrscaler "
           "xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" "
           "kernel-name=v_multi_scaler:{v_multi_scaler_3}"
        << " ! video/x-raw, width=" << 960 << ","
        << " height=" << 540  //
        << "! kmssink driver-name=xlnx plane-id=" << 34 + id
        << " render-rectangle=\"<" << 0 << "," << 540 << "," << 960 << ","
        << 540 << ">\" sync=false";
  else if (id == 3)
    str << "appsrc"
        << " ! videoconvert  "
        << " ! vvas_xabrscaler "
           "xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" "
           "kernel-name=v_multi_scaler:{v_multi_scaler_4}"
        << " ! video/x-raw, width=" << 960 << ","
        << " height=" << 540  //
        << "! kmssink driver-name=xlnx plane-id=" << 34 + id
        << " render-rectangle=\"<" << 960 << "," << 540 << "," << 960 << ","
        << 540 << ">\" sync=false";
  else
    str << "appsrc"
        << " ! videoconvert ! videoscale "
        << " ! video/x-raw, width=" << width * 2 << ","
        << " height=" << height * 2  //
        << "! kmssink driver-name=xlnx plane-id=" << 34 + id
        << " render-rectangle=\"<" << x * 2 << "," << y * 2 << "," << width * 2
        << "," << height * 2 << ">\" sync=false";

  return str.str();
}

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int x, int y, int width, int height, int id) {
  std::string pipeline;
  std::string file_prefix = "gst_writer_";
  auto file_suffix = ".cmd";
  if (ENV_PARAM(DEBUG_VEK280_HDMI)) {
    // std::ostringstream str;
    // str << "appsrc"
    //  << " ! videoconvert ! videoscale "
    //  << " ! video/x-raw, width=" << width << ","
    //  << " height=" << height  //
    //  << "! kmssink driver-name=xlnx plane-id=" << 34 + id
    //  << " render-rectangle=\"<" << x << "," << y << "," << width  << ","
    //  << height  << ">\" sync=false";
    // pipeline = str.str();
    std::string file_name = file_prefix + std::to_string(id) + file_suffix;
    std::ifstream in(file_name.c_str());
    if (in.is_open()) {
      std::stringstream str;
      str << in.rdbuf();
      pipeline = str.str();
    } else {
      LOG(ERROR) << "read gst command file:" << file_name << " fail!";
      exit(0);
    }

  } else {
    pipeline = create_pipeline(x, y, width, height, id);
  }
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_GST_ERROR][failed to open "
                  "gstreamer!] cannot open "
               << pipeline;
    return nullptr;
  } else {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "video writer is created: " << width
                                        << "x" << height << " " << pipeline;
  }
  return video_stream;
}

struct ChannelGuiThread : public MyThread {
  static std::shared_ptr<ChannelGuiThread> instance() {
    static std::weak_ptr<ChannelGuiThread> the_instance;
    std::shared_ptr<ChannelGuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<ChannelGuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  ChannelGuiThread(int id = 0)
      : MyThread{},
        queue_{
            new queue_t{
                10}  // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0},
        id_(id) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT CHANNEL GUI";
  }
  virtual ~ChannelGuiThread() {  //
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    while (!queue_->empty()) {
      queue_->pop(frame_info);
    }
  }
  virtual int run() override {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "gui loop begin";
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 20) {
        // inactive for 10 second, stop
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
        return 1;
      } else {
        return 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
    inactive_counter_ = 0;
    // if (!video_writer_) {
    if (video_writer_ == nullptr && ENV_PARAM(DEMO_USE_VIDEO_WRITER)) {
      auto width = frame_info.mat.cols;
      auto height = frame_info.mat.rows;
      auto& layout = gui_layout()[frame_info.channel_id];
      auto x = layout.tl().x;
      auto y = layout.tl().y;
      video_writer_ = maybe_create_gst_video_writer(x, y, width, height,
                                                    frame_info.channel_id);
    }

    if (video_writer_ == nullptr) {
      LOG(ERROR) << "set use video writer but create fail";
      return 0;
    }

    cv::Mat test;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "write begin";
    // if (ENV_PARAM(DEBUG_NO_SHOW)) {
    //  // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "copy begin";
    //  for (int i = 0; i < 5; ++i) {
    //    frame_info.mat.copyTo(test);
    //  }
    //  // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "copy end";
    //  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //} else {
    // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
    //    << "pre write ch :" << frame_info.channel_id
    //    << " show frame:" << frame_info.frame_id;
    *video_writer_ << frame_info.mat;
    // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
    //    << "write done ch :" << frame_info.channel_id
    //    << " show frame:" << frame_info.frame_id;
    //}
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "write end";
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "ch :" << frame_info.channel_id
                                        << " show frame:" << frame_info.frame_id
                                        << ", size:" << frame_info.mat.size();
    //*video_writer_ << frame_info.mat;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "ch :" << frame_info.channel_id << "pre clean gui queue";
    clean_up_queue();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "ch :" << frame_info.channel_id << "end clean gui queue";
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "gui loop end";
    return 0;
  }
  virtual std::string name() override {
    return std::string{"ChannelGUIThread-"} + std::to_string(id_);
  }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
  int id_;
};  // namespace ai

struct GuiThread : public MyThread {
  static std::shared_ptr<GuiThread> instance() {
    static std::weak_ptr<GuiThread> the_instance;
    std::shared_ptr<GuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<GuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  GuiThread()
      : MyThread{},
        queue_{
            new queue_t{
                10}  // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0} {
    gui_show_image_ = cv::Mat(g_show_height, g_show_width, CV_8UC3);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
  }

  virtual ~GuiThread() {  //
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    while (!queue_->empty()) {
      queue_->pop(frame_info);
      frames_[frame_info.channel_id].frame_info = frame_info;
      frames_[frame_info.channel_id].dirty = true;
    }
  }
  virtual int run() override {
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 10) {
        // inactive for 5 second, stop
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
        return 1;
      } else {
        return 0;
      }
    }
    inactive_counter_ = 0;

    frames_[frame_info.channel_id].frame_info = frame_info;
    frames_[frame_info.channel_id].dirty = true;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "show channel:" << frame_info.channel_id
        << ", frame:" << frame_info.mat.size();
    clean_up_queue();

    if (video_writer_ == nullptr && ENV_PARAM(DEMO_USE_VIDEO_WRITER)) {
      auto width = frame_info.mat.cols;
      auto height = frame_info.mat.rows;
      auto& layout = gui_layout()[frame_info.channel_id];
      auto x = layout.tl().x;
      auto y = layout.tl().y;
      video_writer_ = maybe_create_gst_video_writer(x, y, width, height,
                                                    frame_info.channel_id);
    }

    bool any_dirty = false;
    if (ENV_PARAM(DEMO_USE_VIDEO_WRITER)) {
      if (video_writer_ != nullptr) {
        for (auto& f : frames_) {
          if (f.second.dirty) {
            LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
                << "pre write ch :" << f.second.frame_info.channel_id
                << " show frame:" << f.second.frame_info.frame_id;
            *video_writer_ << f.second.frame_info.mat;
            LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
                << "write done ch :" << f.second.frame_info.channel_id
                << " show frame:" << f.second.frame_info.frame_id;
          }
          f.second.dirty = false;
          any_dirty = true;
          LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
              << "ch :" << f.second.frame_info.channel_id
              << " show frame:" << f.second.frame_info.frame_id;
        }
      } else {
        LOG(WARNING) << "no video writer!";
      }
    } else if (ENV_PARAM(DEBUG_CHANNEL_SPLIT)) {
      for (auto& f : frames_) {
        if (f.second.dirty) {
          cv::imshow(std::string{"CH-"} +
                         std::to_string(f.second.frame_info.channel_id),
                     f.second.frame_info.mat);
          LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
              << " show frame:" << f.second.frame_info.frame_id;
        }
        f.second.dirty = false;
        any_dirty = true;
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
            << "ch :" << f.second.frame_info.channel_id
            << " show frame:" << f.second.frame_info.frame_id;
      }
      if (any_dirty) {
        auto key = cv::waitKey(1);
        if (key == 27) {
          return 1;
        }
      }
    } else {  // default use global show
      for (auto& f : frames_) {
        if (f.second.dirty) {
          auto& layout = gui_layout()[f.second.frame_info.channel_id];
          auto x = layout.tl().x;
          auto y = layout.tl().y;
          auto w = layout.width;
          auto h = layout.height;
          cv::Mat mat_resize;
          cv::resize(f.second.frame_info.mat, mat_resize, cv::Size(w, h));
          mat_resize.copyTo(gui_show_image_(layout));
          LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
              << " show frame:" << f.second.frame_info.frame_id;
        }
        f.second.dirty = false;
        any_dirty = true;
      }

      if (any_dirty) {
        cv::imshow(std::string{"ipu_modelsx4_demo"}, gui_show_image_);
        auto key = cv::waitKey(1);
        if (key == 27) {
          return 1;
        }
      }
    }
    clean_up_queue();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "clean gui queue";
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_;
  struct FrameCache {
    bool dirty;
    FrameInfo frame_info;
  };
  std::map<int, FrameCache> frames_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
  cv::Mat gui_show_image_;
};

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual cv::Mat run(cv::Mat& input) { return cv::Mat(); }
  // virtual std::vector<cv::Mat> run(std::vector<cv::Mat>& input) = 0;
};

// Execute each lib run function and processor your implement
template <typename dpu_model_type_t, typename ProcessResult>
struct DpuFilter : public Filter {
  DpuFilter(std::unique_ptr<dpu_model_type_t>&& dpu_model,
            const ProcessResult& processor)
      : Filter{}, dpu_model_{std::move(dpu_model)}, processor_{processor} {
    LOG(INFO) << "DPU model size=" << dpu_model_->getInputWidth() << "x"
              << dpu_model_->getInputHeight();
  }
  virtual ~DpuFilter() {}
  // std::vector<cv::Mat> run(std::vector<cv::Mat>& images) override {
  //  auto results = dpu_model_->run(images);
  //  return processor_(images, results, false);
  //}

  cv::Mat run(cv::Mat& image) override {
    auto result = dpu_model_->run(image);
    return processor_(image, result, false);
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  // const ProcessResult& processor_;
  ProcessResult& processor_;
};

template <typename FactoryMethod, typename ProcessResult>
std::unique_ptr<Filter> create_dpu_filter(const FactoryMethod& factory_method,
                                          const ProcessResult& process_result) {
  using dpu_model_type_t = typename decltype(factory_method())::element_type;
  return std::unique_ptr<Filter>(new DpuFilter<dpu_model_type_t, ProcessResult>(
      factory_method(), process_result));
}

// Execute dpu filter
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<Filter>&& filter, queue_t* queue_in,
            queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        filter_{std::move(filter)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }
  virtual ~DpuThread() {}

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }
    if (filter_) {
      frame.mat = filter_->run(frame.mat);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t* queue_in_;
  queue_t* queue_out_;
  std::string suffix_;
};

// Implement sorting thread
struct SortingThread : public MyThread {
  SortingThread(queue_t* queue_in, queue_t* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT SORTING";
  }
  virtual ~SortingThread() {}
  virtual int run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond =
        std::function<bool(const FrameInfo&)>{[frame_id](const FrameInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    if (!queue_in_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto now = std::chrono::steady_clock::now();
    float fps = -1.0f;
    long duration = 0;
    if (!points_.empty()) {
      auto end = points_.back();
      duration =
          (long)std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;
      // auto x = 10;
      // auto y = 20;
      auto x = 20;
      auto y = 40;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO) || ENV_PARAM(DEBUG_SHOW_FPS))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_in_->size() << " queue out size" << queue_out_->size()
        << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }

    if (frame.mat.cols > 200)
      cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                  cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(20, 20, 180), 2, 1);
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }

    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
};
inline void usage_video(const char* progname) {
  std::cout << "usage: " << progname << "      -t <num_of_threads>\n"
            << "      <video file name>\n"
            << std::endl;
  return;
}
/*
  global command line options
 */
static std::vector<int> g_num_of_threads;
static std::vector<std::string> g_avi_file;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  while ((opt = getopt(argc, argv, "c:t:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads.emplace_back(std::stoi(optarg));
        break;
      case 'c':  // how many channels
        break;   // do nothing. parse it in outside logic.
      default:
        usage_video(argv[0]);
        exit(1);
    }
  }
  for (int i = optind; i < argc; ++i) {
    g_avi_file.push_back(std::string(argv[i]));
  }
  if (g_avi_file.empty()) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_num_of_threads.empty()) {
    // by default, all channels has at least one thread
    g_num_of_threads.emplace_back(1);
  }
  return;
}

// A class can create a video channel
struct Channel {
  Channel(size_t ch, const std::string& avi_file,
          const std::function<std::unique_ptr<Filter>()>& filter,
          int n_of_threads) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "create channel " << ch << " for " << avi_file;
    auto channel_id = ch;
    decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)channel_id, avi_file, decode_queue.get()});
    dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(5 * n_of_threads));
    std::shared_ptr<GuiThread> g_gui_thread;
    queue_t* gui_queue = nullptr;
    if (ENV_PARAM(DEMO_USE_GLOBAL_GUI)) {
      g_gui_thread = GuiThread::instance();
      gui_queue = g_gui_thread->getQueue();
    } else {
      // auto gui_thread = GuiThread::instance();
      gui_thread = std::make_unique<ChannelGuiThread>((int)channel_id);
      gui_queue = gui_thread->getQueue();
    }

    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix =
          avi_file + "-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
  std::unique_ptr<ChannelGuiThread> gui_thread;
};
}  // namespace ai
}  // namespace vitis
