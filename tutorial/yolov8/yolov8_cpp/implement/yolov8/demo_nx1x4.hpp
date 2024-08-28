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
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <stack>
#include <thread>
#include <type_traits>
#include "vitis/ai/bounded_queue.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DEMO, "0")


// onnx_param
extern int onnx_x;
extern int onnx_y;
extern bool onnx_disable_spinning;
extern bool enable_result_print;
extern bool onnx_disable_spinning_between_run;
extern std::string intra_op_thread_affinities;

// camera and display setting
string set_cap_resolution = "";
extern string set_display_resolution = "";
int cap_width = 1920;
int cap_height = 1080;
int display_width = 1920;
int display_height = 1080;

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}



inline std::string ToUTF8String(const std::string& s) { return s; }
/**
 * Convert a wide character string to a UTF-8 string
 */
std::string ToUTF8String(const std::wstring& s);

//

// set the layout
inline std::vector<cv::Rect>& gui_layout() {
  static std::vector<cv::Rect> rects;
  return rects;
}
// set the wallpaper
inline cv::Mat& gui_background() {
  static cv::Mat img;
  return img;
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
  cv::Rect_<int> local_rect;
  cv::Rect_<int> page_layout;
  std::string channel_name;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;
struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }
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
        << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "waiting for [" << name() << "] ended";
      thread_->join();
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
      cap.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
    }
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    auto& cap = *video_stream_.get();
    cv::Mat image;
    cap >> image;
    auto video_ended = image.empty();
    if (video_ended) {
      // loop the video
      open_stream();
      return 0;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    if (queue_->size() > 0 && is_camera_ == true) {
      return 0;
    }
    while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image},
                         std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
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
    clean_up_queue();

    bool any_dirty = false;
    for (auto& f : frames_) {
      if (f.second.dirty) {
        if (video_writer_ == nullptr) {
          cv::imshow(std::string{"CH-"} +
                         std::to_string(f.second.frame_info.channel_id),
                     f.second.frame_info.mat);
        } else {
          *video_writer_ << f.second.frame_info.mat;
        }
        f.second.dirty = false;
        any_dirty = true;
      }
    }
    if (video_writer_ == nullptr) {
      if (any_dirty) {
        auto key = cv::waitKey(1);
        if (key == 27) {
          return 1;
        }
      }
    }
    clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_=0;
  struct FrameCache {
    bool dirty=false;
    FrameInfo frame_info;
  };
  std::map<int, FrameCache> frames_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
};  // namespace ai



struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual cv::Mat run(cv::Mat& input) = 0;
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
  cv::Mat run(cv::Mat& image) override {
    auto result = dpu_model_->run(image);
    return processor_(image, result, false);
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  const ProcessResult& processor_;
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
  queue_t* queue_in_=NULL;
  queue_t* queue_out_=NULL;
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
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;
      auto x = 13;
      auto y = 28;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(178, 79, 0), 2, 4);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_=NULL;
  queue_t* queue_out_=NULL;
  unsigned long frame_id_=0;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_=0.0;
  float max_fps_=0.0;
};

inline void usage_video(const char* progname) {
  std::cout << progname << " [options...] \n" <<"Options:\n      -c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.\n"
            << "      -s [input_stream] set input stream, E.g. set 0 to use default camera.\n" 
            << "      -x [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.\n" 
            << "      -y [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default. Must >=0.\n" 
            << "      -D [Disable thread spinning]: disable spinning entirely for thread owned by onnxruntime intra-op thread pool.\n"
            << "      -Z [Force thread to stop spinning between runs]: disallow thread from spinning during runs to reduce cpu usage.\n"
            << "      -T [Set intra op thread affinities]: Specify intra op thread affinity string.\n         [Example]: -T 1,2;3,4;5,6 or -T 1-2;3-4;5-6\n         Use semicolon to separate configuration between threads.\n         E.g. 1,2;3,4;5,6 specifies affinities for three threads, the first thread will be attached to the first and second logical processor.\n"
            << "      -R [Set camera resolution]: Specify the camera resolution by string.\n         [Example]: -R 1280x720\n         Default:1920x1080.\n"
            << "      -r [Set Display resolution]: Specify the display resolution by string.\n         [Example]: -r 1280x720\n         Default:1920x1080.\n"
            << "      -L Print detection log when turning on.\n"
            << "      -h: help\n"
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
  std::vector<std::string> sp;
  std::vector<std::string> spd;
  while ((opt = getopt(argc, argv, "s:y:x:c:T:R:r:DhLZ")) != -1) {
    // LOG(INFO) << *argv;
    switch (opt) {
      case 'c':
        LOG(INFO) << "Setting parallelism to " << std::stoi(optarg);
        g_num_of_threads.emplace_back(std::stoi(optarg));
        break;
      case 'x':
        LOG(INFO) << "Setting intra_op_num_threads to " << std::stoi(optarg);
        onnx_x = std::stoi(optarg);
        break;
      case 'y':
        LOG(INFO) << "Setting inter_op_num_threads to " << std::stoi(optarg);
        onnx_y = std::stoi(optarg);
        break;
      case 'D':
        onnx_disable_spinning = true;
        LOG(INFO) << "[Disable thread spinning]";
        break;
      case 'L':
        enable_result_print = true;
        LOG(INFO) << "[Result printing is on]";
        break;
      case 'Z':
        onnx_disable_spinning_between_run = true;
        LOG(INFO) << "[Force thread to stop spinning between runs]";
        break;
      case 'T':
        intra_op_thread_affinities = ToUTF8String(optarg);
        LOG(INFO) << "[Set intra op thread affinities]: " << intra_op_thread_affinities;
        break;
      case 'h':
        usage_video(argv[0]);
        exit(1);
      case 's':
        LOG(INFO) << "stream: " << optarg;
        g_avi_file.push_back(optarg);
        break;
      case 'R':
        sp = split(optarg, "x");
        cap_width = stoi(sp[0].c_str());
        cap_height = stoi(sp[1].c_str());
        LOG(INFO) << "[Set camera resolution]: " << cap_width << "x" << cap_height;
        break;
      case 'r':
        spd = split(optarg, "x");
        display_width = stoi(spd[0].c_str());
        display_height = stoi(spd[1].c_str());
        LOG(INFO) << "[Set display resolution]: " << display_width << "x" << display_height;
        break;
      default:
        usage_video(argv[0]);
        exit(1);
    }
  }
  // for (int i = optind; i < argc; ++i) {
    // g_avi_file.push_back(std::string(argv[i]));
  // }
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

// Entrance of single channel video demo
template <typename FactoryMethod, typename ProcessResult>
int main_for_video_demo(int argc, char* argv[],
                        const FactoryMethod& factory_method,
                        const ProcessResult& process_result) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  {
    auto channel_id = 0;
    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    auto decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{channel_id, g_avi_file[0], decode_queue.get()});
    auto dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    auto sorting_queue =
        std::unique_ptr<queue_t>(new queue_t(5 * g_num_of_threads[0]));
    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    for (int i = 0; i < g_num_of_threads[0]; ++i) {
      dpu_thread.emplace_back(new DpuThread(
          create_dpu_filter(factory_method, process_result), decode_queue.get(),
          sorting_queue.get(), std::to_string(i)));
    }
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));
    // start everything
    MyThread::start_all();
    gui_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}



}  // namespace ai
}  // namespace vitis
