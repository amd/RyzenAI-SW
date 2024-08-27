#pragma once
#include <chrono>
#include <map>
#include <memory>

#include "fps_record.hpp"
#include "frame_info.hpp"
#include "global.hpp"
#include "queue.hpp"
#include "task.hpp"
struct SortTask : public AsyncTask {
 public:
  SortTask() {}
  virtual ~SortTask() {}
  void init(const Config& config) override {
    input_queue_ =
        std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
    CONFIG_GET(config, int, channel_matrix_id, "channel_matrix_id")
    channel_id_ = channel_matrix_id;
  }
  void run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;

    auto cond =
        std::function<bool(const FrameInfo&)>{[frame_id](const FrameInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    // PRINT("sort pop"<<input_queue_->size())
    if (!input_queue_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return;
    }
    frame.channel_id = channel_id_;
    fps_recorder.record();
    int fps = fps_recorder.fps();
    frame.fps = fps;
    static auto x = 20;
    static auto y = 40;
    if (frame.mat.cols > 200) {
      cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                  cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(20, 20, 180), 2, 1);
    }
    // PRINT("sort push"<<output_queue_->size())
    while (!output_queue_->push(frame, std::chrono::milliseconds(500))) {
      if (g_is_stopped()) {
        return;
      }
    }
  }

 public:
  std::shared_ptr<BoundedFrameQueue> output_queue_{};
  std::shared_ptr<BoundedFrameQueue> input_queue_{};

 private:
  int channel_id_{0};
  unsigned long frame_id_{0};
  FpsRecorder fps_recorder{10};
};