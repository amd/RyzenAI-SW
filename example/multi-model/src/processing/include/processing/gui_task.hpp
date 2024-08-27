#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <memory>

#include "frame_info.hpp"
#include "global.hpp"
#include "queue.hpp"
#include "task.hpp"
struct FrameCache {
  bool dirty;
  FrameInfo frame_info;
};
class GuiTask : public AsyncTask {
 public:
  GuiTask() {}
  virtual ~GuiTask() {}
  void init(const Config& config) override {
    input_queue_ =
        std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
    CONFIG_GET(config, int32_t, screen_width, "width")
    CHECK(screen_width > 1)
    CONFIG_GET(config, int32_t, screen_height, "height")
    CHECK(screen_height > 1)
    CONFIG_GET(config, int32_t, split_channel_matrix_size,
               "split_channel_matrix_size")
    CHECK(split_channel_matrix_size >= 1)
    gui_show_image_ =
        cv::Mat(screen_height, screen_width, CV_8UC3, cv::Scalar(255, 255, 0));
    layouts_ =
        cal_gui_layout(screen_height, screen_width, split_channel_matrix_size);
    PRINT("GUI Layout: ")
    for (auto& layout : layouts_) {
      PRINT("\t"<<layout)
    }
  }
  void run() override {
    FrameInfo frame_info;
    if (!input_queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 10) {
        PRINT("gui is starvatting!!")
      }
      return;
    }
    inactive_counter_ = 0;

    frames_[frame_info.channel_id].frame_info = frame_info;
    frames_[frame_info.channel_id].dirty = true;
    drain_queue();
    bool any_dirty{false};
    for (auto& f : frames_) {
      if (f.second.dirty) {
        auto& layout = layouts_[f.second.frame_info.channel_id];
        auto x = layout.tl().x;
        auto y = layout.tl().y;
        auto w = layout.width;
        auto h = layout.height;
        cv::Mat mat_resize;
        cv::resize(f.second.frame_info.mat, mat_resize, cv::Size(w, h));
        mat_resize.copyTo(gui_show_image_(layout));
      }
      f.second.dirty = false;
      any_dirty = true;
    }
    if (any_dirty) {
      cv::imshow(GLOBAL_APP_NAME, gui_show_image_);
      auto key = cv::waitKey(1);
      if (key == 27) {
        return;
      }
    }
    drain_queue();
  }
  std::shared_ptr<BoundedFrameQueue> input_queue_{nullptr};

 private:
  void drain_queue() {
    FrameInfo frame_info;

    while (!input_queue_->empty()) {
      input_queue_->pop(frame_info);
      if (frame_info.mat.empty()) {
        PRINT("Got empty mat")
        return;
      }
      frames_[frame_info.channel_id].frame_info = frame_info;
      frames_[frame_info.channel_id].dirty = true;
    }
  }
  std::vector<cv::Rect> cal_gui_layout(size_t height, size_t width,
                                       size_t split_matrix_size) {
    std::vector<cv::Rect> rects;
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
  std::map<int, FrameCache> frames_;

  std::unique_ptr<cv::VideoWriter> video_writer_{nullptr};
  int inactive_counter_{0};
  cv::Mat gui_show_image_;
  std::vector<cv::Rect> layouts_;
};