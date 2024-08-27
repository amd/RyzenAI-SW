#pragma once
#include <chrono>
#include <map>
#include <memory>

#include "frame_info.hpp"
#include "global.hpp"
#include "queue.hpp"
#include "sync_image_to_image_model.hpp"
#include "task.hpp"
class ModelTask : public AsyncTask {
 public:
  ModelTask() {}
  virtual ~ModelTask() {}
  void init(const Config& config) override {
    CONFIG_GET(config, std::string, model_type, "type")
    model_ = ModelRegister::instance().build(model_type);
    CONFIG_GET(config, Config, model_config, "config")
    model_->init(model_config);
  }
  void run() override {
    FrameInfo frame;
    // PRINT("model input size"<<input_queue_->size())
    if (!input_queue_->pop(frame, std::chrono::milliseconds(500))) {
      return;
    }
    if (model_) {
      frame.mat = model_->run(frame.mat);
    }
    // PRINT("model push"<<output_queue_->size())
    while (!output_queue_->push(frame, std::chrono::milliseconds(500))) {
      if (g_is_stopped()) {
        return;
      }
    }
    return;
  }

 public:
  std::shared_ptr<BoundedFrameQueue> output_queue_;
  std::shared_ptr<BoundedFrameQueue> input_queue_;

 private:
  std::unique_ptr<SyncImageToImageModel> model_;
};