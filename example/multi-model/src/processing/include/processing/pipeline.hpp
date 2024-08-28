#pragma once
#include <vector>

#include "decode_task.hpp"
#include "gui_task.hpp"
#include "model_task.hpp"
#include "sort_task.hpp"
#include "util/config.hpp"
std::vector<std::shared_ptr<AsyncTask>> create_model_pipeline(
    const Config& config, std::shared_ptr<GuiTask>& gui_task) {
  std::vector<std::shared_ptr<AsyncTask>> tasks;
  CONFIG_GET(config, Config, decode_config, "decode")
  CONFIG_GET(decode_config, std::string, decode_file, "video_file_path")
  auto decode_task = make_decode_task(decode_file);
  // auto decode_task = std::make_shared<DecodeTask>();
  // CONFIG_GET(config, Config, decode_config, "decode")
  decode_task->init(decode_config);
  {
    auto async_task = std::dynamic_pointer_cast<AsyncTask>(decode_task);
    CHECK(async_task != nullptr)
    tasks.push_back(async_task);
  }
  PRINT("Building decode task finished!!")
  auto sort_task = std::make_shared<SortTask>();
  CONFIG_GET(config, Config, sort_config, "sort")
  sort_task->init(sort_config);
  sort_task->output_queue_ = gui_task->input_queue_;
  PRINT("Building sort task finished!!")
  CONFIG_GET(config, int, model_thread_num, "thread_num")
  PRINT("Need model task num: " << model_thread_num)
  CONFIG_GET(config, Config, model_config, "model")
  for (int i = 0; i < model_thread_num; i++) {
    auto model_task = std::make_shared<ModelTask>();
    model_task->init(model_config);
    model_task->input_queue_ = decode_task->output_queue_;
    model_task->output_queue_ = sort_task->input_queue_;
    {
      auto async_task = std::dynamic_pointer_cast<AsyncTask>(model_task);
      CHECK(async_task != nullptr)
      tasks.push_back(async_task);
    }
    PRINT("Building model task " << i << " finished!!")
  }
  {
    auto async_task = std::dynamic_pointer_cast<AsyncTask>(sort_task);
    CHECK(async_task != nullptr)
    tasks.push_back(async_task);
  }
  return tasks;
}