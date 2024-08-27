#pragma once
#include <thread>

#include "global.hpp"
#include "task.hpp"
class ThreadExcutor {
 public:
  ~ThreadExcutor() { wait(); }
  void run(std::vector<std::shared_ptr<AsyncTask>>& tasks) {
    tasks_ = tasks;
    for (auto& task : tasks_) {
      auto worker = std::make_shared<std::thread>([task] {
        while (!g_is_stopped()) {
          task->run();
        }
      });
      threads_.push_back(worker);
    }
  }
  void wait() {
    for (auto& w : threads_) {
      if (w->joinable()) {
        w->join();
      }
    }
  }
  std::vector<std::shared_ptr<AsyncTask>> tasks_;
  std::vector<std::shared_ptr<std::thread>> threads_;
};
class TestExcutor {
 public:
  ~TestExcutor() {}
  void run(std::vector<std::shared_ptr<AsyncTask>>& tasks) {
    while (!g_is_stopped()) {
      for (auto& task : tasks) {
        task->run();
      }
    }
  }
  void wait() {}
};