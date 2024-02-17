/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

/* Based on :
 * https://raw.githubusercontent.com/progschj/ThreadPool/master/ThreadPool.h */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "utils.h"

class ThreadPool {
public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(int tid, F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  // N task queues, each one for each worker
  std::vector<std::queue<std::function<void()>>> tasks;

  // synchronization
  std::vector<std::mutex> queue_mutex;
  std::vector<std::condition_variable> condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    : tasks(threads), queue_mutex(threads), condition(threads), stop(false) {
  workers.reserve(threads);
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this, i] {
      for (;;) {
        auto &task_queue = this->tasks.at(i);
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex.at(i));
          this->condition.at(i).wait(lock, [this, i] {
            return this->stop || !this->tasks.at(i).empty();
          });
          if (this->stop && this->tasks.at(i).empty())
            return;
          task = std::move(this->tasks.at(i).front());
          this->tasks.at(i).pop();
        }
        task();
      }
    });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(int tid, F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex.at(tid));

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.at(tid).emplace([task]() { (*task)(); });
  }
  condition.at(tid).notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    // Acquire all locks before "STOP"ing.
    std::vector<std::unique_lock<std::mutex>> ulocks;
    for (int i = 0; i < workers.size(); ++i)
      ulocks.emplace_back(queue_mutex.at(i));
    stop = true;
  }
  for (int i = 0; i < workers.size(); ++i)
    condition.at(i).notify_all();
  for (std::thread &worker : workers)
    worker.join();
}

class ThreadPoolSingleton {
public:
  static ThreadPoolSingleton &getInstance() {
    static ThreadPoolSingleton tps;
    return tps;
  }

  ThreadPool pool;

private:
  ThreadPoolSingleton() : pool(Utils::get_qlinear_num_threads()) {}
};

#endif