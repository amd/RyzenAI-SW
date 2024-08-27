#pragma once
#include "util/config.hpp"
class AsyncTask {
 public:
  AsyncTask() {}
  virtual ~AsyncTask() {}
  virtual void init(const Config& config) = 0;
  virtual void run() = 0;
};