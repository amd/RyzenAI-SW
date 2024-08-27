#pragma once
#include <atomic>
#include <csignal>
#include <iomanip>
#include <opencv2/core.hpp>
#include <chrono>

static std::atomic<bool> GLOBAL_IS_STOP{false};
void g_stop() { GLOBAL_IS_STOP.store(true); };
bool g_is_stopped() { return GLOBAL_IS_STOP.load(); }
void signal_handler(int signum) { g_stop(); }
void register_interrupt_handler() { signal(SIGINT, signal_handler); }

static int GLOBAL_BOUNDED_QUEUE_CAPACITY{10}; 
static std::vector<cv::Rect> GLOBAL_LAYOUTS;
static std::string GLOBAL_APP_NAME{"demo"};
static int GLOBAL_VIDEO_FILE_MAX_FRAME_COUNT{500};
static std::chrono::milliseconds GLOBAL_DECODE_TASK_SLEEP_DURATION{10};