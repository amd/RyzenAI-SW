#pragma once
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <thread>

#include "frame_info.hpp"
#include "global.hpp"
#include "queue.hpp"
#include "task.hpp"
#include "util/fs.hpp"

bool is_camera(const std::string& file) {
  return file.size() == 1 && file[0] >= '0' && file[0] <= '9';
}
std::unique_ptr<cv::VideoCapture> open_stream(const std::string& video_file) {
  auto video_stream =
      std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(video_file));
  if (!video_stream->isOpened()) {
    PRINT("can't open: " << video_file);
    g_stop();
  }
  return video_stream;
}

struct VideoCache {
 public:
  static VideoCache& instance() {
    static VideoCache instance{};
    return instance;
  }
  void add(const std::string& name) {
    auto iter = video_caches_.find(name);
    if (iter != video_caches_.end()) {
      return;
    }
    CHECK_WITH_INFO(!is_camera(name), name);
    auto video_stream = open_stream(name);
    auto images = std::make_unique<std::vector<cv::Mat>>();
    images->reserve(GLOBAL_VIDEO_FILE_MAX_FRAME_COUNT);
    for (int i = 0; i < GLOBAL_VIDEO_FILE_MAX_FRAME_COUNT; i++) {
      cv::Mat image;
      if (!video_stream->read(image)) {
        break;
      }
      images->push_back(image);
    }
    video_caches_[name] = std::move(images);
  }

  bool exist(const std::string& name) {
    auto iter = video_caches_.find(name);
    return iter != video_caches_.end();
  }
  void add(const std::string& name,
           std::unique_ptr<std::vector<cv::Mat>>& images) {
    auto iter = video_caches_.find(name);
    if (iter != video_caches_.end()) {
      return;
    }
    CHECK_WITH_INFO(!images->empty(),
                    std::string{"add empty images for key: "} + name)
    video_caches_[name] = std::move(images);
  }
  std::vector<cv::Mat>* get_ref(const std::string& name) {
    auto iter = video_caches_.find(name);
    CHECK(iter != video_caches_.end());
    return iter->second.get();
  }

 private:
  VideoCache() {}
  std::map<std::string, std::unique_ptr<std::vector<cv::Mat>>> video_caches_;
};

// class DecodeTask : public AsyncTask {
//  public:
//   DecodeTask() {}
//   virtual ~DecodeTask() {}
//   void init(const Config& config) override {
//     // Set the queue size to be larger, so that downstream tasks do not feel
//     // that the video file has been reopened
//     output_queue_ = std::make_shared<BoundedFrameQueue>(
//         GLOBAL_BOUNDED_QUEUE_CAPACITY + 100);
//     CONFIG_GET(config, std::string, video_file, "video_file_path")
//     video_file_ = video_file;
//     open_stream();
//   }
//   void run() override {
//     FrameInfo frameinfo{0, ++frame_id_};
//     auto& cap = *video_stream_.get();
//     cv::Mat image;
//     cap.read(image);
//     auto video_ended = image.empty();
//     if (video_ended) {
//       open_stream();
//       return;
//     }
//     frameinfo.mat = image;
//     while (!output_queue_->push(frameinfo, std::chrono::milliseconds(500))) {
//       if (g_is_stopped()) {
//         return;
//       }
//     }
//     using namespace std::chrono_literals;
//     std::this_thread::sleep_for(10ms);
//   }
//   std::shared_ptr<BoundedFrameQueue> output_queue_{nullptr};

//  private:
//   void open_stream() {
//     is_camera_ = is_camera(video_file_);
//     video_stream_ = std::unique_ptr<cv::VideoCapture>(
//         is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
//                    : new cv::VideoCapture(video_file_));

//     if (!video_stream_->isOpened()) {
//       PRINT("can't open: " << video_file_);
//       g_stop();
//     }
//   }
//   std::string video_file_;
//   unsigned long frame_id_{0};
//   std::unique_ptr<cv::VideoCapture> video_stream_{};
//   std::vector<std::unique_ptr<cv::VideoCapture>> video_streams_{};
//   std::vector<cv::Mat> mats_;
//   bool is_camera_{false};
// };
class DecodeTask : public AsyncTask {
 public:
  DecodeTask() {}
  virtual ~DecodeTask() {}
  std::string video_file_;
  std::shared_ptr<BoundedFrameQueue> output_queue_{nullptr};
};
class DecodeCameraTask : public DecodeTask {
 public:
  DecodeCameraTask() {}
  virtual ~DecodeCameraTask() {}
  void init(const Config& config) override {
    output_queue_ =
        std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
    CONFIG_GET(config, std::string, video_file, "video_file_path")
    video_file_ = video_file;
    CHECK_WITH_INFO(is_camera(video_file), video_file_)
    open_stream();
  }
  void run() override {
    FrameInfo frameinfo{0, ++frame_id_};
    auto& cap = *video_stream_.get();
    cv::Mat image;
    cap.read(image);
    auto video_ended = image.empty();
    if (video_ended) {
      open_stream();
      return;
    }
    frameinfo.mat = image;
    while (!output_queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (g_is_stopped()) {
        return;
      }
    }
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(10ms);
  }

 private:
  void open_stream() {
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        new cv::VideoCapture(std::stoi(video_file_)));
    if (!video_stream_->isOpened()) {
      PRINT("can't open: " << video_file_);
      g_stop();
    }
  }
  unsigned long frame_id_{0};
  std::unique_ptr<cv::VideoCapture> video_stream_{};
};
// template <typename T>
// class Generator {
//   using BoundedQueue = vitis::ai::BoundedQueue<T>;

//  public:
//   Generator(int num, const std::function<T()>& generate_func)
//       : num_{num}, generate_func_{generate_func} {
//     CHECK(num_ >= 1)
//     output_queue_ = std::make_unique<BoundedQueue>(num);
//     worker_ = std::make_shared<std::thread>([this]() {
//       while (!g_is_stopped()) {
//         auto e = generate_func_();
//         while (!output_queue_->push(e, std::chrono::milliseconds(500))) {
//           if (g_is_stopped()) {
//             return;
//           }
//         }
//       }
//     });
//   }
//   ~Generator() {
//     if (worker_ != nullptr && worker_->joinable()) {
//       worker_->join();
//     }
//     T e;
//     while (!output_queue_->empty()) {
//       if (output_queue_->pop(e, std::chrono::milliseconds(100))) {
//         if (std::is_pointer<T>::value) {
//           delete e;
//         }
//       }
//     }
//   }
//   T get_next() {
//     T e;
//     while (!output_queue_->pop(e, std::chrono::milliseconds(500))) {
//       PRINT("get element from async generator failled!!!!");
//     }
//     return e;
//   }

//  private:
//   std::unique_ptr<BoundedQueue> output_queue_{nullptr};
//   std::shared_ptr<std::thread> worker_{nullptr};
//   int num_{0};
//   std::function<T()> generate_func_{nullptr};
// };

// class DecodeVideoTask : public DecodeTask {
//  public:
//   DecodeVideoTask() {}
//   virtual ~DecodeVideoTask() {}
//   void init(const Config& config) override {
//     output_queue_ =
//         std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
//     CONFIG_GET(config, std::string, video_file, "video_file_path")
//     video_file_ = video_file;
//     CHECK(!is_camera(video_file))
//     video_stream_generator_ =
//         std::make_unique<Generator<cv::VideoCapture*>>(1, [video_file]() {
//           auto video_stream = new cv::VideoCapture(video_file);
//           if (!video_stream->isOpened()) {
//             PRINT("can't open: " << video_file);
//             g_stop();
//           }
//           return video_stream;
//         });
//     video_stream_.reset(video_stream_generator_->get_next());
//     // open_stream();
//   }
//   void run() override {
//     // channel_id set in sort work
//     FrameInfo frameinfo{0, ++frame_id_};
//     cv::Mat image;
//     video_stream_->operator>>(image);  // ???? why blocking??? opened file
//     must
//                                        // read in the same thread?
//     // video_stream_->read(image);
//     PRINT_THIS_LINE()
//     auto video_ended = image.empty();
//     if (video_ended) {
//       video_stream_.reset(video_stream_generator_->get_next());
//       return;
//     }
//     frameinfo.mat = image;
//     PRINT("decode ouput size" << output_queue_->size())
//     while (!output_queue_->push(frameinfo, std::chrono::milliseconds(500))) {
//       if (g_is_stopped()) {
//         return;
//       }
//     }
//     using namespace std::chrono_literals;
//     std::this_thread::sleep_for(10ms);
//   }

//  private:
//   // void open_stream() {
//   //   video_stream_ =
//   //       std::shared_ptr<cv::VideoCapture>(new
//   // cv::VideoCapture(video_file_));
//   //   if (!video_stream_->isOpened()) {
//   //     PRINT("can't open: " << video_file_);
//   //     g_stop();
//   //   }
//   // }
//   unsigned long frame_id_{0};
//   std::unique_ptr<cv::VideoCapture> video_stream_{};
//   std::unique_ptr<Generator<cv::VideoCapture*>> video_stream_generator_{
//       nullptr};
// };
class DecodeVideoTask : public DecodeTask {
 public:
  DecodeVideoTask() {}
  virtual ~DecodeVideoTask() {}
  void init(const Config& config) override {
    output_queue_ =
        std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
    CONFIG_GET(config, std::string, video_file, "video_file_path")
    CHECK_WITH_INFO(is_file(video_file), video_file)
    video_file_ = absolute(video_file);
    CHECK_WITH_INFO(!is_camera(video_file), video_file)
    auto& video_cache = VideoCache::instance();
    video_cache.add(video_file_);
    CHECK_WITH_INFO(video_cache.exist(video_file_), video_file_)
    images_ = video_cache.get_ref(video_file_);
  }
  void run() override {
    // channel_id set in sort work
    FrameInfo frameinfo{0, ++frame_id_};
    cv::Mat image;
    images_->operator[](frame_id_ % images_->size()).copyTo(image);
    frameinfo.mat = image;
    while (!output_queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (g_is_stopped()) {
        return;
      }
    }
    std::this_thread::sleep_for(GLOBAL_DECODE_TASK_SLEEP_DURATION);
  }

 private:
  unsigned long frame_id_{0};
  std::vector<cv::Mat>* images_;
};
namespace image_list_helper {
std::pair<int, int> cal_max_height_and_width(std::vector<cv::Mat>& images) {
  int fold_height{0};
  int fold_width{0};
  for (auto& im : images) {
    auto width = im.cols;
    auto height = im.rows;
    fold_width = std::max(width, fold_width);
    fold_height = std::max(height, fold_height);
  }
  return std::make_pair(fold_height, fold_width);
}
cv::Mat embed(const cv::Mat& image, int max_height, int max_width) {
  auto width = image.cols;
  auto height = image.rows;
  int pad_h = (max_height - height) / 2;
  int pad_w = (max_width - width) / 2;
  cv::Mat res;
  cv::copyMakeBorder(image, res, pad_h, pad_h, pad_w, pad_w,
                     cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return res;
}
std::unique_ptr<std::vector<cv::Mat>> read_image_list(
    const std::string& image_list_path) {
  auto list_file_path =
      (std::filesystem::path(image_list_path) / "image_list.txt").string();
  CHECK_WITH_INFO(is_file(list_file_path), list_file_path)
  std::ifstream ifs{list_file_path};
  std::string image_file_name;
  std::vector<cv::Mat> images;
  while (std::getline(ifs, image_file_name)) {
    auto image_file_path =
        (std::filesystem::path(image_list_path) / image_file_name).string();
    auto image = cv::imread(image_file_path);
    CHECK_WITH_INFO(!image.empty(), image_file_path)
    images.push_back(image);
  }
  auto [max_height, max_width] = cal_max_height_and_width(images);
  auto res_images = std::make_unique<std::vector<cv::Mat>>();
  for (auto& image : images) {
    res_images->push_back(embed(image, max_height, max_width));
  }
  return res_images;
}
}  // namespace image_list_helper

class DecodeImageListTask : public DecodeTask {
 public:
  DecodeImageListTask() {}
  virtual ~DecodeImageListTask() {}
  void init(const Config& config) override {
    output_queue_ =
        std::make_shared<BoundedFrameQueue>(GLOBAL_BOUNDED_QUEUE_CAPACITY);
    CONFIG_GET(config, std::string, video_file, "video_file_path")
    CONFIG_GET(config, int, repeat_frame_per_image, "repeat_frame_per_image")
    repeat_frame_per_image_ = repeat_frame_per_image;
    CHECK_WITH_INFO(is_directory(video_file), video_file)
    video_file_ = absolute(video_file);
    CHECK_WITH_INFO(!is_camera(video_file), video_file)
    auto& video_cache = VideoCache::instance();
    if (!video_cache.exist(video_file_)) {
      auto images = image_list_helper::read_image_list(video_file);
      video_cache.add(video_file_, images);
    }
    images_ = video_cache.get_ref(video_file_);
  }
  void run() override {
    // channel_id set in sort work
    FrameInfo frameinfo{0, ++frame_id_};
    cv::Mat image;
    images_->operator[]((frame_id_ / repeat_frame_per_image_) % images_->size())
        .copyTo(image);
    frameinfo.mat = image;
    while (!output_queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (g_is_stopped()) {
        return;
      }
    }
    std::this_thread::sleep_for(GLOBAL_DECODE_TASK_SLEEP_DURATION);
  }

 private:
  unsigned long frame_id_{0};
  int repeat_frame_per_image_{10};
  std::vector<cv::Mat>* images_{nullptr};
};
std::shared_ptr<DecodeTask> make_decode_task(const std::string& file) {
  PRINT("Decoding file: " << absolute(file))
  if (is_camera(file)) {
    PRINT("Building camera decode task")
    return std::dynamic_pointer_cast<DecodeTask>(
        std::make_shared<DecodeCameraTask>());
  } else if (is_directory(file)) {
    PRINT("Building image list decode task")
    return std::dynamic_pointer_cast<DecodeTask>(
        std::make_shared<DecodeImageListTask>());
  } else {
    PRINT("Building video decode task")
    return std::dynamic_pointer_cast<DecodeTask>(
        std::make_shared<DecodeVideoTask>());
  }
}