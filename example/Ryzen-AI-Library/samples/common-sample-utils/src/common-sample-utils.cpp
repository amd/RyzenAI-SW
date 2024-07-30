/*
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include "common-sample-utils.h"

#include <chrono>
#include <filesystem>
#include <iomanip>

using amd::cvml::sample::utils::CamRes;

namespace amd {
namespace cvml {
namespace sample {
namespace utils {

bool SetupCamera(int camera_index, const std::vector<CamRes>& res_list, cv::VideoCapture* camera) {
  // list certain API preferences before CAP_ANY to try them first
  // regardless of opencv's ordering
  static const int camera_api_preference[] = {
#ifdef _WIN32
      cv::CAP_DSHOW, cv::CAP_MSMF,
#endif
      cv::CAP_ANY};

  if (camera == nullptr) {
    return false;
  }

  for (auto api : camera_api_preference) {
    try {
      camera->open(camera_index, api);
      if (camera->isOpened()) {
        break;
      }
    } catch (std::exception& e) {
      std::cout << "SetupCamera exception(" << api << "): " << e.what() << std::endl;
    }
  }

  if (camera->isOpened() != true) {
    std::cout << "Failed to open camera device with id:" << camera_index << std::endl;
    return false;
  }

  bool result = false;

  for (auto res : res_list) {
    camera->set(cv::CAP_PROP_FRAME_WIDTH, res.width);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, res.height);
    auto w = camera->get(cv::CAP_PROP_FRAME_WIDTH);
    auto h = camera->get(cv::CAP_PROP_FRAME_HEIGHT);
    if (w != res.width || h != res.height) {
      std::cout << "Camera doesn't support " << res.width << "x" << res.height << std::endl;
    } else {
      std::cout << "Camera enabled at " << w << "x" << h << std::endl;
      result = true;
      break;
    }
  }
  if (!result) {
    std::cout << "No supported resolution for camera." << std::endl;
    camera->release();
  }
  return result;
}

/**
 * Helper function to determine output display scale factor.
 *
 * @param user: User structure for execution flags, etc.
 * @param frame_out: Reference to unscaled output buffer
 * @return Desired width/height scale factor
 */
static double CalculateDispScaling(const float disp_window_scale, const cv::Mat& frame_out) {
  if (disp_window_scale != 0.0f) {
    return static_cast<double>(disp_window_scale);
  }
  const double TARGET_WIDTH = 960;
  const double TARGET_HEIGHT = 960;
  double scale_width = 1.0;
  double scale_height = 1.0;

  if (frame_out.cols > TARGET_WIDTH) {
    scale_width = TARGET_WIDTH / frame_out.cols;
  }
  if (frame_out.rows > TARGET_HEIGHT) {
    scale_height = TARGET_HEIGHT / frame_out.rows;
  }

  // use the smaller of two scale factors
  return scale_width < scale_height ? scale_width : scale_height;
}

bool RunFeatureClass::GetSingleVideoFrame(uint32_t frame_id) {
  (void)frame_id;

  if (video_input_.isOpened()) {
    // video capture device input
    cv::Mat tmp;
    if (!video_input_.read(tmp)) {
      return false;
    }
    cv::cvtColor(tmp, frame_rgb_, cv::COLOR_BGR2RGB);
  }

  // in single frame case, frame_rgb_ has already been pre-loaded
  return true;
}

bool RunFeatureClass::RunFeatureStreaming() {
  bool user_exit = false;

  uint32_t frame_id;  // frame counter

  // set FPS to be the same as the input device/file, or 30FPS
  if (video_input_.isOpened()) {
    stream_fps_ = video_input_.get(cv::CAP_PROP_FPS);
  }

  // ms time for stream_fps_
  std::chrono::milliseconds test_fps_period_ =
      std::chrono::milliseconds(static_cast<uint32_t>(1000 / stream_fps_));

  // record start time
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  //
  // Iterate over frames
  // No special handling for frame_id overflow, but this can handle 2^32 frames
  // so is good enough for a sample application.
  //
  for (frame_id = 1; GetSingleVideoFrame(frame_id - 1); ++frame_id) {
    // Run feature and measure effective fps (execution time of a single feature call)
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cv::Mat frame_out = Feature(frame_rgb_);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    bool window_shown = false;  // whether or not an output window was shown

    if (!frame_out.empty()) {
      // open requested output file
      if (open_output_file_ && !video_output_.isOpened()) {
        // only attempt to open once
        open_output_file_ = false;
        // fourcc encoding format for output video(s)
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        // set output resolution the same as first output frame
        cv::Size output_size(frame_out.cols * (side_by_side_ ? 2 : 1), frame_out.rows);
        bool result = video_output_.open(output_file_, fourcc, stream_fps_, output_size);
        if (!result) {
          std::cout << "Failed to open output file: " << output_file_ << std::endl;
        }
      }

      //
      // Optionally put input/output side by side
      //
      if (side_by_side_) {
        cv::Mat tmp;

        cv::resize(frame_out, tmp, cv::Size(frame_rgb_.cols, frame_rgb_.rows));
        cv::hconcat(frame_rgb_, tmp, frame_out);
      }

      cv::cvtColor(frame_out, frame_out, cv::COLOR_RGB2BGR);

      //
      // Write output video
      //
      if (video_output_.isOpened()) {
        video_output_.write(frame_out);
      }

      // optionally show input window
      if (input_window_name_ != nullptr) {
        cv::Mat tmp;
        cv::cvtColor(frame_rgb_, tmp, cv::COLOR_RGB2BGR);
        cv::imshow(input_window_name_, tmp);  // Show the input frame
        window_shown = true;
      }

      //
      // Display the output frame at resized width/height
      //
      if (!output_window_name_.empty()) {
        cv::Mat frame_disp{};  // OpenCV display buffer

        double disp_scaling = CalculateDispScaling(disp_window_scale_, frame_out);
        cv::resize(frame_out, frame_disp, cv::Size(), disp_scaling, disp_scaling);

        // Add fps to window name
        std::stringstream window_title_ss;
        window_title_ss
            << output_window_name_ << " | Inference time: " << std::fixed << std::setprecision(1)
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms";

        cv::imshow(output_window_name_.c_str(), frame_disp);
        cv::setWindowTitle(output_window_name_.c_str(),
                           window_title_ss.str());  // Update fps in window
        window_shown = true;
      }
    }  // if (!frame_out.empty())

    if (window_shown) {
      char c = static_cast<char>(cv::pollKey());
      if (tolower(c) == tolower(frame_shot_key_)) {  // save frame shot
        cv::imwrite(GetTimestamp() + "_frame_" + std::to_string(frame_id) + ".png", frame_out);
      }

      //
      // Quit if window was closed
      // OpenCV throws an exception if the window is invalid, so catch it here.
      //
      try {
        if (input_window_name_ != nullptr &&
            cv::getWindowProperty(input_window_name_, cv::WND_PROP_AUTOSIZE) == -1) {
          user_exit = true;
          break;
        }
        if (!output_window_name_.empty() &&
            cv::getWindowProperty(output_window_name_.c_str(), cv::WND_PROP_AUTOSIZE) == -1) {
          user_exit = true;
          break;
        }
      } catch (std::exception& e) {
        (void)e;  // ignore the error
        user_exit = true;
        break;
      }
    }

    // Rough simulation of test FPS. Figure out how much time should have
    // passed based on how many iterations have executed, and inject some
    // additional delay if ahead of schedule.
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);

    // extra sleep to simulate expected FPS
    if (elapsed_time < test_fps_period_ * frame_id) {
      std::this_thread::sleep_for(test_fps_period_ * frame_id - elapsed_time);
    }
  }

  // final clean up if not repeating
  if (user_exit == true || repeat_image_video_ == false) {
    // clean up OpenCV windows
    cv::destroyAllWindows();

    // perform cleanup and prepare to exit
    if (video_output_.isOpened()) {
      video_output_.release();
      std::cout << "Output file saved: " << output_file_ << std::endl;
    }
  }

  return user_exit;
}

void RunFeatureClass::RunFeatureVideoFile(const std::string& input_file) {
  std::cout << "Opening video file: " << input_file << std::endl;
  bool user_exit = true;
  do {
    video_input_ = cv::VideoCapture(input_file);
    if (!video_input_.isOpened()) {
      std::cout << "Failed to open video file: " << input_file << std::endl;
    } else {
      user_exit = RunFeatureStreaming();

      video_input_.release();
    }
  } while (user_exit == false && repeat_image_video_ == true);
}

void RunFeatureClass::RunFeature(const std::string& input, const std::string& output_file,
                                 const std::string& window_title,
                                 std::vector<CamRes>* supported_res) {
  // attempt to open output file later if name specified
  open_output_file_ = output_file.size() > 0;
  output_file_ = output_file;
  output_window_name_ = window_title;

  // default frame rate
  stream_fps_ = 30;

  //
  // Determine input type based on the incoming string.
  // Default to camera index 0 if no input specified.
  // e.g.,
  //   still jpeg - "image.jpg"
  //   video clip - "clip.mp4"
  //   camera 2   - "2"
  //
  const std::string input_str = input.empty() ? "0" : input;
  std::string ext = static_cast<std::filesystem::path>(input_str).extension().string();
  bool is_image{false};
  bool is_video{false};
  bool is_camera{false};

  // assume camera index if a number is provided
  if (ext.length() == 0 && std::isdigit(input_str[0])) {
    is_camera = true;
  } else {
    // check if we can treat the input as an image
    frame_rgb_ = cv::imread(input_str);
    if (!frame_rgb_.empty()) {
      cv::cvtColor(frame_rgb_, frame_rgb_, cv::COLOR_BGR2RGB);
      is_image = true;
    } else {
      // assume the input is a video file
      is_video = true;
    }
  }

  if (is_camera) {
    //
    // Camera
    //
    // preferred camera resolution list
    const std::vector<CamRes> camera_res_list = {{1920, 1080}, {1280, 720}};
    // requested camera index
    int camera_index = static_cast<int>(std::strtod(input_str.c_str(), nullptr));

    std::cout << "Opening camera index: " << camera_index << std::endl;
    if (amd::cvml::sample::utils::SetupCamera(
            camera_index, supported_res == nullptr ? camera_res_list : *supported_res,
            &video_input_)) {
      RunFeatureStreaming();
      video_input_.release();
    }
  } else if (is_image) {
    //
    // Still image file, contents read earlier
    //
    std::cout << "Image file read: " << input_str << std::endl;
    if (repeat_image_video_) {
      RunFeatureStreaming();
    } else {
      cv::Mat frame_out = Feature(frame_rgb_);
      cv::cvtColor(frame_out, frame_out, cv::COLOR_RGB2BGR);

      if (output_file_.size() > 0) {
        cv::imwrite(output_file_, frame_out);
        std::cout << "Output file saved: " << output_file_ << std::endl;
      }
    }
  } else if (is_video) {
    //
    // Video file
    //
    RunFeatureVideoFile(input_str);
  }
}

std::string CreateFolderWithTimestamp() {
  std::string file_save_path = GetTimestamp();
  namespace fs = std::filesystem;
  if (fs::create_directories(file_save_path))
    return file_save_path;
  else
    return {};
}

std::string GetTimestamp() {
  std::string timestamp{};
  struct tm ltm;
  time_t now = time(0);
  localtime_s(&ltm, &now);
  std::stringstream mon_s, day_s, hour_s, min_s, sec_s;
  mon_s << std::setw(2) << std::setfill('0') << (ltm.tm_mon + 1);
  day_s << std::setw(2) << std::setfill('0') << ltm.tm_mday;
  hour_s << std::setw(2) << std::setfill('0') << ltm.tm_hour;
  min_s << std::setw(2) << std::setfill('0') << ltm.tm_min;
  sec_s << std::setw(2) << std::setfill('0') << ltm.tm_sec;
  timestamp = mon_s.str() + day_s.str() + hour_s.str() + min_s.str() + sec_s.str();
  return timestamp;
}

void GetPlatformInformation() {
  // print supported platform information
  amd::cvml::SupportedPlatformInformation info{};
  amd::cvml::Context::GetSupportedPlatformInformation(&info);

  for (size_t i = 0; i < info.supported_platform_count; i++) {
    std::cout << "supported APU devide-id: 0x" << std::hex << info.platform[i].device_id << std::dec
              << std::endl;
    std::cout << "required minimal-vulkan-driver-version: 0x" << std::hex
              << info.platform[i].required_gpu_minimal_vulkan_driver_version << std::dec
              << std::endl;
  }
  std::cout << "supported_platform_count=" << info.supported_platform_count << std::endl;
}

bool ParseArguments(int argc, char** const argv, std::string* input_str, std::string* output_file,
                    const char* arg_help) {
  if (input_str == nullptr || output_file == nullptr || argv == nullptr) {
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-i" && ((i + 1) < argc)) {
      *input_str = argv[++i];
    } else if (std::string(argv[i]) == "-o" && ((i + 1) < argc)) {
      *output_file = argv[++i];
    } else {
      std::string app_name{"sample"};

      try {
        std::filesystem::path app_path = argv[0];
        app_name = app_path.stem().string();
      } catch (std::exception& e) {
        // do nothing
        (void)e;
      }
      if (arg_help == nullptr) {
        std::cout << "Usage: " << app_name << ".exe"
                  << " [-i input] [-o file]\n"
                     "    -i\tSpecify an input image/video file or camera device index\n"
                     "    -o\tSpecify output image/video file name\n";
      } else {
        // use argument help override string
        std::cout << "Usage: " << app_name << " " << arg_help;
      }
      std::cout
          << "\n"
             "  Opens the specified input device and runs the feature against it. Results are\n"
             "  displayed in an output window and optionally saved to a file. If no arguments\n"
             "  are provided, the application attempts to capture input from camera index 0\n"
          << std::endl;
      return false;
    }
  }
  return true;
}

void PutRectangle(cv::Mat* image, const cv::Rect& rect, const cv::Scalar& color) {
  if (image == nullptr) {
    return;
  }

  auto alpha = color[3];
  if (alpha == 0 || alpha == 255 || image->type() != CV_8UC3) {
    // simple rectangle
    cv::rectangle(*image, rect, color, -1);
  } else {
    // alpha blend
    auto x_min = (std::clamp)(rect.x, 0, image->cols);
    auto y_min = (std::clamp)(rect.y, 0, image->rows);
    auto x_max = (std::clamp)(rect.x + rect.width, 0, image->cols);
    auto y_max = (std::clamp)(rect.y + rect.height, 0, image->rows);
    auto r = color[0] * alpha / 255.0;
    auto g = color[1] * alpha / 255.0;
    auto b = color[2] * alpha / 255.0;
    auto alpha_1 = (255.0 - alpha) / 255.0;
    for (auto y = y_min; y < y_max; ++y) {
      // image type checked above, 3 bytes / pixel
      auto ptr = image->ptr(y) + x_min * 3;
      for (auto x = x_min; x < x_max; ++x) {
        *ptr = static_cast<uint8_t>(*ptr * alpha_1 + r);
        ++ptr;
        *ptr = static_cast<uint8_t>(*ptr * alpha_1 + g);
        ++ptr;
        *ptr = static_cast<uint8_t>(*ptr * alpha_1 + b);
        ++ptr;
      }
    }
  }
}

void PutText(cv::Mat* image, const std::string& display_text, const int text_row,
             cv::Scalar text_color, const int center_x, const int text_height,
             const bool fill_background, cv::Scalar background_color) {
  static int TEXT_HEIGHT = 30;     // hard coded text height, because getTextSize isn't reliable
  static int TEXT_BOX_OFFSET = 5;  // offset for background box
  static int TEXT_PADDING = 3;     // space between rows of text
  static const int TEXT_THICKNESS = 2;
  static const double TEXT_SCALE = 1.0;

  if (image == nullptr || text_row < 0) {
    // silently return
    return;
  }

  double text_scale = TEXT_SCALE;
  int text_h;

  if (text_height == 0) {
    // default to 1.0 text scaling
    text_h = static_cast<int>(TEXT_HEIGHT * text_scale);
  } else {
    // update text scale based on desired height and image
    text_h = text_height * image->rows / 100;
    text_scale = static_cast<float>(text_h) / TEXT_HEIGHT;
  }

  // cppcheck-suppress knownConditionTrueFalse
  int text_font = text_scale > 1.0 ? cv::FONT_HERSHEY_COMPLEX : cv::FONT_HERSHEY_DUPLEX;
  int text_box_offset = static_cast<int>(TEXT_BOX_OFFSET * text_scale + 0.5);
  int text_padding = static_cast<int>(TEXT_PADDING * text_scale + 0.5);
  int text_thickness = static_cast<int>(TEXT_THICKNESS * text_scale);

  // calculate text height/width
  auto text_size = cv::getTextSize(display_text, text_font, text_scale, text_thickness, nullptr);

  // constant left starting point for english text
  int origin_x = TEXT_PADDING;

  // handle text centering
  if (center_x != 0) {
    origin_x += center_x - text_size.width / 2;
  }

  cv::Point2i origin = cv::Point2i(origin_x, (text_h + text_padding) * (text_row + 1));

  if (fill_background) {
    // draw rectangle on frame for each text
    cv::Rect rectangle(origin - cv::Point2i(0, text_h - text_box_offset),
                       cv::Size(text_size.width, text_h));
    PutRectangle(image, rectangle, background_color);
  }

  // actually display the text
  cv::putText(*image, display_text, origin, text_font, text_scale, text_color, text_thickness);
}

}  // namespace utils
}  // namespace sample
}  // namespace cvml
}  // namespace amd
