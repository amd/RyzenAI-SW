/*
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef SAMPLES_COMMON_SAMPLE_UTILS_INCLUDE_COMMON_SAMPLE_UTILS_H_
#define SAMPLES_COMMON_SAMPLE_UTILS_INCLUDE_COMMON_SAMPLE_UTILS_H_

#include <string>
#include <vector>

#include "cvml-context.h"
#include "opencv2/opencv.hpp"

namespace amd {
namespace cvml {
namespace sample {
namespace utils {

/**
 * camera resolution
 */
typedef struct CamRes {
  int width;
  int height;
} CamRes;

/**
 * Sets up the camera with the specified camera id according to the preferred resolution list
 * @param camera_index: camera to open
 * @param res_list: a list of resolutions that can be used, the first resolution in the list will be
 * tried first
 * @param camera: opencv camera handle if camera openeed successfully
 * @return camera open successfully or not
 */
bool SetupCamera(int camera_index, const std::vector<CamRes>& res_list, cv::VideoCapture* camera);

/**
 * Create a folder with timestamp corresponding to the current local time of the system
 *
 * @return folder named with timestamp
 */
std::string CreateFolderWithTimestamp();

/**
 * Get a string of timestamp corresponding to the current local time of the system
 *
 * @return string of timestamp
 */
std::string GetTimestamp();

/**
 * Local class definition for passing information to \a RunFeature
 * callbacks. To use, inherit a local class/struct from this and provide
 * it to the \a RunFeature function.
 *
 * This class also contains additional flags/configuration parameters
 * to modify the behavior of the RunFeature() function.
 */
class RunFeatureClass {
 public:
  /// destructor
  virtual ~RunFeatureClass() {}

  /// Press the specified key to save a frame shot, not case sensitive
  char frame_shot_key_{'s'};

  /// Repeatedly iterate on image or video until window is closed
  bool repeat_image_video_{false};

  /// Scaling factor for display window, auto-scaling if zero
  float disp_window_scale_{0.0f};

  /// Specify window title to enable showing input frame
  const char* input_window_name_{nullptr};

  /// Concatenate input/output images beside each other
  bool side_by_side_{false};

  // Called for specific run feature code in each feature
  virtual cv::Mat Feature(const cv::Mat& input_frame_rgb) { return input_frame_rgb; }

  /**
   * Opens video source and executes the feature.
   *
   * This function throws exceptions on errors.
   *
   * The provided callback function is called for each frame of
   * camera/video/still with an OpenCV RGB input buffer. It should
   * return an output/processed RGB buffer.
   *
   * The input extension is used to differentiate between video clips
   * and still images. To select a camera input, provide the desired
   * camera index as the input. If an empty input string is provided,
   * the function attempts to open camera index 0.
   *
   * @param input: Input file name, or "<camera index>" if camera desired
   * @param output_file: Output file name
   * @param window_title: Optional application window title, can be empty
   * @param supported_res: Pointer to Camera resolution list supported by feature, can be NULL if
   * using default value
   */
  virtual void RunFeature(const std::string& input, const std::string& output_file,
                          const std::string& window_title,
                          std::vector<CamRes>* supported_res = nullptr);

 protected:
  cv::VideoCapture video_input_;  ///< OpenCV video capture device
  cv::VideoWriter video_output_;  ///< Video writer for main output

  bool open_output_file_;             ///< Whether to attempt opening output file
  std::string output_file_{};         ///< Output file name for \a RunFeatureStreaming
  std::string output_window_name_{};  ///< Output window name for \a RunFeatureStreaming

  cv::Mat frame_rgb_;  ///< Input RGB frame data
  double stream_fps_;  ///< Video/camera input frame rate

 protected:
  /**
   * Helper function to run feature against video files.
   *
   * @param input_file Video file name
   */
  virtual void RunFeatureVideoFile(const std::string& input_file);

  /**
   * Helper function to run feature against streaming inputs.
   *
   * @return true if user exit
   */
  virtual bool RunFeatureStreaming();

  /**
   * Helper function to run feature against a video file.
   */
  /**
   * Fill local frame_rgb with next input frame.
   *
   * @param frame_id Zero-based frame id
   * @return true if local's frame_rgb is ready to be processed
   */
  virtual bool GetSingleVideoFrame(uint32_t frame_id);
};

/**
 * Function to print the supported platform details.
 */
void GetPlatformInformation();

/**
 * Parse command line arguments for RunFeature.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param input_str Pointer to input device/file string
 * @param output_file Pointer to output file string
 * @param arg_help Optional replacement string of argument option help text
 * @return true if the application should continue
 */
bool ParseArguments(int argc, char** const argv, std::string* input_str, std::string* output_file,
                    const char* arg_help = nullptr);

/**
 * Render rectangle into the frame.
 *
 * @param image Pointer to target image buffer
 * @param rect OpenCV rectangle definition
 * @param color Rectangle color, as an RGBA scalar
 */
void PutRectangle(cv::Mat* image, const cv::Rect& rect, const cv::Scalar& color);

/**
 * Render text strings into the frame.
 *
 * @param image Target image buffer
 * @param display_text String of text to render
 * @param row Zero-based row number to render text, assuming text console
 * @param text_color Color of text to render
 * @param center_x If non-zero, text will be centered around this point
 * @param text_height If non-zero, specifies text height as a percentage of the frame height
 * @param fill_background Whether or not an opaque background should be added
 * @param background_color Color of background, if specified
 */
void PutText(cv::Mat* image, const std::string& display_text, const int text_row,
             cv::Scalar text_color, const int center_x, const int text_height,
             const bool fill_background = false, cv::Scalar background_color = cv::Scalar(0, 0, 0));

}  // namespace utils
}  // namespace sample
}  // namespace cvml
}  // namespace amd

#endif  // SAMPLES_COMMON_SAMPLE_UTILS_INCLUDE_COMMON_SAMPLE_UTILS_H_
