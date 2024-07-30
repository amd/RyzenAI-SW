/*
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <common-sample-utils.h>
#include <cvml-depth-estimation.h>

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

using amd::cvml::DepthEstimation;
using amd::cvml::Image;
using amd::cvml::ImageType;

/**
 * Declare local class for sample variables and functions.
 */
class DepthEstimationSample : public amd::cvml::sample::utils::RunFeatureClass {
 public:
  amd::cvml::DepthEstimation* depth_estimation_{nullptr};  ///< Depth Estimation feature
  std::string input_str_{};    ///< frame source: image or video or camera
  std::string output_file_{};  ///< Output file path/name
  bool use_fp16_;              ///< depth output type
  amd::cvml::DepthEstimation::DepthModelType de_model_{};  ///< Depth model to use

  /**
   * Post process depth map for opencv visualization.
   *
   * @param depth_map Pointer to depth map
   * @return Postprocessed depth-map
   */
  cv::Mat DepthEstimationCvmlToOpenCV(const Image* depth_map);

  /**
   * Run Depth Estimation on single frame
   *
   * @param frame_rgb Incoming RGB frame
   * @return Output RGB frame
   */
  cv::Mat Feature(const cv::Mat& frame_rgb) override;
};

cv::Mat DepthEstimationSample::Feature(const cv::Mat& frame_rgb) {
  cv::Mat frame_out = frame_rgb.clone();  // OpenCV output buffer
  cv::Mat display_buffer;                 // display buffer for model output

  if (depth_estimation_ == nullptr) {
    // return empty output
    return frame_out;
  }

  Image input_frame_amd_image(amd::cvml::Image::Format::kRGB, amd::cvml::Image::DataType::kUint8,
                              frame_rgb.cols, frame_rgb.rows, frame_rgb.data);
  use_fp16_ = (depth_estimation_->GetOutputType() == amd::cvml::ImageType::kGrayScaleFloat16);

  // Create destination output
  amd::cvml::Image output_img(amd::cvml::Image::Format::kGrayScale,
                              amd::cvml::Image::DataType::kFloat32, frame_rgb.cols, frame_rgb.rows,
                              nullptr);

  // Depth Estimation
  bool depth_map_generated =
      depth_estimation_->GenerateDepthMap(input_frame_amd_image, &output_img);
  if (!depth_map_generated) {
    std::cout << "Failed to generate depth map" << std::endl;
    throw std::exception("Failed to generate depth map!");
  }
  return DepthEstimationCvmlToOpenCV(&output_img);
}

cv::Mat DepthEstimationSample::DepthEstimationCvmlToOpenCV(const Image* depth_map) {
  cv::Mat depth_map_or_mat_raw;
  cv::Mat frame_out;

  if (depth_map != nullptr) {
    float* depth_map_or_p =
        reinterpret_cast<float*>(reinterpret_cast<void*>(depth_map->GetBuffer()));
    if (depth_map_or_p == nullptr) {
      throw std::exception("Failed to get depth map data!");
    }
    if (use_fp16_) {
      cv::Mat depth_map_or_mat_raw2 =
          cv::Mat{static_cast<int>(depth_map->GetHeight()), static_cast<int>(depth_map->GetWidth()),
                  CV_16FC1, depth_map_or_p};

      depth_map_or_mat_raw2.convertTo(depth_map_or_mat_raw, CV_32FC1);
    } else {
      depth_map_or_mat_raw =
          cv::Mat{static_cast<int>(depth_map->GetHeight()), static_cast<int>(depth_map->GetWidth()),
                  CV_32FC1, depth_map_or_p};
    }
    cv::Mat tmp;
    depth_map_or_mat_raw.convertTo(tmp, CV_8U, 255);
    cv::cvtColor(tmp, frame_out, cv::COLOR_GRAY2RGB);
    cv::applyColorMap(frame_out, frame_out, cv::COLORMAP_SPRING);
  }
  return frame_out;
}

void PrintHelpMessage() {
  std::cout << "Usage: cvml-sample-depth-estimation.exe [-i input image/video] [-o output "
               "image/video] [-m depth model] [-h]"
            << std::endl;
  std::cout << "    -i\tSpecify an input image/video file or camera device index" << std::endl;
  std::cout << "    -o\tSpecify output image/video file name" << std::endl;
  std::cout << "    -m\tspecify depth estimation model. e.g. <fast/precise>. Optional. Fast "
               "is the default"
            << std::endl;
  std::cout << "    -h\tshow usage" << std::endl;

  std::cout << "Example 1: cvml-sample-depth-estimation.exe -i image.jpg" << std::endl;
  std::cout << "Example 2: cvml-sample-depth-estimation.exe -i image.jpg -m precise" << std::endl;
}

bool ParseArguments(int argc, char** argv, DepthEstimationSample* local_data) {
  std::string de_model_str;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-i" && ((i + 1) < argc)) {
      local_data->input_str_ = argv[i + 1];
    } else if (std::string(argv[i]) == "-o" && ((i + 1) < argc)) {
      local_data->output_file_ = argv[i + 1];
    } else if (std::string(argv[i]) == "-m" && ((i + 1) < argc)) {
      de_model_str = argv[i + 1];
    } else if (std::string(argv[i]) == "-h") {
      PrintHelpMessage();
      return false;
    }
  }

  // choose depth model
  if (de_model_str == "precise") {
    local_data->de_model_ = amd::cvml::DepthEstimation::DepthModelType::Precise;
    std::cout << "Running with precise Depth Estimation model" << std::endl;
  } else {  // default
    local_data->de_model_ = amd::cvml::DepthEstimation::DepthModelType::Fast;
    std::cout << "Running with fast Depth Estimation model" << std::endl;
  }

  return true;
}

/**
 * Main entry point of the sample application.
 *
 * @param argc: Number of command line arguments
 * @param argv: Array of command line arguments
 * @return 0 on success
 */
int main(int argc, char** argv) {
  DepthEstimationSample de_sample;

  // show both input and output images
  de_sample.side_by_side_ = true;

  // parse command line arguments
  if (!ParseArguments(argc, argv, &de_sample)) {
    return -1;
  }

  try {
    // create CVML SDK context for the feature
    auto context = amd::cvml::CreateContext();
    if (!context) {
      std::cerr << "Failed to create context" << std::endl;
    } else {
      // select backend (optional)
      context->SetInferenceBackend(amd::cvml::Context::InferenceBackend::AUTO);

      // initialize depth estimation class
      amd::cvml::DepthEstimation depth_estimation(context, de_sample.de_model_);

      // execute main sample application loop with the created feature
      de_sample.depth_estimation_ = &depth_estimation;

      // run the feature against input frames and local_data
      de_sample.RunFeature(de_sample.input_str_, de_sample.output_file_, "AMD Depth Estimation");
    }

    // release previously created context
    if (context) {
      context->Release();
    }
  } catch (std::exception& e) {
    std::cerr << "Sample application error: " << e.what() << std::endl;
  }
  return 0;
}
