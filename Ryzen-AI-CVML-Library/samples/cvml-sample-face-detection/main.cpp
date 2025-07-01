/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <common-sample-utils.h>
#include <cvml-face-detector.h>

#include <filesystem>

#include "opencv2/opencv.hpp"

/**
 * Declare local structure for sample variables.
 */
class FaceDetectionSample : public amd::cvml::sample::utils::RunFeatureClass {
 public:
  amd::cvml::FaceDetector* face_detector_{nullptr};  /// Face Detection feature
  std::string src_path_{};                           /// Input path/device
  // cppcheck-suppress duplInheritedMember
  std::string output_file_{};                        /// Output file path/name
  amd::cvml::FaceDetector::FDModelType fd_model_{};  /// FD model to use

  /**
   * Run Face Detection on single frame
   *
   * @param frame_rgb Incoming RGB frame
   * @return Output RGB frame
   */
  cv::Mat Feature(const cv::Mat& frame_rgb) override;
};

/**
 * Draw detected faces.
 *
 * @param rgb_img Pointer to the target image
 * @param faces Detected face structure
 * @param bbox_color Draw color for bounding boxes
 * @param landmark_color Draw color for landmarks
 * @param landmark_size Size of landmark circle
 */
void DrawFaces(cv::Mat* rgb_img, const amd::cvml::Array<amd::cvml::Face>& faces,
               const cv::Scalar& bbox_color = cv::Scalar(0, 255, 0),
               const cv::Scalar& landmark_color = cv::Scalar(0, 0, 255), int landmark_size = 2) {
  if (rgb_img == nullptr) {
    return;
  }

  // go through all the detected faces
  for (size_t k = 0; k < faces.size(); ++k) {
    const amd::cvml::Face& curr_face = faces[k];
    // convert amd::face to cv::Rect
    cv::Rect cv_face(curr_face.face_.x_, curr_face.face_.y_, curr_face.face_.width_,
                     curr_face.face_.height_);

    // draw a bounding box for each face
    cv::rectangle(*rgb_img, cv_face, bbox_color);

    // print confidence score
    std::ostringstream out_text;
    out_text.precision(2);
    out_text << std::fixed << curr_face.confidence_score_;
    cv::putText(*rgb_img, out_text.str(), cv::Point(cv_face.x, cv_face.y), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 255, 255), 2);

    // draw landmarks
    for (size_t j = 0; j < curr_face.landmarks_.size(); ++j) {
      const amd::cvml::Point2i& landmark = curr_face.landmarks_[j];
      cv::Point center(landmark.x_, landmark.y_);
      cv::circle(*rgb_img, center, landmark_size, landmark_color, cv::FILLED, cv::FILLED);
    }
  }
}

cv::Mat FaceDetectionSample::Feature(const cv::Mat& frame_rgb) {
  std::string title{};
  cv::Mat frame_w_faces = frame_rgb;
  if (face_detector_ == nullptr) {
    // return empty output
    return cv::Mat{};
  }
  // convert to amd::cvml::Image
  amd::cvml::Image input_img(amd::cvml::Image::Format::kRGB, amd::cvml::Image::DataType::kUint8,
                             frame_rgb.cols, frame_rgb.rows, frame_rgb.data);
  // step 1: run face detect on input frame
  auto faces = face_detector_->Detect(input_img);
  // step 2: draw faces on output frame
  DrawFaces(&frame_w_faces, faces);

  return frame_w_faces;
}

void PrintHelpMessage() {
  std::cout << "Usage: cvml-sample-face-detection [-i path to image/video] [-o output "
               "image/video filename] [-h] [-m fd model]"
            << std::endl;
  std::cout << "    -i\trun face detection on video or image given the path" << std::endl;
  std::cout << "    -o\tspecify output video or image file name. e.g. <filename.mp4/jpg>. Optional."
            << std::endl;
  std::cout << "    -m\tspecify face detection model. e.g. <fast/precise>. Optional. Fast "
               "is the default"
            << std::endl;
  std::cout << "    -t\t Enable or disable face tracking. Disabled by default" << std::endl;
  std::cout << "    -h\tshow usage" << std::endl;

  std::cout << "Example 1: cvml-sample-face-detection -i image.jpg" << std::endl;
  std::cout << "Example 2: cvml-sample-face-detection -h" << std::endl;
  std::cout << "Example 3: cvml-sample-face-detection -i image.jpg -m precise" << std::endl;
}

bool ParseArguments(int argc, char** argv, FaceDetectionSample* local_data) {
  std::string fd_model_str;
  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "-i" && ((i + 1) < argc)) {
      local_data->src_path_ = argv[i + 1];
    } else if (std::string(argv[i]) == "-o" && ((i + 1) < argc)) {
      local_data->output_file_ = argv[i + 1];
    } else if (std::string(argv[i]) == "-m" && ((i + 1) < argc)) {
      fd_model_str = argv[i + 1];
    } else if (std::string(argv[i]) == "-h") {
      PrintHelpMessage();
      return false;
    }
  }

  // choose fd model
  if (fd_model_str == "precise") {
    local_data->fd_model_ = amd::cvml::FaceDetector::FDModelType::Precise;
    std::cout << "Running with precise Retinaface model" << std::endl;
  } else {  // default
    local_data->fd_model_ = amd::cvml::FaceDetector::FDModelType::Fast;
    std::cout << "Running with fast Retinaface model" << std::endl;
  }
  return true;
}

void SetContextStreamingMode(const std::string& src_path, amd::cvml::Context* context) {
  // assume camera index if a number is provided
  const std::string input_str = src_path.empty() ? "0" : src_path;
  std::string ext = static_cast<std::filesystem::path>(input_str).extension().string();
  if (ext.length() == 0 && std::isdigit(input_str[0])) {
    context->SetStreamingMode(amd::cvml::Context::StreamingMode::ONLINE_STREAMING);
  } else {
    // check if we can treat the input as an image
    auto frame_rgb_ = cv::imread(input_str);
    if (!frame_rgb_.empty()) {
      context->SetStreamingMode(amd::cvml::Context::StreamingMode::ONE_SHOT);
    } else {
      // assume the input is a video file
      context->SetStreamingMode(amd::cvml::Context::StreamingMode::OFFLINE_STREAMING);
    }
  }
}

/**
 * Options:
 * -i: run face detection on video or image, provide full path to video
 * -o: specify output video clip or image file name
 * -m specify FD model - precise/fast(default)
 * -h: to show usage
 */
int main(int argc, char** argv) {
  try {
    FaceDetectionSample fd_sample;
    if (!ParseArguments(argc, argv, &fd_sample)) {
      return -1;
    }

    // create CVML SDK context for the feaeture
    auto context = amd::cvml::CreateContext();
    if (!context) {
      std::cerr << "Failed to create context" << std::endl;
    } else {
      context->SetInferenceBackend(amd::cvml::Context::InferenceBackend::AUTO);
      SetContextStreamingMode(fd_sample.src_path_, context);

      // create the facedetector feature instances
      amd::cvml::FaceDetector face_detector(context, fd_sample.fd_model_);

      // execute main sample application loop with the created feature
      fd_sample.face_detector_ = &face_detector;
      fd_sample.RunFeature(fd_sample.src_path_, fd_sample.output_file_, "AMD Face Detection");
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
