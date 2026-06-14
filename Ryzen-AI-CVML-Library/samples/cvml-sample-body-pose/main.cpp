/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <common-sample-utils.h>
#include <cvml-body-pose.h>
#include <stdint.h>

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

/**
 * Declare local class for sample variables and functions.
 */
class BodyPoseSample : public amd::cvml::sample::utils::RunFeatureClass {
 public:
  static constexpr float detection_threshold_ = 0.2f;
  amd::cvml::BodyPose* body_pose_;
  std::string input_str_{};
  // cppcheck-suppress duplInheritedMember
  std::string output_file_{};

  /**
   * Run Body Pose on single frame
   *
   * @param frame_rgb Incoming RGB frame
   * @return Output RGB frame
   */
  cv::Mat Feature(const cv::Mat& frame_rgb) override;

  /**
   * Helper functions to draw person skeleton on input image.
   *
   * @param out_img image to draw on
   * @param bp detected Person struct
   */
  void DrawPersonEdges(cv::Mat* out_img, const amd::cvml::Person& bp);
  void DrawPersonLandmarks(cv::Mat* out_img, const amd::cvml::Person& bp);
  void DrawPersonBoundingBox(cv::Mat* out_img, const amd::cvml::Person& bp);
};

cv::Mat BodyPoseSample::Feature(const cv::Mat& frame_rgb) {
  // Return on invalid input
  if (frame_rgb.empty()) {
    return frame_rgb;
  }

  cv::Mat frame_out = frame_rgb;

  //
  // Generate body pose results from the received input frame
  //
  auto results = body_pose_->Generate(
      amd::cvml::Image(amd::cvml::Image::Format::kRGB, amd::cvml::Image::DataType::kUint8,
                       frame_rgb.cols, frame_rgb.rows, frame_rgb.data));

  //
  // Draw the keypoints, edges, bounding boxes on the output image
  //
  for (size_t index = 0; index < results.size(); ++index) {
    const amd::cvml::Person& bp = results[index];

    // draw edges between landmarks for each person
    DrawPersonEdges(&frame_out, bp);

    // draw individual landmarks for each person
    DrawPersonLandmarks(&frame_out, bp);

    // draw bounding box for person instance
    DrawPersonBoundingBox(&frame_out, bp);
  }

  return frame_out;
}

void BodyPoseSample::DrawPersonEdges(cv::Mat* out_img, const amd::cvml::Person& bp) {
  static const struct {
    amd::cvml::BodyPose::Keypoint start;
    amd::cvml::BodyPose::Keypoint end;
  } edge_list[] = {
      // left side body
      {amd::cvml::BodyPose::Keypoint::kLeftShoulder, amd::cvml::BodyPose::Keypoint::kLeftElbow},
      {amd::cvml::BodyPose::Keypoint::kLeftElbow, amd::cvml::BodyPose::Keypoint::kLeftWrist},
      {amd::cvml::BodyPose::Keypoint::kLeftShoulder, amd::cvml::BodyPose::Keypoint::kLeftHip},
      {amd::cvml::BodyPose::Keypoint::kLeftHip, amd::cvml::BodyPose::Keypoint::kLeftKnee},
      {amd::cvml::BodyPose::Keypoint::kLeftKnee, amd::cvml::BodyPose::Keypoint::kLeftAnkle},

      // right side body
      {amd::cvml::BodyPose::Keypoint::kRightShoulder, amd::cvml::BodyPose::Keypoint::kRightElbow},
      {amd::cvml::BodyPose::Keypoint::kRightElbow, amd::cvml::BodyPose::Keypoint::kRightWrist},
      {amd::cvml::BodyPose::Keypoint::kRightShoulder, amd::cvml::BodyPose::Keypoint::kRightHip},
      {amd::cvml::BodyPose::Keypoint::kRightHip, amd::cvml::BodyPose::Keypoint::kRightKnee},
      {amd::cvml::BodyPose::Keypoint::kRightKnee, amd::cvml::BodyPose::Keypoint::kRightAnkle},

      // center body
      {amd::cvml::BodyPose::Keypoint::kLeftShoulder, amd::cvml::BodyPose::Keypoint::kRightShoulder},
      {amd::cvml::BodyPose::Keypoint::kLeftHip, amd::cvml::BodyPose::Keypoint::kRightHip},
      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kLeftShoulder},
      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kRightShoulder},

      // head
      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kLeftEye},
      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kLeftEar},
      {amd::cvml::BodyPose::Keypoint::kLeftEye, amd::cvml::BodyPose::Keypoint::kLeftEar},

      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kRightEye},
      {amd::cvml::BodyPose::Keypoint::kNose, amd::cvml::BodyPose::Keypoint::kRightEar},
      {amd::cvml::BodyPose::Keypoint::kRightEye, amd::cvml::BodyPose::Keypoint::kRightEar}};

  const cv::Scalar color = cv::Scalar(255, 165, 0);
  int line_thickness = 3;

  if (out_img == nullptr || out_img->data == nullptr) {
    std::cout << "Invalid output image" << std::endl;
    return;
  }

  for (int k = 0; k < static_cast<int>(sizeof(edge_list) / sizeof(edge_list[0])); ++k) {
    int start_idx = static_cast<int>(edge_list[k].start);
    int end_idx = static_cast<int>(edge_list[k].end);

    cv::Point p1(bp.landmarks_[start_idx].x_, bp.landmarks_[start_idx].y_);
    cv::Point p2(bp.landmarks_[end_idx].x_, bp.landmarks_[end_idx].y_);
    // If either start/end landmark has conf == -1, clip line to image boundaries
    if ((bp.landmark_scores_[start_idx] == -1) || (bp.landmark_scores_[end_idx] == -1)) {
      if (!cv::clipLine(out_img->size(), p1, p2)) {  // skip if line is entirely out of bounds
        continue;
      }
    }
    // if conf score < threshold, do not draw edge
    if ((bp.landmark_scores_[start_idx] < detection_threshold_) ||
        (bp.landmark_scores_[end_idx] < detection_threshold_)) {
      continue;
    }
    cv::line(*out_img, p1, p2, color, line_thickness);
  }
}

void BodyPoseSample::DrawPersonLandmarks(cv::Mat* out_img, const amd::cvml::Person& bp) {
  const cv::Scalar color = cv::Scalar(0, 255, 0);
  const int radius = 10;  // radius for landmarks

  if (out_img == nullptr || out_img->data == nullptr) {
    std::cout << "Invalid output image" << std::endl;
    return;
  }

  for (size_t k = 0; k < bp.landmarks_.size(); k++) {
    if (bp.landmark_scores_[k] == -1) {
      continue;
    }
    if (bp.landmark_scores_[k] < detection_threshold_) {
      continue;
    }
    cv::Point p(bp.landmarks_[k].x_, bp.landmarks_[k].y_);

    cv::circle(*out_img, p, radius, color, -1);
  }
}

void BodyPoseSample::DrawPersonBoundingBox(cv::Mat* out_img, const amd::cvml::Person& bp) {
  const cv::Scalar color = cv::Scalar(125, 18, 255);
  const int thickness = 4;  // line thickness

  if (out_img == nullptr || out_img->data == nullptr) {
    std::cout << "Invalid output image" << std::endl;
    return;
  }
  cv::Rect r(bp.person_.x_, bp.person_.y_, bp.person_.width_, bp.person_.height_);

  cv::rectangle(*out_img, r, color, thickness);
}

/**
 * Main entry point of the sample application.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success
 */
int main(int argc, char** const argv) {
  BodyPoseSample bp_sample;

  // parse command line arguments
  if (!amd::cvml::sample::utils::ParseArguments(argc, argv, &bp_sample.input_str_,
                                                &bp_sample.output_file_)) {
    return -1;
  }

  try {
    // create a CVML context for the feature
    auto context = amd::cvml::CreateContext();
    if (!context) {
      std::cerr << "Failed to create context" << std::endl;
    } else {
      // select backend (optional)
      context->SetInferenceBackend(amd::cvml::Context::InferenceBackend::AUTO);

      // set streaming mode based on input file
      bp_sample.SetContextStreamingModeBySrc(context, bp_sample.input_str_);

      // initialize body pose class
      amd::cvml::BodyPose body_pose(context);

      // execute main sample application loop with the created feature
      bp_sample.body_pose_ = &body_pose;

      // run the feature against input frames and local_data
      bp_sample.RunFeature(bp_sample.input_str_, bp_sample.output_file_, "AMD Body Pose");
    }

    // release previously created context
    if (context) {
      context->Release();
    }
  } catch (std::exception& e) {
    std::cerr << "Sample application error:" << e.what() << std::endl;
  }
  return 0;
}
