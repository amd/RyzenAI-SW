/*
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_BODY_POSE_INCLUDE_CVML_BODY_POSE_H_
#define EDGEML_FEATURES_BODY_POSE_INCLUDE_CVML_BODY_POSE_H_

#include "cvml-api-common.h"
#include "cvml-context.h"
#include "cvml-image.h"
#include "cvml-types.h"

using amd::cvml::Array;
using amd::cvml::Context;
using amd::cvml::Image;
using amd::cvml::Person;

namespace amd {
namespace cvml {

/**
 * Body Pose Detection feature class.
 *
 * The body pose feature takes in an image or video stream as input. For each frame,
 * it returns predictions for up to 6 people in the frame containing:
 *
 * - Bounding box (x, y, width, height) in image space
 * - 17 landmark coordinates (x, y, z) in image space
 * - Confidence score for each landmark (0.0 to 1.0)
 * - Confidence score for the overall person (0.0 to 1.0)
 *
 * The landmarks correspond to 17 unique joint positions on the human body in accordance with
 * the COCO-Pose dataset (see \a BodyPose::Keypoint). Any landmark that is occluded or falls outside
 * of the frame will have its associated confidence score set to -1. Unless 3D detection is enabled,
 * the z coordinate of each landmark is set to 0.
 *
 * If the input streaming mode != ONE_SHOT, the API may enable additional postprocessing to smoothen
 * detections across frames.
 *
 * Example
 *
 *     // create Ryzen AI context
 *     auto context = amd::cvml::CreateContext();
 *
 *     // create body pose feature
 *     amd::cvml::BodyPose feature(context);
 *
 *     // iterate over input frames
 *     for (auto frame ... ) {
 *         // encapsulate input image
 *         amd::cvml::Image img( ... );
 *
 *         // detect people/poses
 *         auto output = feature.Generate(img);
 *     }
 */
class CVML_SDK_EXPORT BodyPose {
  AMD_CVML_INTERFACE(BodyPose);

 public:
  /**
   * Constructor
   *
   * @param context Pointer to CVML SDK context
   */
  explicit BodyPose(Context* context);

  /**
   * Defines the landmark indices of the Array<Point3i> landmarks
   * within a \a Person object
   */
  enum class Keypoint {
    kNose,           ///< Nose
    kLeftEye,        ///< Left eye
    kRightEye,       ///< Right eye
    kLeftEar,        ///< Left ear
    kRightEar,       ///< Right ear
    kLeftShoulder,   ///< Left shoulder
    kRightShoulder,  ///< Right shoulder
    kLeftElbow,      ///< Left elbow
    kRightElbow,     ///< Right elbow
    kLeftWrist,      ///< Left wrist
    kRightWrist,     ///< Right wrist
    kLeftHip,        ///< Left hip
    kRightHip,       ///< Right hip
    kLeftKnee,       ///< Left knee
    kRightKnee,      ///< Right knee
    kLeftAnkle,      ///< Left ankle
    kRightAnkle,     ///< Right ankle
    kNumPoints       ///< Total number of returned landmarks
  };

  /**
   * Main feature entry point.
   *
   * Applications/clients should call this function once for every
   * frame in the video or live stream.
   *
   * @param img amd::cvml::Image object containing input image
   * @return the Array of Person structs representing detected people
   */
  Array<Person> Generate(const Image& img);

  /**
   * Set the detection threshold for people within a scene.
   *
   * Detected persons under the specified threshold are not returned by
   * the \a Generate() function.
   *
   * @param threshold Person detection threshold, from 0.0 to 1.0
   */
  void SetDetectionThreshold(float threshold);

  class Impl;

 protected:
  Impl* impl_;  ///< Implementation of body pose interface.
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_BODY_POSE_INCLUDE_CVML_BODY_POSE_H_
