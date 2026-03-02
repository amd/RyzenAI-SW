/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_FACE_DETECTOR_INCLUDE_CVML_FACE_DETECTOR_H_
#define EDGEML_FEATURES_FACE_DETECTOR_INCLUDE_CVML_FACE_DETECTOR_H_

#include "cvml-context.h"
#include "cvml-image.h"
#include "cvml-types.h"

namespace amd {
namespace cvml {

/**
 * Face Detection feature class.
 *
 * The FaceDetector class offers an interface for efficient face detection in various model types.
 *
 * Supporting Fast and Precise model types, the class is initialized with a specified
 * CVML context, allowing users to switch between models during runtime. Key features include:
 *
 *     - Detecting faces and their landmarks in input images
 *     - Adjusting the detection threshold
 *     - Tracking detected faces across consecutive frames
 *
 * By utilizing the detect and tracking functions, the FaceDetector class allows precise face
 * detection and seamless integration in applications.
 *
 * Example:
 *
 *     // create Ryzen AI context
 *     auto context = amd::cvml::CreateContext();
 *
 *     // create face detector feature
 *     amd::cvml::FaceDetector feature(context);
 *
 *     // encapsulate input image
 *     amd::cvml::Image input( ... );
 *
 *     // detect faces in the input image
 *     auto faces = feature.Detect(input);
 */

class CVML_SDK_EXPORT FaceDetector {
  AMD_CVML_INTERFACE(FaceDetector);

 public:
  /**
   * Fast - Optimized for performance
   * Precise - Optimized for acccuracy
   */
  enum class FDModelType { Fast, Precise };

  /**
   * Constructor for FaceDetector class.
   *
   * Constructs a FaceDetector object with a specified CVML context and face detection model type.
   * Face detection model types include Fast, and Precise.
   * It throws exceptions if any errors occur during processing.
   * @param context CVML context
   * @param model_type Face detection model type (default: FDModelType::Precise)
   */
  explicit FaceDetector(Context* context, FDModelType model_type = FDModelType::Precise);

  /**
   * Main face detection function.
   * This function is the main entry point for face detection in the provided image.
   * It should be called for each frame in an application or video stream.
   * The function detects faces and associated landmarks in the input image.
   * It throws exceptions if any errors occur during processing.
   * @param img Input image of type amd::cvml::Image.
   * @return Array of detected faces of type amd::cvml::Face.
   */
  Array<Face> Detect(const Image& img) const;

  /**
   * Set the detection threshold for face detection.
   *
   * This function sets the minimum confidence score required for
   * faces to be included in the detection output. Faces with a confidence
   * score below the specified threshold will be ignored.
   *
   * The default threshold is 0.5
   */
  void SetDetectionThreshold(float detection_threshold);

  /**
   * Set the face detection model type.
   *
   * This function allows the user to switch between different face detection
   * models while the application is running. The available model types are:
   * Fast - Optimized for performance.
   * Precise - Optimized for accuracy.
   * Changing the model type can affect the speed and accuracy of the face
   * detection results. The optimal model type may vary depending on the
   * requirements of the application and the specific use case.
   * @param model_type required model type
   */
  void SetModelType(FDModelType model_type);

  /**
   * Get tracking IDs for face detection.
   *
   * This function provides consistent face IDs for detected faces in consecutive frames.
   * IDs in the array will be in the same order as the Face Array from the last Detect call.
   * Array will be empty if Detect has not been called before.
   */
  Array<int> GetTrackedIDs() const;

  /**
   * Get the Transformation Matrix from World Space to Camera space
   * based on the provided face.
   *
   * @param face  face object used to map world coordinates to image points
   * @return Array<double> flattened 3x4 [R|t] matrix represented as [r_00, r01, r02, t0, r_10, ...]
   * where translations are in meters
   */
  Array<double> GetTransformationMatrix(const Face& face) const;

  /**
   * Get position of the middle point between both eyes in
   *  Camera space based on the provided face in centimeters
   *
   * @param face  face object used to map world coordinates to image points
   * @return Point3d position of head in camera space
   */
  Point3d GetHeadPosition(const Face& face) const;

  /**
   * Get the distance from the middle point between both eyes to the camera in centimeters
   *
   * @param face face to get the head distance from camera for
   * @return double distance of the camera in meters
   */
  double GetHeadDistanceFromCamera(const Face& face) const;

  /**
   * Set camera focal length for more accurate transofrmation matrix calculation
   *
   * @param fx focal length of the camera in the x axis
   * @param fy focal length of the camera in the y axis
   */
  void SetFocalLength(double fx, double fy);

  class Impl;

 protected:
  Impl* impl_;  ///< Implementation of face detector interface.
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_FACE_DETECTOR_INCLUDE_CVML_FACE_DETECTOR_H_
