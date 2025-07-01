/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#ifndef EDGEML_FEATURES_FACE_MESH_INCLUDE_CVML_FACE_MESH_H_
#define EDGEML_FEATURES_FACE_MESH_INCLUDE_CVML_FACE_MESH_H_

#include "cvml-api-common.h"
#include "cvml-context.h"
#include "cvml-face-detector.h"
#include "cvml-image.h"
#include "cvml-types.h"

namespace amd {
namespace cvml {

/**
 * Face Mesh Detection feature class.
 *
 * The FaceMesh class enables the generation of high-quality 3D face meshes from input images using
 * different model types. It simplifies creating instances of face mesh objects and estimating their
 * 3D landmarks with the following features:
 *   -  Allowing configuration of the maximum number of faces for mesh creation
 *   -  Enabling/disabling the computation of head poses
 *   -  Generating face meshes for the largest detected face, a specific Face object, or a region of
 *      interest (ROI) in the image
 *
 * Example:
 *
 *     // create Ryzen AI context
 *     auto context = amd::cvml::CreateContext();
 *
 *     // create face detector feature
 *     amd::cvml::FaceDetector fd(context);
 *
 *     // create face mesh feature
 *     amd::cvml::FaceMesh feature(context);
 *
 *     // encapsulate input image
 *     amd::cvml::Image input( ... );
 *
 *     // detect faces in the input image
 *     auto faces = feature.Detect(input);
 *
 *     // generate mesh of first detected face
 *     auto mesh = face_mesh_->CreateMesh(input, faces[0]);
 */
class CVML_SDK_EXPORT FaceMesh {
  AMD_CVML_INTERFACE(FaceMesh);

 public:
  /**
   * Struct containing output of FaceMesh
   */
  struct CVML_SDK_EXPORT Mesh {
    Array<Point3f> landmarks_;  /// Array of 3D landmarks
  };

  /// Face Mesh Implementaiton class
  class Impl;

  /**
   * Constructor.
   * Creates a FaceMesh instance using the specified CVML context.
   * @param context: Pointer to the CVML context used for initializing the object
   */
  explicit FaceMesh(Context* context);

  /**
   * Set the maximum number of faces for mesh creation. This function is used to
   * determine the face mesh for the largest face in an image by limiting the
   * number of faces generated during the mesh creation process.
   * @param max_num_faces Maximum number of faces for mesh generation.
   */
  void SetMaxNumFaces(int max_num_faces) const;

  /**
   * Generate face mesh for the largest detected face in the image.
   * @param img Input image to generate the face mesh for.
   * @return An Array of Mesh objects containing 3D landmarks and mesh transformation matrix.
   */
  Array<Mesh> CreateMesh(const Image& img) const;

  /**
   * Generate face mesh for the given Face object in the image.
   * Useful when skipping face detection step.
   * @param img Input image to generate the face mesh for.
   * @param face Face structure obtained from face detection.
   * @return A Mesh object containing 3D landmarks and mesh transformation matrix.
   */
  Mesh CreateMesh(const Image& img, const Face& face) const;

  /**
   * Generate face mesh for the given region of interest (ROI) in the image.
   * Useful when skipping face detection step.
   * @param img Input image to generate the face mesh for.
   * @param roi Image region of interest containing the face.
   * @return A Mesh object containing 3D landmarks and mesh transformation matrix.
   */
  Mesh CreateMesh(const Image& img, const Rect_i& roi) const;

  /////////////////   Head Pose API    /////////////////

  /**
   * Get the Transformation Matrix from World Space to Camera space
   * based on the provided face.
   *
   * @param mesh  mesh object used to map world coordinates to image points
   * @return Array<double> flattened 3x4 [R|t] matrix represented as [r_00, r01, r02, t0, r_10, ...]
   */
  Array<double> GetTransformationMatrix(const Mesh& mesh) const;

  /**
   * Get position of the middle point between both eyes in
   *  Camera space based on the provided mesh in centimeters
   *
   * @param face  face object used to map world coordinates to image points
   * @return Point3d position of head in camera space
   */
  Point3d GetHeadPosition(const Mesh& mesh) const;

  /**
   * Get the distance from the middle point between both eyes to the camera in centimeters
   *
   * @param face_mesh mesh to get the head distance from camera for
   * @return double distance of the camera in meters
   */
  double GetHeadDistanceFromCamera(const Mesh& face_mesh) const;

  /**
   * Set camera focal length for more accurate transofrmation matrix calculation
   *
   * @param fx_pxl focal length of the camera in the x axis
   * @param fy_pxl focal length of the camera in the y axis
   */
  void SetFocalLengthInPixels(double fx_pxl, double fy_pxl);

  /**
   * Set the Focal Length and sensor width of the camera
   *
   * @param focal_length_mm focal length in mm
   * @param sensor_width_mm    camera sensor width in mm
   */
  void SetFocalLengthInMillimeters(double focal_length_mm, double sensor_width_mm);

  /**
   * Get the x,y image coordinates for the center of the left eye
   * and the right eye
   *
   * @param face_mesh mesh to calculate the left eye coordinates for
   * @param left_eye_center pointer to left eye point
   * @param right_eye_center pointer to right eye point
   * @return bool if both centers were computed sucessfully
   */
  bool GetEyeCenterCoordinates(const Mesh& face_mesh, Point2f* left_eye_center,
                               Point2f* right_eye_center) const;

 private:
  Impl* impl_;  ///< Implementation of face mesh interface.
};
/**
 * Interface class for face mesh array
 */
template class CVML_SDK_EXPORT Array<FaceMesh::Mesh>;
}  // namespace cvml
}  // namespace amd
#endif  // EDGEML_FEATURES_FACE_MESH_INCLUDE_CVML_FACE_MESH_H_
