/*
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_DEPTH_ESTIMATION_INCLUDE_CVML_DEPTH_ESTIMATION_H_
#define EDGEML_FEATURES_DEPTH_ESTIMATION_INCLUDE_CVML_DEPTH_ESTIMATION_H_

#include "cvml-api-common.h"
#include "cvml-context.h"
#include "cvml-image.h"
#include "cvml-types.h"

namespace amd {
namespace cvml {

/**
 * Depth Estimation feature class.
 *
 * Based on the provided images, the feature calculates a relative depth map
 * for each invocation of the \a GenerateDepthMap() function. Appropriate
 * resize and normalization is done during pre/post processing by the
 * \a GenerateDepthMap() function to generate a depth map for each frame.
 *
 * Example
 *
 *     // create Ryzen AI context
 *     auto context = amd::cvml::CreateContext();
 *
 *     // create depth estimation feature
 *     amd::cvml::DepthEstimation feature(context);
 *
 *     // iterate over input frames
 *     for (auto frame ... ) {
 *         // encapsulate input image
 *         amd::cvml::Image input( ... );
 *
 *         // encapsulate output image
 *         amd::cvml::Image output( ... );
 *
 *         // generate depth map
 *         feature.GenerateDepthMap(input, &output);
 *     }
 */
class CVML_SDK_EXPORT DepthEstimation {
  AMD_CVML_INTERFACE(DepthEstimation);

 public:
  /**
   * Constructor for the Depth Estimation feature.
   *
   * @param context Pointer to CVML SDK context
   */
  explicit DepthEstimation(Context* context);

  /**
   * Generate depth map from an image.
   *
   * This function throws exceptions on errors.
   *
   * Each call of this function returns a depth map of
   * floating point values representing the relative depth of the pixels
   * corresponding to the width/height of the uncropped image frame.
   *
   * @param input Reference to the Image input
   * @param output Pointer to the Image output as a floating point grayscale buffer
   * @return true if the output Image has been populated with inference information
   */
  bool GenerateDepthMap(const Image& input, Image* output);

  /**
   * Set the image type of the depth map output data.
   *
   * This function throws exceptions on errors.
   *
   * @param t The desired ImageType of the depth map output data
   *        \n Valid values are: kGrayScaleFloat16, kGrayScaleFloat32.
   */
  [[deprecated("Output type is determined by the provided output image buffer")]]
  void SetOutputType(ImageType t);

  /**
   * Get the image type of the depth map output data.
   *
   * @return Image type of the depth estimation output (ImageType)
   */
  ImageType GetOutputType() const;

 protected:
  class Impl;
  Impl* impl_;  ///< Implementation of depth estimation interface.
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_DEPTH_ESTIMATION_INCLUDE_CVML_DEPTH_ESTIMATION_H_
