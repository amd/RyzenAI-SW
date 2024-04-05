/*
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * Interface class for the depth estimation feature.
 */
class CVML_SDK_EXPORT DepthEstimation {
  AMD_CVML_INTERFACE(DepthEstimation);

 public:
  // Depth Estimation model types
  enum class DepthModelType { Fast, Precise };

  /**
   * Constructor
   *
   * @param context Pointer to CVML SDK context
   * @param model_type Whether to prefer Fast or Precise depth estimation
   */
  explicit DepthEstimation(Context* context, DepthModelType model_type = DepthModelType::Fast);

  /**
   * Generate depth map from an image
   *
   * @param input A reference to the Image input; format: rgb-interleaved-uint8_t
   * @param output a pointer to the Image output; format: NCHW grayscale-float32
   * @return true, if inference output is assigned to output
   *         \n data from image is valid as long as the feature has not been destroyed
   */
  bool GenerateDepthMap(const Image& input, Image* output);

  /**
   * Set the image type of the depth map output data/
   *
   * @param t The desired ImageType of the depth map output data
   *        \n valid values are: kGrayScaleFloat16, kGrayScaleFloat32.
   */
  void SetOutputType(ImageType t);

  /**
   * Get the image type of the depth map output data.
   *
   * @return Image type of the depth estimation output (ImageType)
   */
  ImageType GetOutputType() const;

 protected:
  class Impl;
  Impl* impl_;
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_DEPTH_ESTIMATION_INCLUDE_CVML_DEPTH_ESTIMATION_H_
