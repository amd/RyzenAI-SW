/*
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_IMAGE_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_IMAGE_H_

#include <inttypes.h>

#include "cvml-types.h"

namespace amd {

namespace cvml {

class Context;

/**
 * The class representing an image
 */
class CVML_SDK_EXPORT Image {
 public:
  /**
   * An enumeration of image formats
   */
  enum Format {
    kGrayScale,
    kRGBA,
    kRGBAp,
    kBGRA,
    kBGRAp,
    kRGB,
    kRGBp,
    kBGR,
    kBGRp,
    kNV12,
    kNV21,
    kYUV420p,
    kYUYV422,
    kP010,
  };

  /**
   * An enumeration of image data types
   */
  enum DataType {
    kUint8,
    kInt8,
    kFloat16,
    kFloat32,
    kFixedPoint16I8F2LE, /**< P010 little endian */
    kFixedPoint16I8F2BE, /**< P010 big endian */
    kUint16,
  };

  /**
   * A valid set of flags used to describe the image
   */
  enum Flags {
    /**
     * A hint that indicates the image will be used as an image source
     * Potentially helpful for accelerating the processing
     */
    kSource = 1 << 0,

    /**
     * A hint that indicates the image will be used as an image target
     * Potentially helpful for accelerating the processing
     */
    kTarget = 1 << 1,

    /**
     * Indicate the image will be created by importing a device local memory
     * (for example, vulkan device local memory)
     */
    kDeviceMemoryImport = 1 << 2,

    /**
     * Indicate the image will be created on device local memory (for example, vulkan
     * device local memory) and the image can be later exported via Image::Export().
     */
    kDeviceMemoryExport = 1 << 3,
  };

  /**
   * Construct a CVML Image class object
   * @param format: Image format
   * @param data_type: Image data type
   * @param width: The pixel width of the image
   * @param height: The pixel height of the image
   * @param buffer: (optional): A pointer to the image data.
   * If buffer is null, the data will be allocated by the CVML context specified via Map()
   * If buffer not null:
   *   - If Flags::kDeviceMemoryImport is not specified, the buffer is expected to be a host buffer.
   *   - If Flags::kDeviceMemoryImport is specified, the buffer shall point to a HANDLE/fd
   *     to a device local memory (for example, vulkan device local memory).
   * @param stride: (optional) if stride is not specified, image will be stored continuously.
   * @param flags: (optional): bit mask of Flags specifying the valid usage of the image.
   * See Image::Flags for more information. Defaults to both source and target.
   * If not specified, the default value is Flags::kSource | Flags::kTarget
   */
  Image(Format format, DataType data_type, uint32_t width, uint32_t height,
        uint8_t* buffer = nullptr, uint32_t stride = 0,
        uint32_t flags = Flags::kSource | Flags::kTarget);

  /**
   * @deprecated
   * Construct an Image class wrapper of host buffer
   * Using this throws an exception. Use the other constructor.
   */
  [[deprecated]] Image(ImageType img_type, uint32_t width, uint32_t height, uint32_t stride,
                       uint8_t* data_buf);

  /**
   * @deprecated
   * Returns the image type
   * Using this throws an exception. Use GetFormat()/GetDataType().
   */
  [[deprecated("Use GetFormat()/GetDataType().")]] ImageType GetImageType() const;

  /**
   * @deprecated
   * @return address fo the CVML Image buffer
   */
  uint8_t* GetBuffer() const;

  /**
   * Map CVML Image buffer using the specified CVML context
   * @param context: CVML context to be associated with the image
   * @param flags: flags for the operation
   * @return address of the mapped CVML Image buffer
   */
  uint8_t* Map(Context* context, uint32_t flags = 0);

  /**
   * Get the width of the image.
   * @return: The width of the image.
   */
  uint32_t GetWidth() const;

  /**
   * Get the height of the image.
   * @return: The height of the image.
   */
  uint32_t GetHeight() const;

  /**
   * Get format for this image.
   * @return The format that the image was created with.
   */
  Format GetFormat() const;

  /**
   * Get data type for this image.
   * @return The data type that the image was created with.
   */
  DataType GetDataType() const;

  /**
   * Get the usage flag bit mask for this image.
   * @return The bit mask of flags that the image was created with.
   */
  uint32_t GetFlags() const;

  /**
   * Export the image so that it can be imported in a different device context
   * (for example, vulkan context). To make an image exportable, the image must
   * be created with Flags::kExport in constructor.
   * @param handle: pointer to a handle the image wil be exported to.
   * For windows, the pointer shall point to windows HANDLE struct.
   * For linux, the pointer shall point to file desriptor (int).
   * @return Returns true on success, false on failure.
   */
  bool Export(void* handle);

  virtual ~Image();
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&&) noexcept = delete;
  Image& operator=(Image&&) noexcept = delete;

  class Impl;
  Impl* impl_;
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_IMAGE_H_
