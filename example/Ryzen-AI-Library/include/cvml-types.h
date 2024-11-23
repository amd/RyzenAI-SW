/*!
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * @file
 *
 * Defines common types and structures for the CVML SDK.
 */

#ifndef EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_TYPES_H_
#define EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_TYPES_H_

#include <inttypes.h>

#include <stdexcept>
#include <utility>

#include "cvml-api-common.h"

namespace amd {
namespace cvml {

/**
 * @deprecated
 * Image types represent colour space and
 * number of bits per pixel.
 */
enum ImageType {
  kRgbUint8 = 1,
  kRgbFloat16 = 2,
  kRgbFloat32 = 3,
  kGrayScaleUint8 = 4,
  kGrayScaleFloat16 = 5,
  kGrayScaleFloat32 = 6,
  kRgbInt8 = 7,
  kGrayScaleInt8 = 8,
  kNV12Float32 = 9,
  kNV12Uint8 = 10,
  kRgbaUint8 = 11,
  kRgbaFloat16 = 12,
  kRgbaFloat32 = 13,
  kRgbaInt8 = 14,
  kNV12Uint16 = 15,
};

/**
 * Structure for describing rectangular regions.
 */
template <typename _Tp>
struct CVML_SDK_EXPORT Rect {
  /**
   * Default constructor.
   */
  Rect() : x_(0), y_(0), width_(0), height_(0) {}

  /**
   * Initializing constructor.
   *
   * @param x X cordinate of top left corner
   * @param y Y cordinate of top left corner
   * @param width Rectange width
   * @param height Rectange height
   */
  Rect(_Tp x, _Tp y, _Tp width, _Tp height) : x_(x), y_(y), width_(width), height_(height) {}

  /// X cordinate of top left corner
  _Tp x_;

  /// Y cordinate of top left corner
  _Tp y_;

  /// Rectange width
  _Tp width_;

  /// Rectange height
  _Tp height_;
};

typedef Rect<int> Rect_i;
typedef Rect<float> Rect_f;
typedef Rect<double> Rect_d;
typedef Rect<uint32_t> Rect_u;

/**
 * Structure for 2-dimensional Point values.
 */
template <typename _Tp>
struct CVML_SDK_EXPORT Point {
  /**
   * Default constructor.
   */
  Point() : x_(0), y_(0) {}

  /**
   * Initializing constructor.
   *
   * @param x X coordinate
   * @param y Y coordinate
   */
  Point(_Tp x, _Tp y) : x_(x), y_(y) {}

  /// X cordinate of top left corner
  _Tp x_;

  /// Y cordinate of top left corner
  _Tp y_;
};

typedef Point<int> Point2i;
typedef Point<float> Point2f;
typedef Point<double> Point2d;

/**
 * Structure for 3-dimensional Point values.
 */
template <typename _Tp>
struct CVML_SDK_EXPORT Point3 {
  /**
   * Default constructor.
   */
  Point3() : x_(0), y_(0), z_(0) {}

  /**
   * Initializing constructor.
   *
   * @param x X coordinate
   * @param y Y coordinate
   * @param z Z coordinate
   */
  Point3(_Tp x, _Tp y, _Tp z) : x_(x), y_(y), z_(z) {}

  /// X cordinate of top left corner
  _Tp x_;

  /// Y cordinate of top left corner
  _Tp y_;

  /// Z cordinate of top left corner
  _Tp z_;
};

typedef Point3<int> Point3i;
typedef Point3<float> Point3f;
typedef Point3<double> Point3d;

/**
 * Structure for quadrilaterals at arbitrary angles.
 */
template <typename _Tp>
struct CVML_SDK_EXPORT Quad {
  /**
   * Default constructor.
   */
  Quad() = default;

  /**
   * Initializing constructor using Points.
   *
   * @param top_left coordinates of top left point
   * @param top_right coordinates of top right point
   * @param bottom_left coordinates of bottom left point
   * @param bottom_right coordinates of bottom right point
   */
  Quad(Point<_Tp> top_left, Point<_Tp> top_right, Point<_Tp> bottom_left, Point<_Tp> bottom_right)
      : top_left_(top_left),
        top_right_(top_right),
        bottom_left_(bottom_left),
        bottom_right_(bottom_right) {}

  /**
   * Initializing contructor using explict x and y values.
   *
   * @param x_tl top left x value
   * @param y_tl top left y value
   * @param x_tr top right x value
   * @param y_tr top right y value
   * @param x_bl bottom left x value
   * @param y_bl bottom left y value
   * @param x_br bottom right x value
   * @param y_br bottom right y value
   */
  Quad(_Tp x_tl, _Tp y_tl, _Tp x_tr, _Tp y_tr, _Tp x_bl, _Tp y_bl, _Tp x_br, _Tp y_br)
      : top_left_(Point<_Tp>(x_tl, y_tl)),
        top_right_(Point<_Tp>(x_tr, y_tr)),
        bottom_left_(Point<_Tp>(x_bl, y_bl)),
        bottom_right_(Point<_Tp>(x_br, y_br)) {}

  /// coordinates of top left point
  Point<_Tp> top_left_;
  /// coordinates of top left point
  Point<_Tp> top_right_;
  /// coordinates of top left point
  Point<_Tp> bottom_left_;
  /// coordinates of top left point
  Point<_Tp> bottom_right_;
};

/// Alias to older 'BoundingQuad' template definition.
template <class T>
using BoundingQuad = Quad<T>;

typedef Quad<int> Quadi;
typedef Quad<float> Quadf;
typedef Quad<double> Quadd;

/**
 * Fixed size array template class for multiple instances of class T.
 */
template <class T>
class CVML_SDK_EXPORT Array {
 public:
  /**
   * Default constructor.
   */
  Array() : v_(nullptr), size_(0) {}

  /**
   * Move constructor.
   *
   * @param other Source array
   */
  Array(Array&& other) noexcept : v_(std::move(other.v_)), size_(std::exchange(other.size_, 0)) {
    other.v_ = nullptr;
    other.size_ = 0;
  }

  /**
   * Move assignment operator.
   *
   * @param other Source array
   * @return Reference to updated object
   */
  Array& operator=(Array&& other) noexcept {
    if (this != &other) {
      if (v_) delete[] v_;
      size_ = other.size_;
      v_ = other.v_;
      other.v_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /**
   * Constructor that initilize required number of classes T.
   *
   * This function throws exceptions on errors.
   *
   * @param size Desired size of the array
   */
  explicit Array(size_t size) : v_{new T[size]}, size_(size) {}

  /**
   * Copy constructor.
   *
   * This function throws exceptions on errors.
   *
   * @param other Source array
   */
  Array(const Array& other) : v_{new T[other.size()]}, size_(other.size()) {
    for (size_t i = 0; i < other.size(); i++)  // copy elements
      v_[i] = other[i];
  }

  /**
   * Assignment operator.
   *
   * This operator throws exceptions on errors.
   *
   * @param other Source array
   * @return Reference to updated object
   */
  Array& operator=(const Array& other) {
    if (&other != this) {
      T* p = new T[other.size()];
      for (size_t i = 0; i != other.size(); ++i) p[i] = other[i];
      if (v_) delete[] v_;  // delete old elements
      v_ = p;
      size_ = other.size();
    }
    return *this;
  }

  /**
   * Read only operator[] for const objects.
   *
   * This operator throws exceptions on out-of-range subscripts.
   *
   * @param i Index to array
   * @return Array value
   */
  const T& operator[](size_t i) const {
    if (i >= size_) throw std::runtime_error("Invalid subscript access");
    return v_[i];
  }

  /**
   * operator[] for subscript access.
   *
   * This operator throws exceptions on out-of-range subscripts.
   *
   * @param i Index to array
   * @return Reference to array entry
   */
  T& operator[](size_t i) {
    if (i >= size_) throw std::runtime_error("Invalid subscript access");
    return v_[i];
  }

  /**
   * Returns the size of the array.
   *
   * @return Current size of the array
   */
  size_t size() const { return size_; }

  /**
   * Destructor
   */
  ~Array() {
    if (v_) delete[] v_;
    v_ = nullptr;
  }

 private:
  T* v_;         ///< Internal array storage
  size_t size_;  ///< Current size of the array
};

/**
 * Structure representing face location and landmarks for a single person.
 */
struct CVML_SDK_EXPORT Face {
  /// Constructor
  Face() : confidence_score_(0.f) {}

  /// Destructor
  virtual ~Face() {}

  /// Face bounding box
  Rect_i face_;

  /// Face detection confidence score
  float confidence_score_;

  /// Image coordinates of landmarks
  /// Facial landmarks are used to localize and represent important regions of the face, such as:
  /// mouth, eyes, eyebrows, nose
  Array<Point2i> landmarks_;

  /// Get bounding box
  Rect_i GetROI() { return face_; }
};

/// explicitly exporting template definition
template class CVML_SDK_EXPORT Array<Face>;

/**
 * Structure representing landmarks and bounding box for a single person.
 */
struct Person {
  /// Bounding box for this person
  Rect_i person_;

  /// Person detection confidence score
  float confidence_score_;

  /// Detected landmarks for this person
  Array<Point3i> landmarks_;

  /// Detected landmark scores for this person
  Array<float> landmark_scores_;

  /// Get bounding box
  Rect_i GetROI() { return person_; }
};

}  // namespace cvml
}  // namespace amd

#endif  // EDGEML_FEATURES_COMMON_FRAMEWORK_PUBLIC_INCLUDE_CVML_TYPES_H_
