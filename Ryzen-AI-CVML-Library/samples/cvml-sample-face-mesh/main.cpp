/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <common-sample-utils.h>
#include <cvml-face-detector.h>
#include <cvml-face-mesh.h>

#include <filesystem>
#include <iomanip>
#include <vector>

#include "opencv2/opencv.hpp"

namespace ml = amd::cvml;

/**
 * Declare local class for sample variables and functions.
 */
class FaceMeshSample : public amd::cvml::sample::utils::RunFeatureClass {
 public:
  amd::cvml::FaceDetector* face_detector_{nullptr};  /// face detection feature
  amd::cvml::FaceDetector::FDModelType fd_model_type_{
      amd::cvml::FaceDetector::FDModelType::Precise};  /// face detection model type
  amd::cvml::FaceMesh* face_mesh_{nullptr};            /// face mesh feature
  std::string src_path_{};                             /// Input path/device
  // cppcheck-suppress duplInheritedMember
  std::string output_file_{};  /// Output file path/name
  /**
   * Run Face Mesh on single frame
   *
   * @param frame_rgb Incoming RGB frame
   * @return Output RGB frame
   */
  cv::Mat Feature(const cv::Mat& frame_rgb) override;
};

/**
 * Draw face mesh on the input RGB image with given mesh and head pose information
 *
 * @param rgb_img Original RGB image
 * @param mesh Mesh containing face landmarks
 * @param head_pose 3D head pose information
 * @return RGB image with face mesh drawn
 */
cv::Mat DrawFaceMesh(const cv::Mat& rgb_img, const ml::FaceMesh::Mesh& mesh) {
  // opencv point variables
  double point_color[] = {0.0, 0.0, 255.0};
  int scale_size = 360;

  // Going through all the detected landmarks
  double highest_point_y = mesh.landmarks_[0].y_;

  for (size_t i = 0; i < mesh.landmarks_.size(); i++) {
    auto landmark = mesh.landmarks_[i];
    if (landmark.y_ < highest_point_y) {
      highest_point_y = landmark.y_;
    }
    cv::Point point(static_cast<int>(landmark.x_), static_cast<int>(landmark.y_));

    cv::circle(rgb_img, point, static_cast<int>(rgb_img.rows / scale_size),
               cv::Scalar(point_color[0], point_color[1], point_color[2]),
               static_cast<int>(rgb_img.rows / scale_size), cv::FILLED);
  }

  return rgb_img;
}

/**
 * Run FaceMesh algorithm on given frame and draw the meshes on the input frame
 *
 * @param frame_rgb Input image frame in RGB format
 * @return Output image frame with face meshes drawn
 */
cv::Mat FaceMeshSample::Feature(const cv::Mat& frame_rgb) {
  cv::Mat frame_out = frame_rgb;
  std::vector<ml::FaceMesh::Mesh> meshes;
  if (face_mesh_ == nullptr) {
    throw std::runtime_error("Incomplete local data");
  }

  // convert to amd::Image
  ml::Image amd_img(ml::Image::Format::kRGB, ml::Image::DataType::kUint8, frame_rgb.cols,
                    frame_rgb.rows, frame_rgb.data);

  // start the clock
  auto faces = face_detector_->Detect(amd_img);

  double small_face_size_thr = 0.05;  // small face size relative to the frame size

  for (size_t i = 0; i < faces.size(); ++i) {
    auto face_width = faces[i].face_.width_;
    if (static_cast<double>(face_width) / static_cast<double>(frame_rgb.cols) <
        small_face_size_thr)  // face is too small
      continue;
    auto mesh = face_mesh_->CreateMesh(amd_img, faces[i]);
    meshes.push_back(mesh);
  }

  for (size_t i = 0; i < meshes.size(); ++i) {
    frame_out = DrawFaceMesh(frame_out, meshes[i]);
  }

  return frame_out;
}

/**
 * Print usage message for the command-line arguments
 */
void PrintUsageMessage() {
  std::cout
      << "Usage: "
      << "cvml-sample-facemesh "
      << "[-i path_to_video/image] [-h] [-o output "
         "image/video filename] [-?]\n"
         "option\n"
         "    -i\tSpecify an input image or video\n"
         "    -o\tSpecify output image or video file name e.g .mp4/.jpg\n"
         "    -fd\tSpecify face detection model (precise/fast). Precise is the default (Optional)\n"
         "    -h\tShow usage\n"
         "    -?\tShow usage\n"
         "\n"
         "  Opens the specified input device and runs the feature against it. Results are\n"
         "  displayed in an output window and optionally saved to a video file. If no\n"
         "  arguments are provided, the application attempts to capture input from\n"
         "  camera index 0"
      << std::endl;
}

/**
 * Parse command-line arguments and update local data class accordingly
 *
 * @param local_data Pointer to the local data object to store parsed arguments
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return true if arguments are parsed successfully, false otherwise
 */
bool ParseArguments(FaceMeshSample* local_data, int argc, char** const argv) {
  if (local_data == nullptr || argv == nullptr) {
    return false;
  }

  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "-i" && ((i + 1) < argc)) {
      local_data->src_path_ = std::string(argv[i + 1]);
    } else if (std::string(argv[i]) == "-o" && ((i + 1) < argc)) {
      local_data->output_file_ = std::string(argv[i + 1]);
    } else if (std::string(argv[i]) == "-fd" && ((i + 1) < argc)) {
      if (std::string(argv[i + 1]) == "fast") {
        local_data->fd_model_type_ = amd::cvml::FaceDetector::FDModelType::Fast;
      } else if (std::string(argv[i + 1]) == "precise") {
        local_data->fd_model_type_ = amd::cvml::FaceDetector::FDModelType::Precise;
      } else {
        std::cout << "Invalid Face Detection model type. Defaulting to Precise" << std::endl;
      }
    } else if (std::string(argv[i]) == "-h") {
      PrintUsageMessage();
      return false;
    }
  }

  return true;
}

/**
 * Main function - initializes and runs the FaceMesh sample
 */
int main(int argc, char** const argv) {
  FaceMeshSample fm_sample;
  try {
    // parse arguments
    bool parse_ok = ParseArguments(&fm_sample, argc, argv);
    if (!parse_ok) return -1;

    // create CVML SDK context for the feaeture
    std::shared_ptr<ml::Context> context(ml::CreateContext(), [](ml::Context* ctx) {
      if (ctx) ctx->Release();
    });
    if (!context) {
      std::cerr << "Failed to create context" << std::endl;
      return 1;
    }

    fm_sample.SetContextStreamingModeBySrc(context.get(), fm_sample.src_path_);

    // initialize FaceDetector class
    ml::FaceDetector fd(context.get(), fm_sample.fd_model_type_);
    fm_sample.face_detector_ = &fd;

    // intialize FaceMesh class
    ml::FaceMesh fm(context.get());
    fm.SetMaxNumFaces(-1);
    fm_sample.face_mesh_ = &fm;

    fm_sample.RunFeature(fm_sample.src_path_, fm_sample.output_file_, "AMD Face Mesh");
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
