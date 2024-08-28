#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <sstream>
// A struct that can storage data and info for each frame
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  // float max_fps;
  float fps;
  // std::string channel_name;
};

std::string to_string(const FrameInfo& frame_info) {
  std::stringstream ss;
  ss << "frame_info=>"
     << "channel_id: " << frame_info.channel_id
     << ", frame_id: " << frame_info.frame_id
     << ", fps: " << frame_info.fps;
    //  << ", max_fps: " << frame_info.max_fps;
  return ss.str();
}
