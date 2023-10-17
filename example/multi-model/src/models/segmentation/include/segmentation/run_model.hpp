#pragma once
#include <vector>

#include "segmentation.hpp"
cv::Mat process_result_segmentation(cv::Mat& image,
                                    const SegmentationResult& result,
                                    bool is_jpeg) {
  cv::Mat resized_seg;
  cv::resize(result.segmentation,resized_seg,image.size());
  cv::Mat thresholded_seg;
  cv::threshold(resized_seg,thresholded_seg,1,255,cv::THRESH_BINARY);
  cv:Mat colored_seg;
  cv::applyColorMap(thresholded_seg,colored_seg,cv::COLORMAP_JET);
  cv::Mat mixed_image;
  cv::addWeighted(image,0.5,colored_seg,0.5,0.0,mixed_image);
  cv::putText(mixed_image, std::string("SEGMENTATION"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return mixed_image;
}