#pragma once
#include <vector>

#include "yolovx_nano_onnx.hpp"
namespace yolovx_helper {
inline std::vector<int> b = {
    144, 89,  30,  3,   16,  69,  237, 54,  4,   89,  15,  141, 87,  65,
    118, 150, 117, 119, 19,  90,  33,  53,  39,  11,  228, 93,  40,  164,
    46,  228, 48,  163, 114, 182, 232, 103, 21,  49,  116, 54,  62,  160,
    159, 163, 212, 117, 237, 169, 94,  16,  79,  124, 68,  154, 190, 70,
    203, 178, 64,  55,  206, 79,  25,  230, 43,  52,  255, 230, 116, 3,
    135, 175, 78,  158, 254, 50,  161, 223, 204, 108, 63};
inline std::vector<int> g = {
    246, 80,  103, 0,   134, 12,  197, 233, 7,   31,  118, 88,  161, 221,
    236, 228, 71,  81,  26,  143, 188, 5,   154, 6,   152, 224, 39,  126,
    196, 216, 177, 149, 161, 65,  192, 133, 5,   254, 151, 66,  158, 117,
    193, 173, 56,  252, 5,   197, 37,  143, 131, 220, 229, 73,  176, 60,
    124, 46,  36,  44,  200, 92,  126, 216, 248, 151, 189, 162, 135, 145,
    244, 158, 135, 188, 34,  33,  99,  10,  146, 107, 139};
inline std::vector<int> r = {
    100, 9,   243, 216, 240, 65,  11,  32,  124, 164, 27,  14,  83,  143,
    49,  27,  59,  131, 138, 184, 178, 176, 73,  16,  10,  226, 189, 30,
    150, 31,  126, 95,  144, 13,  205, 128, 226, 39,  158, 12,  202, 155,
    210, 255, 250, 106, 93,  25,  56,  36,  51,  20,  134, 108, 120, 41,
    118, 163, 162, 55,  122, 160, 89,  173, 240, 218, 187, 150, 231, 78,
    177, 184, 160, 246, 36,  11,  152, 221, 108, 249, 216};
inline std::vector<std::string> classes{
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};
cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}
}  // namespace yolovx_helper
// for YolovxnanoOnnx
static cv::Mat process_result_yolovx(cv::Mat& image,
                                     const YolovxnanoOnnxResult& result,
                                     bool is_jpeg) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
                  cv::Scalar(yolovx_helper::b[label], yolovx_helper::g[label],
                             yolovx_helper::r[label]),
                  2, 1, 0);
    cv::putText(
        image, yolovx_helper::classes[label] + " " + std::to_string(res.score),
        cv::Point(box[0] + 5, box[1] + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(yolovx_helper::b[label], yolovx_helper::g[label],
                   yolovx_helper::r[label]),
        2, 4);
  }
  cv::putText(image, std::string("YOLOVX"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}