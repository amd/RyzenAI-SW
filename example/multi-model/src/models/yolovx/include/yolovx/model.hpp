#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace yolovx {

static float overlap(float x1, float w1, float x2, float w2) {
  float left = (float)std::max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = (float)std::min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(const std::vector<float>& box,
                     const std::vector<float>& truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return (float)(inter_area * 1.0 / union_area);
}

static void applyNMS(const std::vector<std::vector<float>>& boxes,
                     const std::vector<float>& scores, const float nms,
                     const float conf, std::vector<size_t>& res) {
  const size_t count = boxes.size();
  std::vector<std::pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  sort(order.begin(), order.end(),
       [](const std::pair<float, size_t>& ls,
          const std::pair<float, size_t>& rs) { return ls.first > rs.first; });
  std::vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  std::vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

static void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om,
                      float& scale) {
  scale = std::min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}
cv::Mat preprocess_one(const cv::Mat& image, cv::Size size) {
  cv::Mat resized_image;
  if (image.size() != size) {
    cv::resize(image, resized_image, cv::Size(size.height, size.width));
  } else {
    resized_image = image;
  }
  return resized_image;
}
void set_input_image_internal(const cv::Mat& image, float* data,
                              const std::vector<float>& mean,
                              const std::vector<float>& scale, bool btrans) {
  // BGR->RGB (maybe) and HWC->CHW
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = btrans ? abs(c - 2) : c;
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scale[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}
void set_input_image_bgr(const cv::Mat& image, float* data,
                         const std::vector<float>& mean,
                         const std::vector<float>& scale) {
  return set_input_image_internal(image, data, mean, scale, false);
}
// return value
struct Result {
  /**
   *@struct BoundingBox
   *@brief Struct of detection result with an object.
   */
  struct BoundingBox {
    /// Classification.
    int label;
    /// Confidence. The value ranges from 0 to 1.
    float score;
    /// (x0,y0,x1,y1). x0, x1 Range from 0 to the input image columns.
    /// y0,y1. Range from 0 to the input image rows.
    std::vector<float> box;
  };
  /// All objects, The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};
inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }
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
Image show_reusult(Image& image, const Result& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
                  cv::Scalar(yolovx_helper::b[label], yolovx_helper::g[label],
                             yolovx_helper::r[label]),
                  2, 1, 0);
    cv::putText(
        image, yolovx_helper::classes[label] + " " + std::to_string(res.score),
        cv::Point(box[0] + 5, box[1] + 10), cv::FONT_HERSHEY_SIMPLEX, 0.4,
        cv::Scalar(yolovx_helper::b[label], yolovx_helper::g[label],
                   yolovx_helper::r[label]),
        1, 4);
  }
  cv::putText(image, std::string("YOLOVX"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}
}  // namespace yolovx
class Yolovx : public Model {
 public:
  Yolovx() {}
  virtual ~Yolovx() {}
  void init(const Config& config) {
    Model::init(config);
    CONFIG_GET(config, float, thresh, "confidence_threshold")
    conf_thresh = thresh;
    conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
    output_shapes_.resize(output_tensor_size);
    output_shapes_[0] = session_->get_output_shape_from_model(0);
    output_shapes_[1] = session_->get_output_shape_from_model(1);
    output_shapes_[2] = session_->get_output_shape_from_model(2);
    stride[0] = 8;
    stride[1] = 16;
    stride[2] = 32;
  }
  void preprocess(const std::vector<Image>& images) override {
    std::vector<float>& input_data_0 = session_->get_input(0);
    std::vector<int64_t>& input_shape_0 = session_->get_input_shape(0);
    int batch_size = images.size();
    input_shape_0[0] = batch_size;
    int64_t total_number_elements =
        std::accumulate(input_shape_0.begin(), input_shape_0.end(), int64_t{1},
                        std::multiplies<int64_t>());
    input_data_0.resize(size_t(total_number_elements));
    auto channel = input_shape_0[1];
    auto height = input_shape_0[2];
    auto width = input_shape_0[3];
    auto batch_element_size = channel * height * width;
    auto size = cv::Size((int)width, (int)height);
    scales.resize(batch_size);
    for (auto index = 0; index < batch_size; ++index) {
      cv::Mat resized_image;
      float& scale = scales[index];
      yolovx::letterbox(images[index], width, height, resized_image, scale);
      yolovx::set_input_image_bgr(
          resized_image, input_data_0.data() + batch_size * index,
          std::vector<float>{0, 0, 0}, std::vector<float>{1, 1, 1});
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    auto batch_size = images.size();
    std::vector<yolovx::Result> results;
    for (auto index = 0; index < batch_size; ++index) {
      results.emplace_back(postprocess_one(index));
    }
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = results[index];
      auto image = images[index];
      image_results.push_back(yolovx::show_reusult(image, result));
    }
    return image_results;
  }
  yolovx::Result postprocess_one(int idx) {
    std::vector<std::vector<float>> boxes;

    int conf_box = 5 + num_classes;

    for (int i = 0; i < output_tensor_size; i++) {
      // 3 output layers  // 85x52x52  85x26x26 85x13x13:
      int ca = output_shapes_[i][1];
      int ha = output_shapes_[i][2];
      int wa = output_shapes_[i][3];
      //   if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
      //     LOG(INFO) << "channel=" << ca << ", height=" << ha << ", width=" <<
      //     wa
      //               << ", stride=" << stride[i] << ", conf=" << conf_thresh
      //               << ", idx=" << idx << endl;
      //   }
      boxes.reserve(boxes.size() + ha * wa);
      float* output_ptr = session_->get_output(i);
#define POS(C) ((C) * ha * wa + h * wa + w)
      for (int h = 0; h < ha; ++h) {
        for (int w = 0; w < wa; ++w) {
          for (int c = 0; c < anchor_cnt; ++c) {
            float score =
                output_ptr[POS(c * conf_box + 4) + idx * ca * ha * wa];
            if (score < conf_desigmoid) continue;
            std::vector<float> box(6);
            std::vector<float> out(4);
            for (int index = 0; index < 4; index++) {
              out[index] =
                  output_ptr[POS(c * conf_box + index) + idx * ca * ha * wa];
            }
            box[0] = (w + out[0]) * stride[i];
            box[1] = (h + out[1]) * stride[i];
            box[2] = exp(out[2]) * stride[i];
            box[3] = exp(out[3]) * stride[i];
            float obj_score = yolovx::sigmoid(score);
            auto conf_class_desigmoid = -logf(obj_score / conf_thresh - 1.0f);
            int max_p = -1;
            box[0] = box[0] - box[2] * 0.5;
            box[1] = box[1] - box[3] * 0.5;
            for (int p = 0; p < num_classes; p++) {
              float cls_score =
                  output_ptr[POS(c * conf_box + 5 + p) + idx * ca * ha * wa];
              if (cls_score < conf_class_desigmoid) continue;
              max_p = p;
              conf_class_desigmoid = cls_score;
            }
            if (max_p != -1) {
              box[4] = max_p;
              box[5] = obj_score * yolovx::sigmoid(conf_class_desigmoid);
              boxes.push_back(box);
            }
          }
        }
      }
    }
    /* Apply the computation for NMS */
    std::vector<std::vector<std::vector<float>>> boxes_for_nms(num_classes);
    std::vector<std::vector<float>> scores(num_classes);

    for (const auto& box : boxes) {
      boxes_for_nms[box[4]].push_back(box);
      scores[box[4]].push_back(box[5]);
    }

    std::vector<std::vector<float>> res;
    for (auto i = 0; i < num_classes; i++) {
      std::vector<size_t> result_k;
      yolovx::applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh,
                       result_k);
      res.reserve(res.size() + result_k.size());
      transform(result_k.begin(), result_k.end(), back_inserter(res),
                [&](auto& k) { return boxes_for_nms[i][k]; });
    }

    std::vector<yolovx::Result::BoundingBox> results;
    for (const auto& r : res) {
      if (r[5] > conf_thresh) {
        yolovx::Result::BoundingBox result;
        result.score = r[5];
        result.label = r[4];
        result.box.resize(4);
        result.box[0] = r[0] / scales[idx];
        result.box[1] = r[1] / scales[idx];
        result.box[2] = result.box[0] + r[2] / scales[idx];
        result.box[3] = result.box[1] + r[3] / scales[idx];
        results.push_back(result);
      }
    }
    return yolovx::Result{results};
  }

 private:
  int output_tensor_size{3};
  std::vector<float> scales;
  float stride[3];
  float conf_thresh{0.f};
  float conf_desigmoid{0.f};
  float nms_thresh{0.65f};
  int num_classes{80};
  int anchor_cnt{1};
  std::vector<std::vector<int64_t>> output_shapes_;
};
REGISTER_MODEL(yolovx, Yolovx)