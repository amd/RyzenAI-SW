#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace retinaface {
struct Result {
  struct Face {
    std::vector<float> bbox;  // x, y, width, height
    float score;
  };
  std::vector<Face> faces;
};

struct SelectedOutput {
  float score;
  int index;
  std::vector<float> box;
  std::vector<float> box_decoded;
  friend inline bool operator<(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return lhs.score < rhs.score;
  }

  friend inline bool operator>(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return lhs.score > rhs.score;
  }
};
std::vector<std::vector<float>> generate_anchors() {
  int width = 320;
  int height = 320;
  std::vector<std::vector<int>> min_sizes{{16, 32}, {64, 128}, {256, 512}};
  std::vector<int> steps{8, 16, 32};
  std::vector<std::vector<int>> feat_maps;
  int anchor_cnt = 0;
  for (auto k = 0u; k < min_sizes.size(); ++k) {
    feat_maps.emplace_back(
        std::vector<int>{int(std::ceil(((float)height) / steps[k])),
                         int(std::ceil(((float)width) / steps[k]))});
    anchor_cnt += feat_maps[k][0] * feat_maps[k][1] *
                  static_cast<int>(min_sizes[k].size());
  }
  // std::cout << "anchor_cnt:" << anchor_cnt;
  std::vector<std::vector<float>> anchors(anchor_cnt,
                                          std::vector<float>(4, 0.f));
  auto index = 0;
  for (auto k = 0u; k < min_sizes.size(); ++k) {
    auto x_step = ((float)steps[k]) / width;
    auto y_step = ((float)steps[k]) / height;
    auto x_start = 0.5f * x_step;
    auto y_start = 0.5f * y_step;
    for (auto i = 0; i < feat_maps[k][0]; ++i) {
      for (auto j = 0; j < feat_maps[k][1]; ++j) {
        for (auto min_size : min_sizes[k]) {
          auto s_kx = (float)min_size / width;
          auto s_ky = (float)min_size / height;
          anchors[index][0] = x_start + j * x_step;
          anchors[index][1] = y_start + i * y_step;
          anchors[index][2] = s_kx;
          anchors[index][3] = s_ky;
          index++;
        }
      }
    }
  }
  return anchors;
}

void decode(const float* src, const float* anchor, float* dst) {
  std::vector<float> variance{0.1f, 0.2f};
  dst[0] = anchor[0] + variance[0] * anchor[2] * src[0];
  dst[1] = anchor[1] + variance[0] * anchor[3] * src[1];
  dst[2] = anchor[2] * std::exp(src[2] * variance[1]);
  dst[3] = anchor[3] * std::exp(src[3] * variance[1]);
  dst[0] -= dst[2] / 2;
  dst[1] -= dst[3] / 2;
  dst[2] += dst[0];
  dst[3] += dst[1];
}
std::vector<std::vector<SelectedOutput>> select(
    float* loc_ptr, float* conf_ptr, const std::vector<int64_t>& conf_shape,
    int64_t batch_size, float score_thresh) {
  auto batch = batch_size;
  auto feat_map_size = conf_shape[1];

  auto conf_last_dim = conf_shape[2];
  static auto loc_last_dim = 4;
  std::vector<std::vector<SelectedOutput>> batch_result(batch);
  for (auto b = 0; b < batch; ++b) {
    auto& result = batch_result[b];
    result.reserve(200);
    auto cur_conf_ptr = conf_ptr + b * feat_map_size * conf_last_dim;
    auto cur_loc_ptr = loc_ptr + b * feat_map_size * loc_last_dim;
    for (auto i = 0; i < feat_map_size; ++i) {
      if (cur_conf_ptr[i * conf_last_dim + 1] > score_thresh) {
        auto index = i;
        auto score = cur_conf_ptr[i * conf_last_dim + 1];
        auto box = std::vector<float>(cur_loc_ptr + i * loc_last_dim,
                                      cur_loc_ptr + (i + 1) * loc_last_dim);

        auto select = SelectedOutput{score, index, box, box};
        result.emplace_back(select);
      }
    }
  }
  return batch_result;
}

std::vector<SelectedOutput> topK(const std::vector<SelectedOutput>& input,
                                 int k) {
  // assert(k >= 0);
  int size = (int)input.size();
  int num = std::min(size, k);
  std::vector<SelectedOutput> result(input.begin(), input.begin() + num);
  std::make_heap(result.begin(), result.begin() + num, std::greater<>());
  for (auto i = num; i < size; ++i) {
    if (input[i] > result[0]) {
      std::pop_heap(result.begin(), result.end(), std::greater<>());
      result[num - 1] = input[i];
    }
  }

  for (auto i = 0; i < num; ++i) {
    std::pop_heap(result.begin(), result.begin() + num - i, std::greater<>());
  }
  // std::stable_sort(result.begin(), result.end(), compare);
  return result;
}

static float overlap(float x1, float w1, float x2, float w2) {
  float left = std::max(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
  float right = std::min(x1 + w1 / 2.0f, x2 + w2 / 2.0f);
  return right - left;
}

float cal_iou_xywh(std::vector<float>& box, std::vector<float>& truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0f / union_area;
}

float cal_iou_xyxy(std::vector<float>& box, std::vector<float>& truth) {
  float box_w = box[2] - box[0];
  float box_h = box[3] - box[1];
  float truth_w = truth[2] - truth[0];
  float truth_h = truth[3] - truth[1];
  float w = overlap(box[0], box_w, truth[0], truth_w);
  float h = overlap(box[1], box_h, truth[1], truth_h);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0f / union_area;
}

float cal_iou_yxyx(std::vector<float> box, std::vector<float> truth) {
  float box_h = box[2] - box[0];
  float box_w = box[3] - box[1];
  float truth_h = truth[2] - truth[0];
  float truth_w = truth[3] - truth[1];
  float h = overlap(box[0], box_h, truth[0], truth_h);
  float w = overlap(box[1], box_w, truth[1], truth_w);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0f / union_area;
}

std::vector<SelectedOutput> nms(std::vector<SelectedOutput>& candidates,
                                float nms_thresh, float score_thresh,
                                int max_output_num, bool need_sort) {
  std::vector<SelectedOutput> result;
  auto compare = [](const SelectedOutput& l, const SelectedOutput& r) {
    return l.score >= r.score;
  };

  // Todo: sort
  if (need_sort) {
    std::stable_sort(candidates.begin(), candidates.end(), compare);
  }

  // nms;
  auto size = candidates.size();
  std::vector<bool> exist_box(size, true);
  for (size_t i = 0; i < size; ++i) {
    if (!exist_box[i]) {
      continue;
    }
    if (candidates[i].score < score_thresh) {
      exist_box[i] = false;
      continue;
    }
    result.push_back(candidates[i]);
    for (size_t j = i + 1; j < size; ++j) {
      if (!exist_box[j]) {
        continue;
      }
      if (candidates[j].score < score_thresh) {
        exist_box[j] = false;
        continue;
      }
      float overlap = 0.0;
      overlap =
          cal_iou_xyxy(candidates[i].box_decoded, candidates[j].box_decoded);
      if (overlap >= nms_thresh) {
        exist_box[j] = false;
      }
    }
  }

  if (result.size() > (unsigned int)max_output_num) {
    result.resize(max_output_num);
  }
  return result;
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
Image show_reusult(Image& image, const Result& result) {
  for (auto i = 0u; i < result.faces.size(); ++i) {
    auto& face = result.faces[i];
    int x = static_cast<int>(face.bbox[0] * image.cols);
    int y = static_cast<int>(face.bbox[1] * image.rows);
    int w = static_cast<int>((face.bbox[2] - face.bbox[0]) * image.cols);
    int h = static_cast<int>((face.bbox[3] - face.bbox[1]) * image.rows);
    cv::rectangle(image, cv::Point(x, y), cv::Point(x + w, y + h),
                  cv::Scalar(0, 255, 255), 2, 1, 0);
  }
  cv::putText(image, std::string("RETINAFACE"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}
}  // namespace retinaface
class Retinaface : public Model {
 public:
  Retinaface() {
    anchors_ = retinaface::generate_anchors();
    means_ = std::vector<float>{104.0f, 117.0f, 123.0f};
    scales_ = std::vector<float>{1, 1, 1};
  }
  virtual ~Retinaface() {}
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
    for (auto index = 0; index < batch_size; ++index) {
      auto resize_image = retinaface::preprocess_one(images[index], size);
      retinaface::set_input_image_bgr(
          resize_image, input_data_0.data() + batch_element_size * index,
          means_, scales_);
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    static int pre_nms_num = 1000;
    static auto nms_thresh = 0.2f;
    static int max_output_num = 200;
    static auto score_thresh = 0.048f;

    auto ouput_shape_conf = session_->get_output_shape_from_tensor(1);
    auto batch_size = images.size();
    std::vector<retinaface::Result> batch_results(batch_size);
    auto loc_ptr = session_->get_output(0);
    auto conf_ptr = session_->get_output(1);
    // 1. select all scores over score_thresh
    auto batch_selected = retinaface::select(
        loc_ptr, conf_ptr, ouput_shape_conf, batch_size, score_thresh);

    for (auto b = 0; b < batch_size; ++b) {
      // 2. topk
      auto topk_selected = topK(batch_selected[b], pre_nms_num);
      // 3. decode
      for (auto i = 0u; i < topk_selected.size(); ++i) {
        auto index = topk_selected[i].index;
        // 3.1 decode box
        retinaface::decode(topk_selected[i].box.data(), anchors_[index].data(),
                           topk_selected[i].box_decoded.data());
      }
      // 4. nms
      auto nms_result = retinaface::nms(topk_selected, nms_thresh, score_thresh,
                                        max_output_num, false);
      //  5. make result
      batch_results[b].faces.resize(nms_result.size());
      for (auto i = 0u; i < nms_result.size(); ++i) {
        batch_results[b].faces[i].score = nms_result[i].score;
        batch_results[b].faces[i].bbox = nms_result[i].box_decoded;
      }
    }

    //   return batch_results;
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = batch_results[index];
      auto image = images[index];
      image_results.push_back(retinaface::show_reusult(image, result));
    }
    return image_results;
  }

 private:
  std::vector<float> means_;
  std::vector<float> scales_;
  std::vector<std::vector<float>> anchors_;
};
REGISTER_MODEL(retinaface, Retinaface)