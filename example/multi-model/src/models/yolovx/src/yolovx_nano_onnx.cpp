#include "yolovx/yolovx_nano_onnx.hpp"

void YolovxnanoOnnx::preprocess(const cv::Mat& image, int idx, float& scale) {
  cv::Mat resized_image;
  onnx_yolovx::letterbox(image, sWidth, sHeight, resized_image, scale);
  set_input_image_bgr(resized_image,
                      input_tensor_values.data() + batch_size * idx,
                      std::vector<float>{0, 0, 0}, std::vector<float>{1, 1, 1});
  return;
}

// preprocess
void YolovxnanoOnnx::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  scales.resize(real_batch);
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index, scales[index]);
  }
  return;
}

inline float sigmoid(float src) { return (1.0f / (1.0f + exp(-src))); }

// postprocess
YolovxnanoOnnxResult YolovxnanoOnnx::postprocess(int idx) {
  vector<vector<float>> boxes;

  int conf_box = 5 + num_classes;

  for (int i = 0; i < output_tensor_size; i++) {
    // 3 output layers  // 85x52x52  85x26x26 85x13x13:
    int ca = output_shapes_[i][1];
    int ha = output_shapes_[i][2];
    int wa = output_shapes_[i][3];
    if (ENV_PARAM(ENABLE_YOLO_DEBUG)) {
      LOG(INFO) << "channel=" << ca << ", height=" << ha << ", width=" << wa
                << ", stride=" << stride[i] << ", conf=" << conf_thresh
                << ", idx=" << idx << endl;
    }
    boxes.reserve(boxes.size() + ha * wa);
#define POS(C) ((C)*ha * wa + h * wa + w)
    for (int h = 0; h < ha; ++h) {
      for (int w = 0; w < wa; ++w) {
        for (int c = 0; c < anchor_cnt; ++c) {
          float score =
              output_tensor_ptr[i][POS(c * conf_box + 4) + idx * ca * ha * wa];
          if (score < conf_desigmoid) continue;
          vector<float> box(6);
          vector<float> out(4);
          for (int index = 0; index < 4; index++) {
            out[index] = output_tensor_ptr[i][POS(c * conf_box + index) +
                                              idx * ca * ha * wa];
          }
          box[0] = (w + out[0]) * stride[i];
          box[1] = (h + out[1]) * stride[i];
          box[2] = exp(out[2]) * stride[i];
          box[3] = exp(out[3]) * stride[i];
          float obj_score = sigmoid(score);
          auto conf_class_desigmoid = -logf(obj_score / conf_thresh - 1.0f);
          int max_p = -1;
          box[0] = box[0] - box[2] * 0.5;
          box[1] = box[1] - box[3] * 0.5;
          for (int p = 0; p < num_classes; p++) {
            float cls_score = output_tensor_ptr[i][POS(c * conf_box + 5 + p) +
                                                   idx * ca * ha * wa];
            if (cls_score < conf_class_desigmoid) continue;
            max_p = p;
            conf_class_desigmoid = cls_score;
          }
          if (max_p != -1) {
            box[4] = max_p;
            box[5] = obj_score * sigmoid(conf_class_desigmoid);
            boxes.push_back(box);
          }
        }
      }
    }
  }
  /* Apply the computation for NMS */
  vector<vector<vector<float>>> boxes_for_nms(num_classes);
  vector<vector<float>> scores(num_classes);

  for (const auto& box : boxes) {
    boxes_for_nms[box[4]].push_back(box);
    scores[box[4]].push_back(box[5]);
  }

  vector<vector<float>> res;
  for (auto i = 0; i < num_classes; i++) {
    vector<size_t> result_k;
    onnx_yolovx::applyNMS(boxes_for_nms[i], scores[i], nms_thresh, conf_thresh,
                          result_k);
    res.reserve(res.size() + result_k.size());
    transform(result_k.begin(), result_k.end(), back_inserter(res),
              [&](auto& k) { return boxes_for_nms[i][k]; });
  }

  vector<YolovxnanoOnnxResult::BoundingBox> results;
  for (const auto& r : res) {
    if (r[5] > conf_thresh) {
      YolovxnanoOnnxResult::BoundingBox result;
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
  return YolovxnanoOnnxResult{results};
}

std::vector<YolovxnanoOnnxResult> YolovxnanoOnnx::postprocess() {
  std::vector<YolovxnanoOnnxResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

YolovxnanoOnnx::YolovxnanoOnnx(const std::string& model_name,
                               const float conf_thresh_, const OnnxConfig& onnx_config)
    : OnnxTask(model_name,onnx_config) {
  int total_number_elements = calculate_product(input_shapes_[0]);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  channel = input_shapes_[0][1];
  sHeight = input_shapes_[0][2];
  sWidth = input_shapes_[0][3];
  batch_size = channel * sHeight * sWidth;
  input_tensor_ptr.resize(1);
  output_tensor_ptr.resize(output_tensor_size);
  conf_thresh = conf_thresh_;
  conf_desigmoid = -logf(1.0f / conf_thresh - 1.0f);
}

YolovxnanoOnnxResult YolovxnanoOnnx::run(const cv::Mat& mats) {
  return run(vector<cv::Mat>(1, mats))[0];
}

std::vector<YolovxnanoOnnxResult> YolovxnanoOnnx::run(
    const std::vector<cv::Mat>& mats) {
  preprocess(mats);
  Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Value::CreateTensor<float>(
        info, input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0].data(), input_shapes_[0].size());

  } else {
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        info, input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0].data(), input_shapes_[0].size()));
  }

  run_task(input_tensors, output_tensors);
  for (int i = 0; i < output_tensor_size; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }

  std::vector<YolovxnanoOnnxResult> ret = postprocess();
  return ret;
}
