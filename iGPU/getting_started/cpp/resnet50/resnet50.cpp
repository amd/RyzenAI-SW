/***********************************************************************************
MIT License

Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ************************************************************************************/
#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <algorithm> // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#if _WIN32
extern "C" {
#include "util/getopt.h"
}
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <cstdlib>
#include <stdlib.h>

using namespace std;


static cv::Mat read_image(const std::string files);
static cv::Mat croppedImage(const cv::Mat& image, int height, int width);
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size);
static void set_input_image(const cv::Mat& image, Ort::Float16_t* data);
static std::vector<float> softmax(float* data, int64_t size);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
    int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
static const char* lookup(int index);

std::vector<float> convertToFloat32(const Ort::Float16_t* input, size_t size) {
    std::vector<float> output(size);
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
    return output;
}

// preprocess
static void preprocess_resnet(const string file,
    std::vector<Ort::Float16_t>& input_tensor_values,
    std::vector<int64_t>& input_shape) {
    auto channel = input_shape[1];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto size = cv::Size((int)width, (int)height);
    auto image = read_image(file);
    cv::Mat resized_image;
    cv::resize(image, image,
        cv::Size(224, 224));
    try
    {
        set_input_image(image, input_tensor_values.data());
    }
    catch (const std::exception& exception) {
        cout << "ERROR running set_input_image: " << exception.what() << endl;
        exit(-1);
    }

}

// postprocess
static string postprocess_resnet(const string file,
    Ort::Value& output_tensor) {
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto channel = output_shape[1];
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<Ort::Float16_t>();
    std::vector<float> float_output_tensor = convertToFloat32(output_tensor_ptr, channel);
    auto softmax_output = softmax(float_output_tensor.data(), channel);
    auto tb_top5 = topk(softmax_output, 5);
    //print_topk(tb_top5);
    auto top1 = tb_top5[0];
    auto cls = std::string("") + lookup(top1.first) + " prob. " +
        std::to_string(top1.second);
    return lookup(top1.first);
}

#define CHECK_STATUS_OK(expr)                                                  \
  do {                                                                         \
    Status _tmp_status = (expr);                                               \
    CHECK(_tmp_status.IsOK()) << _tmp_status;                                  \
  } while (0)


// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

static int calculate_product(const std::vector<int64_t>& v) {
    int total = 1;
    for (auto& i : v)
        total *= (int)i;
    return total;
}

static void usage() {
    std::cout << "usage: resnet50 <path to onnx model> <provider> <path to image>"<< std::endl;
}

int main(int argc, char* argv[]) {

    vector<pair<string, string>> results;
    int opt = 0;
    int64_t batch_number = 1;
    auto model_name = strconverter.from_bytes(std::string(argv[optind]));
    cout << "model name:" << std::string(argv[optind]) << endl;
    auto ep = std::string(argv[optind + 1]);
    auto curr_file = std::string(argv[optind + 2]);
    cout << "ep:" << ep << endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50");
    auto session_options = Ort::SessionOptions();

    auto config_key = std::string{ "config_file" };
    auto cache_dir = std::filesystem::current_path().string();

    if (ep == "dml")
    {
        OrtApi const& ortApi = Ort::GetApi();
        OrtDmlApi const* ortDmlApi = nullptr;
        ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<void const**>(&ortDmlApi));

        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // For DML EP
        session_options.DisableMemPattern(); // For DML EP

        int deviceIndex = 0;

        ortDmlApi->SessionOptionsAppendExecutionProvider_DML(session_options, deviceIndex);
        cout << "Finished appending DML session options" << endl;

    }
    auto session = Ort::Session(env, model_name.data(), session_options);
    // print name/shape of inputs and outputs
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_count = session.GetInputCount();
    auto input_names = std::vector<const char*>();
    auto input_names_ptr = std::vector<Ort::AllocatedStringPtr>();
    auto input_shapes = std::vector<std::vector<int64_t>>();
    input_shapes.reserve(input_count);
    input_names_ptr.reserve(input_count);
    input_names.reserve(input_count);
    std::cout << "Input Node Name/Shape (" << input_count << "):" << std::endl;
    for (size_t i = 0; i < input_count; i++)
    {
        input_shapes.push_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names.push_back(name.get());
        input_names_ptr.push_back(std::move(name));
        std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
    }

    auto output_count = session.GetOutputCount();
    auto output_shapes = std::vector<std::vector<int64_t>>();
    auto output_names_ptr = std::vector<Ort::AllocatedStringPtr>();
    auto output_names = std::vector<const char*>();
    output_shapes.reserve(output_count);
    output_names_ptr.reserve(output_count);
    output_names.reserve(output_count);

    for (size_t i = 0; i < output_count; i++)
    {
        auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        output_shapes.push_back(shape);
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(name.get());
        output_names_ptr.push_back(std::move(name));
        std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
    }
    for (int i = 0; i < 1; i++)
    {
        auto input_shape = input_shapes[0];
        if (input_shape[0] == -1) {
            input_shape[0] = batch_number;
        }
        int total_number_elements = calculate_product(input_shape);
        std::vector<Ort::Float16_t> input_tensor_values(total_number_elements);
        try
        {
            preprocess_resnet(curr_file, input_tensor_values, input_shape);
        }
        catch (const Ort::Exception& exception) {
            cout << "ERROR running preprocess: " << exception.what() << endl;
            exit(-1);
        }
        std::vector<Ort::Value> input_tensors;
        Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()));

        // double-check the dimensions of the input tensor
        assert(input_tensors[0].IsTensor() &&
            input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() ==
            input_shape);
        try {
            auto output_tensors = session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(), input_count, output_names.data(), output_count);
            assert(output_tensors.size() == session.GetOutputNames().size() &&
                output_tensors[0].IsTensor());
            auto output_shape =
                output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            string predicted = postprocess_resnet(curr_file, output_tensors[0]);
            cout << "Finished inference. output label: " << predicted << endl;
        }
        catch (const Ort::Exception& exception) {
            cout << "ERROR running model inference: " << exception.what() << endl;
            exit(-1);
        }
    }

    return 0;
}

static cv::Mat read_image(const string file) {
    cv::Mat image;
    image = cv::imread(file);
    return image;
}

static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
    cv::Mat cropped_img;
    int offset_h = (image.rows - height) / 2;
    int offset_w = (image.cols - width) / 2;
    cv::Rect box(offset_w, offset_h, width, height);
    cropped_img = image(box).clone();
    return cropped_img;
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
    float smallest_side = 256;
    float scale = smallest_side / ((image.rows > image.cols) ? (float)image.cols
        : (float)image.rows);
    cv::Mat resized_image;
    cv::resize(image, resized_image,
        cv::Size(image.cols * (int)scale, image.rows * (int)scale));
    return croppedImage(resized_image, size.height, size.width);
}

//(image_data - mean) * scale, BRG2RGB and hwc2chw
static void set_input_image(const cv::Mat& image, Ort::Float16_t* data) {
    float mean[3] = { 0.0f, 0.0f, 0.0f };
    float scales[3] = { 1.0f, 1.0f, 1.0f };
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < image.rows; h++) {
            for (int w = 0; w < image.cols; w++) {
                auto c_t = abs(c - 2); // BRG to RGB
                auto image_data =
                    ((image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scales[c_t]) / 255;
                data[c * image.rows * image.cols + h * image.cols + w] =
                    (Ort::Float16_t)image_data;
            }
        }
    }
}

static std::vector<float> softmax(float* data, int64_t size) {
    auto output = std::vector<float>(size);
    std::transform(data, data + size, output.begin(), expf);
    auto sum =
        std::accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
    std::transform(output.begin(), output.end(), output.begin(),
        [sum](float v) { return v / sum; });
    return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
    int K) {
    auto indices = std::vector<int>(score.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
        [&score](int a, int b) { return score[a] > score[b]; });
    auto ret = std::vector<std::pair<int, float>>(K);
    std::transform(
        indices.begin(), indices.begin() + K, ret.begin(),
        [&score](int index) { return std::make_pair(index, score[index]); });
    return ret;
}

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
    for (const auto& v : topk) {
        std::cout << std::setiosflags(std::ios::left) << std::setw(11)
            << "score[" + std::to_string(v.first) + "]"
            << " =  " << std::setw(12) << v.second
            << " text: " << lookup(v.first)
            << std::resetiosflags(std::ios::left) << std::endl;
    }
}

static const char* lookup(int index) {
    static const char* table[] = {
  #include "imagenet_labels.inc"
    };

    if (index < 0) {
        return "";
    }
    else {
        return table[index];
    }
}