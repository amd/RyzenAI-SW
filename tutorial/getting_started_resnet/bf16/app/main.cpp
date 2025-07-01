/***********************************************************************************
MIT License

Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ************************************************************************************/
#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>

#define NOMINMAX  // Prevent Windows min/max macros from interfering with std::min/max

#include "npu_util.h"

// CIFAR-10 class labels
const std::vector<std::string> CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
};


static int get_num_elements(const std::vector<int64_t>& v) {
    int total = 1;
    for (auto& i : v)
        total *= (int)i;
    return total;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

static std::string print_tensor(Ort::Value& tensor) {
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto nelem = get_num_elements(shape);
    auto tensor_ptr = tensor.GetTensorMutableData<float>();

    std::stringstream ss("");
    for (auto i = 0; i < nelem; i++)
        ss << tensor_ptr[i] << " ";
    return ss.str();
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

std::string get_program_dir()
{
    char* exe_path; _get_pgmptr(&exe_path); // full path and name of the executable
    return std::filesystem::path(exe_path).parent_path().string(); // directory in which the executable is located
}

// Function to load a binary image file (for CIFAR-10: label + 32x32x3 pixels)
std::vector<float> load_cifar_image(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open image file: " << filename << std::endl;
        // Return a random image if file doesn't exist
        std::vector<float> random_image(3 * 32 * 32);
        std::generate(random_image.begin(), random_image.end(), [&] { return (float)(rand() % 256) / 255.0f; });
        return random_image;
    }
    
    // CIFAR-10 binary format: 1 byte label + 3072 bytes image data (32x32x3)
    // Skip the label byte if present
    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);
    
    std::vector<uint8_t> buffer(3 * 32 * 32);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    file.close();
    
    // Convert to float and normalize [0, 255] -> [0, 1]
    // CIFAR-10 format is: all red pixels, then all green pixels, then all blue pixels
    std::vector<float> image(3 * 32 * 32);
    for (size_t i = 0; i < buffer.size(); ++i) {
        image[i] = static_cast<float>(buffer[i]) / 255.0f;
    }
    
    return image;
}

// Function to get the predicted class from model output
int get_predicted_class(Ort::Value& output_tensor) {
    auto shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto output_ptr = output_tensor.GetTensorMutableData<float>();
    
    // Find the index with maximum probability
    int predicted_class = 0;
    float max_prob = output_ptr[0];
    
    for (int i = 1; i < shape[1]; ++i) {
        if (output_ptr[i] > max_prob) {
            max_prob = output_ptr[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

// Function to print top-k predictions
void print_top_predictions(Ort::Value& output_tensor, int top_k = 3) {
    auto shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto output_ptr = output_tensor.GetTensorMutableData<float>();
    
    // Create pairs of (probability, class_index)
    std::vector<std::pair<float, int>> prob_class_pairs;
    for (int i = 0; i < shape[1]; ++i) {
        prob_class_pairs.emplace_back(output_ptr[i], i);
    }
    
    // Sort by probability in descending order
    std::sort(prob_class_pairs.begin(), prob_class_pairs.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // Print top-k predictions
    std::cout << "Top " << top_k << " predictions:" << std::endl;
    int num_predictions = (top_k < (int)prob_class_pairs.size()) ? top_k : (int)prob_class_pairs.size();
    for (int i = 0; i < num_predictions; ++i) {
        int class_idx = prob_class_pairs[i].second;
        float prob = prob_class_pairs[i].first;
        std::cout << "  " << (i + 1) << ". " << CIFAR10_CLASSES[class_idx] 
                  << " (probability: " << std::fixed << std::setprecision(4) << prob << ")" << std::endl;
    }
}


int runtest(std::string& model_name, std::unordered_map<std::string, std::string> vai_ep_options = {}, bool run_classification = false)
{    
    int64_t batch_size = 1;

    printf("Creating ORT env\n");
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "quicktest");

    printf("Initializing session options\n");
    auto session_options = Ort::SessionOptions();

    if (vai_ep_options.empty()==false) // If VAI EP options are provided, initialize the VitisAI EP
    {
        printf("Configuring VAI EP\n");
        try {
            session_options.AppendExecutionProvider_VitisAI(vai_ep_options);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception occurred in appending execution provider: " << e.what() << std::endl;
        }
    }
    
    printf("Creating ONNX Session\n");
    
    // Check if model file exists first
    if (!std::filesystem::exists(model_name)) {
        std::cerr << "Error: Model file not found at: " << model_name << std::endl;
        std::cerr << "Please ensure you have the model file before running this application." << std::endl;
        return -1;
    }
    
    // Create session - this might throw an exception if the model can't be loaded
    Ort::Session session(env, std::basic_string<ORTCHAR_T>(model_name.begin(), model_name.end()).c_str(), session_options);

    try {
        // Get names and shapes of model inputs and outputs
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_count       = session.GetInputCount();
        auto input_names       = std::vector<std::string>();
        auto input_names_char  = std::vector<const char*>();
        auto input_shapes      = std::vector<std::vector<int64_t>>();
        auto output_count      = session.GetOutputCount();
        auto output_names      = std::vector<std::string>();
        auto output_names_char = std::vector<const char*>();
        auto output_shapes     = std::vector<std::vector<int64_t>>();
        for (size_t i = 0; i < input_count; i++)
        {
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::string name = session.GetInputNameAllocated(i, allocator).get();
            input_names.emplace_back(name);
            input_names_char.emplace_back(input_names.at(i).c_str());
            input_shapes.emplace_back(shape);
        }
        for (size_t i = 0; i < output_count; i++)
        {
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::string name = session.GetOutputNameAllocated(i, allocator).get();
            output_names.emplace_back(name);
            output_names_char.emplace_back(output_names.at(i).c_str());
            output_shapes.emplace_back(shape);
        }

        // Display model info
        std::cout << "ONNX model : " << model_name << std::endl;
        for (size_t i = 0; i < input_count; i++)
            std::cout << "  " << input_names.at(i) << " " << print_shape(input_shapes.at(i)) << std::endl;
        for (size_t i = 0; i < output_count; i++)
            std::cout << "  " << output_names.at(i) << " " << print_shape(output_shapes.at(i)) << std::endl;

        // The code which follows expects the model to have 1 input node and 1 output node.
        if (output_count != 1 || input_count != 1) {
            std::cout << "This version of the program only supports models with 1 input node and 1 output node. Exiting." << std::endl;
            return -1;
        }

        // If input shape has dynamic batch size, set it to a fixed value
        auto input_shape = input_shapes[0];
        if (input_shape[0] < 0) {
            std::cout << "Dynamic batch size detected. Setting batch size to " << batch_size << "." << std::endl;        
            input_shape[0] = batch_size;
        }

        if (run_classification) {
            // Run classification on sample images
            std::cout << "Running classification on sample images..." << std::endl;
            
            std::string exe_dir = get_program_dir();
            
            // Check if test_images directory exists
            std::string test_images_dir = exe_dir + "\\test_images";
            if (!std::filesystem::exists(test_images_dir)) {
                std::cout << "Warning: Test images directory not found at: " << test_images_dir << std::endl;
                std::cout << "Creating directory: " << test_images_dir << std::endl;
                std::filesystem::create_directory(test_images_dir);
            }
            
            std::vector<std::string> test_images = {
                exe_dir + "\\test_images\\airplane.bin",
                exe_dir + "\\test_images\\automobile.bin", 
                exe_dir + "\\test_images\\cat.bin",
                exe_dir + "\\test_images\\ship.bin",
                exe_dir + "\\test_images\\dog.bin"
            };
            
            for (const auto& image_path : test_images) {
                std::cout << "\n--- Testing image: " << std::filesystem::path(image_path).filename().string() << " ---" << std::endl;
                
                // Load image data
                std::vector<float> input_tensor_values = load_cifar_image(image_path);
                
                // Initialize input tensor
                std::vector<Ort::Value> input_tensors;
                input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

                // Run inference
                try 
                {
                    auto output_tensors = session.Run(
                            Ort::RunOptions(), 
                            input_names_char.data(), input_tensors.data(), input_names_char.size(), 
                            output_names_char.data(), output_names_char.size()
                    ); 
                    
                    // Get predicted class
                    int predicted_class = get_predicted_class(output_tensors[0]);
                    std::cout << "Predicted class: " << CIFAR10_CLASSES[predicted_class] << std::endl;
                    
                    // Print top predictions
                    print_top_predictions(output_tensors[0]);
                }
                catch (const Ort::Exception& exception) {
                    std::cout << "ERROR running model inference: " << exception.what() << std::endl;
                    return -1;
                }
            }
        } else {
            // Run performance benchmark
            auto n = 100;
            std::cout << "Running " << n << " inferences of the model" << std::endl;
            // Get the current time before the operation
            auto start = std::chrono::high_resolution_clock::now(); 
            for (int i = 0; i < n; i++)
            {
                // Initialize input data with random numbers in the range [0, 1]
                std::vector<float> input_tensor_values(get_num_elements(input_shape));
                std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return (float)(rand() % 256) / 255.0f; });

                // Initialize input tensor with input data
                std::vector<Ort::Value> input_tensors;
                input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

                // Pass input tensors through model
                try 
                {
                    auto output_tensors = session.Run(
                            Ort::RunOptions(), 
                            input_names_char.data(), input_tensors.data(), input_names_char.size(), 
                            output_names_char.data(), output_names_char.size()
                    ); 
                    // std::cout << i << " : " << print_tensor(output_tensors[0]) << std::endl;
                }
                catch (const Ort::Exception& exception) {
                    std::cout << "ERROR running model inference: " << exception.what() << std::endl;
                    return -1;
                }
            }
            // Get the current time after the operation
            auto end = std::chrono::high_resolution_clock::now();
            // Calculate the duration of the operation
            std::chrono::duration<double> duration = end - start;
            // Print the duration in seconds
            std::cout << "Operation took " << duration.count() << " seconds" << std::endl;
        }
    }
    catch (const Ort::Exception& exception) {
        std::cerr << "ERROR initializing model: " << exception.what() << std::endl;
        return -1;
    }
    catch (const std::exception& exception) {
        std::cerr << "ERROR: " << exception.what() << std::endl;
        return -1;
    }

    printf("Done\n");
    printf("-------------------------------------------------------\n");
    printf("\n");
    
    return 0;
}

int main(int argc, char* argv[]) 
{
    std::string exe_dir = get_program_dir();

    // Default values
    std::unordered_map<std::string, std::string> vai_ep_options;
    vai_ep_options["config_file"] = exe_dir + "\\vitisai_config.json";
    vai_ep_options["cache_dir"]   = exe_dir + "\\my_cache_dir"; 
    vai_ep_options["cache_key"]   = "resnet_trained_for_cifar10"; 
    
    // Ensure models directory exists
    std::string models_dir = exe_dir + "\\models";
    if (!std::filesystem::exists(models_dir)) {
        std::cout << "Warning: Models directory not found at: " << models_dir << std::endl;
        std::cout << "Creating directory: " << models_dir << std::endl;
        std::filesystem::create_directory(models_dir);
    }
    
    std::string model_path = exe_dir + "\\models\\resnet_trained_for_cifar10.onnx";

    bool run_classification = true; // Default to classification mode
    
    std::cout << "usage: app.exe <onnx model> <json_config> [mode]" << std::endl;
    std::cout << "  mode: 'classification' (default) or 'benchmark'" << std::endl;
    
    if (argc > 1) {
        model_path = std::string(argv[1]); // First argument: model path    
    }
    if (argc > 2) {
        vai_ep_options["config_file"] = std::string(argv[2]); // Second argument config file
    }
    if (argc > 3) {
        std::string mode = std::string(argv[3]); // Third argument: mode
        if (mode == "benchmark") {
            run_classification = false;
        } else if (mode == "classification") {
            run_classification = true;
        } else {
            std::cout << "Unknown mode '" << mode << "'. Using classification mode." << std::endl;
        }
    }

    printf("-------------------------------------------------------\n");
    printf("Performing compatibility check for VitisAI EP 1.5.0    \n");
    printf("-------------------------------------------------------\n");
    auto npu_info = npu_util::checkCompatibility_RAI_1_5();

    std::cout << " - NPU Device ID     : 0x" << std::hex << npu_info.device_id << std::dec << std::endl;
    std::cout << " - NPU Device Name   : " << npu_info.device_name << std::endl;
    std::cout << " - NPU Driver Version: " << npu_info.driver_version_string << std::endl;  
    switch (npu_info.check) {
        case npu_util::Status::OK:          
            std::cout << "Environment compatible for VitisAI EP" << std::endl;
            break;
        case npu_util::Status::NPU_UNRECOGNIZED:
            std::cout << "NPU type not recognized." << std::endl;
            std::cout << "Skipping run with VitisAI EP." << std::endl;
            return -1;           
            break;
        case npu_util::Status::DRIVER_TOO_OLD: 
            std::cout << "Installed drivers are too old." << std::endl;
            std::cout << "Skipping run with VitisAI EP." << std::endl;
            return -1;           
            break;
        case npu_util::Status::EP_TOO_OLD:
            std::cout << "VitisAI EP is too old." << std::endl;
            std::cout << "Skipping run with VitisAI EP." << std::endl;
            return -1;           
            break;
        default:
            std::cout << "Unknown state." << std::endl;
            std::cout << "Skipping run with VitisAI EP." << std::endl;
            return -1;           
            break;
    }
    switch(npu_info.device_id) {
        case 0x17F0: // STX/KRK NPU
            std::cout << "STX/KRK NPU device detected." << std::endl;
            break;
        case 0x1502: // PHX/HPT NPU
        default:
            std::cout << "Unsupported NPU device ID." << std::endl;
            return -1;
            break;
    }
    std::cout << std::endl;

    // Set environment variables
    _putenv("XLNX_VART_FIRMWARE=");
    _putenv("XLNX_TARGET_NAME=");  
    _putenv("XLNX_ENABLE_CACHE=0");  

    // Run test
    printf("-------------------------------------------------------\n");
    printf("Running model on CPU                                   \n");    
    printf("-------------------------------------------------------\n");
    runtest(model_path, {}, run_classification);

    printf("-------------------------------------------------------\n");
    printf("Running model on NPU                                   \n");    
    printf("-------------------------------------------------------\n");
    runtest(model_path, vai_ep_options, run_classification);


    std::cout << "Test Done." << std::endl;

    return 0;
}
