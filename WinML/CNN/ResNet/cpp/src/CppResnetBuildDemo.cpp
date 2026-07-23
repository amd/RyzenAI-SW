// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.md in the repo root for license information.
#include "ResnetModelHelper.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <winml/onnxruntime_cxx_api.h>
#include <winrt/base.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>

using namespace winrt::Microsoft::Windows::AI::MachineLearning;

// Helper function to parse command line arguments
struct CommandLineArgs {
    std::wstring ep_policy = L"NPU";
    std::wstring model_path;
    std::wstring compiled_output_path;
    std::wstring image_path;
};

void PrintUsage() {
    std::wcout << L"Windows ML C++ Inference Script\n";
    std::wcout << L"Usage: CppResnetBuildDemo.exe [options]\n";
    std::wcout << L"Options:\n";
    std::wcout << L"  --ep_policy <NPU|CPU|GPU|DEFAULT>  Set execution provider policy (default: NPU)\n";
    std::wcout << L"  --model <path>                      Path to the input ONNX model (default: resnet50.onnx in model directory)\n";
    std::wcout << L"  --compiled_output <path>            Path for compiled output model (default: resnet50_ctx.onnx in model directory)\n";
    std::wcout << L"  --image_path <path>                 Path to the input image (default: dog.jpg in executable directory)\n";
    std::wcout << L"  --help                              Show this help message\n";
}

CommandLineArgs ParseCommandLine(int argc, wchar_t* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::wstring arg = argv[i];
        
        if (arg == L"--help" || arg == L"-h") {
            PrintUsage();
            exit(0);
        }
        else if (arg == L"--ep_policy" && i + 1 < argc) {
            args.ep_policy = argv[++i];
        }
        else if (arg == L"--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        }
        else if (arg == L"--compiled_output" && i + 1 < argc) {
            args.compiled_output_path = argv[++i];
        }
        else if (arg == L"--image_path" && i + 1 < argc) {
            args.image_path = argv[++i];
        }
    }
    
    return args;
}

int wmain(int argc, wchar_t* argv[]) noexcept
{
    CommandLineArgs args = ParseCommandLine(argc, argv);

    auto env = Ort::Env();
    std::cout << "ONNX Version string: " << Ort::GetVersionString() << std::endl;
    std::cout << "Getting available providers..." << std::endl;
    // Get the default ExecutionProviderCatalog
    auto catalog = ExecutionProviderCatalog::GetDefault();
    // Ensure execution providers compatible with device are present (downloads if necessary)
    // and then registers all present execution providers with ONNX Runtime
    catalog.EnsureAndRegisterCertifiedAsync().get();
    auto providers = catalog.FindAllProviders();
    // for (const auto& provider : providers)
    // {
    //     std::wcout << "Provider: " << provider.Name().c_str() << std::endl;
    //     auto readyState = provider.ReadyState();
    //     if (readyState == ExecutionProviderReadyState::Ready)
    //     {
    //         std::wcout << "Provider:" << provider.Name().c_str() << " is ready to use." << std::endl;
    //         continue;
    //     }
    //     else if (readyState == ExecutionProviderReadyState::NotReady)
    //     {
    //         std::wcout << "Provider:" << provider.Name().c_str() << " is not ready on this device. The EP is installed on device, trying to add the EP to runtime." << std::endl;
    //         provider.EnsureReadyAsync().get();
    //     }
    //     else if (readyState == ExecutionProviderReadyState::NotPresent)
    //     {
    //         std::wcout << "Provider:" << provider.Name().c_str() << " is not present on this device. Trying to download and install the EP." << std::endl;
    //         provider.EnsureReadyAsync().get();
    //     }
    //     else
    //     {
    //         std::wcout << "Provider is in unknown state." << std::endl;
    //         continue; // Skip registration if provider state is unknown
    //     }
    //     provider.TryRegister();
    // }

    bool needsDownload = false;
    for (const auto& provider : providers)
    {
        if (provider.ReadyState() == ExecutionProviderReadyState::NotPresent)
        {
            needsDownload = true;
            break;
        }
        std::wcout << "Provider:" << provider.Name().c_str() << " is ready to use." << std::endl;
    }
    if (needsDownload) {
        std::cout << "Downloading required execution providers..." << std::endl;
        catalog.EnsureAndRegisterCertifiedAsync().get();
    }

    auto devices = env.GetEpDevices();
    std::cout << "ONNX providers registered: " << std::endl;
    for (const auto& device : devices)
    {
        std::cout << device.EpName() << " " << std::endl;
    }

    // Map ep_policy string to enum value
    std::map<std::wstring, OrtExecutionProviderDevicePolicy> policyMap = {
        {L"NPU", OrtExecutionProviderDevicePolicy_PREFER_NPU},
        {L"CPU", OrtExecutionProviderDevicePolicy_PREFER_CPU},
        {L"DEFAULT", OrtExecutionProviderDevicePolicy_DEFAULT}
    };

    // Set the EP selection policy based on command line argument
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetLogSeverityLevel(1); // 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    auto policyIt = policyMap.find(args.ep_policy);
    if (policyIt != policyMap.end()) {
        sessionOptions.SetEpSelectionPolicy(policyIt->second);
        std::wcout << L"Using execution provider policy: " << args.ep_policy << std::endl;
    }
    else {
        std::wcerr << L"Warning: Invalid ep_policy '" << args.ep_policy << L"'. Using default policy." << std::endl;
        sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_DEFAULT);
    }

    // Prepare paths for model and labels using command-line arguments or defaults
    std::filesystem::path executableFolder = ResnetModelHelper::GetExecutablePath().parent_path();
    std::filesystem::path resourcePath = executableFolder / ".." / ".." / ".." / "..";
    std::filesystem::path labelsPath = executableFolder / "ResNet50Labels.txt";
    
    // Set model path from args or use default
    std::filesystem::path modelPath;
    if (!args.model_path.empty()) {
        modelPath = args.model_path;
    }
    else {
        modelPath = resourcePath / "model" / "resnet50.onnx";
    }
    
    // Set compiled model path from args or use default
    std::filesystem::path compiledModelPath;
    if (!args.compiled_output_path.empty()) {
        compiledModelPath = args.compiled_output_path;
    }
    else {
        compiledModelPath = resourcePath / "model" / "resnet50_ctx.onnx";
    }
    
    // Set image path from args or use default
    std::filesystem::path imagePath;
    if (!args.image_path.empty()) {
        imagePath = args.image_path;
    }
    else {
        imagePath = executableFolder / "dog.jpg";
    }
    // WinRT StorageFile::GetFileFromPathAsync requires a fully-qualified path,
    // so normalize any relative --image_path to absolute.
    if (std::filesystem::exists(imagePath)) {
        imagePath = std::filesystem::absolute(imagePath);
    }

    std::cout << "Model path: " << modelPath << std::endl;
    std::cout << "Compiled model path: " << compiledModelPath << std::endl;
    std::cout << "Image path: " << imagePath << std::endl;

    bool isCompiledModelAvailable = std::filesystem::exists(compiledModelPath);

    if (modelPath.empty())
    {
        std::cerr << "Please specify the path to the ResNet model using --model argument or ensure default path exists." << std::endl;
        std::cerr << "Exiting..." << std::endl;
        return 1;
    }

    // Compile model if compiled model does not exist
    if (!isCompiledModelAvailable)
    {
        std::cout << "No compiled model found, attempting to create compiled model at " << compiledModelPath << std::endl;

        Ort::ModelCompilationOptions compile_options(env, sessionOptions);
        compile_options.SetInputModelPath(modelPath.c_str());
        compile_options.SetOutputModelPath(compiledModelPath.c_str());

        std::cout << "Starting compile, this may take a few moments..." << std::endl;
        Ort::Status compileStatus = Ort::CompileModel(env, compile_options);
        if (compileStatus.IsOK())
        {
            std::cout << "Model compiled successfully!" << std::endl;
            isCompiledModelAvailable = std::filesystem::exists(compiledModelPath);
        }
        else
        {
            std::cerr << "Failed to compile model: " << compileStatus.GetErrorCode() << ", " << compileStatus.GetErrorMessage()
                      << std::endl;
            std::cerr << "Falling back to uncompiled model" << std::endl;
        }
    }
    else
    {
        std::cout << "Using existing compiled model: " << compiledModelPath << std::endl;
    }
    
    std::filesystem::path modelPathToUse = isCompiledModelAvailable ? compiledModelPath : modelPath;

    // Create the session and load the model
    Ort::Session session(env, modelPathToUse.c_str(), sessionOptions);
    std::cout << "ResNet model loaded" << std::endl;

    // Load and Preprocess image
    std::cout << "Loading image: " << imagePath << std::endl;
    // Check if image file exists
    if (!std::filesystem::exists(imagePath)) {
        std::cerr << "Error: Image file not found: " << imagePath << std::endl;
        return 1;
    }
    // winrt::hstring imagePathHString{imagePath.c_str()};
    // auto imageFrameResult = ResnetModelHelper::LoadImageFileAsync(imagePathHString);
    // auto inputTensorData = ResnetModelHelper::BindSoftwareBitmapAsTensor(imageFrameResult.get());

    std::cout << "Image file exists, starting WinRT image loading..." << std::endl;
    winrt::hstring imagePathHString{imagePath.c_str()};
    std::cout << "Calling LoadImageFileAsync..." << std::endl;
    auto imageFrameResult = ResnetModelHelper::LoadImageFileAsync(imagePathHString);
    std::cout << "Waiting for async result..." << std::endl;
    auto softwareBitmap = imageFrameResult.get();
    if (!softwareBitmap) {
        std::cerr << "Error: LoadImageFileAsync returned null" << std::endl;
        return 1;
    }
    std::cout << "Converting to tensor..." << std::endl;
    auto inputTensorData = ResnetModelHelper::BindSoftwareBitmapAsTensor(softwareBitmap);
    std::cout << "Image processing completed successfully!" << std::endl;
    // Prepare input tensor
    auto inputInfo = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto inputType = inputInfo.GetElementType();
    auto expectedShape = inputInfo.GetShape();

    auto inputShape = std::array<int64_t, 4>{1, 3, 224, 224};

    // Validate input shape matches expected model input
    if (expectedShape.size() != inputShape.size()) {
        std::cerr << "Error: Input tensor dimension mismatch!" << std::endl;
        std::cerr << "Expected " << expectedShape.size() << " dimensions, but got " << inputShape.size() << std::endl;
        return 1;
    }
    
    for (size_t i = 0; i < expectedShape.size(); ++i) {
        if (expectedShape[i] != -1 && expectedShape[i] != inputShape[i]) {
            std::cerr << "Error: Input shape mismatch at dimension " << i << "!" << std::endl;
            std::cerr << "Expected: [";
            for (size_t j = 0; j < expectedShape.size(); ++j) {
                std::cerr << expectedShape[j];
                if (j < expectedShape.size() - 1) std::cerr << ", ";
            }
            std::cerr << "], but got: [";
            for (size_t j = 0; j < inputShape.size(); ++j) {
                std::cerr << inputShape[j];
                if (j < inputShape.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            return 1;
        }
    }
    
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<uint8_t> rawInputBytes;

    if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        auto converted = ResnetModelHelper::ConvertFloat32ToFloat16(inputTensorData);
        rawInputBytes.assign(
            reinterpret_cast<uint8_t*>(converted.data()),
            reinterpret_cast<uint8_t*>(converted.data()) + converted.size() * sizeof(uint16_t));
    }
    else
    {
        rawInputBytes.assign(
            reinterpret_cast<uint8_t*>(inputTensorData.data()),
            reinterpret_cast<uint8_t*>(inputTensorData.data()) + inputTensorData.size() * sizeof(float));
    }

    OrtValue* ortValue = nullptr;

    try {
        Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(
            memoryInfo, rawInputBytes.data(), rawInputBytes.size(), inputShape.data(), inputShape.size(), inputType, &ortValue));
    } catch (const Ort::Exception& e) {
        std::cerr << "Error creating input tensor: " << e.what() << std::endl;
        std::cerr << "This might be due to input shape or data size mismatch." << std::endl;
        return 1;
    }
    
    Ort::Value inputTensor{ortValue};

    const int iterations = 20;
    std::cout << "Running inference for " << iterations << " iterations" << std::endl;
    auto before = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        // std::cout << "---------------------------------------------" << std::endl;
        // std::cout << "Running inference for " << i + 1 << "th time" << std::endl;
        // std::cout << "---------------------------------------------"<< std::endl;
        std::cout << ".";

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> inputNames = {inputName.get()};
        std::vector<const char*> outputNames = {outputName.get()};

        // Run inference
        std::vector<Ort::Value> outputTensors;
        try {
            outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);
        } catch (const Ort::Exception& e) {
            std::cerr << "\nError during inference: " << e.what() << std::endl;
            std::cerr << "This might be due to input tensor shape or type mismatch with the model." << std::endl;
            inputName.release();
            outputName.release();
            return 1;
        }

        // Extract results
        std::vector<float> results;
        if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            auto outputData = outputTensors[0].GetTensorMutableData<uint16_t>();
            size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            std::vector<uint16_t> outputFloat16(outputData, outputData + outputSize);
            results = ResnetModelHelper::ConvertFloat16ToFloat32(outputFloat16);
        }
        else
        {
            auto outputData = outputTensors[0].GetTensorMutableData<float>();
            size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            results.assign(outputData, outputData + outputSize);
        }

        if (i == iterations - 1)
        {
            // Load labels and print result
            std::cout << "\nOutput for the last iteration" << std::endl;
            auto labels = ResnetModelHelper::LoadLabels(labelsPath);
            ResnetModelHelper::PrintResults(labels, results);
        }
        inputName.release();
        outputName.release();
    }
    std::cout << "---------------------------------------------" << std::endl;
    auto after = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
    std::cout << "Time taken for " << iterations << " iterations: " << duration.count() / 1000ull << " seconds" << std::endl;
    std::cout << "Avg time per iteration : " << duration.count() / static_cast<long>(iterations) << " milliseconds" << std::endl;

    return 0;
}
