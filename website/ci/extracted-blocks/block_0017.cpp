# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:256
#include <onnxruntime_cxx_api.h>

auto onnx_model = "resnet50_int8.onnx";
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_int8");
auto session_options = Ort::SessionOptions();
auto vai_ep_options = std::unordered_map<std::string,std::string>({});
vai_ep_options["cache_dir"]   = exe_dir + "\\my_cache_dir";
vai_ep_options["cache_key"]   = "resnet_trained_for_cifar10";
vai_ep_options["enable_cache_file_io_in_mem"]   = "0";
vai_ep_options["target"]   = "X2";
session_options.AppendExecutionProvider_VitisAI(vai_ep_options);
auto session = Ort::Session(
    env,
    std::basic_string<ORTCHAR_T>(onnx_model.begin(), onnx_model.end()).c_str(),
    session_options);
