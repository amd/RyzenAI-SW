# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:188
#include <onnxruntime_cxx_api.h>

auto onnx_model = "resnet50.onnx";
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_bf16");
auto session_options = Ort::SessionOptions();
auto vai_ep_options = std::unordered_map<std::string,std::string>({});
vai_ep_options["config_file"] = "vai_ep_config.json";
session_options.AppendExecutionProvider_VitisAI(vai_ep_options);
auto session = Ort::Session(
    env,
    std::basic_string<ORTCHAR_T>(onnx_model.begin(), onnx_model.end()).c_str(),
    session_options);
