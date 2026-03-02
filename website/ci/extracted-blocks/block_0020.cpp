# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:382
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort");

// VAI EP Provider options
auto vai_ep_options = std::unordered_map<std::string,std::string>({});
vai_ep_options["encryption_key"] = "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2";

// Session options
auto session_options = Ort::SessionOptions();
session_options.AppendExecutionProvider_VitisAI(vai_ep_options);

// Inference session
auto onnx_model = "context_model.onnx"; // The EP context model
auto session = Ort::Session(
    env,
    std::basic_string<ORTCHAR_T>(onnx_model.begin(), onnx_model.end()).c_str(),
    session_options);
