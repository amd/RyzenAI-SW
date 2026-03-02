# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\cnn-examples.mdx:312
auto session_options = Ort::SessionOptions();

auto cache_dir = std::filesystem::current_path().string();

if(ep=="npu")
{
auto options =
    std::unordered_map<std::string, std::string>{ {"cacheDir", cache_dir}, {"cacheKey", "modelcachekey"}};
session_options.AppendExecutionProvider_VitisAI(options);
}

auto session = Ort::Session(env, model_name.data(), session_options);
