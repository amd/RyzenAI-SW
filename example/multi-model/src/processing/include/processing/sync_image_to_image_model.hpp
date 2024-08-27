#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <thread>
#include "util/config.hpp"
// #include "fake_cv.hpp"
using Image = cv::Mat;

// struct OrtValue {
//   void* data_ptr_;
//   size_t data_byte_count_;
//   int64_t* shape_;
//   size_t* shape_len_;
//   // ONNXTensorElementDataType type_;
// };

class SyncImageToImageModel {
 public:
  SyncImageToImageModel() {}
  virtual ~SyncImageToImageModel() {}
  virtual void init(const Config& config) = 0;
  // virtual void preprocess(const Image& image) = 0;
  // virtual Image postprocess() = 0;
  virtual Image run(const Image& image) {
    // preprocess(images);
    // ineternal_run();
    // return postprocess();
    return image;
  }
  // std::vector<OrtValue> inputs_;
  // std::vector<OrtValue> output_;
};

class ModelRegister {
 public:
  static ModelRegister& instance() {
    static ModelRegister instance{};
    return instance;
  }
  void register_model(
      const std::string& name,
      const std::function<std::unique_ptr<SyncImageToImageModel>()>&
          builder_func) {
    auto iter = model_buidler_funcs.find(name);
    CHECK_WITH_INFO(
        iter == model_buidler_funcs.end(),
        std::string("name :") + name + std::string(" redefine!!!!"));
    model_buidler_funcs[name] = builder_func;
  }
  std::unique_ptr<SyncImageToImageModel> build(const std::string& name) {
    auto iter = model_buidler_funcs.find(name);
    CHECK_WITH_INFO(iter != model_buidler_funcs.end(),
                    std::string("name :") + name + std::string(" not find!!!!"))
    return iter->second();
  }

 private:
  ModelRegister() {}
  std::map<std::string, std::function<std::unique_ptr<SyncImageToImageModel>()>>
      model_buidler_funcs;
};

#define REGISTER_MODEL(NAME, MODEL_CLASS_TYPE)               \
  struct NAME##_##MODEL_CLASS_TYPE##RegisterStruct {         \
    NAME##_##MODEL_CLASS_TYPE##RegisterStruct() {            \
      ModelRegister::instance().register_model(#NAME, []() { \
        auto model = new MODEL_CLASS_TYPE();                 \
        return std::unique_ptr<SyncImageToImageModel>(       \
            (SyncImageToImageModel*)model);                  \
      });                                                    \
    }                                                        \
  };                                                         \
  static NAME##_##MODEL_CLASS_TYPE##RegisterStruct           \
      NAME##_##MODEL_CLASS_TYPE##_register_struct_instance{};

class IdentityModel : public SyncImageToImageModel {
 public:
  IdentityModel() {}
  virtual ~IdentityModel() {}
  void init(const Config& config) override {}
  Image run(const Image& image) override { 
    // using namespace std::chrono_literals;
    // std::this_thread::sleep_for(10ms);
    return image; }
};


REGISTER_MODEL(identity, IdentityModel)
