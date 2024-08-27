
add_library(my_onnx INTERFACE)
target_include_directories(my_onnx INTERFACE include)
target_link_libraries(my_onnx INTERFACE my_util onnxruntime)


