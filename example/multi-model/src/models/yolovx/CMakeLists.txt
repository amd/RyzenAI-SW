
# set(SOURCE src/yolovx_nano_onnx.cpp)
# add_library(yolovx STATIC ${SOURCE})
# target_include_directories(yolovx PUBLIC include)
# target_link_libraries(yolovx PUBLIC my_onnx ${OpenCV_LIBS})

add_library(yolovx INTERFACE)
target_include_directories(yolovx INTERFACE include)
