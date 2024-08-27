set(COMMNON_LIBS processing my_onnx my_util ${OpenCV_LIBS} onnxruntime glog)

add_executable(npu_multi_models npu_multi_models.cpp)
target_link_libraries(npu_multi_models 
                      mobile_net_v2
                      resnet50
                      retinaface
                      segmentation
                      yolovx
                      ${COMMNON_LIBS})
install(TARGETS npu_multi_models RUNTIME DESTINATION bin)
