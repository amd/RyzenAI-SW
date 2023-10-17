set(COMMNON_LIBS processing  my_onnx my_util ${OpenCV_LIBS} onnxruntime glog::glog)

add_executable(ipu_multi_models ipu_multi_models.cpp)
target_link_libraries(ipu_multi_models 
                     resnet50
                     yolovx 
                     mobile_net_v2
                     retinaface
                     segmentation
                     ${COMMNON_LIBS})
install(TARGETS ipu_multi_models RUNTIME DESTINATION bin)