add_executable(camera_yolov8 demo_yolov8_onnx.cpp util/getopt.c
)
target_link_libraries(camera_yolov8 ${ORT_LIBRARY} ${OpenCV_LIBS} onnxruntime glog::glog)
install(TARGETS camera_yolov8 RUNTIME DESTINATION bin)

add_executable(camera_yolov8_nx1x4 demo_yolov8_onnx_n.cpp util/getopt.c
)
target_link_libraries(camera_yolov8_nx1x4 ${ORT_LIBRARY} ${OpenCV_LIBS} onnxruntime glog::glog)
install(TARGETS camera_yolov8_nx1x4 RUNTIME DESTINATION bin)

add_executable(test_jpeg_yolov8 test_yolov8_onnx.cpp util/getopt.c
)
target_link_libraries(test_jpeg_yolov8 ${ORT_LIBRARY} ${OpenCV_LIBS} onnxruntime glog::glog)
install(TARGETS test_jpeg_yolov8 RUNTIME DESTINATION bin)

