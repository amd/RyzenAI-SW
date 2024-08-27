      std::cout<< "Error: no config!!!\n";
      std::cout<< "Usage: ipu_multi_models.exe  <path to models_config.json>\n\n";
      std::cout<< "Examples of model_config.json is located in the config folder of the repository.\n";
      std::cout<< "The meaning of the fields in the models_config.json file:\n";
      std::cout<< "    split_channel_matrix_size         If set to 1, your screen will be splited to 1x1 uniformly; if set to 2,your screen will be split to 2x2 uniformly; and so on.\n";
      std::cout<< "    height                            Window sreen height\n";
      std::cout<< "    width                             Window sreen width\n";
      std::cout<< "    thread_num                        How many thread to feed data to IPU.\n";
      std::cout<< "    onnx_model_path                   Your onnx model for the program to find.\n";
      std::cout<< "    video_file_path                   Your video file for the program to consume; you can set it to string \"0\" for defuatl camera;\n";
      std::cout<< "    confidence_threshold              Bewteen [0,1];The larger the value, the higher the model accuracy;only for model: yolov8 and yolovx\n";
      std::cout<< "    onnx_x                            Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.\n";
      std::cout<< "    onnx_y                            Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default.Must >=0.\n";
      std::cout<< "    onnx_disable_spinning_between_run Disallow thread from spinning during runs to reduce cpu usage.\n";
      std::cout<< "    onnx_disable_spinning             Disable spinning entirely for thread owned by onnxruntime intra-op thread pool.\n";
      std::cout<< "    intra_op_thread_affinities        Not support now;\n";
      std::cout<< "Config example:\n";
      std::cout<< R"({
    "screen": {
        "width": 1024,
        "height": 640,
        "split_channel_matrix_size": 2
    },
    "pipelines": [
        {
            "thread_num": 4,
            "decode": {
                 "video_file_path": "resource\\detection",
                "repeat_frame_per_image":30
            },
            "model": {
                "type": "yolovx",
                "config": {
                    "onnx_config": {
                        "onnx_x": 1,
                        "onnx_y": 1,
                        "onnx_disable_spinning": false,
                        "onnx_disable_spinning_between_run": false,
                        "intra_op_thread_affinities": "0"
                    },
                    "confidence_threshold": 0.3,
                    "onnx_model_path": "resource\\nano-YOLOX_int.onnx"
                }
            },
            "sort": {
                "channel_matrix_id": 0
            }
        },
        {
            "thread_num": 4,
            "decode": {
                 "video_file_path": "resource\\face",
                "repeat_frame_per_image":30
            },
            "model": {
                "type": "retinaface",
                "config": {
                    "onnx_config": {
                        "onnx_x": 1,
                        "onnx_y": 1,
                        "onnx_disable_spinning": false,
                        "onnx_disable_spinning_between_run": false,
                        "intra_op_thread_affinities": "0"
                    },
                    "onnx_model_path": "resource\\RetinaFace_int.onnx"
                }
            },
            "sort": {
                "channel_matrix_id": 1
            }
        },
        {
            "thread_num": 4,
            "decode": {
                "video_file_path": "resource\\seg_512_288.avi"
            },
            "model": {
                "type": "segmentation",
                "config": {
                    "onnx_config": {
                        "onnx_x": 1,
                        "onnx_y": 1,
                        "onnx_disable_spinning": false,
                        "onnx_disable_spinning_between_run": false,
                        "intra_op_thread_affinities": "0"
                    },
                    "onnx_model_path": "resource\\pointpainting-nus-FPN_int.onnx"
                }
            },
            "sort": {
                "channel_matrix_id": 2
            }
        },
        {
            "thread_num": 4,
            "decode": {
                 "video_file_path": "resource\\detection",
                "repeat_frame_per_image":30
            },
            "model": {
                "type": "mobile_net_v2",
                "config": {
                    "onnx_config": {
                        "onnx_x": 1,
                        "onnx_y": 1,
                        "onnx_disable_spinning": false,
                        "onnx_disable_spinning_between_run": false,
                        "intra_op_thread_affinities": "0"
                    },
                    "onnx_model_path": "resource\\mobilenetv2_1.4_int.onnx",
                    "confidence_threshold": 0.3
                }
            },
            "sort": {
                "channel_matrix_id": 3
            }
        }
    ]
})";