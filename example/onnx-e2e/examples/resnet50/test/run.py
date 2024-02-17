#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import subprocess
import sys
import argparse
import struct
import math
import glob
import shutil
import os
import time
from typing import Tuple

import numpy as np
import onnxruntime as ort
import time

import pandas as pd
import tabulate as tab


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install("opencv-python")
    import cv2

try:
    from ultralytics.utils import ROOT, yaml_load
    from ultralytics.utils.checks import check_requirements, check_yaml
except ImportError:
    install("ultralytics")
    from ultralytics.utils import ROOT, yaml_load
    from ultralytics.utils.checks import check_requirements, check_yaml


class Resnet50:
    def __init__(self, args):
        """
        Initializes an instance of the Resnet50 class.
        """

    def softmax(self, res):
        x = np.array(res)
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        res_list = (e_x / e_x.sum(axis=0)).tolist()
        return res_list

    def sort_idx(self, res):
        sort_idx = np.flip(np.squeeze(np.argsort(res)))
        return sort_idx

    def preprocess(self, original_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        mean_vec = np.array([103.94, 116.78, 123.68])
        scale = 0.006

        self.img_height, self.img_width, _ = original_image.shape

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        resize_data = cv2.resize(img, (self.input_width, self.input_height))

        float_data = resize_data.astype("float32")
        norm_img_data = np.zeros(float_data.shape).astype("float32")
        norm_img_data[:, :, :] = (float_data[:, :, :] - mean_vec[:]) * scale
        tran_data = np.array(norm_img_data).transpose(2, 0, 1)
        image_data = np.expand_dims(tran_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract classification labels.

        Args:
            output (numpy.ndarray): The output of the model.
        """

        res_list = self.softmax(output[0])
        indices = self.sort_idx(res_list)
        return indices, res_list

    def model_output(self, e2e, class_id, scores):
        class_file_path = "../models/words.txt"
        with open(class_file_path, "rt") as f:
            classes = f.read().rstrip("\n").split("\n")
        if e2e :
            print("============ Top 5 labels are: ============================")
            for k in range(5):
                # print(scores[0:5])
                print(
                    "- Score: {:.8f}, Class: {}".format(scores[k], classes[class_id[k]])
                )
                with open("output.txt", "a") as f:
                    f.write(
                        "- Score: {:.8f}, Class: {}\n".format(
                            scores[k], classes[class_id[k]]
                        )
                    )
            print("===========================================================")
        else:
            print("============ Top 5 labels are: ============================")
            for k in class_id[:5]:
                print("- Score: {:.8f}, Class: {}".format(scores[k], classes[k]))
                with open("output.txt", "a") as f:
                    f.write(
                        "- Score: {:.8f}, Class: {}\n".format(scores[k], classes[k])
                    )
            print("===========================================================")

    def run(self, args):
        """
        Performs inference using an ONNX model
        """
        print("- Run Resnet50")
        print("- E2E Model (with pre processing): ", args.with_pre)
        print("- E2E Model (with pre and post processing): ", args.e2e)
        current_path = os.path.dirname(os.path.realpath(__file__))

        # Tools path
        power_tools_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..\\..\\..", "tools\\power_profiling"
            )
        )
        sys.path.append(power_tools_path)
        import agm_stats as stats
        import onnx
        from onnx import version_converter

        if args.with_pre and args.vai_ep:
            model_path = os.path.join(
                current_path, "../models/resnet50_pt.v14.with_pre.onnx"
            )
        elif args.e2e and args.vai_ep:
            model_path = os.path.join(
                current_path, "../models/resnet50_pt.v14.e2e.onnx"
            )
        else:
            model_path = os.path.join(current_path, "../models/resnet50_pt.v14.onnx")
        print("model path",model_path)

        # Remove power profiling files in current directory if exist
        if args.power_profile:
            if os.path.exists("power_profile_cpu.csv"):
                os.remove("power_profile_cpu.csv")
            if os.path.exists("power_profile_vitisai.csv"):
                os.remove("power_profile_vitisai.csv")

        #  Run both CPU and Vitis-AI EP
        if args.run_both_ep:
            self.num_ep_iter = 2
            args.vai_ep = True
        else:
            self.num_ep_iter = 1

        # enabling the profiling option
        options = ort.SessionOptions()
        if args.operator_profile:
            options.enable_profiling = True

        # enabling the profiling option
        options = ort.SessionOptions()

        if args.operator_profile:
            options.enable_profiling = True

        # Set Onnx thread options
        options.intra_op_num_threads = args.num_intra_op_threads

        if args.num_inter_op_threads:
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        options.inter_op_num_threads = args.num_inter_op_threads

        options.add_session_config_entry(
            "session.intra_op.allow_spinning", args.intra_op_spinning
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning", args.inter_op_spinning
        )

        # create a cache dir
        if args.vai_ep:
            cache_dir_name = "cache_vai_ep"
            if args.with_pre:
                cache_dir_name = "cache_vai_ep_e2e"
            if os.path.exists(".\{}".format(cache_dir_name)):
                shutil.rmtree(".\{}".format(cache_dir_name))

        # Create an inference session using the ONNX model and specify execution providers
        for i in range(self.num_ep_iter):
            if args.vai_ep:
                print("- VitisAIExecutionProvider: ", args.vai_ep)
                session = ort.InferenceSession(
                    model_path,
                    options,
                    providers=["VitisAIExecutionProvider"],
                    provider_options=[
                        {
                            "config_file": args.config,
                            "cacheDir": ".\{}".format(cache_dir_name),
                        }
                    ],
                )
                # args.vai_ep = False
                print("- print the current execution provider", session.get_providers())
                power_profile_name = "power_profile_vitisai.csv"
                op_profile_name = "op_profile_vitisai.csv"
            else:
                print("- CPUExecutionProvider: ", "True")
                session = ort.InferenceSession(
                    model_path, options, providers=["CPUExecutionProvider"]
                )
                print("- print the current execution provider", session.get_providers())
                power_profile_name = "power_profile_cpu.csv"
                op_profile_name = "op_profile_cpu.csv"

            # Get the model inputs
            model_inputs = session.get_inputs()

            # Store the shape of the input for later use
            for inp in model_inputs:
                print("- Name: {}, Shape: {}".format(inp.name, inp.shape))

            self.input_width = 224
            self.input_height = 224

            print("\n- Input Image: {}".format(args.img))

            original_image = cv2.imread(args.img)

            # Preprocess the image data
            if (args.with_pre or args.e2e) and args.vai_ep :
                input_data_org = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGBA)
            else:
                preprocessed_img = self.preprocess(original_image)

            # Run over iterations
            num_iterations = args.iterations
            print("num of iterations", num_iterations)
            if args.with_pre and args.vai_ep:
                print("in the right place")
                if args.power_profile:
                    with stats.AGMCollector("tmp", power_profile_name):
                        starts = time.time()
                        for i in range(num_iterations):
                            outputs = session.run(
                                None, {model_inputs[0].name: input_data_org}
                            )
                        print("time here in sec", time.time() - starts)
                else:
                    start = time.perf_counter()
                    # Run inference session
                    for i in range(num_iterations):
                        outputs = session.run(
                            None, {model_inputs[0].name: input_data_org}
                        )
                    end = time.perf_counter()
                    elapsed = end - start
                    if args.e2e_profile:
                        print(
                            "- Average Time Elapsed (ms): ",
                            (elapsed * 1000) / num_iterations,
                        )
                class_id, scores = self.postprocess(outputs)

            elif args.e2e and args.vai_ep:
                if args.power_profile:
                    with stats.AGMCollector("tmp", power_profile_name):
                        starts = time.time()
                        for i in range(num_iterations):
                            outputs = session.run(
                                None, {model_inputs[0].name: input_data_org}
                            )
                        print("time here in sec", time.time() - starts)
                else:
                    start = time.perf_counter()
                    # Run inference session
                    for i in range(num_iterations):
                        outputs = session.run(
                            None, {model_inputs[0].name: input_data_org}
                        )
                    end = time.perf_counter()
                    elapsed = end - start
                    if args.e2e_profile:
                        print(
                            "- Average Time Elapsed (ms): ",
                            (elapsed * 1000) / num_iterations,
                        )
                    scores, class_id = outputs[0], outputs[1]

            else:
                if args.power_profile:
                    with stats.AGMCollector("tmp", power_profile_name):
                        for i in range(num_iterations):
                            outputs = session.run(
                                None, {model_inputs[0].name: preprocessed_img}
                            )
                else:
                    start = time.perf_counter()
                    for i in range(num_iterations):
                        outputs = session.run(
                            None, {model_inputs[0].name: preprocessed_img}
                        )
                    end = time.perf_counter()
                    elapsed = end - start
                    if args.e2e_profile:
                        print(
                            "- Average Time Elapsed (ms): ",
                            (elapsed * 1000) / num_iterations,
                        )
                # Perform post-processing on the outputs.
                print("post-processing on the outputs.")
                class_id, scores = self.postprocess(outputs)

            if (
                args.power_profile
                and os.path.exists("power_profile_cpu.csv")
                and os.path.exists("power_profile_vitisai.csv")
            ):
                subprocess.run(
                    [
                        "python",
                        power_tools_path + "\\agm_visualizer.py",
                        "power_profile_cpu.csv",
                        "power_profile_vitisai.csv",
                    ]
                )

            if args.operator_profile:
                prof_file = session.end_profiling()
                # convert prof_file into csv
                df = pd.read_json(prof_file)
                df.to_csv(op_profile_name)
                print("Runtime profiler file name : {}".format(op_profile_name))

            # Print the models output with scores and Classification labels.
            if args.e2e and args.vai_ep:
               self.model_output(True, class_id, scores)
            else:
                self.model_output(False, class_id, scores)


def main(args):
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-pre",
        action="store_true",
        default=False,
        help="Run e2e model (with pre/post processing)",
    )
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--vai-ep",
        action="store_true",
        default=False,
        help="Run with VitisAIExecutionProvider",
    )
    parser.add_argument("--config", type=str, default=None, help="VitisAI EP config")
    parser.add_argument(
        "--power-profile", action="store_true", default=False, help="Power profiling"
    )
    parser.add_argument(
        "--operator-profile",
        action="store_true",
        default=False,
        help="Operator level profiling",
    )
    parser.add_argument(
        "--e2e-profile", action="store_true", default=False, help="e2e average latency"
    )
    parser.add_argument(
        "--run-both-ep",
        action="store_true",
        default=False,
        help="Run with VitisAIExecutionProvider,CPUExecutionProvider with both power,operator level profiling enabled",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the model to get the performance number",
    )
    parser.add_argument(
        "--num_intra_op_threads",
        help="Number of intra op num threads",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--num_inter_op_threads",
        help="Number of inter op num threads",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--intra_op_spinning",
        help="Disable intra op spinning",
        type=str,
        default="0",
        choices=["0", "1"],
    )
    parser.add_argument(
        "--inter_op_spinning",
        help="Disable inter op spinning",
        type=str,
        default="0",
        choices=["0", "1"],
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        default=False,
        help="Run e2e model (with pre and post processing)",
    )

    args = parser.parse_args()
    # Create an instance of the Resnet50 class with the specified arguments
    if args.vai_ep and args.config is None:
        print(
            "\n- Error: Please provide config file (vaip_config.json) path. Exiting!\n"
        )
        return

    Classifier = Resnet50(args)

    # Perform object detection and obtain the output image
    Classifier.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
