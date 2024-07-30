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


class Yolo:
    def __init__(self, confidence_thresh, score_thresh, iou_thresh):
        """
        Initializes an instance of the Yolo class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thresh: Confidence threshold for filtering detections.
            score_thresh: Confidence threshold for filtering detections.
            iou_thresh: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.confidence_thresh = confidence_thresh
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def preprocess(self, original_image, letterbox=True):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        self.img_height, self.img_width, _ = original_image.shape

        # Letter-box
        if letterbox:
            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGBA)

            # Image is opened with opencv, in shape(h, w, c), this is the original image shape
            img_h, img_w, _ = img.shape

            # Desired input shape for the model
            new_h, new_w = self.input_height, self.input_width

            # Initialize the offset
            offset_h, offset_w = 0, 0

            # If the resizing scale of width is lower than that of height
            if (new_w / img_w) <= (new_h / img_h):
                # get a new_h that is with the same resizing scale of width
                new_h = int(img_h * new_w / img_w)
                # update the offset_h
                offset_h = (self.input_height - new_h) // 2
            # If the resizing scale of width is higher than that of height, update new_w
            else:
                # get a new_w that is with the same resizing scale of height
                new_w = int(img_w * new_h / img_h)
                # update the offset_w
                offset_w = (self.input_width - new_w) // 2

            # Resize the image using new_w and new_h
            resized = cv2.resize(img, (new_w, new_h))

            # Initialize a img with pixel value 127, gray color
            output = np.full(
                (self.input_height, self.input_width, 3), 127, dtype=np.uint8
            )

            # Fill resized image data
            output[
                offset_h : (offset_h + new_h), offset_w : (offset_w + new_w), :
            ] = resized

        # No Letter-box
        else:
            output = cv2.resize(original_image, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(output) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        self.img_height, self.img_width, _ = input_image.shape

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thresh:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thresh, self.iou_thresh
        )

        table = [["Class-ID", "Score", "xmin", "ymin", "xmax", "ymax"]]
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            table.append([class_id, score, box[1], box[0], box[3], box[2]])
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        table_data = tab.tabulate(table, headers="firstrow", tablefmt="grid")
        with open("output.txt", "w") as fp:
            fp.write(table_data)
        print(tab.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
        # Return the modified input image
        return input_image

    def run(self, args):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Model path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # Tools path
        power_tools_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..\\..\\..", "tools\\power_profiling"
            )
        )
        sys.path.append(power_tools_path)
        import agm_stats as stats

        if args.e2e_model:
            model_path = os.path.join(
                current_path, "../models/yolov8_int_si.v14.with_pre.onnx"
            )
        else:
            model_path = os.path.join(current_path, "../models/yolov8_int_si.v14.onnx")

        # Remove power profiling files in current directory if exist
        if args.power_profile:
            if os.path.exists("power_profile_cpu.csv"):
                os.remove("power_profile_cpu.csv")
            if os.path.exists("power_profile_vitisai.csv"):
                os.remove("power_profile_vitisai.csv")

        #  Run both CPU and Vitis-AI EP
        if args.run_both_ep:
            num_ep_iter = 2
            args.vai_ep = True
        else:
            num_ep_iter = 1

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
            if args.e2e_model:
                cache_dir_name = "cache_vai_ep_e2e"
            if os.path.exists(".\{}".format(cache_dir_name)):
                shutil.rmtree(".\{}".format(cache_dir_name))

        # Create an inference session using the ONNX model and specify execution providers
        for i in range(num_ep_iter):
            if args.vai_ep:
                print("- VitisAIExecutionProvider: ", args.vai_ep)
                print("- Creating session with VitisAIExecutionProvider ...")
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
                args.vai_ep = False
                print("- print the current execution provider", session.get_providers())
                power_profile_name = "power_profile_vitisai.csv"
                op_profile_name = "op_profile_vitisai.csv"
            else:
                print("- CPUExecutionProvider: ", "True")
                print("- Creating session with CPUExecutionProvider ...")
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

            self.input_width = 640
            self.input_height = 640

            print("\n- Input Image: {}".format(args.img))

            # Read the input image using OpenCV
            orig_img = cv2.imread(args.img)

            # Preprocess the image data
            if args.e2e_model:
                input_data_org = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
            else:
                resized_img = self.preprocess(orig_img, False)

            # Iterations
            num_iterations = args.iterations
            if args.e2e_model:
                if args.power_profile:
                    with stats.AGMCollector("tmp", power_profile_name):
                        for i in range(num_iterations):
                            outputs = session.run(
                                None, {model_inputs[0].name: input_data_org}
                            )
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
            else:
                if args.power_profile:
                    with stats.AGMCollector("tmp", power_profile_name):
                        for i in range(num_iterations):
                            outputs = session.run(
                                None, {model_inputs[0].name: resized_img}
                            )
                else:
                    start = time.perf_counter()
                    for i in range(num_iterations):
                        outputs = session.run(None, {model_inputs[0].name: resized_img})
                    end = time.perf_counter()
                    elapsed = end - start
                    if args.e2e_profile:
                        print(
                            "- Average Time Elapsed (ms): ",
                            (elapsed * 1000) / num_iterations,
                        )

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

            # Perform post-processing on the outputs to obtain output image.
            output_img = self.postprocess(orig_img, outputs)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e2e-model",
        action="store_true",
        default=False,
        help="Run e2e model (with pre/post processing)",
    )
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--conf-thres", type=float, default=float(0.25), help="Confidence threshold"
    )
    parser.add_argument(
        "--score-thres", type=float, default=float(0.45), help="Score threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=float(0.5), help="NMS IoU threshold"
    )
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
    args = parser.parse_args()

    # Create an instance of the Yolo class with the specified arguments
    detector = Yolo(args.conf_thres, args.score_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    detector.run(args)
