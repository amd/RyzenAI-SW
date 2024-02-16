import os
import subprocess
import pandas as pd
import argparse
import pathlib


class TestModel:
    def __init__(self, model_dir_path, img_path):
        # Set the path to the directory containing ONNX models
        self.onnx_directory = model_dir_path
        self.img_path = img_path
        self.Ref_id = "n01917289"

    def compare_id(self, cpu_id, vai_id):
        return (
            cpu_id == self.Ref_id,
            vai_id == self.Ref_id,
        )

    def get_id_from_line(self, line):
        # Assuming the ID is the first word in the line
        return line.split()[0]

    def read_ids(self, file1_path, file2_path):
        # Read the first line from each file
        with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
            line1 = file1.readline().strip()
            line2 = file2.readline().strip()

        # Extract the IDs from the first lines
        id1 = self.get_id_from_line(line1)
        id2 = self.get_id_from_line(line2)

        # Compare the IDs
        cpu_result, vai_result = self.compare_id(id1, id2)
        return [self.Ref_id, cpu_result, vai_result, id1 == id2]

    def create_csv(self, list):
        columns = [
            "Model Name",
            "Model Path",
            "Ref",
            "CPU-EP vs Ref",
            "VAI-EP vs Ref",
            "CPU-EP vs VAI-EP",
        ]
        df = pd.DataFrame(list, columns=[columns])
        df.to_csv("Nightly.csv")
        print("\n-- Results stored in: {}".format("Nightly.csv"))

    def run(self):
        # Get a list of ONNX models in the specified directory
        print(self.onnx_directory)
        onnx_models = []
        files = list(pathlib.Path(self.onnx_directory).rglob("*.onnx"))
        onnx_models = [file for file in files]

        results = []
        # Create output directory
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Iterate over the ONNX models and run the batch file for each model
        for onnx_model in onnx_models:
            # Construct the full path to the ONNX model
            onnx_model_path = str(onnx_model)
            onnx_model_name = onnx_model_path.split("\\")[-1]
            print("- Model Path: {}".format(onnx_model_path))
            print("- Model Name: {}".format(onnx_model_name))
            if os.path.isfile(onnx_model_path):
                # Run the ONNX model with on cpu ep
                # cpu_output_log = os.path.join(output_dir + '\\' + onnx_model_name + ".cpu")
                print("  -- Running with CPU-EP")
                subprocess.run(
                    [
                        "classify.bat",
                        "--model",
                        onnx_model_path,
                        "--ep",
                        "cpu",
                        "--log-to-file",
                        "--img",
                        self.img_path,
                    ]
                )

                # Run the ONNX model with on vai ep (CPU Runner)
                # vai_output_log = os.path.join(output_dir + '\\' + onnx_model_name + ".vai.cpu")
                print("  -- Running with VAI-EP (CPU Runner) ")
                subprocess.run(
                    [
                        "classify.bat",
                        "--model",
                        onnx_model_path,
                        "--ep",
                        "vai",
                        "--log-to-file",
                        "--cpu-runner",
                        "--img",
                        self.img_path,
                    ]
                )

                # Run the ONNX model with on vai ep (DPU Runner)
                # vai_output_log = os.path.join(output_dir + '\\' + onnx_model_name + ".vai.dpu")
                print("  -- Running with VAI-EP (DPU Runner) ")
                subprocess.run(
                    [
                        "classify.bat",
                        "--model",
                        onnx_model_path,
                        "--ep",
                        "vai",
                        "--log-to-file",
                        "--img",
                        self.img_path,
                    ]
                )
            else:
                print("-- File not found: {}".format(onnx_model_path))

            print("\n")
            # result = self.read_ids(cpu_output_log, vai_output_log)
            # l = [onnx_model_name, onnx_model_path]
            # l.extend(result)
            # results.append(l)

        # # Create results CSV
        # self.create_csv(results)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir", type=str, required=True, help="Path to onnx models directory."
    )
    parser.add_argument(
        "--img", type=str, required=True, help="Path to image."
    )
    args = parser.parse_args()
    # Create an instance of the TestModel class with the specified arguments
    TestModel(args.models_dir, args.img).run()
