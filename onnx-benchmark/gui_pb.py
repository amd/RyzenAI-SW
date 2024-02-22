import argparse
import tkinter as tk
from tkinter import filedialog
import subprocess
import json

def create_argparse_window(parser, root, parameter_values):
    for action in parser._actions:
        if not isinstance(action, argparse._HelpAction):
            label_text = f"{', '.join(action.option_strings)}: {action.help}"
            label = tk.Label(root, text=label_text, wraplength=600, anchor='w')
            label.grid(sticky='w', column=2, pady=(10, 0))

            if "path to the file" in action.help.lower():
                default_value = action.default if action.default is not None else ''
                entry = tk.Entry(root)
                entry.grid(row=root.grid_size()[1]-1, column=0, sticky='ew', pady=(10, 0))
                browse_button = tk.Button(root, text="Browse...", command=lambda entry=entry: browse_file_path(entry))
                browse_button.grid(row=root.grid_size()[1]-1, column=1, sticky='ew')
                parameter_values[action.dest] = entry
            elif "path to the folder" in action.help.lower():
                default_value = action.default if action.default is not None else ''
                entry = tk.Entry(root)
                entry.grid(row=root.grid_size()[1]-1, column=0, sticky='ew', pady=(10, 0))
                browse_button = tk.Button(root, text="Browse...", command=lambda entry=entry: browse_dir_path(entry))
                browse_button.grid(row=root.grid_size()[1]-1, column=1, sticky='ew')
                parameter_values[action.dest] = entry
            elif action.choices:
                option_var = tk.StringVar(root)
                default_value = action.default if action.default is not None else action.choices[0]
                option_var.set(default_value)
                option_menu = tk.OptionMenu(root, option_var, *action.choices)
                option_menu.grid(row=root.grid_size()[1]-1, column=0, columnspan=2, sticky='ew', pady=(10, 0))
                parameter_values[action.dest] = option_var
            elif action.type == int:
                default_value = str(action.default) if action.default is not None else ''
                entry = tk.Entry(root)
                entry.insert(0, default_value)
                entry.grid(row=root.grid_size()[1]-1, column=0, sticky='ew', pady=(10, 0))
                parameter_values[action.dest] = entry
            else:
                default_value = action.default if action.default is not None else ''
                entry = tk.Entry(root)
                entry.insert(0, default_value)
                entry.grid(row=root.grid_size()[1]-1, column=0, sticky='ew', pady=(10, 0))
                parameter_values[action.dest] = entry

def browse_file_path(entry):
    filename = filedialog.askopenfilename()
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

def browse_dir_path(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def display_results(result_frame):
    # Read results from file
    try:
        with open("report_performance.json", "r") as file:
            results = json.load(file)
            throughput = results["results"]["performance"]["total_throughput"]
            latency = results["results"]["performance"]["average_latency"]
            label = tk.Label(result_frame, text=f"Measured throughput: {throughput:.2f}", font=("Arial", 12, "bold"))
            label.grid(row=1, column=0, sticky='w')
            label = tk.Label(result_frame, text=f"Measured latency:    {latency:.2f}", font=("Arial", 12, "bold"))
            label.grid(row=2, column=0, sticky='w')

    except FileNotFoundError:
        print("File report_performance.json not found.")

def launch_benchmark(parameter_values):
    composed_string = ""
    for param, value in parameter_values.items():
        if isinstance(value, tk.StringVar):
            entry_value = value.get().strip()
        else:
            entry_value = value.get().strip()

        if entry_value:
            #composed_string += f"--{param.replace('-', '_')} {entry_value} "
            composed_string += f"--{param} {entry_value} "
    command = f"python performance_benchmark.py {composed_string.strip()}"
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True)

def cancel_labels(result_frame):
    for widget in result_frame.winfo_children():
        widget.destroy()

def startnewtest(parameter_values, result_frame):
    cancel_labels(result_frame)
    result_frame.update()
    launch_benchmark(parameter_values)  
    display_results(result_frame)

def main():
    # copy from utilities.py and reordered
    parser = argparse.ArgumentParser(description='Argument Parser for Tkinter Window')
    parser.add_argument(
        "--json",
        type=str,
        help="Path to the file of parameters. This file overrides all other choices",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet50_int.onnx",
        help="Path to the file of ONNX model",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="CPU",
        choices=["CPU", "VitisAIEP"],
        help="Execution Provider selection. Default=CPU",
    )
    parser.add_argument(
        "--core",
        default="1x4",
        type=str,
        choices=["1x4", "4x4"],
        help="Which core to use. Possible values are 1x4 and 4x4. Default=1x4",
    )

    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="vaip_config.json", 
        help="path to the file of VitisAI EP compiler configuration. Default=vaip_config.json"
    )
    parser.add_argument(
        "--intra_op_num_threads", 
        type=int, 
        default=2, 
        help="Number of CPU threads enabled when an operator is resolved by the CPU. Affects the performances but also the CPU power consumption. Default=2"
        )

    parser.add_argument(
        "--num", "-n", type=int, default=100, help="The number of images loaded into memory and subsequently sent to the model. Default=100"
    )

    parser.add_argument(
        "--instance_count",
        "-i",
        type=int,
        default=1,
        help="This parameter governs the parallelism of job execution. When the Vitis AI EP is selected, this parameter controls the number of DPU runners. The workload is always equally divided per each instance count. Default=1",
    )
    parser.add_argument(
        "--threads", 
        "-t", 
        type=int, 
        default=1, 
        help="CPU threads. Default=1"
    )

    parser.add_argument(
        "--batchsize", 
        "-b", 
        type=int, 
        default=1, 
        help="batch size: number of images processed at the same time by the model. VitisAIEP supports batchsize = 1. Default is 1"
    )

    parser.add_argument(
        "--infinite",
        type=str,
        default="1",
        choices=["0", "1"],
        help="if 1: Executing an infinite loop, when combined with a time limit, enables the test to run for a specified duration. Default=1",
    )
    parser.add_argument(
        "--timelimit",
        "-l",
        type=int,
        default=10,
        help="When used in conjunction with the --infinite option, it represents the maximum duration of the experiment. The default value is set to 10 seconds.",
    )
    parser.add_argument(
        "--renew",
        "-r",
        type=str,
        default="1",
        choices=["0", "1"],
        help="if set to 1 cancel the cache and recompile the model. Set to 0 to keep the old compiled file. Default=1",
    )

    parser.add_argument(
        "--min_interval",
        type=float,
        default=0,
        help="Minimum time interval (s) for running inferences. Default=0",
    )
    parser.add_argument(
        "--no_inference",
        type=str,
        default="0",
        choices=["0", "1"],
        help="When set to 1 the benchmark runs without inference for power measurements baseline. Default=0",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=40,
        help="Perform warmup runs, default = 40",
    )
    parser.add_argument(
        "--log_csv",
        "-k",
        type=str,
        default="0",
        choices=["0", "1"],
        help="If this option is set to 1, measurement data will appended to a CSV file. Default=0",
    )
    parser.add_argument(
        "--verbose", "-v",
        type=str,
        default="0",
        choices=["0", "1", "2"],
        help="0 (default): no debug messages, 1: few debug messages, 2: all debug messages"
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=".\\Imagenet\\val",
        help=f"path to the folder of Imagenet database, used for quantization with calibration. Default= .\Imagenet\val ",
    )
    parser.add_argument(
        "--num_calib", 
        type=int, 
        default=10, 
        help="The number of images for calibration. Default=10"
    )
    
    args = parser.parse_args()


    # Create a Tkinter window
    root = tk.Tk()
    root.title("ONNX Performance Benchmark")
    root.resizable(True, True)  # Allow resizing both horizontally and vertically

    # Create a frame to hold the widgets
    frame = tk.Frame(root)
    frame.grid(row=0, column=0, padx=10, pady=10)
    
    # Dictionary to store parameter values
    parameter_values = {}

    # Create widgets for each argument found
    create_argparse_window(parser, frame, parameter_values)

    # frame for the results
    result_frame = tk.Frame(root)
    result_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # frame for the results
    buttons_frame = tk.Frame(root)
    buttons_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Create button to compose the string    
    compose_button = tk.Button(buttons_frame, text="RUN", command=lambda: startnewtest(parameter_values, result_frame))
    compose_button.grid(row=2, column=3, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
