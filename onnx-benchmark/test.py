import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Menu
import subprocess
import pyperclip
import os
import onnx
import sys

def show_frame(selected_frame):
    """Show the selected frame and hide the others."""
    for frame in frames.values():
        frame.grid_remove()  # Hide all frames
    frames[selected_frame].grid(row=1, column=0, padx=10, pady=10)  # Show the selected frame

def show_info(info_text):
    """Open a new window with information."""
    info_window = tk.Toplevel(root)
    info_window.title("Information")
    info_window.geometry("300x200")
    ttk.Label(info_window, text=info_text, wraplength=250).pack(pady=20, padx=20)
    ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

def browse_file_path(entry):
    file_path = filedialog.askopenfilename()
    entry.set(file_path)

def browse_dir_path(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def create_widget_with_info(frame, row, text, info):
    widget_label = ttk.Label(frame, text=text)
    widget_label.grid(row=row, column=2, padx=10, pady=5, sticky="ew")
    info_button = ttk.Button(frame, text="Info", command=lambda: show_info(info))
    info_button.grid(row=row, column=3, padx=10, pady=5, sticky="ew")
    return widget_label, info_button

def startnewtest():
    parameters = {}
    parameters = launch_benchmark(frames[selected_option.get()])

def launch_benchmark(frame):
    global command_history
    commandline = "python performance_benchmark.py"

    labels_by_row = {}

    for widget in frame.winfo_children():
        if isinstance(widget, ttk.Label):
            row = widget.grid_info().get('row', 'unknown')
            labels_by_row[row] = widget.cget('text')

    for widget in frame.winfo_children():
        row = widget.grid_info().get('row', 'unknown')
        label_text = labels_by_row.get(row, "No Label")

        if isinstance(widget, tk.Entry):
            try:
                if row != 'unknown':
                    value = widget.get()
                    if " " in value:
                        value = f'"{value}"'  # Enclose in quotes if it contains a space
                    commandline += f" {label_text} {value}"
            except tk.TclError:
                print("Error accessing Entry widget.")

    for custom_menu in custom_option_menus:
        if custom_menu.option_menu.winfo_viewable():
            row = custom_menu.option_menu.grid_info().get('row', 'unknown')
            value = custom_menu.getvalue()
            if " " in value:
                value = f'"{value}"'  # Enclose in quotes if it contains a space
            commandline += f" {custom_menu.label.cget('text')} {value}"

    print(f"Executing command: {commandline}")
    subprocess.run(commandline, shell=True)
    pyperclip.copy(commandline)

    
    log_file = "test.log"
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(commandline + "\n")

    return {}


def update_advance_visibility(*args):
    for widget in frame_0.winfo_children():
        widget.grid_remove()
    if selected_option.get() == "Complete":
        #update_vitisaiep_visibility()      
        if entry2_ep.getvalue() == "VitisAIEP": 
            entry1_model.show()
            entry2_ep.show()
            entry3_core.show()
            entry4_instance_count.show()
            entry5_threads.show()
            entry6_renew.show()
            entry7_config.show()
            entry8_timelimit.show()
            entry9_num.show()
            entry10_intraop.show()
            entry11_calib.show()
            entry12_num_calib.show()
            entry13_power.show()
            entry14_log_csv.show()
            entry15_min_interval.show()
            entry16_infinite.show()
            entry17_warmup.show()
            entry18_no_inference.show()
            entry19_update_opset.show()
            entry22_quarkpreset.show()
        elif entry2_ep.getvalue() == "CPU":
            entry1_model.show()
            entry2_ep.show()
            entry5_threads.show()
            entry8_timelimit.show()
            entry9_num.show()
            entry13_power.show()
            entry14_log_csv.show()
            entry15_min_interval.show()
            entry16_infinite.show()
            entry17_warmup.show()
            entry18_no_inference.show()
    if selected_option.get() == "File driven":
        entry100_json.show()  
    
    frame_0.update_idletasks()


def list_operators(onnx_model_path):
    # Load the ONNX model
    oplist = []
    onnx_model = onnx.load(onnx_model_path)

    # Iterate through the nodes in the model's graph
    for node in onnx_model.graph.node:
        # Print the operator type for each node
        #print("Operator:", node.op_type)
        oplist.append(node.op_type)
    return oplist



class CustomOptionMenu:
    def __init__(self, parent, row, column, columnspan, choices, default, label_text, info_text):

        self.var = tk.StringVar(parent)
        self.var.set(default if default else choices[0])  # Ensure default value is set
                
        # Create the OptionMenu widget
        self.option_menu = tk.OptionMenu(parent, self.var, *choices)
        self.option_menu.grid(row=row, column=column, columnspan=columnspan, sticky='ew', pady=(10, 0))

        # Create the label and info button
        self.label, self.info_button = create_widget_with_info(parent, row, label_text, info_text)

        # Initially hide the widgets
        self.hide()

    def show(self):
        """Show the option menu, label, and info button."""
        self.option_menu.grid()
        self.label.grid()
        self.info_button.grid()

    def hide(self):
        """Hide the option menu, label, and info button."""
        self.option_menu.grid_remove()
        self.label.grid_remove()
        self.info_button.grid_remove()

    def getvalue(self):
        """Return the current value of the option menu."""
        return self.var.get()

    def setvalue(self, value):
        """
        Set the value of the option menu.
        :param value: The value to set. Must be one of the choices.
        :raises ValueError: If the value is not in the list of choices.
        """
        menu = self.option_menu["menu"]  # Access the OptionMenu's menu
        choices = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)]
        
        if value not in choices:
            raise ValueError(f"Value '{value}' is not a valid choice. Valid choices are: {choices}")
        self.var.set(value)

    def add_trace(self, mode, callback):
        """Add a trace to the StringVar."""
        self.var.trace_add(mode, callback)

class CustomFileEntry:
    def __init__(self, parent, row, column, label_text, info_text, default=""):
        self.var = tk.StringVar(value=default)  # Creiamo un StringVar per tracciare il valore dell'Entry
        self.entry = tk.Entry(parent, textvariable=self.var)  # Colleghiamo l'Entry a StringVar
        self.entry.grid(row=row, column=0, columnspan=1, sticky='ew', pady=(10, 0))

        self.button = tk.Button(parent, text="Browse...", command=lambda: browse_file_path(self.var))
        self.button.grid(row=row, column=1, columnspan=1, sticky='ew', pady=(10, 0))

        # Creiamo l'etichetta e il pulsante di info
        self.label, self.info_button = create_widget_with_info(parent, row, label_text, info_text)

        # Nascondiamo inizialmente i widget
        self.hide()

    def show(self):
        """Mostra tutti i widget."""
        self.entry.grid()
        self.button.grid()
        self.label.grid()
        self.info_button.grid()

    def hide(self):
        """Nascondi tutti i widget."""
        self.entry.grid_remove()
        self.button.grid_remove()
        self.label.grid_remove()
        self.info_button.grid_remove()

    def getvalue(self):
        """Restituisce il valore corrente dell'Entry."""
        return self.var.get()

    def setvalue(self, value):
        self.var.set(value)

    def add_trace(self, mode, callback):
        """Aggiunge una trace allo StringVar."""
        self.var.trace_add(mode, callback)

class simpleentry:
    def __init__(self, parent, row, column, columnspan, default, label_text, info_text):
        self.var = tk.Entry(parent)
        self.default_value = default 
        self.var.insert(0, self.default_value)
        self.var.grid(row=row, column=column, columnspan=columnspan, sticky='ew', pady=(10, 0))
        # Create the label and info button
        self.label, self.info_button = create_widget_with_info(parent, row, label_text, info_text)
        # Initially hide the widgets
        self.hide()

    def show(self):
        """Show the option menu, label, and info button."""
        self.var.grid()
        self.label.grid()
        self.info_button.grid()

    def hide(self):
        """Hide the option menu, label, and info button."""
        self.var.grid_remove()
        self.label.grid_remove()
        self.info_button.grid_remove()

    def getvalue(self):
        """Return the current value of the option menu."""
        return self.var.get()

    def setvalue(self, value):
        self.var.delete(0, tk.END)
        self.var.insert(0, value)
    
    def add_trace(self, mode, callback):
        """Add a trace to the StringVar."""
        self.var.trace_add(mode, callback)

class CustomFolderEntry:
    def __init__(self, parent, row, column, label_text, info_text, default=""):
        self.var = tk.StringVar(value=default)  # Creiamo un StringVar per tracciare il valore dell'Entry
        self.entry = tk.Entry(parent, textvariable=self.var)  # Colleghiamo l'Entry a StringVar
        self.entry.grid(row=row, column=0, columnspan=1, sticky='ew', pady=(10, 0))

        self.button = tk.Button(parent, text="Browse...", command=lambda: browse_dir_path(self.entry))
        self.button.grid(row=row, column=1, columnspan=1, sticky='ew', pady=(10, 0))

        # Creiamo l'etichetta e il pulsante di info
        self.label, self.info_button = create_widget_with_info(parent, row, label_text, info_text)

        # Nascondiamo inizialmente i widget
        self.hide()

    def show(self):
        """Mostra tutti i widget."""
        self.entry.grid()
        self.button.grid()
        self.label.grid()
        self.info_button.grid()

    def hide(self):
        """Nascondi tutti i widget."""
        self.entry.grid_remove()
        self.button.grid_remove()
        self.label.grid_remove()
        self.info_button.grid_remove()

    def getvalue(self):
        """Restituisce il valore corrente dell'Entry."""
        return self.var.get()

    def setvalue(self, value):
        self.var.set(value)

    def add_trace(self, mode, callback):
        """Aggiunge una trace allo StringVar."""
        self.var.trace_add(mode, callback)


# Required environment variables
required_vars = [
    "RYZEN_AI_CONDA_ENV_NAME",
    "RYZEN_AI_INSTALLATION_PATH",
    "DEVICE",
    "VAIP_CONFIG_HOME"
]

# Check if all required environment variables are set
missing_vars = [var for var in required_vars if os.getenv(var) is None]

if missing_vars:
    print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
    print("Please run `set_env.bat` to set up the required environment variables.")
    sys.exit(1)

# Get the expected Conda environment name
expected_env = os.getenv("RYZEN_AI_CONDA_ENV_NAME")

# Check the actual Conda environment
actual_env = os.getenv("CONDA_DEFAULT_ENV")

if actual_env != expected_env:
    print(f"Error: Conda environment '{expected_env}' is required but '{actual_env}' is currently active.")
    print(f"Please activate the correct environment using:")
    print(f"    conda activate {expected_env}")
    sys.exit(1)

print("All checks passed. Environment is correctly set up.")

# Create the main window
root = tk.Tk()
root.title("AIG-SAIS ONNX Benchmark")

# Create a dictionary to store the frames
frames = {}

# Create a list to track all CustomOptionMenu instances
custom_option_menus = []
# Create a selection frame with a dropdown
selection_frame = ttk.Frame(root, padding=10)
selection_frame.grid(row=0, column=0, sticky='ew')
label = ttk.Label(selection_frame, text="Frame style:")
label.grid(row=0, column=0, padx=10, pady=10)
selected_option = tk.StringVar()

options = ["Complete", "File driven"]
dropdown = ttk.Combobox(selection_frame, textvariable=selected_option, values=options)
dropdown.grid(row=0, column=1, padx=10, pady=10)
dropdown.set(options[0])  # Set default selection

run_button0 = tk.Button(selection_frame, text="RUN", command=lambda: startnewtest())
run_button0.grid(row=0, column=2, sticky='ew')

parameters = {}
frame_0 = ttk.Frame(root, padding=10, relief='ridge')

frames["Complete"] = frame_0
frames["File driven"] = frame_0

entry1_model = CustomFileEntry(
    parent=frame_0,
    row=1,
    column=0,
    label_text="--model",
    info_text="Path to the file of ONNX model"
)

entry2_ep = CustomOptionMenu(
    parent=frame_0,
    row=2,
    column=0,
    columnspan=2,
    choices=["CPU", "VitisAIEP"],
    default="CPU",
    label_text="--execution_provider",
    info_text="Execution Provider selection. Default=CPU"
)
custom_option_menus.append(entry2_ep)

def update_vitisaiep_visibility(*args):
    for widget in frame_0.winfo_children():
        widget.grid_remove()
    if entry2_ep.getvalue() == "VitisAIEP":
        entry1_model.show()
        entry2_ep.show()
        entry3_core.show()
        entry4_instance_count.show()
        entry5_threads.show()
        entry6_renew.show()
        entry7_config.show()
        entry8_timelimit.show()
        entry9_num.show()
        entry10_intraop.show()
        entry11_calib.show()
        entry12_num_calib.show()
        entry13_power.show()
        entry14_log_csv.show()
        entry15_min_interval.show()
        entry16_infinite.show()
        entry17_warmup.show()
        entry18_no_inference.show()
        entry19_update_opset.show()
        entry22_quarkpreset.show()
    if entry2_ep.getvalue() == "CPU":
        entry1_model.show()
        entry2_ep.show()
        entry5_threads.show()
        entry8_timelimit.show()
        entry9_num.show()
        entry13_power.show()
        entry14_log_csv.show()
        entry15_min_interval.show()
        entry16_infinite.show()
        entry17_warmup.show()
        entry18_no_inference.show()

    frame_0.update_idletasks()

entry2_ep.add_trace("write", update_vitisaiep_visibility)

# Core selection
dchoices=[]
ddefault=""
device = os.getenv("DEVICE").lower()
if device == "strix_50tops":
    dchoices=["STX_1x4", "STX_4x4"]
    ddefault="STX_1x4"
if device == "strix_55tops":
    dchoices=["STX_1x4", "STX_4x4"]
    ddefault="STX_1x4"
if device == "phoenix":
    dchoices=["PHX_1x4", "PHX_4x4"]
    ddefault="PHX_1x4"
elif device == "hawk":
    dchoices=["PHX_1x4", "PHX_4x4"]
    ddefault="PHX_1x4"

entry3_core = CustomOptionMenu(
    parent=frame_0,
    row=3,
    column=0,
    columnspan=2,
    choices=dchoices,
    default=ddefault,
    label_text="--core",
    info_text="Which core to use with STRIX silicon. Default=1x4"
)
custom_option_menus.append(entry3_core)

# DPU runners
entry4_instance_count = simpleentry(
    parent=frame_0,
    row=4,
    column=0,
    columnspan=2,
    default="1",
    label_text="--instance_count",
    info_text="This parameter governs the parallelism of job execution. When the Vitis AI EP is selected, this parameter controls the number of DPU runners. The workload is always equally divided per each instance count. Default=1"
)

# Threads
entry5_threads = simpleentry(
    parent=frame_0,
    row=5,
    column=0,
    columnspan=2,
    default="1",
    label_text="--threads",
    info_text="CPU threads. Default=1"
)

# renew cache
entry6_renew = CustomOptionMenu(
    parent=frame_0,
    row=6,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="1",
    label_text="--renew",
    info_text="if set to 1 cancel the cache and recompile the model. Set to 0 to keep the old compiled file. Default=1"
)
custom_option_menus.append(entry6_renew)

# custom compiler config file
entry7_config = CustomFileEntry(
    parent=frame_0,
    row=7,
    column=0,
    label_text="--config",
    info_text="path to config json file. Default= <release>\\vaip_config.json"
)

ryzen_ai_path = os.getenv("RYZEN_AI_INSTALLATION_PATH")
if ryzen_ai_path:
    config_file_path = os.path.join(ryzen_ai_path.replace('"', '').replace('/', '\\'), "voe-4.0-win_amd64", "vaip_config.json")
    if os.path.isfile(config_file_path):
        entry7_config.setvalue(config_file_path)
    else:
        print(f"The file does not exist: {config_file_path}")
else:
    print("The RYZEN_AI_INSTALLATION_PATH environment variable is not set.")


selected_option.trace_add("write", update_advance_visibility)

# timelimit
entry8_timelimit = simpleentry(
    parent=frame_0,
    row=8,
    column=0,
    columnspan=2,
    default="10",
    label_text="--timelimit",
    info_text="When used in conjunction with the --infinite option, it represents the maximum duration of the experiment. The default value is set to 10 seconds."
)


# num
entry9_num = simpleentry(
    parent=frame_0,
    row=9,
    column=0,
    columnspan=2,
    default="100",
    label_text="--num",
    info_text="The number of images loaded into memory and subsequently sent to the model. Default=100"
)

# intra op num threads
entry10_intraop = simpleentry(
    parent=frame_0,
    row=10,
    column=0,
    columnspan=2,
    default="1",
    label_text="--intra_op_num_threads",
    info_text="Number of CPU threads enabled when an operator is resolved by the CPU. Affects the performances but also the CPU power consumption. Default=1"
)

# calibration folder
entry11_calib = CustomFolderEntry(
    parent=frame_0,
    row=11,
    column=0,
    default="./Imagenet_small",
    label_text="--calib",
    info_text="path to images folder, used for quantization with calibration."
)

# num_calib
entry12_num_calib = simpleentry(
    parent=frame_0,
    row=12,
    column=0,
    columnspan=2,
    default="10",
    label_text="--num_calib",
    info_text="The number of images for calibration. Default=10"
)

# power
entry13_power = CustomOptionMenu(
    parent=frame_0,
    row=13,
    column=0,
    columnspan=2,
    choices=["no power", "HWINFO"],
    default="no power",
    label_text="--power",
    info_text="This option controls which tool is used for power measurement. No power measurement by default."
)
custom_option_menus.append(entry13_power)

# log_csv
entry14_log_csv = CustomOptionMenu(
    parent=frame_0,
    row=14,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--log_csv",
    info_text="If this option is set to 1, measurement data will appended to a CSV file. Default=0"
)
custom_option_menus.append(entry14_log_csv)

# min_interval
entry15_min_interval = simpleentry(
    parent=frame_0,
    row=15,
    column=0,
    columnspan=2,
    default="0",
    label_text="--min_interval",
    info_text="Minimum time interval (s) for running inferences. Default=0"
)

# infinite loop
entry16_infinite = CustomOptionMenu(
    parent=frame_0,
    row=16,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="1",
    label_text="--infinite",
    info_text="if 1: Executing an infinite loop, when combined with a time limit, enables the test to run for a specified duration. Default=1"
)
custom_option_menus.append(entry16_infinite)

# warmup
entry17_warmup = simpleentry(
    parent=frame_0,
    row=17,
    column=0,
    columnspan=2,
    default="40",
    label_text="--warmup",
    info_text="Perform warmup runs, default = 40"
)

# no_inference
entry18_no_inference = CustomOptionMenu(
    parent=frame_0,
    row=18,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--no_inference",
    info_text="When set to 1 the benchmark runs without inference for power measurements baseline. Default=0"
)
custom_option_menus.append(entry18_no_inference)

# update_opset
entry19_update_opset = CustomOptionMenu(
    parent=frame_0,
    row=19,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--update_opset",
    info_text="if set to 1 and the model has opset < 11, automatically update the model opset to 17. Default=0"
)
custom_option_menus.append(entry19_update_opset)

presets=['XINT8']

entry22_quarkpreset = CustomOptionMenu(
    parent=frame_0,
    row=22,
    column=0,
    columnspan=2,
    choices=presets,
    default=presets[0],
    label_text="--quarkpreset",
    info_text="This option sets the Quark quantization preset to be used."
)
custom_option_menus.append(entry22_quarkpreset)

entry100_json = CustomFileEntry(
    parent=frame_0,
    row=9,
    column=0,
    label_text="--json",
    info_text="Test configuration file overriding all other choices"
)

entry1_model.show()
entry2_ep.show()
entry5_threads.show()
entry8_timelimit.show()
entry9_num.show()
entry13_power.show()
entry14_log_csv.show()
entry15_min_interval.show()
entry16_infinite.show()
entry17_warmup.show()
entry18_no_inference.show()

show_frame("Complete")

# Update the visible frame when the dropdown selection changes
dropdown.bind("<<ComboboxSelected>>", lambda e: show_frame(selected_option.get()))

# Start the Tkinter event loop
root.mainloop()

