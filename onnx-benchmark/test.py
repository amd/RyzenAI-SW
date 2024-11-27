import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Menu
import subprocess
import pyperclip
import json
import os

def detect_device():
    def get_driver_info(device_name):
        # PowerShell command to get device details
        command = f'powershell -Command "Get-WmiObject Win32_PnPEntity | Where-Object {{ $_.Name -like \'*{device_name}*\' }} | Select-Object Name, DeviceID"'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        device_info = result.stdout.strip()
    
        return device_info

    device_name = "NPU Compute Accelerator Device"
    device_info = get_driver_info(device_name)
    #print("Device Info:\n", device_info)

    if "17F0" in device_info:
        device = "STRIX"
    elif "1502" in device_info:
        device = "PHOENIX"
    else: device = "ERROR Device Unknown"

    #print(f'Device = {device}')
    return device   

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
    entry.set(folder)

# Function to create a widget with an info button
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
    commandline = "python performance_benchmark.py"
   
    # Store labels by row for easy lookup
    labels_by_row = {}

    # collect all labels by row
    for widget in frame.winfo_children():
        if isinstance(widget, ttk.Label):
            row = widget.grid_info().get('row', 'unknown')
            labels_by_row[row] = widget.cget('text')  # Store the label text by row

   
    for widget in frame.winfo_children():
        row = widget.grid_info().get('row', 'unknown')
        label_text = labels_by_row.get(row, "No Label")  # Get the label for the current row

        # Check if it's a tk.Entry and print its value
        if isinstance(widget, tk.Entry):
            try:
                if row != 'unknown':
                    #print(f"Label: {label_text}, Entry at row {row}: {widget.get()}")
                    commandline += f" {label_text} {widget.get()}"
            except tk.TclError:
                print("Error accessing grid info or getting value from Entry widget.")

    # Print values of all CustomOptionMenu instances tracked
    for custom_menu in custom_option_menus:
        if custom_menu.option_menu.winfo_viewable():
            row = custom_menu.option_menu.grid_info().get('row', 'unknown')
            #print(f"CustomOptionMenu ({custom_menu.label.cget('text')}) at row {row}: {custom_menu.getvalue()}")
            commandline += f" {custom_menu.label.cget('text')} {custom_menu.getvalue()}"

    print(f"Executing command: {commandline}")
    subprocess.run(commandline, shell=True)
    pyperclip.copy(commandline)

    return {}

def update_advance_visibility(*args):
    if selected_option.get() == "Complete":
        for widget in frame_0.winfo_children():
            widget.grid_remove()

        if entry2_ep.getvalue() == "VitisAIEP": 
            entry1_model.show()
            entry2_ep.show()
            entry3_core.show()
            entry4_instance_count.show()
            entry5_threads.show()
            #
            entry6_renew.show()
            entry7_config.show()
            entry8_timelimit.show()
            entry9_num.show()
            entry10_intraop.show()
            entry11_calib.show()
            entry12_num_calib.show()
            entry13_NHWC.show()
            entry15_log_csv.show()
            entry16_min_interval.show()
            entry17_infinite.show()
            entry18_warmup.show()
            entry19_no_inference.show()
            entry20_update_opset.show()
            entry23_profile.show()

        elif entry2_ep.getvalue() == "CPU":
            entry1_model.show()
            entry2_ep.show()
            entry5_threads.show()
            #
            entry8_timelimit.show()
            entry9_num.show()
            entry10_intraop.show()
            entry15_log_csv.show()
            entry16_min_interval.show()
            entry17_infinite.show()
            entry18_warmup.show()
            entry19_no_inference.show()
            entry23_profile.show()
            
            entry4_instance_count.hide()
            entry11_calib.hide()
            entry12_num_calib.hide()
            entry13_NHWC.hide()
            entry20_update_opset.hide()

    if selected_option.get() == "Compact":
        for widget in frame_0.winfo_children():
            widget.grid_remove()
        if entry2_ep.getvalue() == "VitisAIEP":
            entry1_model.show()
            entry2_ep.show()
            entry3_core.show()
            entry4_instance_count.show()
            entry5_threads.show()

        elif entry2_ep.getvalue() == "CPU":
            entry1_model.show()
            entry2_ep.show()
            entry5_threads.show()

    if selected_option.get() == "File driven":
        for widget in frame_0.winfo_children():
            widget.grid_remove()
        entry100_json.show()
    
    frame_0.update_idletasks()
'''
class CustomOptionMenu:
    def __init__(self, parent, row, column, columnspan, choices, default, label_text, info_text):
        self.var = tk.StringVar(parent)
        self.default_value = default if default is not None else choices[0]
        self.var.set(self.default_value)

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
    
    def add_trace(self, mode, callback):
        """Add a trace to the StringVar."""
        self.var.trace_add(mode, callback)
'''

class CustomOptionMenu:
    def __init__(self, parent, row, column, columnspan, choices, default, label_text, info_text):
        self.var = tk.StringVar(parent)
        self.default_value = default if default is not None else choices[0]
        self.var.set(self.default_value)

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

    def add_trace(self, mode, callback):
        """Aggiunge una trace allo StringVar."""
        self.var.trace_add(mode, callback)


class CustomPathEntry:
    def __init__(self, parent, row, column, label_text, info_text, default=""):
        self.var = tk.StringVar(value=default)  # Creiamo un StringVar per tracciare il valore dell'Entry
        self.entry = tk.Entry(parent, textvariable=self.var)  # Colleghiamo l'Entry a StringVar
        self.entry.grid(row=row, column=0, columnspan=1, sticky='ew', pady=(10, 0))

        self.button = tk.Button(parent, text="Browse...", command=lambda: browse_dir_path(self.var))
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

# Create the main window
root = tk.Tk()

device = detect_device()

root.title(f"Ryzen AI ONNX Benchmark - {device}")

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

options = ["Compact", "Complete", "File driven"]
dropdown = ttk.Combobox(selection_frame, textvariable=selected_option, values=options)
dropdown.grid(row=0, column=1, padx=10, pady=10)
dropdown.set(options[0])  # Set default selection

run_button0 = tk.Button(selection_frame, text="RUN", command=lambda: startnewtest())
run_button0.grid(row=0, column=2, sticky='ew')

parameters = {}
frame_0 = ttk.Frame(root, padding=10, relief='ridge')
frame_3 = ttk.Frame(root, padding=10, relief='ridge')

frames["Compact"] = frame_0
frames["Complete"] = frame_0
frames["File driven"] = frame_0
frames["C++"] = frame_3

entry1_model = CustomFileEntry(
    parent=frame_0,
    row=1,
    column=0,
    label_text="--model",
    info_text="Path to the file of ONNX model"
)
entry1_model.show()

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
entry2_ep.show()
custom_option_menus.append(entry2_ep)

def update_vitisaiep_visibility(*args):
    entry6_renew.setvalue("1")
    if entry2_ep.getvalue() == "VitisAIEP":
        entry3_core.show()
        entry4_instance_count.show()
        entry6_renew.show()
    if entry2_ep.getvalue() == "CPU":
        entry3_core.hide()
        entry4_instance_count.hide()
        entry6_renew.hide()
        entry11_calib.hide()
        entry12_num_calib.hide()
        entry13_NHWC.hide()
    frame_0.update_idletasks()

entry2_ep.add_trace("write", update_vitisaiep_visibility)

def trace_entry3_core(*args):
    entry6_renew.setvalue("1")

# Core selection
entry3_core = CustomOptionMenu(
    parent=frame_0,
    row=3,
    column=0,
    columnspan=2,
    choices=["STX_1x4", "STX_4x4"] if device=="STRIX" else ["PHX_1x4", "PHX_4x4"],
    default="STX_1x4" if device=="STRIX" else "PHX_1x4",
    label_text="--core",
    info_text="Which core to use with STRIX silicon. Default=STX_1x4"
)
custom_option_menus.append(entry3_core)
entry3_core.hide()
entry3_core.add_trace("write", trace_entry3_core)


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
entry4_instance_count.hide()

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
entry5_threads.show()

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
entry6_renew.hide()
custom_option_menus.append(entry6_renew)

# custom compiler config file
entry7_config = CustomFileEntry(
    parent=frame_0,
    row=7,
    column=0,
    default=os.path.join(os.environ.get('VAIP_CONFIG_HOME'), 'vaip_config.json'),
    label_text="--config",
    info_text="path to config json file. Default= <release>\\vaip_config.json"
)
entry7_config.hide()
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
entry8_timelimit.hide()

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
entry9_num.hide()

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
entry10_intraop.hide()

# calibration folder
entry11_calib = CustomPathEntry(
    parent=frame_0,
    row=11,
    column=0,
    default=".",
    label_text="--calib",
    info_text="path to images folder, used for quantization with calibration."
)
entry11_calib.hide()

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
entry12_num_calib.hide()

# Core selection
entry13_NHWC = CustomOptionMenu(
    parent=frame_0,
    row=13,
    column=0,
    columnspan=2,
    choices=["0", "1"],
    default="0",
    label_text="--nhwc",
    info_text="When set to 1 enables the input tensor conversion from NCHW to NHWC. Default=0"
)
custom_option_menus.append(entry13_NHWC)
entry13_NHWC.hide()

# log_csv
entry15_log_csv = CustomOptionMenu(
    parent=frame_0,
    row=15,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--log_csv",
    info_text="If this option is set to 1, measurement data will appended to a CSV file. Default=0"
)
entry15_log_csv.hide()
custom_option_menus.append(entry15_log_csv)

# min_interval
entry16_min_interval = simpleentry(
    parent=frame_0,
    row=16,
    column=0,
    columnspan=2,
    default="0",
    label_text="--min_interval",
    info_text="Minimum time interval (s) for running inferences. Default=0"
)
entry16_min_interval.hide()

# infinite loop
entry17_infinite = CustomOptionMenu(
    parent=frame_0,
    row=17,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="1",
    label_text="--infinite",
    info_text="if 1: Executing an infinite loop, when combined with a time limit, enables the test to run for a specified duration. Default=1"
)
entry17_infinite.hide()
custom_option_menus.append(entry17_infinite)

# warmup
entry18_warmup = simpleentry(
    parent=frame_0,
    row=18,
    column=0,
    columnspan=2,
    default="40",
    label_text="--warmup",
    info_text="Perform warmup runs, default = 40"
)
entry18_warmup.hide()

# no_inference
entry19_no_inference = CustomOptionMenu(
    parent=frame_0,
    row=19,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--no_inference",
    info_text="When set to 1 the benchmark runs without inference for power measurements baseline. Default=0"
)
entry19_no_inference.hide()
custom_option_menus.append(entry19_no_inference)

# update_opset
entry20_update_opset = CustomOptionMenu(
    parent=frame_0,
    row=20,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--update_opset",
    info_text="if set to 1 and the model has opset < 11, automatically update the model opset to 17. Default=0"
)
entry20_update_opset.hide()
custom_option_menus.append(entry20_update_opset)

# onnx profiler
entry23_profile = CustomOptionMenu(
    parent=frame_0,
    row=23,
    column=0,
    columnspan=2,
    choices=["0","1"],
    default="0",
    label_text="--profile",
    info_text="If this option is set to 1 the Vitis AI profiler is enabled. Default = 0"
)
entry23_profile.hide()
custom_option_menus.append(entry23_profile)

entry100_json = CustomFileEntry(
    parent=frame_0,
    row=9,
    column=0,
    label_text="--json",
    info_text="Test configuration file overriding all other choices"
)
entry100_json.hide()

# Show the initial frame
show_frame("Compact")

# Update the visible frame when the dropdown selection changes
dropdown.bind("<<ComboboxSelected>>", lambda e: show_frame(selected_option.get()))

create_widget_with_info(frame_3, 0, "Frame 3 Widget", "C++ test code")

# Start the Tkinter event loop
root.mainloop()
