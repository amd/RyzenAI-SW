from typing import List
import os
import shlex
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pyperclip

from utilities import get_downloadable_model_selectors

MODEL_SELECTOR_PLACEHOLDER = "Select a model"

INFO_PANEL = None


class InfoPanel:
    """Side panel used to surface contextual guidance for the current control."""

    def __init__(self, parent: ttk.Frame) -> None:
        self.frame = ttk.LabelFrame(parent, text="Details", padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.label = ttk.Label(
            self.frame,
            text="Select a control to see details.",
            wraplength=260,
            justify="left",
        )
        self.label.grid(row=0, column=0, sticky="nw")

    def display(self, text: str) -> None:
        message = text or ""
        self.label.configure(text=message)


class FormBuilder:
    """Utility to keep form rows consistent inside label frames."""

    def __init__(self, parent: ttk.Frame) -> None:
        self.parent = parent
        self.row = 0
        self.parent.columnconfigure(1, weight=1)

    def add_entry(self, label_text: str, default: str = "", width: int = 28):
        label = ttk.Label(self.parent, text=label_text)
        var = tk.StringVar(value=default)
        entry = ttk.Entry(self.parent, textvariable=var, width=width)
        label.grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry.grid(row=self.row, column=1, sticky="ew", pady=4)
        self.row += 1
        return label, entry, var

    def add_combobox(self, label_text: str, values, default=None):
        label = ttk.Label(self.parent, text=label_text)
        var = tk.StringVar(value=default or (values[0] if values else ""))
        combo = ttk.Combobox(
            self.parent,
            textvariable=var,
            values=values,
            state="readonly",
        )
        label.grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo.grid(row=self.row, column=1, sticky="ew", pady=4)
        self.row += 1
        return label, combo, var

    def add_checkbutton(self, label_text: str, default: bool = False, button_text: str = "Enabled"):
        label = ttk.Label(self.parent, text=label_text)
        var = tk.BooleanVar(value=default)
        check = ttk.Checkbutton(self.parent, variable=var, text=button_text)
        label.grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=4)
        check.grid(row=self.row, column=1, sticky="w", pady=4)
        self.row += 1
        return label, check, var

    def add_file_selector(self, label_text: str, default: str = "", filetypes=None):
        label = ttk.Label(self.parent, text=label_text)
        var = tk.StringVar(value=default)
        entry = ttk.Entry(self.parent, textvariable=var)
        button = ttk.Button(self.parent, text="Browse...")
        label.grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry.grid(row=self.row, column=1, sticky="ew", pady=4)
        button.grid(row=self.row, column=2, sticky="w", padx=(6, 0), pady=4)
        self.row += 1

        def open_dialog():
            selected = filedialog.askopenfilename(filetypes=filetypes)
            if selected:
                var.set(selected)

        button.configure(command=open_dialog)
        return label, entry, button, var

    def add_folder_selector(self, label_text: str, default: str = ""):
        label = ttk.Label(self.parent, text=label_text)
        var = tk.StringVar(value=default)
        entry = ttk.Entry(self.parent, textvariable=var)
        button = ttk.Button(self.parent, text="Browse...")
        label.grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry.grid(row=self.row, column=1, sticky="ew", pady=4)
        button.grid(row=self.row, column=2, sticky="w", padx=(6, 0), pady=4)
        self.row += 1

        def open_dialog():
            selected = filedialog.askdirectory()
            if selected:
                var.set(selected)

        button.configure(command=open_dialog)
        return label, entry, button, var


def register_info(widgets, text: str) -> None:
    if not isinstance(widgets, (list, tuple)):
        widgets = (widgets,)
    for widget in widgets:
        for sequence in ("<FocusIn>", "<Enter>"):
            widget.bind(sequence, lambda _event, message=text: INFO_PANEL.display(message))


def _set_widget_enabled(widget, enabled: bool) -> None:
    if isinstance(widget, ttk.Combobox):
        widget.configure(state="readonly" if enabled else "disabled")
    elif isinstance(widget, ttk.Entry):
        widget.state(["!disabled"] if enabled else ["disabled"])
    elif isinstance(widget, ttk.Checkbutton):
        widget.state(["!disabled"] if enabled else ["disabled"])
    elif isinstance(widget, ttk.Button):
        widget.state(["!disabled"] if enabled else ["disabled"])


class Field:
    """Encapsulates the metadata required to turn a widget into CLI arguments."""

    def __init__(
        self,
        *,
        flag: str,
        focus_widgets,
        control_widgets,
        info_text: str,
        value_getter,
        include_condition=None,
        value_formatter=None,
    ) -> None:
        self.flag = flag
        self.focus_widgets = tuple(focus_widgets if isinstance(focus_widgets, (list, tuple)) else [focus_widgets])
        self.control_widgets = tuple(control_widgets if isinstance(control_widgets, (list, tuple)) else [control_widgets])
        register_info(self.focus_widgets, info_text)
        self.value_getter = value_getter
        self.include_condition = include_condition or (lambda value, _context: value not in ("", None))
        self.value_formatter = value_formatter or (lambda value: value)

    def set_enabled(self, enabled: bool) -> None:
        for widget in self.control_widgets:
            _set_widget_enabled(widget, enabled)

    def get_cli_parts(self, context) -> List[str]:
        value = self.value_getter()
        if isinstance(value, str):
            value = value.strip()
        if not self.include_condition(value, context):
            return []
        formatted = self.value_formatter(value)
        if formatted is None:
            return []
        if isinstance(formatted, (list, tuple)):
            values = [str(item) for item in formatted if str(item) != ""]
        else:
            values = [str(formatted)]
        if not values:
            return []
        return [self.flag, *values]


manual_fields: List[Field] = []
vitis_option_fields: List[Field] = []
calibration_fields: List[Field] = []


def register_manual_field(field: Field) -> Field:
    manual_fields.append(field)
    return field


def register_vitis_field(field: Field) -> Field:
    vitis_option_fields.append(field)
    return field


def register_calibration_field(field: Field) -> Field:
    calibration_fields.append(field)
    return field


def format_command(cmd: List[str]) -> str:
    if hasattr(shlex, "join"):
        return shlex.join(cmd)
    return " ".join(shlex.quote(part) for part in cmd)


def execute_command(arguments: List[str]) -> None:
    cmd = ["python", "performance_benchmark.py", *arguments]
    cmd_display = format_command(cmd)
    print(f"Executing command: {cmd_display}")
    subprocess.run(cmd, check=False)
    try:
        pyperclip.copy(cmd_display)
    except pyperclip.PyperclipException:
        pass
    with open("test.log", "a", encoding="utf-8") as log_file:
        log_file.write(cmd_display + "\n")


root = tk.Tk()
root.title("ONNX CNN Benchmark")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass
style.configure("TNotebook", padding=5)
style.configure("TLabel", padding=2)

main_frame = ttk.Frame(root, padding=20)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

mode_notebook = ttk.Notebook(main_frame)
mode_notebook.grid(row=0, column=0, sticky="nsew")

manual_tab = ttk.Frame(mode_notebook, padding=10)
json_tab = ttk.Frame(mode_notebook, padding=10)
mode_notebook.add(manual_tab, text="Manual Setup")
mode_notebook.add(json_tab, text="JSON Config")

manual_tab.columnconfigure(0, weight=1)
manual_tab.columnconfigure(1, weight=0)
manual_tab.rowconfigure(0, weight=1)
manual_tab.rowconfigure(1, weight=0)

manual_notebook = ttk.Notebook(manual_tab)
manual_notebook.grid(row=0, column=0, sticky="nsew")

basic_tab = ttk.Frame(manual_notebook, padding=10)
advanced_tab = ttk.Frame(manual_notebook, padding=10)
manual_notebook.add(basic_tab, text="Basic")
manual_notebook.add(advanced_tab, text="Advanced")

basic_tab.columnconfigure(0, weight=1)
advanced_tab.columnconfigure(0, weight=1)

INFO_PANEL = InfoPanel(manual_tab)
INFO_PANEL.frame.grid(row=0, column=1, sticky="nsw", padx=(15, 0))

# ---------------------------------------------------------------------------
# Basic tab
model_frame = ttk.LabelFrame(basic_tab, text="Model", padding=10)
model_frame.grid(row=0, column=0, sticky="ew", pady=5)
model_form = FormBuilder(model_frame)

model_choices = [MODEL_SELECTOR_PLACEHOLDER, *get_downloadable_model_selectors()]
model_label, model_combo, model_var = model_form.add_combobox(
    "Model selector (--model)",
    model_choices,
    default=MODEL_SELECTOR_PLACEHOLDER,
)
register_manual_field(
    Field(
        flag="--model",
        focus_widgets=[model_label, model_combo],
        control_widgets=model_combo,
        info_text="Model selector with built-in download support. Optional when using --model_path.",
        value_getter=lambda var=model_var: var.get(),
        include_condition=lambda value, _ctx: value not in ("", MODEL_SELECTOR_PLACEHOLDER),
    )
)

model_path_label, model_path_entry, model_path_button, model_path_var = model_form.add_file_selector(
    "Model path (--model_path)",
    filetypes=[("ONNX Models", "*.onnx"), ("All files", "*.*")],
)
register_manual_field(
    Field(
        flag="--model_path",
        focus_widgets=[model_path_label, model_path_entry, model_path_button],
        control_widgets=[model_path_entry, model_path_button],
        info_text="Path to the ONNX model file or directory.",
        value_getter=lambda var=model_path_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

runtime_basic_frame = ttk.LabelFrame(basic_tab, text="Run Settings", padding=10)
runtime_basic_frame.grid(row=1, column=0, sticky="ew", pady=5)
runtime_basic_form = FormBuilder(runtime_basic_frame)

provider_label, provider_combo, provider_var = runtime_basic_form.add_combobox(
    "Execution provider (--execution_provider)",
    ["CPU", "VitisAIEP", "iGPU", "dGPU"],
    default="CPU",
)
register_manual_field(
    Field(
        flag="--execution_provider",
        focus_widgets=[provider_label, provider_combo],
        control_widgets=provider_combo,
        info_text="Execution Provider selection. Default=CPU.",
        value_getter=lambda var=provider_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

batch_label, batch_entry, batch_var = runtime_basic_form.add_entry(
    "Batch size (--batchsize)",
    default="1",
)
register_manual_field(
    Field(
        flag="--batchsize",
        focus_widgets=[batch_label, batch_entry],
        control_widgets=batch_entry,
        info_text="Number of images processed per batch. VitisAIEP supports batch size = 1.",
        value_getter=lambda var=batch_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

force_batch_label, force_batch_check, force_batch_var = runtime_basic_form.add_checkbutton(
    "Force batch dimension to 1 (--force_batch)",
    default=True,
    button_text="Force dynamic batch to 1",
)
register_manual_field(
    Field(
        flag="--force_batch",
        focus_widgets=[force_batch_label, force_batch_check],
        control_widgets=force_batch_check,
        info_text="Rewrite dynamic batch dimensions to 1 to avoid shape mismatches.",
        value_getter=lambda var=force_batch_var: var.get(),
        include_condition=lambda _value, _ctx: True,
        value_formatter=lambda value: "1" if value else "0",
    )
)

num_label, num_entry, num_var = runtime_basic_form.add_entry(
    "Images to process (--num)",
    default="100",
)
register_manual_field(
    Field(
        flag="--num",
        focus_widgets=[num_label, num_entry],
        control_widgets=num_entry,
        info_text="Number of images loaded into memory and sent to the model.",
        value_getter=lambda var=num_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

timelimit_label, timelimit_entry, timelimit_var = runtime_basic_form.add_entry(
    "Time limit seconds (--timelimit)",
    default="10",
)
register_manual_field(
    Field(
        flag="--timelimit",
        focus_widgets=[timelimit_label, timelimit_entry],
        control_widgets=timelimit_entry,
        info_text="Maximum duration of the experiment when combined with --infinite.",
        value_getter=lambda var=timelimit_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

# ---------------------------------------------------------------------------
# Advanced tab
runtime_adv_frame = ttk.LabelFrame(advanced_tab, text="Runtime", padding=10)
runtime_adv_frame.grid(row=0, column=0, sticky="ew", pady=5)
runtime_adv_form = FormBuilder(runtime_adv_frame)

threads_label, threads_entry, threads_var = runtime_adv_form.add_entry(
    "CPU threads (--threads)",
    default="1",
)
register_manual_field(
    Field(
        flag="--threads",
        focus_widgets=[threads_label, threads_entry],
        control_widgets=threads_entry,
        info_text="Total CPU threads available to the benchmark.",
        value_getter=lambda var=threads_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

intra_label, intra_entry, intra_var = runtime_adv_form.add_entry(
    "Intra-op threads (--intra_op_num_threads)",
    default="1",
)
register_manual_field(
    Field(
        flag="--intra_op_num_threads",
        focus_widgets=[intra_label, intra_entry],
        control_widgets=intra_entry,
        info_text="Number of threads used inside each operator.",
        value_getter=lambda var=intra_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

min_interval_label, min_interval_entry, min_interval_var = runtime_adv_form.add_entry(
    "Minimum interval seconds (--min_interval)",
    default="0",
)
register_manual_field(
    Field(
        flag="--min_interval",
        focus_widgets=[min_interval_label, min_interval_entry],
        control_widgets=min_interval_entry,
        info_text="Minimum delay between inferences. Useful to simulate live framerates.",
        value_getter=lambda var=min_interval_var: var.get(),
        include_condition=lambda value, _ctx: value not in ("", "0"),
    )
)

warmup_label, warmup_entry, warmup_var = runtime_adv_form.add_entry(
    "Warmup iterations (--warmup)",
    default="40",
)
register_manual_field(
    Field(
        flag="--warmup",
        focus_widgets=[warmup_label, warmup_entry],
        control_widgets=warmup_entry,
        info_text="Number of warmup runs before measurements start.",
        value_getter=lambda var=warmup_var: var.get(),
        include_condition=lambda value, _ctx: value != "",
    )
)

infinite_label, infinite_check, infinite_var = runtime_adv_form.add_checkbutton(
    "Run continuously (--infinite)",
    default=True,
    button_text="Continue until time limit",
)
register_manual_field(
    Field(
        flag="--infinite",
        focus_widgets=[infinite_label, infinite_check],
        control_widgets=infinite_check,
        info_text="If enabled, the loop keeps running until --timelimit expires.",
        value_getter=lambda var=infinite_var: var.get(),
        include_condition=lambda _value, _ctx: True,
        value_formatter=lambda value: "1" if value else "0",
    )
)

no_inference_label, no_inference_check, no_inference_var = runtime_adv_form.add_checkbutton(
    "Disable inference (--no_inference)",
    default=False,
    button_text="Measure power only",
)
register_manual_field(
    Field(
        flag="--no_inference",
        focus_widgets=[no_inference_label, no_inference_check],
        control_widgets=no_inference_check,
        info_text="Run without inference to capture baseline power.",
        value_getter=lambda var=no_inference_var: var.get(),
        include_condition=lambda value, _ctx: value,
        value_formatter=lambda value: "1" if value else "0",
    )
)

logging_frame = ttk.LabelFrame(advanced_tab, text="Logging & Power", padding=10)
logging_frame.grid(row=1, column=0, sticky="ew", pady=5)
logging_form = FormBuilder(logging_frame)

log_csv_label, log_csv_check, log_csv_var = logging_form.add_checkbutton(
    "Append CSV summary (--log_csv)",
    default=False,
    button_text="Append measurements",
)
register_manual_field(
    Field(
        flag="--log_csv",
        focus_widgets=[log_csv_label, log_csv_check],
        control_widgets=log_csv_check,
        info_text="Append results to a CSV summary when enabled.",
        value_getter=lambda var=log_csv_var: var.get(),
        include_condition=lambda value, _ctx: value,
        value_formatter=lambda value: "1" if value else "0",
    )
)

power_label, power_combo, power_var = logging_form.add_combobox(
    "Power tool (--power)",
    ["no power", "HWINFO"],
    default="no power",
)
register_manual_field(
    Field(
        flag="--power",
        focus_widgets=[power_label, power_combo],
        control_widgets=power_combo,
        info_text="Select the power measurement backend.",
        value_getter=lambda var=power_var: var.get(),
        include_condition=lambda value, _ctx: value not in ("", "no power"),
    )
)

vitis_frame = ttk.LabelFrame(advanced_tab, text="Vitis AI Options", padding=10)
vitis_frame.grid(row=2, column=0, sticky="ew", pady=5)
vitis_form = FormBuilder(vitis_frame)

autoquant_label, autoquant_check, autoquant_var = vitis_form.add_checkbutton(
    "Enable auto quantization (--autoquant)",
    default=False,
    button_text="Quantize with Quark",
)
autoquant_field = register_vitis_field(
    register_manual_field(
        Field(
            flag="--autoquant",
            focus_widgets=[autoquant_label, autoquant_check],
            control_widgets=autoquant_check,
            info_text="Enable Quark auto-quantization for FP32 models.",
            value_getter=lambda var=autoquant_var: var.get(),
            include_condition=lambda _value, ctx: ctx["provider"] == "VitisAIEP",
            value_formatter=lambda value: "1" if value else "0",
        )
    )
)

renew_label, renew_check, renew_var = vitis_form.add_checkbutton(
    "Recompile model (--renew)",
    default=True,
    button_text="Force recompilation",
)
register_vitis_field(
    register_manual_field(
        Field(
            flag="--renew",
            focus_widgets=[renew_label, renew_check],
            control_widgets=renew_check,
            info_text="Cancel the cache and recompile the model when enabled.",
            value_getter=lambda var=renew_var: var.get(),
            include_condition=lambda _value, ctx: ctx["provider"] == "VitisAIEP",
            value_formatter=lambda value: "1" if value else "0",
        )
    )
)

config_label, config_entry, config_button, config_var = vitis_form.add_file_selector(
    "Compiler config (--config)",
)
register_vitis_field(
    register_manual_field(
        Field(
            flag="--config",
            focus_widgets=[config_label, config_entry, config_button],
            control_widgets=[config_entry, config_button],
            info_text="Path to the Vitis AI configuration JSON file.",
            value_getter=lambda var=config_var: var.get(),
            include_condition=lambda value, ctx: ctx["provider"] == "VitisAIEP" and value != "",
        )
    )
)

calib_label, calib_entry, calib_button, calib_var = vitis_form.add_folder_selector(
    "Calibration images (--calib)",
    default=".\\Imagenet_small",
)
register_calibration_field(
    register_manual_field(
        Field(
            flag="--calib",
            focus_widgets=[calib_label, calib_entry, calib_button],
            control_widgets=[calib_entry, calib_button],
            info_text="Images used for calibration during quantization.",
            value_getter=lambda var=calib_var: var.get(),
            include_condition=lambda value, ctx: ctx["provider"] == "VitisAIEP" and ctx["autoquant"] == "1" and value != "",
        )
    )
)

num_calib_label, num_calib_entry, num_calib_var = vitis_form.add_entry(
    "Calibration samples (--num_calib)",
    default="10",
)
register_calibration_field(
    register_manual_field(
        Field(
            flag="--num_calib",
            focus_widgets=[num_calib_label, num_calib_entry],
            control_widgets=num_calib_entry,
            info_text="Number of images used during calibration.",
            value_getter=lambda var=num_calib_var: var.get(),
            include_condition=lambda value, ctx: ctx["provider"] == "VitisAIEP" and ctx["autoquant"] == "1" and value != "",
        )
    )
)

quarkpreset_label, quarkpreset_combo, quarkpreset_var = vitis_form.add_combobox(
    "Quark preset (--quarkpreset)",
    ["XINT8"],
    default="XINT8",
)
register_calibration_field(
    register_vitis_field(
        register_manual_field(
            Field(
                flag="--quarkpreset",
                focus_widgets=[quarkpreset_label, quarkpreset_combo],
                control_widgets=quarkpreset_combo,
                info_text="Quark quantization preset.",
                value_getter=lambda var=quarkpreset_var: var.get(),
                include_condition=lambda value, ctx: ctx["provider"] == "VitisAIEP" and ctx["autoquant"] == "1" and value != "",
            )
        )
    )
)

update_opset_label, update_opset_check, update_opset_var = vitis_form.add_checkbutton(
    "Update opset to 17 (--update_opset)",
    default=False,
    button_text="Rewrite opset when < 11",
)
register_vitis_field(
    register_manual_field(
        Field(
            flag="--update_opset",
            focus_widgets=[update_opset_label, update_opset_check],
            control_widgets=update_opset_check,
            info_text="Automatically bump the opset when the model uses opset < 11.",
            value_getter=lambda var=update_opset_var: var.get(),
            include_condition=lambda value, ctx: ctx["provider"] == "VitisAIEP" and value,
            value_formatter=lambda value: "1" if value else "0",
        )
    )
)

# ---------------------------------------------------------------------------
# JSON tab
json_tab.columnconfigure(0, weight=1)
json_frame = ttk.LabelFrame(json_tab, text="Configuration", padding=10)
json_frame.grid(row=0, column=0, sticky="ew")
json_frame.columnconfigure(1, weight=1)

json_info = ttk.Label(
    json_frame,
    text="Run the benchmark using a JSON parameter file. All manual options are ignored.",
    wraplength=400,
    justify="left",
)
json_info.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

json_label = ttk.Label(json_frame, text="Parameters file (--json)")
json_var = tk.StringVar()
json_entry = ttk.Entry(json_frame, textvariable=json_var)
json_button = ttk.Button(json_frame, text="Browse...")

json_label.grid(row=1, column=0, sticky="w", padx=(0, 8))
json_entry.grid(row=1, column=1, sticky="ew", pady=4)
json_button.grid(row=1, column=2, sticky="w", padx=(6, 0), pady=4)


def select_json_file():
    selected = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if selected:
        json_var.set(selected)


json_button.configure(command=select_json_file)

# ---------------------------------------------------------------------------
# Actions


def update_autoquant_dependents(*_args) -> None:
    provider = provider_var.get()
    is_vitis = provider == "VitisAIEP"

    if is_vitis:
        if not vitis_frame.winfo_ismapped():
            vitis_frame.grid(row=2, column=0, sticky="ew", pady=5)
    else:
        if vitis_frame.winfo_manager():
            vitis_frame.grid_remove()

    for field in vitis_option_fields:
        field.set_enabled(is_vitis)

    needs_calibration = is_vitis and autoquant_var.get()
    for field in calibration_fields:
        field.set_enabled(needs_calibration)

    if not is_vitis and autoquant_var.get():
        autoquant_var.set(False)


provider_combo.bind("<<ComboboxSelected>>", lambda _event: update_autoquant_dependents())
autoquant_var.trace_add("write", update_autoquant_dependents)


def collect_manual_args() -> List[str]:
    context = {
        "provider": provider_var.get(),
        "autoquant": "1" if autoquant_var.get() else "0",
    }
    arguments: List[str] = []
    for field in manual_fields:
        arguments.extend(field.get_cli_parts(context))
    return arguments


def run_manual_benchmark() -> None:
    model_selected = model_var.get()
    model_path = model_path_var.get().strip()
    if model_selected in ("", MODEL_SELECTOR_PLACEHOLDER) and not model_path:
        messagebox.showerror("Missing model", "Select a model from the dropdown or provide a model path before running.")
        manual_notebook.select(basic_tab)
        return
    execute_command(collect_manual_args())


run_manual_button = ttk.Button(manual_tab, text="Run Benchmark", command=run_manual_benchmark)
run_manual_button.grid(row=1, column=0, sticky="e", pady=(10, 0))


def run_json_benchmark() -> None:
    config_path = json_var.get().strip()
    if not config_path:
        messagebox.showerror("Missing file", "Select a JSON configuration file before running.")
        return
    execute_command(["--json", config_path])


run_json_button = ttk.Button(json_tab, text="Run with JSON", command=run_json_benchmark)
run_json_button.grid(row=1, column=0, sticky="e", pady=(10, 0))


# Initialise default config path if available
ryzen_ai_path = os.getenv("RYZEN_AI_INSTALLATION_PATH")
if ryzen_ai_path:
    candidate = os.path.join(
        ryzen_ai_path.strip().strip('"').replace('/', '\\'),
        "voe-4.0-win_amd64",
        "vaip_config.json",
    )
    if os.path.isfile(candidate):
        config_var.set(candidate)

update_autoquant_dependents()
manual_notebook.select(basic_tab)

root.mainloop()





