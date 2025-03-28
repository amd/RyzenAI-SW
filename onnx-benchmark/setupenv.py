import os
import tkinter as tk
from tkinter import filedialog, messagebox

class EnvVariableSetter(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Set Environment Variables")
        self.geometry("500x400")
        self.resizable(False, False)

        self.conda_env_name = os.getenv("RYZEN_AI_CONDA_ENV_NAME", "")
        # RYZEN_AI_CONDA_ENV_NAME Entry
        self.conda_label = tk.Label(self, text="RYZEN_AI_CONDA_ENV_NAME:")
        self.conda_label.pack(pady=(10, 0))       
        self.conda_entry = tk.Entry(self, width=50)
        self.conda_entry.insert(0, self.conda_env_name)
        self.conda_entry.pack(pady=5)

        self.installation_path = os.getenv("RYZEN_AI_INSTALLATION_PATH", "")        
        # RYZEN_AI_INSTALLATION_PATH Selection
        self.installation_label = tk.Label(self, text="RYZEN_AI_INSTALLATION_PATH:")       
        self.installation_label.pack(pady=(10, 0))
        self.installation_entry = tk.Entry(self, width=50)
        self.installation_entry.insert(0, self.installation_path)
        self.installation_entry.pack(pady=5)
        
        self.installation_button = tk.Button(self, text="Browse", command=lambda: self.browse_installation_path("ryzenai"))
        self.installation_button.pack()
        
        self.device = os.getenv("DEVICE", "STRIX_50TOPS").upper()
        # DEVICE Selection
        self.device_label = tk.Label(self, text="Select DEVICE:")
        self.device_label.pack(pady=(10, 0))
        
        self.device_var = tk.StringVar(value=self.device)
        self.device_optionmenu = tk.OptionMenu(self, self.device_var, "STRIX_50TOPS", "STRIX_55TOPS", "PHOENIX", "HAWK")
        self.device_optionmenu.pack()
       
        self.hwinfoinstallation_path = os.getenv("HWINFO_INSTALLATION_PATH", "")
        # Power measurement HWINFO Path
        self.hwinifoinstallation_label = tk.Label(self, text="Optional: path to HWINFO if HWINFO is used to measure power")
        self.hwinifoinstallation_label.pack(pady=(10, 0))
        self.hwinifoinstallation_entry = tk.Entry(self, width=50)
        self.hwinifoinstallation_entry.insert(0, self.hwinfoinstallation_path)

        self.hwinifoinstallation_entry.pack(pady=5)
        
        self.hwinifoinstallation_button = tk.Button(self, text="Browse", command=lambda: self.browse_installation_path("hwinfo"))
        self.hwinifoinstallation_button.pack()

        # OK Button
        self.ok_button = tk.Button(self, text="OK", command=self.set_env_variables)
        self.ok_button.pack(pady=10)
        
    def browse_installation_path(self, target):
        if target == "ryzenai":
            directory_path = filedialog.askdirectory(title="Select RyzenAI Installation Directory")
            if directory_path:
                self.installation_entry.delete(0, tk.END)
                self.installation_entry.insert(0, directory_path)
        elif target == "agm":
            directory_path = filedialog.askdirectory(title="Select AGM Installation Directory")
            if directory_path:
                self.agminstallation_entry.delete(0, tk.END)
                self.agminstallation_entry.insert(0, directory_path)
        elif target == "hwinfo":
            directory_path = filedialog.askdirectory(title="Select HWINFO Installation Directory")
            if directory_path:
                self.hwinifoinstallation_entry.delete(0, tk.END)
                self.hwinifoinstallation_entry.insert(0, directory_path)

    def set_env_variables(self):
        """Generate set_env.bat with selected environment variables."""
        conda_env_name = self.conda_entry.get().strip()
        installation_path = self.installation_entry.get().strip()
        hwinfoinstallation_path = self.hwinifoinstallation_entry.get().strip()
        device = self.device_var.get().strip().lower()

        if not conda_env_name or not installation_path:
            messagebox.showerror("Error", "Please fill in all required fields before proceeding.")
            return

        script_lines = ["@echo off"]
        
        if conda_env_name:
            script_lines.append(f"set RYZEN_AI_CONDA_ENV_NAME={conda_env_name}")
        if installation_path:
            script_lines.append(f"set RYZEN_AI_INSTALLATION_PATH={installation_path}")
        if hwinfoinstallation_path:
            script_lines.append(f"set HWINFO_INSTALLATION_PATH={hwinfoinstallation_path}")
        if device:
            script_lines.append(f"set DEVICE={device}")
        
        if device == "strix_50tops" or device == "strix_55tops":
            dev_folder = "strix"
        else:
            dev_folder = "phoenix"


        script_lines.extend([
            "set XLNX_TARGET_NAME=AMD_AIE2P_Nx4_Overlay",
            f"set XCLBINHOME=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/{dev_folder}",
            "set VAIP_CONFIG_HOME=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64",
            "set PATH=%CONDA_PREFIX%/Lib/site-packages/flexmlrt/lib/;%PATH%",
            "set PATH=%RYZEN_AI_INSTALLATION_PATH%/utils;%PATH%",
            "set XLNX_VART_FIRMWARE=%RYZEN_AI_INSTALLATION_PATH%/voe-4.0-win_amd64/xclbins/%DEVICE%/AMD_AIE2P_Nx4_Overlay.xclbin",
            "echo Environment variables set. Run \"call set_env.bat\" in your terminal."
        ])

        script_content = "\n".join(script_lines)
        script_name = "set_env.bat"

        with open(script_name, "w") as script_file:
            script_file.write(script_content)

        messagebox.showinfo("Success", f"Environment variable script saved as '{script_name}'.\n\n"
                                       f"To apply it, run:\n"
                                       f"- **Windows:** call {script_name}")

if __name__ == "__main__":
    app = EnvVariableSetter()
    app.mainloop()
