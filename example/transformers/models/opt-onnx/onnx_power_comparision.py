#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import subprocess
import os
import sys
class OptPowerComparission:
    def __init__(self):
        # current working directory
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        # Tools path
        self.power_tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\..", "tools\\power_profiling"))

    def run(self):
        subprocess.run(["python", self.current_path + "\\run.py","--model_name", "opt-1.3b", "--target", "cpu", "--local_path", "./opt-1.3b_ortquantized", "--power-profile","--iterations", "35"])       
        subprocess.run(["python", self.current_path + "\\run.py","--model_name", "opt-1.3b", "--target", "aie", "--local_path", "./opt-1.3b_ortquantized", "--impl", "v1", "--power-profile","--iterations", "35"])
        self.compare()

    def compare(self):
        sys.path.append(self.power_tools_path)
        import agm_stats as stats
        if (os.path.exists(self.current_path+"\\power_profile_cpu.csv") and os.path.exists(self.current_path+"\\power_profile_aie_v1.csv")):
                subprocess.run(
                    [
                        "python",
                        self.power_tools_path + "\\agm_visualizer.py",
                        self.current_path+"\\power_profile_cpu.csv",
                        self.current_path+"\\power_profile_aie_v1.csv",
                    ]
                )

if __name__ == "__main__":
    OptPowerComparission().run()
