#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import subprocess
import shlex
import time


class AGMCollector:
    def __init__(self, name, output_file="output_csv"):
        self.name = name
        self.output_file = output_file
        cmd = f'"C:\Program Files\AMD Graphics Manager\AMDGraphicsManager.exe" -pmPeriod=50 -pmLogAll -pmOutput="{output_file}"'
        self.cmds = shlex.split(cmd)
        self.proc = None

    def __wait_until_started(self):
        while self.proc.poll() is not None:
            pass

    def __wait_until_stopped(self):
        while self.proc.poll() is None:
            pass

    def __enter__(self):
        try:
            # print("Opening process")
            self.proc = subprocess.Popen(self.cmds, stdout=subprocess.PIPE)
            # print("Process Opened")
            self.__wait_until_started()
            # print("Process started")
            # Let AGM collect some base data
            time.sleep(2)
            print("\n Power profilie file name : ", self.output_file,"\n")
            return self
        except Exception as E:
            print(
                "\n{}\n\nThe system can not find the AGM executable in the provided path.\n\n{} ".format(
                    "* " * 31, "* " * 31
                )
            )

    def __exit__(self, a, b, c):
        try:
            self.proc.terminate()
            self.__wait_until_stopped()
        except Exception as E:
            print("")
