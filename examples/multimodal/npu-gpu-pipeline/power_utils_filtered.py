#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#
import pandas as pd 
import numpy as np 
import subprocess
import time
import shlex

def StartPowerMeas():
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = "power_" + timestamp + ".csv"
    command = f'"C:\Program Files\AMD Graphics Manager\AMDGraphicsManager.exe" -unilog=PM,CLK -unilogsetup=unilogsetup.cfg -unilogperiod=50 -unilogstopcheck -unilogoutput="{filename}"'
    cmds = shlex.split(command)
    process = subprocess.Popen(
        cmds, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return filename

def StopPowerMeas():
    file_name = "terminate.txt"
    try:
        with open(file_name, "w") as file:
            pass  # The "pass" statement does nothing, but it's needed to create an empty file
        print(
            f"Stopping the power measurement"
        )
    except IOError as e:
        print(f"An error occurred while creating the file: {e}")

def med_pow(filename):
    # real name and short name
    cols_names = {
        "CPU0 Power Correlation VDDCR_VDD Power": "CPU",
        "CPU0 Power Correlation VDDCR_SOC Power": "NPU_SOC",
        "CPU0 Power Correlation VDD MEM rail Power": "MEM_PHY",
        "CPU0 Power Correlation SOCKET Power": "APU",
    }
    relevant_cols = ["Time Stamp"] + list(cols_names.keys())
    return medians(filename, relevant_cols, cols_names, ylabel="[W]")

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(":"))
    return h * 3600 + m * 60 + s

def str_to_sec(date_string):
    date_string = date_string.strip("--[ ]").strip()
    #
    # date_format = "%d.%m.%Y %H:%M:%S.%f"
    time_str = date_string.split(" ")[1]
    seconds = time_to_seconds(time_str)
    return seconds

def medians(filename, relevant_cols, short_names, ylabel):
    # remove first and last samples affected by AGM
    df = pd.read_csv(filename)
    df_selected = df[relevant_cols]
    df_selected = df_selected.rename(columns=short_names)

    df_selected["Time Stamp"] = df_selected["Time Stamp"].apply(str_to_sec)
    df_selected["Time Stamp"] = (
        df_selected["Time Stamp"] - df_selected["Time Stamp"].min()
    )

    medians = {}
    for key, value in short_names.items():
        # Calculate the median (more robust respect to average without utliers)
        median_value = np.median(df_selected[value])
        # print(f"{value}: {median_value} {ylabel}")
        medians[value] = median_value

    return medians
