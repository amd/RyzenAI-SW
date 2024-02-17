#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import re

_identity = lambda x: x


def box_filter(src, win_size):
    assert win_size > 3, "Min. window size of kernel should be 3"
    kernel = np.ones(win_size) / win_size
    return np.convolve(src, kernel)


def gaussian_filter(src, win_size):
    assert win_size > 3, "Min. window size of kernel should be 3"
    pass


def get_cpu_utilization(csv_file):
    pattern = "CPU0_CORES_CORE[0-9]+_C0"
    fields = [
        field
        for field in csv_file.dtype.names
        if re.fullmatch(pattern, field) is not None
    ]
    assert len(fields) > 0, "Couldn't find any CPU Utilization info"
    acc = csv_file[fields[0]].copy()
    for i in range(1, len(fields)):
        acc += csv_file[fields[i]]
    acc /= len(fields)
    return acc


def get_core_utilization(csv_file, core_id):
    pattern = f"CPU0_CORES_CORE{core_id}_C0"
    fields = [
        field
        for field in csv_file.dtype.names
        if re.fullmatch(pattern, field) is not None
    ]
    assert (
        len(fields) > 0
    ), f"Couldn't find any CPU Utilization entry for core:{core_id}"
    assert len(fields) == 1, f"Found multiple utilization entry for core:{core_id}"
    acc = csv_file[fields[0]]
    return acc


def get_cpu_gpu_power(csv_file):
    return csv_file["CPU0_Power_Correlation_VDDCR_VDD_Power"]


def get_ipu_power(csv_file):
    return csv_file["CPU0_Power_Correlation_VDDCR_SOC_Power"]


def get_ddr_mem_phy_power(csv_file):
    return csv_file["CPU0_Power_Correlation_VDD_MEM_rail_Power"]


def get_apu_power(csv_file):
    return csv_file["CPU0_Power_Correlation_SOCKET_Power"]


def get_dram_read_bw(csv_file):
    return csv_file["CPU0_DF_Bandwidth_DRAM_Reads"]


def get_dram_write_bw(csv_file):
    return csv_file["CPU0_DF_Bandwidth_DRAM_Writes"]


def get_dram_rw_bw(csv_file):
    return get_dram_read_bw(csv_file) + get_dram_write_bw(csv_file)


def get_dram_peak_bw(csv_file):
    return csv_file["CPU0_DF_Bandwidth_DRAM_Max_BW"]


def get_cpu_read_bw(csv_file):
    return csv_file["CPU0_DF_Bandwidth_CCX_Reads"]


def get_cpu_write_bw(csv_file):
    return csv_file["CPU0_DF_Bandwidth_CCX_Writes"]


def get_cpu_rw_bw(csv_file):
    return get_cpu_read_bw(csv_file) + get_cpu_write_bw(csv_file)


def get_ipu_read_bw(csv_file):
    """Note: Need to subtract several other components also, but need explore it further"""
    bw = (
        get_dram_read_bw(csv_file)
        - get_cpu_read_bw(csv_file)
        - csv_file["CPU0_DF_Bandwidth_GFX_64B_Reads"]
    )
    return np.maximum(bw, 0)


def get_ipu_write_bw(csv_file):
    """Note: Need to subtract several other components also, but need explore it further"""
    bw = (
        get_dram_write_bw(csv_file)
        - get_cpu_write_bw(csv_file)
        - csv_file["CPU0_DF_Bandwidth_GFX_64B_Writes"]
    )
    return np.maximum(bw, 0)


def get_ipu_rw_bw(csv_file):
    return get_ipu_read_bw(csv_file) + get_ipu_write_bw(csv_file)


def print_full_summary(csv_file, fields):
    summary = [
        round(get_cpu_utilization(csv_file).mean(), 2),
        round(get_cpu_gpu_power(csv_file).mean(), 2),
        round(get_ipu_power(csv_file).mean(), 2),
        round(get_ddr_mem_phy_power(csv_file).mean(), 2),
        round(get_apu_power(csv_file).mean(), 2),
        round(get_apu_power(csv_file).sum() / 1000, 2),
        round(get_cpu_read_bw(csv_file).mean(), 2),
        round(get_cpu_write_bw(csv_file).mean(), 2),
        round(get_ipu_read_bw(csv_file).mean(), 2),
        round(get_ipu_write_bw(csv_file).mean(), 2),
        round(get_dram_read_bw(csv_file).mean(), 2),
        round(get_dram_write_bw(csv_file).mean(), 2),
        round(get_dram_peak_bw(csv_file).mean(), 2),
    ]
    return summary


def plot_cpu_utilization(csv_file, smooth_fn=_identity):
    plt.plot(smooth_fn(get_cpu_utilization(csv_file)))
    plt.title("Avg. CPU Utilization (%)")
    plt.xlabel("samples")
    plt.ylabel("cpu utilization (%)")


def plot_power(csv_file, smooth_fn=_identity):
    plt.plot(smooth_fn(get_cpu_gpu_power(csv_file)))
    plt.plot(smooth_fn(get_ipu_power(csv_file)))
    plt.plot(smooth_fn(get_apu_power(csv_file)))
    plt.title("Power Consumption")
    plt.xlabel("samples")
    plt.ylabel("Power (W)")
    plt.legend(["CPU-GPU", "IPU", "APU"])


def plot_bw(csv_file, smooth_fn=_identity):
    plt.plot(smooth_fn(get_cpu_rw_bw(csv_file)))
    plt.plot(smooth_fn(get_ipu_rw_bw(csv_file)))
    plt.plot(smooth_fn(get_dram_rw_bw(csv_file)))
    plt.plot(smooth_fn(get_dram_peak_bw(csv_file)))
    plt.title("Memory Read-Write BW (GB/s)")
    plt.xlabel("samples")
    plt.ylabel("BW (GB/s)")
    plt.legend(["CPU", "IPU", "Total", "Peak"])


def plot_all(csv_file, fields, smooth_fn=_identity):
    plt.figure()
    plot_cpu_utilization(csv_file, smooth_fn)
    plt.figure()
    plot_power(csv_file, smooth_fn)
    plt.figure()
    plot_bw(csv_file, smooth_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", metavar="FILE", help="Input AGM log files", nargs="+", type=str
    )
    parser.add_argument(
        "--start_idx", help="Starting row of the log file", type=int, default=0
    )
    parser.add_argument(
        "--end_idx", help="Last row of the log file", type=int, default=0
    )
    parser.add_argument(
        "--plot_moving_avg",
        help="Plot a moving average over the actual plot",
        action="store_true",
    )
    parser.add_argument(
        "--moving_avg_window",
        help="Number of samples to consider while calculating moving average",
        type=str,
        default=20,
    )
    args = parser.parse_args()
    print(f"{args}")

    smoothen_avg = lambda x: box_filter(x, args.moving_avg_window)
    smoothen_none = lambda x: x
    smoothen = smoothen_avg

    data = []
    result = []
    columns = []
    for filename in args.input:
        csv_file = np.genfromtxt(filename, delimiter=",", names=True)
        csv_file.flags.writeable = False
        # print(filename)
        columns.append(filename)
        result.append(print_full_summary(csv_file, []))
        data.append((filename, csv_file))
        print()
    # convert the full summary of power details into a csv file
    import pandas as pd

    Rows = [
        "Avg. CPU Utilization %",
        "Avg. CPU-GPU Power (W)",
        "Avg. IPU Power (W)",
        "Avg. DDR Phy Power (W)",
        "Avg. APU Power (W)",
        "Total Energy Consumption (J)",
        "Avg. CPU read BW (GB/s)",
        "Avg. CPU write BW (GB/s)",
        "Avg. IPU read BW (GB/s)",
        "Avg. IPU write BW (GB/s)",
        "Avg. DRAM read BW (GB/s)",
        "Avg. DRAM write BW (GB/s)",
        "Avg. DRAM Peak BW (GB/s)",
    ]
    transposed_list = list(zip(*result))
    df = pd.DataFrame(transposed_list, index=Rows, columns=["CPU","AIE"])
    df.to_csv("power_profiling.csv")

    print("Power profiling comparission file name: ", "power_profiling.csv")
    import tabulate as tab
    table_data = tab.tabulate(
                transposed_list,
                showindex = Rows,
                headers=["CPU","AIE"],
                tablefmt="grid",
                numalign="right",
                maxcolwidths=[None, 30],
            )
    print(table_data)

    #  Plot CPU Utilization
    #
    fig = go.Figure()
    fig.update_layout(
        title="CPU Utilization (%)",
        xaxis_title="sample id",
        yaxis_title="CPU Utilization (%)",
    )
    for filename, csv_file in data:
        cpu = get_cpu_utilization(csv_file)
        fig.add_trace(go.Scatter(y=smoothen(cpu), mode="lines", name=filename))
    # fig.write_html("cpu_utilization.html", auto_open=True)

    #
    #  Plot Core Utilization
    #
    fig = go.Figure()
    fig.update_layout(
        title="CPU Core Utilization (%)",
        xaxis_title="sample id",
        yaxis_title="CPU Utilization (%)",
    )
    for filename, csv_file in data:
        for i in range(8):
            cpu = get_core_utilization(csv_file, i)
            fig.add_trace(
                go.Scatter(
                    y=smoothen(cpu), mode="lines", name=filename + "_core_" + str(i)
                )
            )
    # fig.write_html("cpu_core_utilization.html", auto_open=True)

    #
    #  Plot Power Consumption
    #
    mem = go.Figure()
    mem.update_layout(
        title="Power (W)", xaxis_title="sample id", yaxis_title="Power (W)"
    )
    for filename, csv_file in data:
        cpu_power = get_cpu_gpu_power(csv_file)
        ipu_power = get_ipu_power(csv_file)
        ddr_power = get_ddr_mem_phy_power(csv_file)
        apu_power = get_apu_power(csv_file)
        mem.add_trace(
            go.Scatter(y=smoothen(cpu_power), mode="lines", name=filename + "/cpu")
        )
        mem.add_trace(
            go.Scatter(y=smoothen(ipu_power), mode="lines", name=filename + "/ipu")
        )
        mem.add_trace(
            go.Scatter(y=smoothen(ddr_power), mode="lines", name=filename + "/ddr")
        )
        mem.add_trace(
            go.Scatter(y=smoothen(apu_power), mode="lines", name=filename + "/apu")
        )
    # mem.write_html("power.html", auto_open=True)

    #
    #  Plot Bandwidth Utilization
    #
    bw = go.Figure()
    bw.update_layout(
        title="Bandwidth (GB/s)",
        xaxis_title="sample id",
        yaxis_title="Bandwidth (GB/s)",
    )
    for filename, csv_file in data:
        cpu_bw = get_cpu_rw_bw(csv_file)
        ipu_bw = get_ipu_rw_bw(csv_file)
        dram_bw = get_dram_rw_bw(csv_file)
        peak_bw = get_dram_peak_bw(csv_file)
        bw.add_trace(
            go.Scatter(y=smoothen(cpu_bw), mode="lines", name=filename + "/cpu")
        )
        bw.add_trace(
            go.Scatter(y=smoothen(ipu_bw), mode="lines", name=filename + "/ipu")
        )
        bw.add_trace(
            go.Scatter(y=smoothen(dram_bw), mode="lines", name=filename + "/dram")
        )
        bw.add_trace(
            go.Scatter(y=smoothen(peak_bw), mode="lines", name=filename + "/peak")
        )
    # bw.write_html("bandwidth.html", auto_open=True)
    #  print_full_summary(csv_file, [])
    #  plot_all(csv_file, [], smoothen)
    #  print()
