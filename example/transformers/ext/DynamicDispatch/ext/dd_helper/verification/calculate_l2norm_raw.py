import argparse
import os, re
import numpy as np
from colorama import Fore


def relative_l2_norm_error(cpu_out, aie_out):
    l2_norm_diff = np.linalg.norm((cpu_out - aie_out))
    return l2_norm_diff


def npu_vs_cpu(npu_dir, cpu_dir, verbose=False):
    # Data files with NPU outputs
    npu_out_files = os.listdir(npu_dir)
    npu_out_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    # Data files with CPU outputs
    cpu_out_files = os.listdir(cpu_dir)
    cpu_out_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Number of outputs
    n_outputs = list(
        set(
            [
                int(re.search(r"\d+_\d+", k).group().split("_")[-1])
                for k in npu_out_files
            ]
        )
    )
    n_outputs.sort()

    l2_norms = {}
    l2_norms_per_tensor = {}
    l2n = {}
    l2n_pt = {}

    for n in n_outputs:
        l2_norms[n] = []
        l2_norms_per_tensor[n] = []

    output_index = 0
    for npu_fp, cpu_fp in zip(npu_out_files, cpu_out_files):
        # NPU Data
        npu_fname = os.path.join(npu_dir, npu_fp)
        npu_data = np.fromfile(npu_fname, dtype=np.float32, sep="")

        # CPU Data
        cpu_fname = os.path.join(cpu_dir, npu_fp)
        cpu_data = np.fromfile(cpu_fname, dtype=np.float32, sep="")

        # Check size of both data
        if npu_data.size != cpu_data.size:
            raise ValueError(
                "Data size mismatch: {} vs {}".format(npu_data.size, cpu_data.size)
            )

        # Error calculation (L2 norm)
        error = relative_l2_norm_error(npu_data, cpu_data)

        if verbose:
            print(
                "- Files: {} X {}\n- L2-norm = {}\n".format(
                    os.path.abspath(npu_fname), os.path.abspath(cpu_fname), error
                )
            )

        k = output_index % len(n_outputs)
        output_index += 1

        l2_norms[k].append(error)
        l2_norms_per_tensor[k].append(error / npu_data.size)

    for n in l2_norms.keys():
        l2n[n] = np.average(l2_norms[n])
        l2n_pt[n] = np.average(l2_norms_per_tensor[n])

    return l2n, l2n_pt


def npu_cpu_vs_msft(data_dir, msft_fp32_dir, verbose=False):
    # Data files
    data_files = os.listdir(data_dir)
    data_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    # MSFT Data Files
    ms_data_files = os.listdir(msft_fp32_dir)
    ms_data_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Number of outputs
    n_outputs = list(
        set([int(re.search(r"\d+_\d+", k).group().split("_")[-1]) for k in data_files])
    )
    n_outputs.sort()

    l2_norms = {}
    l2_norms_per_tensor = {}
    l2n = {}
    l2n_pt = {}

    # Filter files based on number of outputs
    ffactor = len(data_files) // len(ms_data_files)
    data_files = data_files[::ffactor]

    for n in n_outputs:
        l2_norms[n] = []
        l2_norms_per_tensor[n] = []

    index = 1
    output_index = 0
    for fp, msfp in zip(data_files, ms_data_files):
        # CPU/NPU Data
        fname = os.path.join(data_dir, fp)
        data = np.fromfile(fname, dtype=np.float32, sep="")

        # MSFT FP32 Data
        ms_fp32_fname = os.path.join(msft_fp32_dir, msfp)
        ms_fp32_data = np.fromfile(ms_fp32_fname, dtype=np.float32, sep="")

        index += 1
        # Check size of both data
        if data.size != ms_fp32_data.size:
            raise ValueError(
                "Data size mismatch: {} vs {}".format(data.size, ms_fp32_data.size)
            )

        # Error calculation (L2 norm)
        error = relative_l2_norm_error(data, ms_fp32_data)

        if verbose:
            print(
                "- Files: {} X {}\n- L2-norm = {}\n".format(
                    os.path.abspath(fname), os.path.abspath(ms_fp32_fname), error
                )
            )

        k = output_index % len(n_outputs)
        output_index += 1
        l2_norms[k].append(error)
        l2_norms_per_tensor[k].append(error / ms_fp32_data.size)

    for n in l2_norms.keys():
        l2n[n] = np.average(l2_norms[n])
        l2n_pt[n] = np.average(l2_norms_per_tensor[n])

    return l2n, l2n_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npu-dir", help="NPU output dir path", required=False, default=None
    )
    parser.add_argument(
        "--cpu-dir", help="CPU output dir path", required=False, default=None
    )
    parser.add_argument(
        "--msft-dir",
        help="MSFT FP32 Golden output dir path",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="Display per tensor L2-norm",
        required=False,
        default=False,
        action="store_true",
    )
    # Parse args
    args = parser.parse_args()

    if args.cpu_dir and args.npu_dir:
        # L2-norm (NPU vs CPU)
        l2_norm_lst, l2_norm_per_tensor_lst = npu_vs_cpu(
            args.npu_dir, args.cpu_dir, args.verbose
        )

        print(Fore.CYAN + "\n- NPU vs CPU ..." + Fore.RESET)
        for k in l2_norm_lst.keys():
            print(
                'L2 norm average for output tensor "{}": {}'.format(k, l2_norm_lst[k])
            )
            print(
                'L2 norm average for output tensor "{}" per element: {:10f}'.format(
                    k, l2_norm_per_tensor_lst[k]
                )
            )

    if args.msft_dir and args.npu_dir:
        # L2-norm (NPU vs MSFT)
        l2_norm_lst, l2_norm_per_tensor_lst = npu_cpu_vs_msft(
            args.npu_dir, args.msft_dir, args.verbose
        )
        print(Fore.CYAN + "\n- NPU vs MSFT FP32 ..." + Fore.RESET)
        for k in l2_norm_lst.keys():
            print(
                'L2 norm average for output tensor "{}": {}'.format(k, l2_norm_lst[k])
            )
            print(
                'L2 norm average for output tensor "{}" per element: {:.10f}'.format(
                    k, l2_norm_per_tensor_lst[k]
                )
            )

    if args.msft_dir and args.cpu_dir:
        # L2-norm (CPU vs MSFT)
        l2_norm_lst, l2_norm_per_tensor_lst = npu_cpu_vs_msft(
            args.cpu_dir, args.msft_dir, args.verbose
        )
        print(Fore.CYAN + "\n- CPU vs MSFT FP32 ..." + Fore.RESET)
        for k in l2_norm_lst.keys():
            print(
                'L2 norm average for output tensor "{}": {}'.format(k, l2_norm_lst[k])
            )
            print(
                'L2 norm average for output tensor "{}" per element: {:.10f}'.format(
                    k, l2_norm_per_tensor_lst[k]
                )
            )
    print("")
