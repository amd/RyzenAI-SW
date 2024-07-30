import argparse
import os
import numpy as np
from colorama import Fore


def relative_l2_norm_error(cpu_out, aie_out):
    l2_norm_diff = np.linalg.norm((cpu_out - aie_out))
    return l2_norm_diff


def calculate_l2_norm(cpu_out_dir, vai_out_dir):
    # CPU-EP output files
    cpu_out_files = os.listdir(cpu_out_dir)
    cpu_out_files.sort()
    # VAI-EP output files
    vai_out_files = os.listdir(vai_out_dir)
    vai_out_files.sort()

    l2_norms = {}
    l2_norms_pertsr = {}
    l2_norm_avg = {}
    l2_norm_avg_pertsr = {}

    print("\n- Computing L2-norm for CPU vs VAI EP outputs ...\n")
    print("- CPU Out Dir: {}".format(os.path.abspath(cpu_out_dir)))
    print("- VAI Out Dir: {}".format(os.path.abspath(vai_out_dir)))

    # Read all outputs and compute l2-norm
    for cpu_out_file, vai_out_file in zip(cpu_out_files, vai_out_files):
        # Read each file
        if cpu_out_file != vai_out_file:
            raise ValueError(
                Fore.RED
                + "- Output filenames do not match. CPU File vs VAI File -> {} vs {}".format(
                    cpu_out_file, vai_out_file
                )
                + Fore.RESET
            )
        cpu_output = np.load(os.path.join(cpu_out_dir, cpu_out_file))
        vai_output = np.load(os.path.join(vai_out_dir, vai_out_file))

        for outname in cpu_output.keys():
            cpu_tensor = cpu_output[outname]
            vai_tensor = vai_output[outname]

            # Depad if required
            depad_vai_tensor = vai_tensor[..., : cpu_tensor.shape[-1]]

            # Error calculation (L2 norm)
            error = relative_l2_norm_error(
                cpu_tensor.flatten(), depad_vai_tensor.flatten()
            )

            if outname in l2_norms.keys():
                l2_norms[outname].append(error)
                l2_norms_pertsr[outname].append(error / depad_vai_tensor.size)
            else:
                l2_norms[outname] = [error]
                l2_norms_pertsr[outname] = [error / depad_vai_tensor.size]

    # Get l2-norm for each output tensor
    for k in l2_norms.keys():
        l2_norm_avg[k] = np.average(l2_norms[k])
        l2_norm_avg_pertsr[k] = np.average(l2_norms_pertsr[k])

    return l2_norm_avg, l2_norm_avg_pertsr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-out-dir", help="CPU output dir", required=True)
    parser.add_argument("--aie-out-dir", help="AIE output dir", required=True)

    args = parser.parse_args()
    l2_norm_avg, l2_norm_avg_pertsr = calculate_l2_norm(
        args.cpu_out_dir, args.aie_out_dir
    )

    print(Fore.CYAN)
    for k in l2_norm_avg.keys():
        print(
            "- L2 norm average for "
            + "{} ".format(k)
            + "output: "
            + "{}".format(l2_norm_avg[k])
        )
        print(
            "- L2 norm average for "
            + "{} ".format(k)
            + "per tensor: "
            + "{}".format(l2_norm_avg_pertsr[k])
        )
    print(Fore.RESET)
