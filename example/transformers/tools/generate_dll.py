#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import os

import opt_mlp
import qlinear_experimental

transpose_dense_kernel_shapes_opt = [
    [(4, 2048), (2048, 2048)],
    [(1, 2048), (2048, 2048)],
    [(8, 2048), (2048, 2048)],
    [(8, 2048), (2048, 6144)],
    [(1, 2048), (2048, 6144)],
]

transpose_dense_kernel_shapes_llama = [
    [(8, 4096), (4096, 4096)],
    [(1, 4096), (4096, 4096)],
    # [(8, 4096), (4096, 11008)], # fails
    # [(1, 4096), (4096, 11008)], # fails
    # [(8, 4096), (4096, 32000)], # fails
    # [(8, 4096), (4096, 12288)],
    # [(1, 4096), (4096, 12288)],
    # [(8, 4096), (4096, 32768)],
    # [(1, 4096), (4096, 32768)],
]


def generate_dll_transpose_dense():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class",
        help="Generate dlls for specific class of models - opt, llama or both",
        type=str,
        default="all",
        choices=["opt", "opt_mlp_fuse", "llama", "all"],
    )
    args = parser.parse_args()
    print(f"{args}")

    dev = os.getenv("DEVICE")
    lib_path = os.getenv("PYTORCH_AIE_PATH") + "/dll/" + dev + "/qlinear/"
    if not os.path.exists(lib_path):
        os.makedirs(lib_path)

    if args.model_class == "opt_mlp_fuse" and dev == "phx":
        mlp = opt_mlp.OPTMLP(
            fc1_in_features=2048,
            fc1_out_features=8192,
            fc2_in_features=8192,
            fc2_out_features=2048,
            use_tvm_dll=False,
            use_cpp_engine=False,
        )

    else:
        if args.model_class == "opt":
            kernel_shapes = transpose_dense_kernel_shapes_opt
        elif args.model_class == "llama":
            kernel_shapes = transpose_dense_kernel_shapes_llama
        else:
            kernel_shapes = (
                transpose_dense_kernel_shapes_llama + transpose_dense_kernel_shapes_opt
            )

        for kernel_shape in kernel_shapes:
            kernel_x_shape = kernel_shape[0]
            kernel_y_shape = kernel_shape[1]
            print(
                f"Generating dll : kernel_x_shape:{kernel_x_shape} kernel_y_shape:{kernel_y_shape}"
            )
            aie_file_name = f"libGemmQnnAie_{kernel_x_shape}_{kernel_y_shape}"
            cpu_file_name = f"libGemmQnnCpu_{kernel_x_shape}_{kernel_y_shape}"
            file_names = [
                f"{aie_file_name}.dll",
                f"{aie_file_name}.lib",
                f"{aie_file_name}.exp",
                f"{cpu_file_name}.dll",
                f"{cpu_file_name}.lib",
                f"{cpu_file_name}.exp",
            ]
            for file_name in file_names:
                if os.path.exists(lib_path + file_name):
                    os.remove(lib_path + file_name)

            gemm = qlinear_experimental.QLinearExperimentalPyTiling(
                in_features=1,
                out_features=1,
                bias=False,
                device="aie",
                kernel_x_shape=kernel_x_shape,
                kernel_y_shape=kernel_y_shape,
                use_tvm_dll=False,
            )


if __name__ == "__main__":
    print("Generating dlls ... ")
    generate_dll_transpose_dense()
    print("Generated dlls ... DONE")
