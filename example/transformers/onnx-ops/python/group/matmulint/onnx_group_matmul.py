#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import argparse
import os

from onnx_group import GroupMatMulInteger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",help="Path to ONNX model", default="")
    parser.add_argument("--output",help="Output ONNX model after grouping", default="")
    args = parser.parse_args()
    print(f"{args}")

    model = args.model
    group_model = args.output
    if group_model == "":
        group_model = "./output/" + os.path.basename(model)
    dir = os.path.dirname(group_model)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    print("Grouping")
    g = GroupMatMulInteger(model)
    status = g.group()
    if status == False:
        print("Unable to perform MatMulInteger group operation....")
    else:
        print("MatMulInteger Grouping successfull....")
        g.save(group_model)