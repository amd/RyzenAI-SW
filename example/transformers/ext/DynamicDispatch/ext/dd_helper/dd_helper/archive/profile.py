##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import argparse

import numpy

import onnx_tool


def get_parser():
    parser = argparse.ArgumentParser(
        "onnx_tool",
        description="Profile MACs(FLOPs) and perform shape inference on fixed-shape ONNX model",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["profile", "export_tensors", "constant_folding", "io_modify"],
        default="profile",
        help="rm_iden: remove Identity layers",
    )
    parser.add_argument(
        "-i", "--in", dest="in_", required=True, help="path of input ONNX model"
    )
    parser.add_argument("-o", "--out", help="path to save the ONNX model with shapes")
    parser.add_argument(
        "--names", nargs="+", default=None, help="tensor names as: --names 410 420"
    )
    parser.add_argument(
        "-d",
        "--dynamic_shapes",
        nargs="+",
        default=None,
        help="dynamic shape for io tensors as: --dynamic_shapes input:f32:1x3x224x224 scale:f32:1x2:0.25x0.25 "
        "or input:1x3x224x224 input:1x3xhxw",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="path to save the ONNX model with shapes"
    )
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help="file to store the MACs result for each node. None: print to console.",
    )
    return parser


parser = get_parser()
args = parser.parse_args()


def __str2numpytype__(strtype):
    if strtype == "f32":
        return numpy.float32
    if strtype == "int32":
        return numpy.int32


def __str2list__(strshape, type):
    strs = strshape.split("x")
    shape = []
    for s in strs:
        shape.append(type(s))
    return shape


def __args2dynamicshapes__(args: [str]):
    dic = {}
    for arg in args:
        strs = arg.split(":")
        dtype = __str2numpytype__(strs[1])
        shape = __str2list__(strs[2], int)
        if len(strs) > 3:
            arr = numpy.array(__str2list__(strs[3], dtype), dtype=dtype).reshape(shape)
        else:
            arr = numpy.zeros(shape, dtype=dtype)
        dic[strs[0]] = arr
    return dic


def __args2strshapes__(args: [str]):
    dic = {}
    for arg in args:
        strs = arg.split(":")
        dic[strs[0]] = strs[1]
    return dic


if args.mode == "profile":
    if args.dynamic_shapes is not None:
        dynamic = __args2dynamicshapes__(args.dynamic_shapes)
    else:
        dynamic = None
    onnx_tool.model_profile(
        args.in_, dynamic, save_profile=args.file, save_model=args.out
    )
elif args.mode == "export_tensors":
    onnx_tool.model_export_tensors_numpy(
        args.in_, tensornames=args.names, savefolder=args.out, fp16=args.fp16
    )
elif args.mode == "constant_folding":
    onnx_tool.model_constant_folding(args.in_, args.out)
elif args.mode == "io_modify":
    if args.dynamic_shapes is not None:
        shapedic = __args2strshapes__(args.dynamic_shapes)
        onnx_tool.model_io_modify(args.in_, args.out, shapedic)
