#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import argparse
import numpy as np
import onnx
import onnxruntime as ort
import os

from onnx import TensorProto
from onnx_group import GroupMatMulInteger
opset_imports = [onnx.helper.make_opsetid("", 14)]

def make_matmultest(model_name, M, K, N):
    X = onnx.helper.make_tensor_value_info('X',TensorProto.FLOAT,[M, K])
    dynamic_quant = onnx.helper.make_node(name="/custom/DynamicQuantizeLinear", op_type='DynamicQuantizeLinear', inputs=['X'], outputs=['Y', 'Y_SCALE', 'Y_ZP'])

    w11 = np.random.randint(low=-128, high=127, size=(K,N))
    W11 = onnx.helper.make_tensor(name="W11", data_type=TensorProto.INT8, dims=w11.shape, vals=w11.flatten().tolist())
    w11_zp = np.array([0])
    W11_ZP = onnx.helper.make_tensor(name="W11_ZP", data_type=TensorProto.INT8, dims=w11_zp.shape, vals=w11_zp.flatten().tolist())
    Z1 = onnx.helper.make_tensor_value_info('Z1',TensorProto.INT32,[M, N])
    matmul1 = onnx.helper.make_node(name="/custom/MatMulInteger1", op_type='MatMulInteger', inputs=['Y', W11.name, 'Y_ZP', W11_ZP.name], outputs=['Z1'])

    w12 = np.random.randint(low=-128, high=127, size=(K,N))
    W12 = onnx.helper.make_tensor(name="W12", data_type=TensorProto.INT8, dims=w12.shape, vals=w12.flatten().tolist())
    w12_zp = np.array([0])
    W12_ZP = onnx.helper.make_tensor(name="W12_ZP", data_type=TensorProto.INT8, dims=w12_zp.shape, vals=w12_zp.flatten().tolist())
    Z2 = onnx.helper.make_tensor_value_info('Z2',TensorProto.INT32,[M, N])
    matmul2 = onnx.helper.make_node(name="/custom/MatMulInteger2", op_type='MatMulInteger', inputs=['Y', W12.name, 'Y_ZP', W12_ZP.name], outputs=['Z2'])

    w13 = np.random.randint(low=-128, high=127, size=(K,N))
    W13 = onnx.helper.make_tensor(name="W13", data_type=TensorProto.INT8, dims=w13.shape, vals=w13.flatten().tolist())
    w13_zp = np.array([0])
    W13_ZP = onnx.helper.make_tensor(name="W13_ZP", data_type=TensorProto.INT8, dims=w13_zp.shape, vals=w13_zp.flatten().tolist())
    Z3 = onnx.helper.make_tensor_value_info('Z3',TensorProto.INT32,[M, N])
    matmul3 = onnx.helper.make_node(name="/custom/MatMulInteger3", op_type='MatMulInteger', inputs=['Y', W13.name, 'Y_ZP', W13_ZP.name], outputs=['Z3'])

    graph = onnx.helper.make_graph([dynamic_quant, matmul1, matmul2, matmul3], 'Custom', inputs=[X], outputs=[Z1, Z2, Z3], initializer=[W11, W11_ZP, W12, W12_ZP, W13, W13_ZP])
    model = onnx.helper.make_model(graph, opset_imports=opset_imports)

    onnx.save(model,model_name)


providers = ['CPUExecutionProvider']
session_options = ort.SessionOptions()

def run_matmul(x, model):
    session = ort.InferenceSession(model, providers=providers, sess_options=session_options)
    outputs = session.run(None, {'X': x.astype(np.float32)})
    return outputs

def run_test(model, group_model, M, K, N):
    x = np.random.uniform(low=0.0, high=100.0, size=(M,K))
    O1 = run_matmul(x, model)
    O2 = run_matmul(x, group_model)
    status = True
    for i in range(3):
        print(f"Diffing outputs {i}")
        if not np.array_equal(O1[i], O2[i]):
            print("\tMismatch .......")
            status = False
    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",help="Path to ONNX model", default="")
    parser.add_argument("--output",help="Output ONNX model after grouping", default="")
    parser.add_argument("--M",help="M dimension", type=int, default=3)
    parser.add_argument("--K",help="K dimension", type=int, default=5)
    parser.add_argument("--N",help="N dimension", type=int, default=5)
    args = parser.parse_args()
    print(f"{args}")

    model = args.model
    if model == "":
        model = "./input/custom.onnx"
    dir = os.path.dirname(model)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    group_model = args.output
    if group_model == "":
        group_model = "./output/" + os.path.basename(model)
    dir = os.path.dirname(group_model)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    print("Generating matmul test")
    make_matmultest(model, args.M, args.K, args.N)

    print("Grouping")
    g = GroupMatMulInteger(model)
    status = g.group()
    if status == False:
        print("Unable to do group operation. Skipping test run.....")
        exit(status)
    g.save(group_model)

    print("Testing")
    status = run_test(model, group_model, args.M, args.K, args.N)

    if status == True:
        print("Test Passed.")
    else:
        print("Test Failed.")