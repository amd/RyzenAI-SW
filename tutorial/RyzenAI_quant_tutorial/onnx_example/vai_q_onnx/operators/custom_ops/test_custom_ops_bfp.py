#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import onnx
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto
import onnxruntime
import numpy as np
from vai_q_onnx import get_library_path


def test_custom_op():
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node("BFPFixNeuron", ["input"], ["out"],
                             domain="com.vai.quantize",
                             bit_width=13,
                             block_size=16,
                             rounding_mode=0,
                             bfp_method="to_bfp_prime_shared")
        ],
        name="test-model",
        inputs=[
            helper.make_tensor_value_info("input",
                                          TensorProto.FLOAT,
                                          shape=None)
        ],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, shape=None)
        ])
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[helper.make_operatorsetid('', 19)])
    onnx.save(model_def, "test.onnx")


def run():
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path("CPU"))
    ort_session = onnxruntime.InferenceSession(
        "test.onnx", so, providers=['CUDAExecutionProvider'])
    inpt = np.random.rand(6).astype(np.float32)
    for _ in range(5):
        ort_inputs = {"input": inpt}
        out = ort_session.run(None, ort_inputs)[0]
    print(inpt - out)


if __name__ == "__main__":
    test_custom_op()
    run()
