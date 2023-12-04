import os
import numpy as np
import onnx
from onnx import numpy_helper
from pathlib import Path

# import sys
# sys.path.insert(0, '..')

from vitis_customop.utils.utils import *


class DecodeAndFilterBoxes:
    def __init__(self, boxes_tensor, score_tensor):
        self.innames = [boxes_tensor.name, score_tensor.name]
        self.outnames = ["detected_boxes"]
        self.decode_box_outnames = ["decoded_boxes"]
        self.nms_sel_box_innames = [boxes_tensor.name, self.decode_box_outnames[0]]
        self.nms_sel_box_outnames = self.outnames

        self.fname = "DecodeAndFilterBoxes"
        self.nodes = []
        self.functions = []
        self.fnodes = []

        self.invalue_info = [boxes_tensor, score_tensor]
        self.outvalue_info = [
            onnx.helper.make_tensor_value_info(
                self.outnames[0], onnx.TensorProto.FLOAT, ["num_boxes", 6]
            )
        ]
        # Data dir for anchor box data
        self.DATA_DIR = (
            os.path.dirname(os.path.abspath(__file__))
            + "/data/decode_center_size_boxes"
        )

        # Create constant nodes
        self.create_constant_nodes()
        # Create decode box nodes
        self.create_decode_box_nodes()
        # Create NMS and select box nodes
        self.create_nms_select_box_nodes()

        # Create function
        self.functions.append(self.create_func())

        # Create function node
        self.fnodes.append(self.create_func_node())

    def create_func(self):
        # Create function
        return onnx.helper.make_function(
            domain=get_domain(),
            fname=self.fname,
            inputs=self.innames,
            outputs=self.outnames,
            nodes=self.nodes,
            opset_imports=get_opsets(),
        )

    def create_func_node(self):
        # Create a node with the fuction
        return onnx.helper.make_node(
            self.fname,
            inputs=self.innames,
            outputs=self.outnames,
            name="FaceDetectPostSubgraph",
            domain=get_domain(),
        )

    def create_constant_nodes(self):
        # Create Tensors
        ## Indices
        _i64_neg1 = onnx.helper.make_tensor(
            "64_neg1", data_type=onnx.TensorProto.INT64, dims=[1], vals=[-1]
        )
        _i64_0 = onnx.helper.make_tensor(
            "64_0", data_type=onnx.TensorProto.INT64, dims=[1], vals=[0]
        )
        _i64_1 = onnx.helper.make_tensor(
            "64_1", data_type=onnx.TensorProto.INT64, dims=[1], vals=[1]
        )
        _i64_2 = onnx.helper.make_tensor(
            "64_2", data_type=onnx.TensorProto.INT64, dims=[1], vals=[2]
        )
        ## Thresholds
        _score_threshold = onnx.helper.make_tensor(
            "score_threshold", onnx.TensorProto.FLOAT, [], [0.5]
        )
        _iou_threshold = onnx.helper.make_tensor(
            "iou_threshold", onnx.TensorProto.FLOAT, [], [0.6000000238418579]
        )
        ## Number of boxes per class
        _max_output_boxes_per_class = onnx.helper.make_tensor(
            "max_output_boxes_per_class", onnx.TensorProto.INT64, [], [10]
        )

        # Create Constant nodes
        self.nodes.extend(
            [
                onnx.helper.make_node(
                    "Constant", inputs=[], outputs=["i64_neg1"], value=_i64_neg1
                ),
                onnx.helper.make_node(
                    "Constant", inputs=[], outputs=["i64_0"], value=_i64_0
                ),
                onnx.helper.make_node(
                    "Constant", inputs=[], outputs=["i64_1"], value=_i64_1
                ),
                onnx.helper.make_node(
                    "Constant", inputs=[], outputs=["i64_2"], value=_i64_2
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["score_threshold"],
                    value=_score_threshold,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["iou_threshold"],
                    value=_iou_threshold,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["max_output_boxes_per_class"],
                    value=_max_output_boxes_per_class,
                ),
            ]
        )

    def create_decode_box_nodes(self):
        input_name = self.innames[0]
        output_name = self.decode_box_outnames[0]
        self.nodes.extend(
            [
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_output_0"],
                    name="/Constant",
                    value=numpy_helper.from_array(np.array(0, dtype="int64"), name=""),
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_1_output_0"],
                    name="/Constant_1",
                    value=numpy_helper.from_array(np.array(1, dtype="int64"), name=""),
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_2_output_0"],
                    name="/Constant_2",
                    value=numpy_helper.from_array(np.array(2, dtype="int64"), name=""),
                ),
                onnx.helper.make_node(
                    "Gather",
                    inputs=[input_name, "/Constant_output_0"],
                    outputs=["/Gather_output_0"],
                    name="/Gather",
                    axis=2,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_3_output_0"],
                    name="/Constant_3",
                    value=numpy_helper.from_array(
                        np.array(10.0, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Div",
                    inputs=["/Gather_output_0", "/Constant_3_output_0"],
                    outputs=["/Div_output_0"],
                    name="/Div",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_4_output_0"],
                    name="/Constant_4",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const0.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Div_output_0", "/Constant_4_output_0"],
                    outputs=["/Mul_output_0"],
                    name="/Mul",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_5_output_0"],
                    name="/Constant_5",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const1.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Add",
                    inputs=["/Mul_output_0", "/Constant_5_output_0"],
                    outputs=["/Add_output_0"],
                    name="/Add",
                ),
                onnx.helper.make_node(
                    "Gather",
                    inputs=[input_name, "/Constant_1_output_0"],
                    outputs=["/Gather_1_output_0"],
                    name="/Gather_1",
                    axis=2,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_6_output_0"],
                    name="/Constant_6",
                    value=numpy_helper.from_array(
                        np.array(10.0, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Div",
                    inputs=["/Gather_1_output_0", "/Constant_6_output_0"],
                    outputs=["/Div_1_output_0"],
                    name="/Div_1",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_7_output_0"],
                    name="/Constant_7",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const2.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Div_1_output_0", "/Constant_7_output_0"],
                    outputs=["/Mul_1_output_0"],
                    name="/Mul_1",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_8_output_0"],
                    name="/Constant_8",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const3.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Add",
                    inputs=["/Mul_1_output_0", "/Constant_8_output_0"],
                    outputs=["/Add_1_output_0"],
                    name="/Add_1",
                ),
                onnx.helper.make_node(
                    "Gather",
                    inputs=[input_name, "/Constant_2_output_0"],
                    outputs=["/Gather_2_output_0"],
                    name="/Gather_2",
                    axis=2,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_9_output_0"],
                    name="/Constant_9",
                    value=numpy_helper.from_array(
                        np.array(5.0, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Div",
                    inputs=["/Gather_2_output_0", "/Constant_9_output_0"],
                    outputs=["/Div_2_output_0"],
                    name="/Div_2",
                ),
                onnx.helper.make_node(
                    "Exp",
                    inputs=["/Div_2_output_0"],
                    outputs=["/Exp_output_0"],
                    name="/Exp",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_10_output_0"],
                    name="/Constant_10",
                    value=numpy_helper.from_array(
                        np.array(0.5, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Exp_output_0", "/Constant_10_output_0"],
                    outputs=["/Mul_2_output_0"],
                    name="/Mul_2",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_11_output_0"],
                    name="/Constant_11",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const4.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Mul_2_output_0", "/Constant_11_output_0"],
                    outputs=["/Mul_3_output_0"],
                    name="/Mul_3",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_12_output_0"],
                    name="/Constant_12",
                    value=numpy_helper.from_array(np.array(3, dtype="int64"), name=""),
                ),
                onnx.helper.make_node(
                    "Gather",
                    inputs=[input_name, "/Constant_12_output_0"],
                    outputs=["/Gather_3_output_0"],
                    name="/Gather_3",
                    axis=2,
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_13_output_0"],
                    name="/Constant_13",
                    value=numpy_helper.from_array(
                        np.array(5.0, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Div",
                    inputs=["/Gather_3_output_0", "/Constant_13_output_0"],
                    outputs=["/Div_3_output_0"],
                    name="/Div_3",
                ),
                onnx.helper.make_node(
                    "Exp",
                    inputs=["/Div_3_output_0"],
                    outputs=["/Exp_1_output_0"],
                    name="/Exp_1",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_14_output_0"],
                    name="/Constant_14",
                    value=numpy_helper.from_array(
                        np.array(0.5, dtype="float32"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Exp_1_output_0", "/Constant_14_output_0"],
                    outputs=["/Mul_4_output_0"],
                    name="/Mul_4",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_15_output_0"],
                    name="/Constant_15",
                    value=numpy_helper.from_array(
                        np.load(os.path.join(self.DATA_DIR, "const5.npy"))
                        .astype("float32")
                        .reshape([1, 1351]),
                        name="",
                    ),
                ),
                onnx.helper.make_node(
                    "Mul",
                    inputs=["/Mul_4_output_0", "/Constant_15_output_0"],
                    outputs=["/Mul_5_output_0"],
                    name="/Mul_5",
                ),
                onnx.helper.make_node(
                    "Sub",
                    inputs=["/Add_output_0", "/Mul_3_output_0"],
                    outputs=["/Sub_output_0"],
                    name="/Sub",
                ),
                onnx.helper.make_node(
                    "Sub",
                    inputs=["/Add_1_output_0", "/Mul_5_output_0"],
                    outputs=["/Sub_1_output_0"],
                    name="/Sub_1",
                ),
                onnx.helper.make_node(
                    "Add",
                    inputs=["/Add_output_0", "/Mul_3_output_0"],
                    outputs=["/Add_2_output_0"],
                    name="/Add_2",
                ),
                onnx.helper.make_node(
                    "Add",
                    inputs=["/Add_1_output_0", "/Mul_5_output_0"],
                    outputs=["/Add_3_output_0"],
                    name="/Add_3",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_16_output_0"],
                    name="/Constant_16",
                    value=numpy_helper.from_array(
                        np.array([2], dtype="int64"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["/Sub_output_0", "/Constant_16_output_0"],
                    outputs=["/Unsqueeze_output_0"],
                    name="/Unsqueeze",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_17_output_0"],
                    name="/Constant_17",
                    value=numpy_helper.from_array(
                        np.array([2], dtype="int64"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["/Sub_1_output_0", "/Constant_17_output_0"],
                    outputs=["/Unsqueeze_1_output_0"],
                    name="/Unsqueeze_1",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_18_output_0"],
                    name="/Constant_18",
                    value=numpy_helper.from_array(
                        np.array([2], dtype="int64"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["/Add_2_output_0", "/Constant_18_output_0"],
                    outputs=["/Unsqueeze_2_output_0"],
                    name="/Unsqueeze_2",
                ),
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["/Constant_19_output_0"],
                    name="/Constant_19",
                    value=numpy_helper.from_array(
                        np.array([2], dtype="int64"), name=""
                    ),
                ),
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["/Add_3_output_0", "/Constant_19_output_0"],
                    outputs=["/Unsqueeze_3_output_0"],
                    name="/Unsqueeze_3",
                ),
                onnx.helper.make_node(
                    "Concat",
                    inputs=[
                        "/Unsqueeze_output_0",
                        "/Unsqueeze_1_output_0",
                        "/Unsqueeze_2_output_0",
                        "/Unsqueeze_3_output_0",
                    ],
                    outputs=[output_name],
                    name="/Concat",
                    axis=2,
                ),
            ]
        )

    def create_nms_select_box_nodes(self):
        innames = [self.decode_box_outnames[0], self.innames[1]]
        outnames = self.nms_sel_box_outnames
        # Create list of nodes
        nodes = []

        # Create Identity Scores and Boxes
        nd_boxes_i = onnx.helper.make_node(
            "Identity", inputs=[innames[0]], outputs=["boxes_i"], name="Id_boxes_i"
        )
        nd_score_i = onnx.helper.make_node(
            "Identity", inputs=[innames[1]], outputs=["score_i"], name="Id_score_i"
        )

        nodes.append(nd_score_i)
        nodes.append(nd_boxes_i)

        # Create Squeeze for Scores and Boxes to get rid of batch dim
        nd_score_is = onnx.helper.make_node(
            "Squeeze",
            inputs=["score_i", "i64_0"],
            outputs=["score_is"],
            name="Sscore_i",
        )
        nd_boxes_is = onnx.helper.make_node(
            "Squeeze",
            inputs=["boxes_i", "i64_0"],
            outputs=["boxes_is"],
            name="Sboxes_i",
        )

        nodes.append(nd_score_is)
        nodes.append(nd_boxes_is)

        # Create the Transpose node
        nd_score_trans = onnx.helper.make_node(
            "Transpose",
            inputs=["score_i"],
            outputs=["score_t"],
            perm=(0, 2, 1),
            name="T_score_t",
        )

        nodes.append(nd_score_trans)

        # Create the NMS node
        ### Inputs and outputs
        nms_inputs = [
            "boxes_i",
            "score_t",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ]
        nms_outputs = ["selected_indices"]
        ### Node
        nd_nms_node = onnx.helper.make_node(
            "NonMaxSuppression", nms_inputs, nms_outputs, name="NMS"
        )

        nodes.append(nd_nms_node)

        # Create Gather for boxes
        nd_gather_classes = onnx.helper.make_node(
            "Gather",
            inputs=[nms_outputs[0], "i64_1"],
            outputs=["classes_i64"],
            axis=-1,
            name="g_classes_i64",
        )

        nodes.append(nd_gather_classes)

        # Create Cast for classes
        nd_class_select = onnx.helper.make_node(
            "Cast",
            inputs=["classes_i64"],
            outputs=["class_select"],
            to=1,
            name="cs_class_sel",
        )

        nodes.append(nd_class_select)

        # Create Gather for box idx
        nd_gather_boxes_idx_us = onnx.helper.make_node(
            "Gather",
            inputs=[nms_outputs[0], "i64_2"],
            outputs=["boxes_select_us"],
            axis=-1,
            name="g_boxes_sel_us",
        )

        nodes.append(nd_gather_boxes_idx_us)

        nd_gather_boxes_idx = onnx.helper.make_node(
            "Squeeze",
            inputs=["boxes_select_us", "i64_neg1"],
            outputs=["boxes_idx"],
            name="s_boxes_idx",
        )

        nodes.append(nd_gather_boxes_idx)

        # Create Gather for box select
        nd_gather_boxes_select = onnx.helper.make_node(
            "Gather",
            inputs=["boxes_is", "boxes_idx"],
            outputs=["boxes_select"],
            axis=0,
            name="g_boxes_select",
        )

        nodes.append(nd_gather_boxes_select)

        # Create Gather for score select
        nd_gather_score_select_nm = onnx.helper.make_node(
            "Gather",
            inputs=["score_is", "boxes_idx"],
            outputs=["score_select_nm"],
            axis=0,
            name="g_score_sel_nm",
        )

        nodes.append(nd_gather_score_select_nm)

        # Reduce Max
        nd_score_select = onnx.helper.make_node(
            "ReduceMax",
            inputs=["score_select_nm"],
            outputs=["score_select"],
            axes=[-1],
            name="RM_score_sel",
        )

        nodes.append(nd_score_select)

        # Concatenate
        nd_concat_output = onnx.helper.make_node(
            "Concat",
            inputs=["boxes_select", "score_select", "class_select"],
            outputs=outnames,
            axis=-1,
            name="cat_det_boxes",
        )
        nodes.append(nd_concat_output)

        return self.nodes.extend(nodes)
