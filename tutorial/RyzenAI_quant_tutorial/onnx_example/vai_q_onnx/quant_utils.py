#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import tempfile
import platform
from enum import Enum
from pathlib import Path
from datetime import datetime
import math
import numpy as np
import onnx
import os
import sys
from onnx import external_data_helper
from onnx import onnx_pb as onnx_proto
import onnx.helper as helper
from onnx.reference import ReferenceEvaluator
from onnx import shape_inference, TensorProto
import onnxruntime as ort
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import QuantType
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path as ext_lib_path
from packaging import version as pv
from vai_q_onnx.version import __version__ as version
from vai_q_onnx.operators.vai_ops.qdq_ops import vai_dquantize

try:
    # The custom op library may not have been compiled
    from vai_q_onnx.gen_files import _COP_DOMAIN, _COP_VERSION, _DEVICE_SUFFIX
    from vai_q_onnx.gen_files import get_library_path as vai_library_path
except:
    # Try to import from original path but may raise an error when call get_library_path
    from vai_q_onnx.operators.custom_ops import _COP_DOMAIN, _COP_VERSION, _DEVICE_SUFFIX
    vai_library_path = None

logger = logging.getLogger(__name__)

__producer__ = "vai_q_onnx"
__version__ = version

VAI_DOMAIN = "ai.onnx.contrib"  # domain for vai ops that inherited from python class
COP_DOMAIN = _COP_DOMAIN  # domain for custom ops that implemented using c api
COP_QUANT_OP_NAME = "VitisQuantizeLinear"
COP_DEQUANT_OP_NAME = "VitisDequantizeLinear"
FIX_OP_NAME = "FixNeuron"
BFPFIX_OP_NAME = "BFPFixNeuron"
CUSTOM_VAI_DOMAIN = "com.vai.quantize"

HARD_SIGMOID_SCALE = (2731. / 16384.) / (1. / 6.)
annotate_op_type = [
    "Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul",
    "Gemm"
]
avg_pool_op_type = ["AveragePool", "GlobalAveragePool"]
remove_qdq_op_type = ["Relu"]



class Int16Method(Enum):
    MinMax = 0


class PowerOfTwoMethod(Enum):
    NonOverflow = 0
    MinMSE = 1


class VitisQuantType(Enum):
    QInt16 = 3
    QUInt16 = 4
    QInt32 = 5
    QUInt32 = 6
    QFloat16 = 7
    QBFloat16 = 8

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(t):
        try:
            return VitisQuantType[t]
        except KeyError:
            raise ValueError()


class VitisQuantFormat(Enum):
    QDQ = 2
    FixNeuron = 3
    # add BFP
    BFPFixNeuron = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(format):
        try:
            return VitisQuantFormat[format]
        except KeyError:
            raise ValueError()


ONNX_TYPE_TO_NP_TYPE = {
    onnx_proto.TensorProto.INT8:
        np.dtype("int8"),
    onnx_proto.TensorProto.UINT8:
        np.dtype("uint8"),
    onnx_proto.TensorProto.INT16:
        np.dtype("int16"),
    onnx_proto.TensorProto.UINT16:
        np.dtype("uint16"),
    onnx_proto.TensorProto.INT32:
        np.dtype("int32"),
    onnx_proto.TensorProto.UINT32:
        np.dtype("uint32"),
    onnx_proto.TensorProto.FLOAT16:
        np.dtype("float16"),
    # This is mismatched conversion,
    # numpy does not support yet
    onnx_proto.TensorProto.BFLOAT16:
        np.dtype("float16"),
}

ONNX_INT_TYPE_RANGE = {
    onnx_proto.TensorProto.UINT8: (0, 255),
    onnx_proto.TensorProto.INT8: (-128, 127),
    onnx_proto.TensorProto.UINT16: (0, 65535),
    onnx_proto.TensorProto.INT16: (-32768, 32767),
    onnx_proto.TensorProto.UINT32: (0, 2**32 - 1),
    onnx_proto.TensorProto.INT32: (-2**31, 2**31 - 1),
}

ONNX_INT_TYPE_SYMMETRIC_RANGE = {
    onnx_proto.TensorProto.INT8: (-127, 127),
    onnx_proto.TensorProto.INT16: (-32767, 32767),
    onnx_proto.TensorProto.INT32: (-(2**31 - 1), 2**31 - 1),
}

ONNX_INT_TYPE_REDUCED_RANGE = {
    onnx_proto.TensorProto.UINT8: (0, 127),
    onnx_proto.TensorProto.INT8: (-64, 64),
    onnx_proto.TensorProto.UINT16: (0, 32767),
    onnx_proto.TensorProto.INT16: (-16384, 16384),
    onnx_proto.TensorProto.UINT32: (0, 2**31 - 1),
    onnx_proto.TensorProto.INT32: (-2**30, 2**30),
}

ONNX_WBIT_QTYPES_LIST = [
    onnx_proto.TensorProto.UINT16,
    onnx_proto.TensorProto.INT16,
    onnx_proto.TensorProto.UINT32,
    onnx_proto.TensorProto.INT32,
    onnx_proto.TensorProto.FLOAT16,
    onnx_proto.TensorProto.BFLOAT16,
]

ONNX_FP_QTYPES_LIST = [
    onnx_proto.TensorProto.FLOAT16,
    onnx_proto.TensorProto.BFLOAT16,
]


def get_tensor_type_from_qType(quant_type):
    if quant_type == QuantType.QUInt8:
        return TensorProto.UINT8
    if quant_type == QuantType.QInt8:
        return TensorProto.INT8
    if quant_type == VitisQuantType.QUInt16:
        return TensorProto.UINT16
    if quant_type == VitisQuantType.QInt16:
        return TensorProto.INT16
    if quant_type == VitisQuantType.QUInt32:
        return TensorProto.UINT32
    if quant_type == VitisQuantType.QInt32:
        return TensorProto.INT32
    if quant_type == VitisQuantType.QFloat16:
        return TensorProto.FLOAT16
    if quant_type == VitisQuantType.QBFloat16:
        return TensorProto.BFLOAT16
    raise ValueError(f"Unexpected value qtype={quant_type!r}.")


def get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=False):
    """
    Return qmin and qmax, the minimum and maximum value representable by the given qType
    :parameter qType: onnx.onnx_pb.TensorProto.UINT8/16 or onnx.onnx_pb.TensorProto.UINT8/16
    :return: qmin, qmax
    """
    if qType in ONNX_FP_QTYPES_LIST:
        raise NotImplementedError(
            "This function is not implemented for (b)float 16 as not needed.")

    qrange = None

    if reduce_range:
        qrange = ONNX_INT_TYPE_REDUCED_RANGE.get(qType)
    elif symmetric and qType in ONNX_INT_TYPE_SYMMETRIC_RANGE:
        qrange = ONNX_INT_TYPE_SYMMETRIC_RANGE[qType]
    else:
        qrange = ONNX_INT_TYPE_RANGE.get(qType)

    if not qrange:
        raise ValueError(
            f"Unexpected data type {qType} requested. Only INT8, UINT8, INT16, and UINT16 are supported."
        )

    return qrange


def get_qrange_for_qType(qType, reduce_range=False, symmetric=False):
    """
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    """
    qmin, qmax = get_qmin_qmax_for_qType(qType,
                                         reduce_range,
                                         symmetric=symmetric)
    return qmax - qmin


def quantize_nparray(qType, arr, scale, zero_point, low=None, high=None):
    assert (
        qType in ONNX_TYPE_TO_NP_TYPE
    ), f"Unexpected data type {qType} requested. Only INT8, UINT8, INT16, UINT16, FLOAT16, and BFLOAT16 are supported."

    if qType in ONNX_FP_QTYPES_LIST:
        arr_fp32 = arr.astype(np.float32) / scale + zero_point
        onnx_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
                "qu",
                [
                    helper.make_tensor_value_info(
                        "X", onnx_proto.TensorProto.FLOAT, None)
                ],
                [helper.make_tensor_value_info("Y", qType, None)],
            ))
        ref = ReferenceEvaluator(onnx_model)
        return ref.run(None, {"X": arr_fp32})[0]
    else:
        dtype = ONNX_TYPE_TO_NP_TYPE[qType]
        (qmin, qmax) = get_qmin_qmax_for_qType(qType,
                                               reduce_range=False,
                                               symmetric=True)

        cliplow = max(qmin, low) if low is not None else qmin
        cliphigh = min(qmax, high) if high is not None else qmax
        arr_fp32 = np.asarray((arr.astype(np.float32) / scale).round() +
                              zero_point)
        np.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
        return arr_fp32.astype(dtype)


class RandomDataReader(CalibrationDataReader):
    """
    A CalibrationDataReader using random data for rapid quantiation.
    """

    def __init__(self, model_path, input_shape=[], input_data_range=None):
        """
        :param model_path : Full path of the input model.
        :param input_shape: If dynamic axes of inputs require specific value, users should provide its shapes.
                            The basic format of shape for single input is `list(int)` or `tuple(int)`,
                            and all dimensions should have concrete values (batch dimensions can be set to 1).
                            For example, input_shape=[1, 3, 224, 224] or input_shape=(1, 3, 224, 224).
                            If the model has multiple inputs, it can be fed in `list(shape)` format,
                            where the list order is the same as the onnxruntime got inputs.
                            For example, input_shape=[[1, 1, 224, 224], [1, 2, 224, 224]] for 2 inputs.
                            Moreover, it is possible to use `dict{name:shape}` to specify a certain input,
                            for example, input_shape={"image":[1, 3, 224, 224]} for the input named "image".
        :param input_data_range: How to deal with input data range in the generated random data.
                            Default is none which means ignore data type, otherwise consider data type.
        """
        self._model_path = model_path
        self._input_shape = input_shape
        self._input_data_range = input_data_range

        self.enum_data_dicts = None
        self.batch_size = 1

    def _parse_input_shape(self, input_index, input_name):
        """
        Parse input shape of model from user's input
        :param input_index: the input index in session.get_inputs()
        :param input_name: the input name string
        :return: input shape required for the input node
        """

        def _deal_shape_value(list_or_tuple_shape):
            if not isinstance(list_or_tuple_shape, (list, tuple)):
                logger.warning(
                    "Invalid input shape {}".format(list_or_tuple_shape))
                return []

            input_shape = []
            for index, shape in enumerate(list_or_tuple_shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)
                    else:
                        logger.warning(
                            "Invalid input shape {} in #{} : {}".format(
                                list_or_tuple_shape, index, shape))
                        return []
            return input_shape

        if isinstance(self._input_shape, dict):
            if input_name in self._input_shape.keys():
                return _deal_shape_value(self._input_shape[input_name])
        elif all(isinstance(n, (list, tuple)) for n in self._input_shape):
            if input_index < len(self._input_shape):
                return _deal_shape_value(self._input_shape[input_index])
        else:
            return _deal_shape_value(self._input_shape)

        return []

    def _get_input_name(self, input_node):
        """
        :param input_node: the input node
        :return: name of the input node
        """
        input_name = input_node.name
        return input_name

    def _get_input_shape(self, input_node):
        """
        :param input_node: the input node
        :return: input shape of the input node
        """
        input_shape = []

        if len(input_node.shape):
            for index, shape in enumerate(input_node.shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(
                            self.batch_size)  # default batch size
                    elif index == 1:
                        if len(input_node.shape) == 2:
                            input_shape.append(16)  # maybe sequence length
                        elif len(input_node.shape) == 4:
                            input_shape.append(3)  # maybe image channel
                        else:
                            input_shape.append(1)
                    elif index == 2:
                        if len(input_node.shape) == 4:
                            input_shape.append(32)  # maybe image height
                        else:
                            input_shape.append(1)
                    elif index == 3:
                        if len(input_node.shape) == 4:
                            input_shape.append(32)  # maybe image width
                        else:
                            input_shape.append(1)
                    else:
                        input_shape.append(1)  # unknown or None

        if input_shape == []:
            # workaround empty shape
            return [self.batch_size]
        else:
            return input_shape

    def _get_input_type(self, input_node):
        """
        :param input_node: the input node
        :return: data type of the input node
        """
        input_type = None

        if 'tensor(int8)' in input_node.type:
            input_type = np.int8
        elif 'tensor(uint8)' in input_node.type:
            input_type = np.uint8
        elif 'tensor(int16)' in input_node.type:
            input_type = np.int16
        elif 'tensor(uint16)' in input_node.type:
            input_type = np.uint16
        elif 'tensor(int32)' in input_node.type:
            input_type = np.int32
        elif 'tensor(uint32)' in input_node.type:
            input_type = np.uint32
        elif 'tensor(int64)' in input_node.type:
            input_type = np.int64
        elif 'tensor(uint64)' in input_node.type:
            input_type = np.uint64
        elif 'tensor(float16)' in input_node.type:
            input_type = np.float16
        elif 'tensor(float)' in input_node.type:
            input_type = np.float32
        elif 'tensor(double)' in input_node.type:
            input_type = np.double
        elif 'tensor(bool)' in input_node.type:
            input_type = np.bool

        return input_type

    def get_next(self):
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_dicts is None:
            so = ort.SessionOptions()
            so.register_custom_ops_library(ext_lib_path())
            if vai_library_path is not None:
                if platform.system().lower() == 'windows':
                    pass
                else:
                    so.register_custom_ops_library(vai_library_path())

            session = ort.InferenceSession(self._model_path,
                                           so,
                                           providers=['CPUExecutionProvider'])
            enum_data = {}
            for input_index, input_node in enumerate(session.get_inputs()):
                input_name = self._get_input_name(input_node)
                input_shape = self._parse_input_shape(input_index, input_name)
                if input_shape == [] or input_shape is None:
                    input_shape = self._get_input_shape(input_node)
                input_type = self._get_input_type(input_node)

                if input_shape is not None:
                    if 'tensor(string)' in input_node.type:
                        input_data = np.chararray(tuple(input_shape))
                    else:
                        if self._input_data_range is None:
                            input_data = np.random.random(input_shape).astype(
                                input_type)
                        else:
                            # TODO : adaptive to more data type
                            if 'uint' in input_node.type:
                                input_data = np.random.randint(
                                    -128, high=128,
                                    size=input_shape).astype(input_type)
                            elif 'int' in input_node.type:
                                input_data = np.random.randint(
                                    0, high=256,
                                    size=input_shape).astype(input_type)
                            else:
                                input_data = np.random.random(
                                    input_shape).astype(input_type)
                else:
                    logger.error(
                        "Unsupported input name {} shape {} type {} ".format(
                            input_node.name, input_node.shape, input_node.type))
                enum_data[input_name] = input_data
                logger.info("Random input name {} shape {} type {} ".format(
                    input_name, input_shape, input_type))
            self.enum_data_dicts = iter([enum_data])

        return next(self.enum_data_dicts, None)


def infer_shape(model):
    """
    :param model: the source model
    :return: the target model contains inferred shape
    """
    inferred_onnx_model = shape_inference.infer_shapes(model)
    return inferred_onnx_model


def get_datatype_shape(tensor):
    """
    :param tensor: the input tensor
    :return: datatype and shape of the tensor
    """
    name = tensor.name
    elem_type_num = tensor.type.tensor_type.elem_type
    data_type = TensorProto.DataType.Name(elem_type_num).lower()
    data_type = data_type if data_type != 'float' else 'float32'
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    shape = [dims[i].dim_value if dims[i].dim_value else -1 for i in range(n)]
    return (data_type, shape)


def dump_model(model,
               dump_data_reader=None,
               random_data_reader_input_shape=[],
               dump_float=False,
               output_dir='./dump_results'):
    """
    This function dumps the simulation results of the quantized model,
    including weights and activation results.
    :param model: the input model
    :param dump_data_reader: data reader for dumpping
    :param random_data_reader_input_shape: if use internal random data reader,
           this is used to configure input node's shape
    :param dump_float: dump results of the float model or not
    :param output_dir: output directory for results
    """
    if isinstance(model, str):
        model_path = model
        model = onnx.load(model)
    else:
        logger.error("The model requires a string of the model path")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # modify_output_nodes
    fn_node_pos = {}
    has_fixneuron = False
    for n in model.graph.node:
        if n.op_type == "FixNeuron":
            fn_node_pos[n.output[0]] = 2**int(n.attribute[1].s)
            has_fixneuron = True
    if not has_fixneuron:
        if not dump_float:
            logger.error(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model."
            )
            return
        else:
            logger.warning(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model "
                "if you want to dump the quantized tensor value.")
            logger.info(
                "The float output results of each node in the model will be saved. "
            )
    node_output = []
    model.graph.ClearField("output")
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            node_output.append(output)
    tmp_dump_model = str(Path(output_dir) / "./tmp_dump_model.onnx")
    onnx.save(model, tmp_dump_model)

    so = ort.SessionOptions()
    so.register_custom_ops_library(ext_lib_path())
    if vai_library_path is not None:
        if platform.system().lower() == 'windows':
            logger.warning("No custom op library was built for this platform")
        else:
            so.register_custom_ops_library(vai_library_path())

    sess = ort.InferenceSession(tmp_dump_model,
                                so,
                                providers=['CPUExecutionProvider'])

    if dump_data_reader is None:
        dump_data_reader = RandomDataReader(
            model_path, input_shape=random_data_reader_input_shape)

    if isinstance(dump_data_reader, CalibrationDataReader):
        input = dump_data_reader.get_next()
        if not input:
            logger.error("dump_data_reader returned None, "
                         "please confirm if the dump_data_reader is correct")
        else:
            logger.info("Dumping activations and weights...")
            results_outputs = sess.run(None, input)
            for node, res in zip(node_output, results_outputs):
                filename = os.path.join(output_dir, node.replace('/', '_'))
                res = res.flatten()
                if node in fn_node_pos:
                    res_q = np.round(res * fn_node_pos[node])
                    res_q = res_q.clip(-128, 127)
                    res_q.astype(np.int8).tofile(filename + ".bin")
                    np.savetxt(filename + ".txt",
                               res_q.astype(np.int8),
                               fmt="%s",
                               delimiter=",")
                if dump_float:
                    res.tofile(filename + '_float.bin')
                    np.savetxt(filename + "_float.txt",
                               res,
                               fmt="%s",
                               delimiter=",")
            os.remove(tmp_dump_model)
    else:
        logger.error("dump_data_reader is used for the dumping process. "
                     "It should be an instance of CalibrationDataReader.")


def is_approximately_equal(a, b, epsilon=1e-6):
    """
    :param a: scalar input
    :param b: scalar input
    :param epsilon: difference tolerance
    :return: equal or not
    """
    if a is None or b is None:
        return False
    return abs(a - b) < epsilon


def check_reduce_mean_condition(model, node):
    has_axes_attr = any(attr.name == 'axes' for attr in node.attribute)
    has_axes_2_3_attr = any(
        attr.name == 'axes' and len(attr.ints) == 2 and attr.ints == [2, 3]
        for attr in node.attribute)
    has_keepdims_attr = any(attr.name == 'keepdims' for attr in node.attribute)
    has_keepdims_1_attr = any(
        attr.name == 'keepdims' and attr.i == 1 for attr in node.attribute)

    if has_axes_attr:
        if has_axes_2_3_attr and (not has_keepdims_attr or has_keepdims_1_attr):
            return True
    # Handling opset >= 18 for Reduce Mean
    elif (not has_keepdims_attr or has_keepdims_1_attr) and len(
            node.input) == 2:
        for init in model.graph.initializer:
            if init.name == node.input[1]:
                axes = onnx.numpy_helper.to_array(init).tolist()
                if axes == [2, 3]:
                    return True

    return False


def check_hard_sigmoid_condition(node):
    """
    :param node: node object
    :return: hard sigmoid or not
    """
    has_beta_attr = any(attr.name == 'beta' for attr in node.attribute)
    has_beta_0_5_attr = any(
        attr.name == 'beta' and is_approximately_equal(attr.f, 0.5)
        for attr in node.attribute)
    has_alpha_attr = any(
        attr.name == 'alpha' and is_approximately_equal(attr.f, 1. / 6.)
        for attr in node.attribute)
    if (not has_beta_attr or has_beta_0_5_attr) and has_alpha_attr:
        return True
    return False


def is_leaky_relu_with_alpha(node, alpha_value=0.1):
    """
    :param node: node object
    :param alpha_value: DPU supported alpha value
    :return: the Leaky ReLU node has a approximately alpha or not
    """
    if node.op_type == "LeakyRelu":
        for attr in node.attribute:
            if attr.name == "alpha" and is_approximately_equal(
                    attr.f, alpha_value):
                return True
    return False


def is_clip_with_min_max(model, node, min_value=0.0, max_value=6.0):
    """
    :param model: model object
    :param node: node object
    :param min_value: supported minimum value of Clip
    :param max_value: supported maximum value of Clip
    :return: the Clip node has supported min and max value or not
    """
    if node.op_type == "Clip" and len(node.input) == 3:
        min_input = node.input[1]
        max_input = node.input[2]

        for init in model.graph.initializer:
            if init.name == min_input:
                try:
                    min = onnx.numpy_helper.to_array(init).item()
                except:
                    continue
                if is_approximately_equal(min, min_value):
                    for init2 in model.graph.initializer:
                        if init2.name == max_input:
                            try:
                                max = onnx.numpy_helper.to_array(init2).item()
                            except:
                                continue
                            if is_approximately_equal(max, max_value):
                                return True

    return False


def is_node_needs_annotated(model, node):
    """
    :param model: model object
    :param node: node object
    :return: the node needs annotated or not
    """

    if (node.op_type in remove_qdq_op_type or
            is_clip_with_min_max(model, node)):
        return True
    return False


def get_annotate_tensors(model):
    """
    Find patterns in the model where qdq needs to be removed, and then return the corresponding tensor names
    annotate_tensors refers to the tensors associated with the input of the qdq that need to be removed
    :param model: model object
    :return: the annotate tensors
    """
    matching_output_tensor = []
    pad_output_tensor = []
    for node in model.graph.node:
        if node.op_type in annotate_op_type:
            matching_output_tensor.append(node.output[0])
        elif node.op_type == "Pad":
            pad_output_tensor.append(node.output[0])

    annotate_tensors = []
    for node in model.graph.node:
        if ((is_node_needs_annotated(model, node) and
             node.input[0] in matching_output_tensor) or
            (node.op_type in avg_pool_op_type and
             node.input[0] in pad_output_tensor)):
            annotate_tensors.append(node.input[0])
    return annotate_tensors


def get_qdq_to_remove(model, relu_input):
    """
    Return the names of nodes to be removed and a dictionary for converting input tensors
    :param model: model object
    :param relu_input: the ReLU node inputs list
    :return: de-quantize & quantize nodes to remove and node mapping dict
    """
    q_nodes_to_remove = []
    dq_nodes_to_remove = []
    q_nodes_output_to_remove = []
    input_node_mapping = {}
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "VitisQuantizeLinear"
                           ) and node.input[0] in relu_input:
            input_node_mapping[node.input[0]] = node.output[0]
            q_nodes_to_remove.append(node)
            q_nodes_output_to_remove.append(node.output[0])
    for node in model.graph.node:
        if node.op_type in ("DequantizeLinear", "VitisDequantizeLinear"
                           ) and node.input[0] in q_nodes_output_to_remove:
            for k, v in input_node_mapping.items():
                if v == node.input[0]:
                    input_node_mapping[k] = node.output[0]
            dq_nodes_to_remove.append(node)
    return dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping


def remove_nodes(model, nodes_list):
    """
    Delete nodes according to the nodes in the list
    :param model: model object
    :param nodes_list: nodes list to remove
    :return: the model that has removed some nodes
    """
    for node in nodes_list:
        model.graph.node.remove(node)
    return model


def remove_initializers(model, init_list):
    """
    Delete initializers according to the initializer in the list
    :param model: model object
    :param init_list: initializer's name list to remove
    :return: the model that has removed some initializers
    """
    for init in init_list:
        for i in model.graph.initializer:
            if init == i.name:
                model.graph.initializer.remove(i)
                break
        for input in model.graph.input:
            if input.name == init:
                model.graph.input.remove(input)
                break
    return model


def modified_annotate_input(model, input_node_mapping):
    """
    Modify the input of ReLU to the output of annotate op, and delete QDQ
    :param model: model object
    :param input_node_mapping: input node mapping dict
    :return: the modified model
    """

    for node in model.graph.node:
        if is_node_needs_annotated(model,
                                   node) or node.op_type in avg_pool_op_type:
            for k, v in input_node_mapping.items():
                if v == node.input[0]:
                    node.input[0] = k
    return model


def scale2pos(scale):
    """
    Obtain the fixed-point position corresponding to the scale.
    To avoid generating infinity during computations,
    the range of scale is limited.
    :param scale: the scale
    :return: the fixed-point position
    """
    scale = min(max(scale, float(2**-127)), float(2**127))
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos):
    """
    Obtain the scale corresponding to the fixed-point position.
    :param scale: the fixed-point position
    :return: the scale
    """
    return float(np.power(2.0, -pos))


def compute_scale_zp_pof2s(rmin,
                           rmax,
                           qmin,
                           qmax,
                           element_type,
                           method,
                           symmetric=False,
                           use_pof2s=True):
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s] of pof2s

    """

    if qmin > 0 or qmax < 0:
        raise ValueError(
            f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}"
        )

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    # Ensure that rmax-rmin is less than or equal to sys.float_info.max
    if rmin == float("-inf"):
        rmin = -sys.float_info.max / 2
    if rmax == float("inf"):
        rmax = sys.float_info.max / 2

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = (rmax - rmin) / float(qmax - qmin)

    if math.isnan(scale):
        logger.error("NaN detected, please check the correctness of the model")

    if scale < np.finfo(np.float32).tiny:
        scale = 1.0
        zero_point = 0
        return [zero_point, scale]
    else:
        zero_point = round(qmin - rmin / scale)

    if use_pof2s == False:
        return [zero_point, scale]

    # Power-of-2 scale calculation
    if method in PowerOfTwoMethod:
        pos = scale2pos(scale)
        pof2_scale = pos2scale(pos)
        new_rmin = min((qmin - zero_point) * pof2_scale, 0)
        new_zero_point = round(qmin - new_rmin / pof2_scale)
        # To meet hardware's requirements
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and new_zero_point == 127:
            new_zero_point = 128

        return [new_zero_point, pof2_scale]
    elif method in Int16Method:
        M, N, diff = find_int16_scale(scale)
        int16_scale = M / 2**N
        logger.debug(f"Find the {M} / 2 ** {N} that is closest to scale {scale}"
                     f"with the difference being {diff}")
        if int16_scale < np.finfo(np.float32).tiny:
            int16_scale = 1 / 2**14
            # zero_point = 0
            # return [zero_point, scale]
        new_rmin = min((qmin - zero_point) * int16_scale, 0)
        new_zero_point = round(qmin - new_rmin / int16_scale)

        return [new_zero_point, int16_scale]
    else:
        logger.error(f"{method} is not supported yet.")
        return [0, 1]


def compute_scale_zp_fp(rmin, rmax, element_type, symmetric=True):
    """Calculate the scale and zero point for a float type.

    :param rmin: minimum value of r
    :param rmax: maximum value of r
    :param element_type: the element data type of the tensor to quantize
    :return: zero and scale [z, s] of pof2s
    """
    if element_type not in ONNX_FP_QTYPES_LIST:
        raise ValueError(
            f"Quantization to element_type={element_type} not implemented.")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    # Ensure that rmax-rmin is less than or equal to sys.float_info.max
    if rmin == float("-inf"):
        rmin = -sys.float_info.max / 2
    if rmax == float("inf"):
        rmax = sys.float_info.max / 2

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = 1.0

    if symmetric:
        zero_point = 0
    else:
        rmid = (rmax + rmin) / 2
        zero_point = 0 - rmid / scale

    return [zero_point, scale]


def dequantize_data(data, scale, zero_point):
    """
    :param data: the input data
    :param scale: the scale for quantization
    :param zero_point: the zero point for quantization
    :return: the de-quantized data
    """
    data = data.astype(np.float32)
    deq_arr = (data - zero_point) * scale
    return deq_arr.astype(np.float32)


def quantize_data_pof2s(data,
                        qType,
                        symmetric,
                        reduce_range=False,
                        method=PowerOfTwoMethod.NonOverflow,
                        pos_range=5,
                        use_pof2s=True):
    """
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8/16 and INT8/16
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8/16.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """

    rmin = 0
    rmax = 0
    zero_point = 0
    scale = 1.0

    if isinstance(data, np.ndarray) and data.size:
        rmin = data.min()
        rmax = data.max()
    elif isinstance(data, list) and len(data):
        rmin = min(data)
        rmax = max(data)
    else:
        logger.warning("Invalid data for quantization {} {}.".format(
            type(data),
            len(data) if isinstance(data, list) else data.shape))
        return rmin, rmax, zero_point, scale, quantize_nparray(
            qType, np.asarray(data), scale, zero_point)

    if qType in ONNX_FP_QTYPES_LIST:
        zero_point, scale = compute_scale_zp_fp(rmin,
                                                rmax,
                                                qType,
                                                symmetric=symmetric)
        quantized_data = quantize_nparray(qType, np.asarray(data), scale,
                                          zero_point)
        return rmin, rmax, zero_point, scale, quantized_data

    qmin, qmax = get_qmin_qmax_for_qType(qType,
                                         reduce_range,
                                         symmetric=symmetric)
    zero_point, scale = compute_scale_zp_pof2s(rmin,
                                               rmax,
                                               qmin,
                                               qmax,
                                               qType,
                                               method,
                                               symmetric=symmetric,
                                               use_pof2s=use_pof2s)

    quantized_data = quantize_nparray(qType, np.asarray(data), scale,
                                      zero_point)

    if method == PowerOfTwoMethod.NonOverflow:
        return rmin, rmax, zero_point, scale, quantized_data
    elif method == PowerOfTwoMethod.MinMSE:
        scale_mse = scale
        zp_mse = zero_point
        quantized_data_mse = quantized_data
        diff_min = float("inf")
        for i in range(pos_range):
            new_scale = pos2scale(scale2pos(scale) + i - 1)
            rmin = (qmin - zero_point) * new_scale

            new_quantized_data = quantize_nparray(qType, np.asarray(data),
                                                  new_scale, zp_mse)
            diff = np.sum(
                (dequantize_data(new_quantized_data, new_scale, zp_mse) -
                 np.asarray(data))**2)
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
                quantized_data_mse = new_quantized_data

        rmin_mse = (qmin - zp_mse) * scale_mse
        rmax_mse = (qmax - zp_mse) * scale_mse
        return rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse
    elif method == Int16Method.MinMax:
        return rmin, rmax, zero_point, scale, quantized_data
    else:
        logger.error(f"{method} is not supported yet")


def get_exclude_nodes(model_path, input_nodes, output_nodes):
    """
    Return the nodes to be excluded based on the given input and output nodes.
    :param model_path: the model path
    :param input_nodes: the nodes to start quantizing
    :param zero_point: the nodes to terminate quantizing
    :return: the nodes excluded from quantization
    """

    def update_exclude_input_nodes(exclude_nodes, name_list, name, input_nodes):
        index = name_list.index(name)
        exclude_nodes_i = name_list[:index]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_i))
        exclude_nodes = list(set(exclude_nodes) - set(input_nodes))
        return exclude_nodes

    def update_exclude_output_nodes(exclude_nodes, name_list, name,
                                    output_nodes):
        index = name_list.index(name) + 1
        exclude_nodes_o = name_list[index:]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
        exclude_nodes = list(set(exclude_nodes) - set(output_nodes))
        return exclude_nodes

    model = ONNXModel(onnx.load(model_path))
    model.topological_sort()

    model_input_to_node = {}
    model_output_to_node = {}
    name_list = []
    exclude_nodes = []

    for i in model.model.graph.input:
        model_input_to_node[i.name] = []
    for o in model.model.graph.output:
        model_output_to_node[o.name] = []
    for n in model.model.graph.node:
        for i in n.input:
            for k, v in model_input_to_node.items():
                if i == k:
                    model_input_to_node[k].append(n.name)
        for o in n.output:
            for k, v in model_output_to_node.items():
                if o == k:
                    model_output_to_node[k].append(n.name)
        name_list.append(n.name)

    if input_nodes:
        for name in input_nodes:
            if name in name_list:
                exclude_nodes = update_exclude_input_nodes(
                    exclude_nodes, name_list, name, input_nodes)
            elif name in model_input_to_node:
                for n in model_input_to_node[name]:
                    exclude_nodes = update_exclude_input_nodes(
                        exclude_nodes, name_list, n, model_input_to_node[name])
            elif name in model_output_to_node:
                for n in model_output_to_node[name]:
                    exclude_nodes = update_exclude_input_nodes(
                        exclude_nodes, name_list, n, model_output_to_node[name])
            else:
                logger.warning(
                    f"Fail to find the {name} in {model_path}, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
                )

    if output_nodes:
        for name in output_nodes:
            if name in name_list:
                exclude_nodes = update_exclude_output_nodes(
                    exclude_nodes, name_list, name, output_nodes)
            elif name in model_output_to_node:
                for n in model_output_to_node[name]:
                    exclude_nodes = update_exclude_output_nodes(
                        exclude_nodes, name_list, n, model_output_to_node[name])
            elif name in model_input_to_node:
                for n in model_input_to_node[name]:
                    exclude_nodes = update_exclude_output_nodes(
                        exclude_nodes, name_list, n, model_input_to_node[name])
            else:
                logger.warning(
                    f"Fail to find the {name} in {model_path}, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
                )
    return exclude_nodes


def run_onnx_model(model_path, data_reader):
    """
    Check if the input ONNX can run successfully
    :param model_path: the model path
    :param data_reader: the data reader for feeding data
    """
    try:
        sess = ort.InferenceSession(model_path,
                                    providers=['CPUExecutionProvider'])
        inputs = data_reader.get_next()
        output = sess.run(None, inputs)
        if output:
            logger.info(
                f"The input ONNX model {model_path} can run inference successfully"
            )
        else:
            logger.warning(
                f"Fail to run inference, please check the {model_path} and the 'calibration_data_reader'."
            )
    except Exception as e:
        logger.error(
            f"Fail to run inference for {model_path}. Exception: {e}. Please check the {model_path} and the 'calibration_data_reader'."
        )


def check_onnx_model(model_path):
    """
    Check if the input ONNX can create InferenceSession successfully
    :param model_path: the model path
    """
    try:
        sess = ort.InferenceSession(model_path,
                                    providers=['CPUExecutionProvider'])
        logger.info(
            f"The input ONNX model {model_path} can create InferenceSession successfully"
        )

    except Exception as e:
        logger.error(
            f"Fail to create InferenceSession for {model_path}. Exception: {e}. Please check the {model_path}."
        )


def dpu_leaky_relu_alpha(x):
    """
    This function implements a DPU-specific Leaky ReLU activation with alpha value correction.
    """
    rounded_value = round(x * 256)
    return rounded_value / 256.


def get_model_node_name_dict(model):
    model_node_name_dict = dict()
    for node in model.node:
        if node.name and not model_node_name_dict.get(node.name):
            model_node_name_dict[node.name] = node
        else:
            if not node.name and node.output[0]:
                model_node_name_dict[node.output[0]] = node
            else:
                logger.warning(
                    f"the node name:{node.name} is not exist in model_node_name_dict."
                )
    return model_node_name_dict


def get_model_weight_name_dict(model):
    model_weight_name_dict = dict()
    for wgt in model.initializer:
        if not model_weight_name_dict.get(wgt.name):
            model_weight_name_dict[wgt.name] = wgt
        else:
            logger.warning(
                f"the weight name:{wgt.name} is exist in model_weight_name_dict."
            )
    return model_weight_name_dict


def get_model_node_output_node_name_dict(model):
    model_node_output_node_name_dict = dict()
    #handle all node
    for node in model.node:
        #the node.output is support multi
        for out in node.output:
            if not model_node_output_node_name_dict.get(out):
                model_node_output_node_name_dict[out] = node.output[0]
            else:
                logger.error(
                    f"the node output var name:{node.output} is exist in model_node_output_node_name_dict."
                )
    return model_node_output_node_name_dict


def get_node_input_var(node):
    if len(node.input) > 0:
        return node.input


def get_node_input_node_name(node, model_output_name_dict,
                             model_weight_name_dict):
    inputs = get_node_input_var(node)
    node_input_node_name = []
    node_weights_bias_node_name = []
    for var in inputs:
        if var in model_output_name_dict.keys():
            node_input_node_name.append(model_output_name_dict[var])
        elif var in model_weight_name_dict.keys():
            node_weights_bias_node_name.append(model_weight_name_dict[var])
        else:
            logger.debug(f"the node: {var} is input or output")
    return node_input_node_name, node_weights_bias_node_name


def get_node_from_node_name(name, model_output_node_dict):
    if model_output_node_dict.get(name):
        return model_output_node_dict[name]
    else:
        logger.error(f"cann't get node:{name} from name.")


def get_weight_from_weight_name(name, model_weight_node_dict):
    if model_weight_node_dict.get(name):
        return model_weight_node_dict[name]
    else:
        logger.warning(f"cann't get weight:{name} from name.")


def get_weights_node_of_node(node, model_output_name_dict,
                             model_weights_node_dict):
    _, all_weights_name = get_node_input_node_name(node, model_output_name_dict,
                                                   model_weights_node_dict)
    weights_nodes = []
    for weight in all_weights_name:
        if (weight):
            weights_nodes.append(weight)
    return weights_nodes


def get_output_nodes_of_node(node, model):
    output_nodes_list = []
    for output in node.output:
        for one_node in model.node:
            for one_node_in in one_node.input:
                if one_node_in == output:
                    if one_node and one_node.name not in output_nodes_list:
                        output_nodes_list.append(one_node)
                    else:
                        logger.info(
                            f"the output_node:{one_node.name} already in list")
    return output_nodes_list


def get_clip_min_max(model, clip_node):
    """
    Get clip min and max value from Clip node.
    :param model: onnx model instance
    :param clip_node: target Clip node
    :return: the min, max value and para type
             The meaning of para type is:
             None - unknown
             0 - attribute
             1 - initializer
             2 - other nodes
    """

    def _get_from_initializer(model, name):
        for init in model.graph.initializer:
            if init.name == name:
                return onnx.numpy_helper.to_array(init)
        return None

    def _get_from_attribute(node):
        for attr in node.attribute:
            if attr.name == 'value':
                if attr.t.data_type == 1:
                    return list(attr.t.float_data)[0]
                else:
                    return list(attr.t.int32_data)[0]
        return None

    def _get_from_other_node(model, name):
        for node in model.graph.node:
            if node.op_type == 'Identity' and name in node.output:
                return _get_from_initializer(model, node.input[0])
            if node.op_type == 'Constant' and name in node.output:
                return _get_from_attribute(node)
        return None

    min_value = None
    max_value = None
    if clip_node.op_type != 'Clip':
        return min_value, max_value, None

    # Get from attributes
    for attr in clip_node.attribute:
        if attr.name == 'min':
            min_value = attr.f
        if attr.name == 'max':
            max_value = attr.f

    if min_value is not None or max_value is not None:
        return min_value, max_value, 0

    # Get from initializers
    if len(clip_node.input) > 1:
        min_value = _get_from_initializer(model, clip_node.input[1])
    if len(clip_node.input) > 2:
        max_value = _get_from_initializer(model, clip_node.input[2])

    if min_value is not None or max_value is not None:
        return min_value, max_value, 1

    # Try to get from other nodes
    if len(clip_node.input) > 1:
        min_value = _get_from_other_node(model, clip_node.input[1])
    if len(clip_node.input) > 2:
        max_value = _get_from_other_node(model, clip_node.input[2])

    if min_value is not None or max_value is not None:
        return min_value, max_value, 2

    return min_value, max_value, None


def check_relu_like_node(model, node):
    """
    Check if the node is a relu-like node
    :param model: the model instance
    :param node: the node to check
    :return: True if it is
    """
    if node.op_type == 'Relu':
        return True
    elif node.op_type == 'Clip':
        min_value, _, _ = get_clip_min_max(model, node)
        if min_value == 0:
            return True
    return False


def print_quantize_info(model_input, model_output, calibration_data_reader,
                        quant_format, input_nodes, output_nodes,
                        op_types_to_quantize, random_data_reader_input_shape,
                        per_channel, reduce_range, activation_type, weight_type,
                        nodes_to_quantize, nodes_to_exclude, optimize_model,
                        use_external_data_format, calibrate_method,
                        execution_providers, enable_dpu, debug_mode,
                        fp16_fp32_convert, convert_nchw_to_nhwc, include_cle,
                        extra_options):
    """
    print os_cpu, time, tool_version, quantized_configuration information.
    """

    def _print_time_info():
        """
        print time information.
        """
        now = datetime.now()
        print("[VAI_Q_ONNX_INFO]: Time information:")
        print(now)

    def _print_os_cpu_info():
        """
        print os_cpu information.
        """
        system_info = platform.system()
        node_info = platform.node()
        release_info = platform.release()
        version_info = platform.version()
        machine_info = platform.machine()
        processor_info = platform.processor()
        print("[VAI_Q_ONNX_INFO]: OS and CPU information:")
        print("{:>50}".format("system ---"), system_info)
        print("{:>50}".format("node ---"), node_info)
        print("{:>50}".format("release ---"), release_info)
        print("{:>50}".format("version ---"), version_info)
        print("{:>50}".format("machine ---"), machine_info)
        print("{:>50}".format("processor ---"), processor_info)

    def _print_tools_version_info():
        """
        print tools version information.
        """
        python_version = platform.python_version()
        onnx_version = onnx.__version__
        onnxruntime_version = ort.__version__
        vai_q_onnx_version = __version__
        print("[VAI_Q_ONNX_INFO]: Tools version information:")
        print("{:>50}".format("python ---"), python_version)
        print("{:>50}".format("onnx ---"), onnx_version)
        print("{:>50}".format("onnxruntime ---"), onnxruntime_version)
        print("{:>50}".format("vai_q_onnx ---"), vai_q_onnx_version)

    def _print_quantized_config_info():
        """
        print quantized configuration information.
        """
        print("[VAI_Q_ONNX_INFO]: Quantized Configuration information:")
        print("{:>50}".format("model_input ---"), model_input)
        print("{:>50}".format("model_output ---"), model_output)
        print("{:>50}".format("calibration_data_reader ---"),
              calibration_data_reader)
        print("{:>50}".format("quant_format ---"), quant_format)
        print("{:>50}".format("input_nodes ---"), input_nodes)
        print("{:>50}".format("output_nodes ---"), output_nodes)
        print("{:>50}".format("op_types_to_quantize ---"), op_types_to_quantize)
        print("{:>50}".format("random_data_reader_input_shape ---"),
              random_data_reader_input_shape)
        print("{:>50}".format("per_channel ---"), per_channel)
        print("{:>50}".format("reduce_range ---"), reduce_range)
        print("{:>50}".format("activation_type ---"), activation_type)
        print("{:>50}".format("weight_type ---"), weight_type)
        print("{:>50}".format("nodes_to_quantize ---"), nodes_to_quantize)
        print("{:>50}".format("nodes_to_exclude ---"), nodes_to_exclude)
        print("{:>50}".format("optimize_model ---"), optimize_model)
        print("{:>50}".format("use_external_data_format ---"),
              use_external_data_format)
        print("{:>50}".format("calibrate_method ---"), calibrate_method)
        print("{:>50}".format("execution_providers ---"), execution_providers)
        print("{:>50}".format("enable_dpu ---"), enable_dpu)
        print("{:>50}".format("debug_mode ---"), debug_mode)
        print("{:>50}".format("fp16_fp32_convert ---"), fp16_fp32_convert)
        print("{:>50}".format("convert_nchw_to_nhwc ---"), convert_nchw_to_nhwc)
        print("{:>50}".format("include_cle ---"), include_cle)
        print("{:>50}".format("extra_options ---"), extra_options)

    try:
        _print_time_info()
        _print_os_cpu_info()
        _print_tools_version_info()
        _print_quantized_config_info()
    except Exception as e:
        pass


def is_ort_version_below_1_16():
    """
    This function checks whether the current version of ONNX Runtime (ORT) is below 1.16.0.
    Returns:
        True if the current ORT version is less than 1.16.0, False otherwise.
    """
    return pv.parse(OrtVersion) < pv.parse("1.16.0")


def find_int16_scale(x):
    """
    Given a float value, find the closest value corresponding to  M and 2**N,
    where the range of M and 2**N is within the representation range of int16 and uint16.
    """
    if x == 0:
        return 0, 0, 0

    closest_m = 0
    closest_n = 0
    closest_diff = float('inf')

    # Loop through possible values of n and m
    for n in range(0, 17):  # Adjust the range as needed
        m_fs = x * 2**n
        if m_fs < -2**15 or m_fs > 2**15 - 1:
            continue
        m_floor = math.floor(m_fs)
        m_ceil = math.ceil(m_fs)
        for m in [m_floor, m_ceil]:  # Adjust the range as needed
            value = m / 2**n
            diff = abs(value - x)
            if diff < closest_diff:
                closest_m = m
                closest_n = n
                closest_diff = diff

    return closest_m, closest_n, closest_diff


def remove_initializer_from_input(model):

    if model.ir_version < 4:
        logger.warning(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return model

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


def fp32_nodes(input_model_path):
    try:
        fp32_nodes_dict = {}
        fp32_model = onnx.load(input_model_path)
        onnx_model = ONNXModel(fp32_model)

        for node in onnx_model.model.graph.node:
            if node.op_type not in fp32_nodes_dict:
                fp32_nodes_dict[node.op_type] = 0
            fp32_nodes_dict[node.op_type] += 1

        return fp32_nodes_dict

    except Exception as e:
        return {}


def print_fp32_nodes(fp32_nodes_dict, output_model_path):
    try:
        fp32_nodes_list = list(fp32_nodes_dict.keys())

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table()
        table.add_column('Op Type')
        table.add_column('Float Model', style='bold green1')

        for node_op_type in fp32_nodes_list:
            node_fp32_count = fp32_nodes_dict[node_op_type]
            table.add_row(node_op_type, str(node_fp32_count))
        table.add_section()
        table.add_row("Quantized model path", output_model_path)

        console.print(table)

    except Exception as e:
        pass
