##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import onnx
import numpy as np
import math
import os

import onnx_tool

symbol_replacement_table = {ord(sym): ord("_") for sym in "/: "}


class GraphNode:
    def __init__(self, name, node=None, isTensor=False):
        self.name = name
        self.op = node
        self.inputs = []
        self.outputs = []
        self.isTensor = isTensor

    # def __repr__(self):
    #     print(f"GraphNode : [name:{self.name}, op:{self.op}, inputs:{self.inputs}, outputs:{self.outputs}")

    def addInput(self, node):
        self.inputs.append(node)

    def addOutput(self, node):
        self.outputs.append(node)


class ONNXGraph:
    def __init__(self, model):
        self.nodev = {}
        self.aux_info = {}  # placeholder to keep any new random hacks coming in.
        self.model = model
        # Dictionary to store all named proto tensors in self.model.
        self._proto_tensors = {}
        ##Exclude list contains initializer items + tensors which are driven by constant operator
        self.exclude_list = {}
        # map [tensor_name -> ValueInfoProto]
        self.value_info_cache = {}
        # map [tensor_name -> TensorProto]
        self.initializer_info_cache = {}

        # Collect all IO tensor objs ...
        for obj in model.graph.value_info:
            self._proto_tensors[obj.name] = obj

        for obj in model.graph.initializer:
            self._proto_tensors[obj.name] = obj

        for obj in model.graph.input:
            self._proto_tensors[obj.name] = obj

        for obj in model.graph.output:
            self._proto_tensors[obj.name] = obj
        # ...

        for vi in model.graph.value_info:
            self.value_info_cache[vi.name] = vi

        for ci in model.graph.initializer:
            # self.exclude_list[ci.name] = ci
            self.initializer_info_cache[ci.name] = ci

        # for node in model.graph.node:
        #     if node.op_type == "Constant":
        #         for out in node.output:
        #             self.exclude_list[out] = node
        #         self.exclude_list[node.name] = node

        for node in model.graph.node:
            if self.is_excluded(node.name):
                continue

            graph_node = self.addNode(node.name, node, isTensor=False)
            for inp in node.input:
                if self.is_excluded(inp):
                    continue

                inp_tensor = self._proto_tensors.get(inp, None)
                inp_node = self.addNode(inp, inp_tensor, isTensor=True)
                inp_node.addOutput(graph_node)
                graph_node.addInput(inp_node)

            for outp in node.output:
                outp_tensor = self._proto_tensors.get(outp, None)
                out_node = self.addNode(outp, outp_tensor, isTensor=True)
                out_node.addInput(graph_node)
                graph_node.addOutput(out_node)

    def is_excluded(self, name):
        if self.exclude_list.get(name) != None:
            return True
        return False

    def addNode(self, name, onnx_node=None, isTensor=False):
        node = self.nodev.get(name)
        if node == None:
            node = GraphNode(name, onnx_node, isTensor)
            self.nodev[name] = node
        return node

    def getPrimaryInputs(self):
        """Returns the GraphNodes representing primary input tensors"""
        ret = []
        # for n in self.nodev.values():
        #     if not n.inputs:
        #         ret.append(n)
        for inp in self.model.graph.input:
            ret.append(self.nodev[inp.name])
        return ret

    def getPrimaryOutputs(self):
        """Returns the GraphNodes representing primary output tensors"""
        ret = []
        for outp in self.model.graph.output:
            ret.append(self.nodev[outp.name])
        return ret

    def getIntermediateTensors(self):
        """Returns the GraphNodes representing intermediate tensors"""
        primary_tensors = dict()
        tmp_pr_tensors = (
            self.getPrimaryInputs() + self.getPrimaryOutputs() + self.getConstTensors()
        )
        for item in tmp_pr_tensors:
            if item not in primary_tensors:
                primary_tensors[item] = 0

        all_tensors = dict()
        for node in self.nodev.values():
            if node.isTensor == True and node not in all_tensors:
                all_tensors[node] = 0

        intermediate_tensors = list(
            item for item in all_tensors.keys() if item not in primary_tensors
        )
        return intermediate_tensors

    def getConstTensors(self):
        """Returns the TensorProto representing const data"""
        return [
            node
            for node in self.nodev.values()
            if node.name in self.initializer_info_cache
        ]

    # list(self.initializer_info_cache.values())

    def getValueInfo(self, name):
        return self.value_info_cache.get(name)

    def getInitializerInfo(self, name):
        return self.initializer_info_cache.get(name)

    """ Get the Predecessor ops for the given op/tensor"""

    def getPredOp(self, name):
        node = self.nodev.get(name)
        ret = {}
        if node != None:
            for inp in node.inputs:
                if inp.isTensor:
                    ret.update(self.getPredOp(inp.name))
                else:
                    ret[inp.name] = inp

        return ret

    """ Get the Successor ops for the given op/tensor"""

    def getSuccOp(self, name):
        node = self.nodev.get(name)
        ret = {}
        if node != None:
            for outp in node.outputs:
                if outp.isTensor:
                    ret.update(self.getSuccOp(outp.name))
                else:
                    ret[outp.name] = outp
        return ret

    def getTensorInfo(self, name):
        tensor = self._proto_tensors[name]
        res = parseValueInfoProto(tensor)
        return res

    def showOpPred_Succ(self, name):
        print(f"Operator - {name}\n")
        pv = self.getPredOp(name)
        nv = self.getSuccOp(name)
        print("Inputs")
        for p in pv.values():
            print(p.op)

        print("Outputs")
        for p in nv.values():
            print(p.op)

    def showTensorInfo(self, name):
        print("Tensor Info")
        print(self.value_info_cache.get(name))

    def writeConsts(self, tensors, dir_path, prefix=""):
        const_file_info = {}
        for idx, tensor in enumerate(tensors):
            if isinstance(tensor.op, onnx.onnx_ml_pb2.TensorProto):
                filename = os.path.join(dir_path, f"{prefix}{idx}.const")
                filename = os.path.abspath(filename)
                size = saveTensorToFile(tensor.op, filename)
                # print(f"Writing {tensor.name}'s data to {filename} ... Done")
                const_file_info[tensor.name] = (filename, size)
        return const_file_info

    """ Returns a list of ops sorted topologically"""

    def topologicalSortOpsviaONNX(self):
        res = [node.name for node in self.model.graph.node]
        return res

    # FIXME : Incorrect functionality. Fix required
    def topologicalSortOps(self):
        res = []
        primary_inputs = self.getPrimaryInputs()
        visited_nodes = set()
        stack = []

        # Get all primary input ops
        for inp in primary_inputs:
            for op in inp.outputs:
                if op.name not in visited_nodes:
                    stack.append(op.name)
                    visited_nodes.add(op.name)

        # For each op in  the stack,
        #   keep op in the stack and continue processing neighbours
        #   Once all its neighbours are processed, push it to the result stack
        while stack:
            op_name = stack[-1]
            op_idx = len(stack) - 1
            op = self.nodev[op_name]
            op_finished = True
            succ_ops = self.getSuccOp(op_name)
            for succ_op in succ_ops:
                if succ_op not in visited_nodes:
                    op_finished = False
                    stack.append(succ_op)
                    visited_nodes.add(succ_op)
            if op_finished:
                op_name = stack.pop(op_idx)
                res.append(op_name)

        op_names = []
        for op_name in reversed(res):
            if not self.nodev[op_name].isTensor:
                op_names.append(op_name)

        return op_names


#
#   Free Functions
#


def getSizeinBytes(dtype, shape):
    """Given dtype and shape of a tensor, compute the total size in bytes"""
    if dtype == onnx.TensorProto.BFLOAT16:
        itemsize = 2
    else:
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dtype)).itemsize
    size_in_bytes = math.prod(shape) * itemsize
    return size_in_bytes


def extract_type(tensor_type):
    """Example : TensorProto.BFLOAT16 -> bfloat16"""
    return tensor_type.split(".")[-1].lower()


def extract_op_attrs(op_proto: onnx.NodeProto):
    res = {}
    if len(op_proto.attribute) == 0:
        return res

    for attr_proto in op_proto.attribute:
        key = attr_proto.name
        value = None
        if attr_proto.type == onnx.AttributeProto.AttributeType.FLOAT:
            dtype = "float"
            value = [str(attr_proto.f)]
        elif attr_proto.type == onnx.AttributeProto.AttributeType.INT:
            dtype = "int"
            value = [str(attr_proto.i)]
        elif attr_proto.type == onnx.AttributeProto.AttributeType.STRING:
            dtype = "str"
            value = [attr_proto.s.decode("utf-8")]
        elif attr_proto.type == onnx.AttributeProto.AttributeType.FLOATS:
            dtype = "float"
            value = [str(attr) for attr in attr_proto.floats]
        elif attr_proto.type == onnx.AttributeProto.AttributeType.INTS:
            dtype = "int"
            value = [str(attr) for attr in attr_proto.ints]
        elif attr_proto.type == onnx.AttributeProto.AttributeType.STRINGS:
            dtype = "str"
            value = [attr.decode("utf-8") for attr in attr_proto.strings]
        else:
            print("WARNING : Cannot write attr:{key} in node:{op_proto.name} to JSON.")

        res[key] = {"type": dtype, "value": value}

    return res


def parseValueInfoProto(tensor):
    """Extract tensor info from ValueInfoProto
    TODO : Label input as ValueInfoProto
    """
    dtype = tensor.type.tensor_type.elem_type
    shape = [d.dim_value for d in tensor.type.tensor_type.shape.dim]
    dtype_str = extract_type(onnx.helper.tensor_dtype_to_string(dtype))
    size_in_bytes = getSizeinBytes(dtype, shape)
    res = {"dtype": dtype_str, "shape": shape, "size_in_bytes": size_in_bytes}
    return res


def parseTensorProto(tensor):
    """Extract tensor info from TensorProto
    TODO : Label input as TensorProto
    """
    dtype = tensor.data_type
    shape = list(tensor.dims)
    dtype_str = extract_type(onnx.helper.tensor_dtype_to_string(dtype))
    size_in_bytes = getSizeinBytes(dtype, shape)
    res = {"dtype": dtype_str, "shape": shape, "size_in_bytes": size_in_bytes}
    return res


def parseTensorGraphNode(graph_node):
    assert graph_node.isTensor == True
    assert hasattr(graph_node, "op") and (graph_node.op != None)
    op = graph_node.op
    if isinstance(op, onnx.onnx_ml_pb2.TensorProto):
        parse_fn = parseTensorProto
    elif isinstance(op, onnx.onnx_ml_pb2.ValueInfoProto):
        parse_fn = parseValueInfoProto
    else:
        raise RuntimeError("Unsupported datatype for parsing")
    return parse_fn(op)


def saveTensorToFile(tensor: onnx.onnx_ml_pb2.TensorProto, filename):
    onnx_dtype = tensor.data_type
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_dtype)
    if onnx_dtype in {onnx.TensorProto.FLOAT}:
        np_array = np.array(tensor.float_data).astype(np.float32)
    elif onnx_dtype in {onnx.TensorProto.INT64}:
        if len(tensor.raw_data) > 0:
            np_array = np.frombuffer(tensor.raw_data, dtype=np_dtype)
        else:
            np_array = np.array(tensor.int64_data).astype(np.int64)
    elif onnx_dtype in {
        onnx.TensorProto.INT32,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT16,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT8,
    }:
        if len(tensor.raw_data) > 0:
            np_array = np.frombuffer(tensor.raw_data, dtype=np_dtype)
        else:
            np_array = np.array(tensor.int32_data).astype(np_dtype)
    elif onnx_dtype in {onnx.TensorProto.BFLOAT16}:
        np_array = np.array(tensor.int32_data).astype(np.uint16)
        # print("weights sum : ", np_array.astype(np.int64).sum(), filename)
    else:
        str_type = onnx.helper.tensor_dtype_to_string(onnx_dtype)
        raise Exception("Unsupported data type : {str_type}")

    if np_array.size == 0:
        raise RuntimeError(f"couldn't find data for {tensor.name}")

    np_array.tofile(filename)
    return np_array.nbytes
