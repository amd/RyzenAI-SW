##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy

from .graph import Graph, ShapeEngine, ValueExpr
import struct


def __write_float2buf(buf, f):
    ba = struct.pack("f", f)
    buf += ba
    return buf


def __write_int2buf(buf, i, bytes=4):
    ba = i.to_bytes(bytes, "little", signed=True)
    buf += ba
    return buf


def __write_len2buf(buf, src, bytes=4):
    ba = len(src).to_bytes(bytes, "little", signed=True)
    buf += ba
    return buf


def __write_str2buf(buf, string):
    if not "\0" in string:
        string += "\0"
    barr = bytes(string, "utf-8")
    buf += barr
    return buf


DTYPE_INT = 0
DTYPE_FLOAT = 1
DTYPE_STR = 2
DTYPE_FP16 = 3
DTYPE_DOUBLE = 4
DTYPE_INT8 = 5
DTYPE_UINT8 = 6
DTYPE_INT16 = 7
DTYPE_UINT16 = 8
DTYPE_INT64 = 9


def __write_data_type(buf, data):
    if isinstance(data, str) or isinstance(data, bytes):
        buf = __write_int2buf(buf, DTYPE_STR)
    if isinstance(data, int):
        buf = __write_int2buf(buf, DTYPE_INT)
    if isinstance(data, float):
        buf = __write_int2buf(buf, DTYPE_FLOAT)
    return buf


def __write_data(buf, data):
    buf = __write_data_type(buf, data)
    if isinstance(data, str):
        buf = __write_str2buf(buf, data)
    if isinstance(data, int):
        buf = __write_int2buf(buf, data)
    if isinstance(data, float):
        buf = __write_float2buf(buf, data)
    return buf


def __write_ndarray(buf, array: numpy.ndarray):
    def write_npdtype(buf, type):
        if type == numpy.float32:
            buf = __write_int2buf(buf, DTYPE_FLOAT)
        elif type == numpy.float16:
            buf = __write_int2buf(buf, DTYPE_FP16)
        elif type == numpy.int32:
            buf = __write_int2buf(buf, DTYPE_INT)
        elif type == numpy.int8:
            buf = __write_int2buf(buf, DTYPE_INT8)
        elif type == numpy.uint8:
            buf = __write_int2buf(buf, DTYPE_UINT8)
        elif type == numpy.int64:
            buf = __write_int2buf(buf, DTYPE_INT64)
        else:
            raise ValueError(f"Unsupported dtype {type}")
        return buf

    buf = write_npdtype(buf, array.dtype)
    buf = __write_len2buf(buf, array.shape)
    for sval in array.shape:
        buf = __write_int2buf(buf, sval)
    buf += array.tobytes()
    return buf


def serialize_shape_engine(engine: ShapeEngine, filepath):
    binfile = open(filepath, "wb")
    writebuf = bytearray(0)
    writebuf = __write_len2buf(writebuf, engine.variables.keys())
    for name in engine.variables.keys():
        writebuf = __write_str2buf(writebuf, name)
    writebuf = __write_len2buf(writebuf, engine.tensor_desc.keys())
    for name in engine.tensor_desc.keys():
        writebuf = __write_str2buf(writebuf, name)
        desc = engine.tensor_desc[name]
        writebuf = __write_len2buf(writebuf, desc)
        for d in desc:
            writebuf = __write_data(writebuf, d)

    writebuf = __write_len2buf(writebuf, engine.tensor_epxr.keys())

    def serialize_expr(buf, expr: ValueExpr):
        buf = __write_float2buf(buf, float(expr.alpha))
        buf = __write_float2buf(buf, float(expr.beta))
        buf = __write_int2buf(buf, int(expr.factor))
        buf = __write_int2buf(buf, int(expr.truncmode))
        return buf

    for name in engine.tensor_epxr.keys():
        writebuf = __write_str2buf(writebuf, name)
        expr = engine.tensor_epxr[name]
        srcname = expr[0]
        writebuf = __write_str2buf(writebuf, srcname)
        writebuf = serialize_expr(writebuf, expr[1])

    binfile.write(writebuf)
    binfile.close()


def serialize_graph(graph: Graph, filepath):
    binfile = open(filepath, "wb")
    writebuf = bytearray(0)
    writebuf = __write_len2buf(writebuf, graph.nodemap.keys())
    for name in graph.nodemap.keys():
        node = graph.nodemap[name]
        writebuf = __write_str2buf(writebuf, name)
        writebuf = __write_str2buf(writebuf, node.op_type)
        writebuf = __write_len2buf(writebuf, node.input)
        for input in node.input:
            writebuf = __write_str2buf(writebuf, input)
        writebuf = __write_len2buf(writebuf, node.output)
        for output in node.output:
            writebuf = __write_str2buf(writebuf, output)

        writebuf = __write_len2buf(writebuf, node.attr.keys())

        def write_attribute(buf, attrval):
            def write_value(buf, val):
                if isinstance(val, str):
                    buf = __write_str2buf(buf, val)
                if isinstance(val, bytes):
                    buf += val + b"\0"
                if isinstance(val, int):
                    buf = __write_int2buf(buf, val)
                if isinstance(val, float):
                    buf = __write_float2buf(buf, val)
                return buf

            if not isinstance(attrval, list):
                attrval = [attrval]
            buf = __write_len2buf(buf, attrval)
            buf = __write_data_type(buf, attrval[0])
            for val in attrval:
                buf = write_value(buf, val)
            return buf

        for key in node.attr.keys():
            writebuf = __write_str2buf(writebuf, key)
            writebuf = write_attribute(writebuf, node.attr[key])
    writebuf = __write_len2buf(writebuf, graph.initials)
    for name in graph.initials:
        writebuf = __write_str2buf(writebuf, name)
        tensor = graph.tensormap[name]
        writebuf = __write_ndarray(writebuf, tensor.numpy)
    binfile.write(writebuf)
    binfile.close()


def serialize_memory_compression(compressed_mem: {}, max_shape: {}, filepath):
    binfile = open(filepath, "wb")
    writebuf = bytearray(0)
    writebuf = __write_int2buf(writebuf, compressed_mem[1], 8)
    compress_map = compressed_mem[0]
    writebuf = __write_len2buf(writebuf, compress_map.keys(), 8)
    for key in compress_map.keys():
        block = compress_map[key]
        writebuf = __write_str2buf(writebuf, key)
        writebuf = __write_int2buf(writebuf, block[0], 8)
        writebuf = __write_int2buf(writebuf, block[1], 8)
    writebuf = __write_len2buf(writebuf, max_shape.keys(), 8)
    for key in max_shape.keys():
        writebuf = __write_str2buf(writebuf, key)
        writebuf = __write_int2buf(writebuf, max_shape[key], 8)
    binfile.write(writebuf)
    binfile.close()
