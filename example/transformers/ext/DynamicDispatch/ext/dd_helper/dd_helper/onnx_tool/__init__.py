##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import os
import warnings

import numpy
import onnx

from .graph import Graph
from .model import Model
from .node import NODE_REGISTRY, Node
from .serialization import serialize_shape_engine, serialize_graph
from .tensor import create_ndarray_f32, create_ndarray_int64
from .utils import timer, VERSION


def loadmodel(f, mcfg={}):
    model = Model(f, mcfg)
    if model.valid is False:
        warnings.warn(f"Invalid onnx model file:{f}")
        exit(-1)
    return model


def __remove_initilisers(graph: onnx.GraphProto):
    graph.ClearField("initializer")


def model_export_tensors_numpy(
    m, tensornames: [str] = None, savefolder: str = None, fp16: bool = False
) -> None:
    model = loadmodel(m)

    def save_numpy(arr: numpy.ndarray, fp16: bool, filename):
        if fp16 and arr.dtype in [numpy.float32, numpy.float64]:
            arr = arr.astype(numpy.float16)
        numpy.save(filename, arr)

    g = model.graph
    if savefolder is not None:
        os.makedirs(savefolder, exist_ok=True)
    else:
        savefolder = "./"
    if tensornames is None:
        for key in g.initials:
            name = key
            if "/" in key:
                name = key.replace("/", "_")
            if "\\" in key:
                name = key.replace("\\", "_")
            narr = g.tensormap[key].numpy
            save_numpy(narr, fp16, os.path.join(savefolder, name + ".npy"))

    else:
        for name in tensornames:
            if name not in g.initials:
                warnings.warn(f"tensor {name} not found ")
                continue
            fname = name
            if "/" in name:
                fname = name.replace("/", "_")
            if "\\" in name:
                fname = name.replace("\\", "_")
            narr = g.tensormap[name].numpy
            save_numpy(narr, fp16, os.path.join(savefolder, fname + ".npy"))


# These ops are created by onnx exporter, they are out of programmer's sense
DefaultFilter = (
    "Identity",
    "Constant",
)

# These ops have no computation
NoMacsOps = (
    "Identity",
    "Constant",
    "Shape",
    "Squeeze",
    "Unsqueeze",
    "Reshape",
    "ConstantOfShape",
    "Cast",
    "Pad",
    "Concat",
    "Slice",
    "Gather",
)


def model_profile(
    m,
    dynamic_shapes: {str: tuple} = None,
    hidden_ops: [str] = NoMacsOps,
    mcfg={"verbose": False},
    save_profile: str = None,
    save_model: str = None,
    shape_only: bool = False,
    no_shape: bool = False,
) -> None:
    model = loadmodel(m, mcfg)
    g = model.graph
    gtmr = timer()
    g.graph_reorder_nodes()
    gtmr.start()
    g.shape_infer(dynamic_shapes)
    g.log(f"infered all tensor shapes, time cost {gtmr.stop():.3f} s")
    gtmr.start()
    g.profile()
    g.log(f"profile all nodes, time cost {gtmr.stop():.3f} s")
    g.print_node_map(save_profile, exclude_ops=hidden_ops)
    if save_model is not None:
        model.save_model(save_model, shape_only=shape_only, no_shape=no_shape)


def model_shape_regress(m, input_desc: {}, input_range: {}):
    model = loadmodel(m)
    graph = model.graph
    graph.graph_reorder_nodes()
    shape_engine = graph.shape_regress(input_desc, input_range)
    cg = graph.get_compute_graph()
    return shape_engine, cg


def model_constant_folding(m, save_model: str):
    model = loadmodel(m, {"constant_folding": True, "verbose": True})
    model.save_model(save_model)


def model_reorder_nodes(
    m,
    save_model: str,
):
    model = loadmodel(m)
    model.graph.graph_reorder_nodes()
    model.save_model(save_model)


def model_io_modify(m, save_model: str, custom_io):
    """
    Args:
        m: onnx.ModelProto or file path
        custom_io: {str:str} e.g. {'input':'Nx3xwidthxheight'}
    Returns:

    """
    model = loadmodel(m)
    graph = model.mproto.graph
    if custom_io is not None:
        keylist = list(custom_io.keys())
        for i, input in enumerate(graph.input):
            if input.name in keylist:
                shapes = custom_io[input.name].split("x")
                # maybe consider create a new valueinfoproto
                dim = input.type.tensor_type.shape.dim
                assert len(shapes) == len(dim)
                for nb, shapeval in zip(dim, shapes):
                    if shapeval.isnumeric():
                        if nb.HasField("dim_param"):
                            nb.ClearField("dim_param")
                        nb.dim_value = int(shapeval)
                    else:
                        if nb.HasField("dim_value"):
                            nb.ClearField("dim_value")
                        nb.dim_param = shapeval

        for i, output in enumerate(graph.output):
            if output.name in keylist:
                shapes = custom_io[output.name].split("x")
                # maybe consider create a new valueinfoproto
                dim = output.type.tensor_type.shape.dim
                assert len(shapes) == len(dim)
                for nb, shapeval in zip(dim, shapes):
                    if shapeval.isnumeric():
                        if nb.HasField("dim_param"):
                            nb.ClearField("dim_param")
                        nb.dim_value = int(shapeval)
                    else:
                        if nb.HasField("dim_value"):
                            nb.ClearField("dim_value")
                        nb.dim_param = shapeval
    graph = Graph(graph, utils.ModelConfig({"verbose": True}))
    graph.save_model(save_model, rawmodel=model.mproto)


def model_subgraph(
    m,
    in_tensor_names: [str] = None,
    out_tensor_names: [str] = None,
    nodenames: [str] = None,
    save_folder="./",
):
    model = loadmodel(m)
    graph = model.graph
    if in_tensor_names is not None and out_tensor_names is not None:
        graph_lvl0, graph_lvl1, graph_lvl2 = graph.get_subgraph(
            inputs=in_tensor_names, outputs=out_tensor_names
        )
        graph_lvl0.save_model(
            os.path.join(save_folder, model.modelname + "_level0.onnx"),
            rawmodel=model.mproto,
        )
        graph_lvl1.save_model(
            os.path.join(save_folder, model.modelname + "_level1.onnx"),
            rawmodel=model.mproto,
        )
        graph_lvl2.save_model(
            os.path.join(save_folder, model.modelname + "_level2.onnx"),
            rawmodel=model.mproto,
        )
    if nodenames is not None:
        rawgraph = graph.get_onnxgraph_by_nodenames(nodenames)
        subgraph = Graph(rawgraph)
        subgraph.save_model(
            os.path.join(save_folder, model.modelname + "_subgraph.onnx"),
            rawmodel=model.mproto,
        )


def model_opfusion(
    m,
    op_type: str,
    op_name: str,
    save_file: str,
    in_tensor_names: [str] = None,
    out_tensor_names: [str] = None,
    nodenames: [str] = None,
    keep_attr=True,
):
    model = loadmodel(m)
    graph = model.graph
    if in_tensor_names is not None and out_tensor_names is not None:
        graph.fuse_subgraph_iotensors(
            inputs=in_tensor_names,
            outputs=out_tensor_names,
            name=op_name,
            nodeop=op_type,
            keep_attr=keep_attr,
        )
        model.save_model(save_file)
    if nodenames is not None:
        graph.fuse_subgraph_node_names(
            nodenames, nodeop=op_type, name=op_name, keep_attr=keep_attr
        )
        model.save_model(save_file)
