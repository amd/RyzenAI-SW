##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy
import onnx.helper

from .graph import Graph
from .node import Node

ConvBN = [
    {
        "name": "conv_0",
        "op": "Conv",
        "attrs": [
            ["kernel_shape", "<=", 0, 7],
            ["kernel_shape", ">=", 0, 1],
            ["kernel_shape", "<=", 1, 7],
            ["kernel_shape", ">=", 1, 1],
        ],
        "inport": [],
        "outport": [[0, "bn_0", 0]],
    },
    {
        "name": "bn_0",
        "op": "BatchNormalization",
        "attrs": [],
        "inport": [[0, "conv_0", 0]],
        "outport": [],
    },
]

Fused_Element = [
    {
        "name": "any",
        "op": "Any",
        "attrs": [],
        "inport": [],
        "outport": [[0, "act_0", 0]],
    },
    {
        "name": "act_0",
        "op": ["Relu", "LeakyRelu", "Add"],
        "attrs": [],
        "inport": [[0, "any", 0]],
        "outport": [],
    },
]

Conv_Res = [
    {
        "name": "conv_0",
        "op": "Conv",
        "attrs": [],
        "inport": [],
        "outport": [[0, "add_0", 1]],
    },
    {
        "name": "add_0",
        "op": [
            "Add",
        ],
        "attrs": [],
        "inport": [[1, "conv_0", 0]],
        "outport": [[0, "act_0", 0]],
    },
    {
        "name": "act_0",
        "op": ["Relu", "LeakyRelu"],
        "attrs": [],
        "inport": [[0, "add_0", 0]],
        "outport": [],
    },
]

MHAint8_Pattern = [
    {
        "name": "MatMul_0",
        "op": "QLinearMatMul",
        "attrs": [],
        "inport": [],
        "outport": [[0, "DequantizeLinea_0", 0]],
    },
    {
        "name": "DequantizeLinea_0",
        "op": "DequantizeLinear",
        "attrs": [],
        "inport": [[0, "MatMul_0", 0]],
        "outport": [[0, "div_0", 0]],
    },
    {
        "name": "div_0",
        "op": "Div",
        "attrs": [],
        "inport": [[0, "DequantizeLinea_0", 0]],
        "outport": [[0, "Add_0", 0]],
    },
    {
        "name": "Add_0",
        "op": "Add",
        "attrs": [],
        "inport": [[0, "div_0", 0]],
        "outport": [[0, "Softmax_0", 0]],
    },
    {
        "name": "Softmax_0",
        "op": "Softmax",
        "attrs": [],
        "inport": [[0, "Add_0", 0]],
        "outport": [[0, "QuantizeLinear_1", 0]],
    },
    {
        "name": "QuantizeLinear_1",
        "op": "QuantizeLinear",
        "attrs": [],
        "inport": [[0, "Add_0", 0]],
        "outport": [[0, "MatMul_1", 0]],
    },
    {
        "name": "MatMul_1",
        "op": "QLinearMatMul",
        "attrs": [],
        "inport": [[0, "QuantizeLinear_1", 0]],
        "outport": [],
    },
]

layernorm_pattern = [
    {
        "name": "ReduceMean_196",
        "op": "ReduceMean",
        "attrs": [],
        "inport": [],
        "outport": [[0, "Sub_197", 1]],
    },
    {
        "name": "Sub_197",
        "op": "Sub",
        "attrs": [],
        "inport": [[1, "ReduceMean_196", 0]],
        "outport": [[0, "Pow_0", 0], [0, "Div_0", 0]],
    },
    {
        "name": "Pow_0",
        "op": "Pow",
        "attrs": [],
        "inport": [[0, "Sub_197", 0]],
        "outport": [[0, "ReduceMean_0", 0]],
    },
    {
        "name": "ReduceMean_0",
        "op": "ReduceMean",
        "attrs": [],
        "inport": [[0, "Pow_0", 0]],
        "outport": [[0, "Add_0", 0]],
    },
    {
        "name": "Add_0",
        "op": "Add",
        "attrs": [],
        "inport": [[0, "ReduceMean_0", 0]],
        "outport": [[0, "Sqrt_0", 0]],
    },
    {
        "name": "Sqrt_0",
        "op": "Sqrt",
        "attrs": [],
        "inport": [[0, "Add_0", 0]],
        "outport": [[0, "Div_0", 1]],
    },
    {
        "name": "Div_0",
        "op": "Div",
        "attrs": [],
        "inport": [[0, "Sub_197", 0], [1, "Sqrt_0", 0]],
        "outport": [],
    },
]

ShapeOps = ["Flatten", "Reshape", "Unsqueeze", "Squeeze"]


def removeShapeOps(g: Graph):
    rmlist = []
    for n in g.nodemap.keys():
        node = g.nodemap[n]
        if node.op_type in ShapeOps:
            rmlist.append(n)
    for n in rmlist:
        g.skip_node(n)
    g.update_tensor_relations()
    return g


def createSerialOpChain(oplist: list[str]):
    chain = []
    for i, op in enumerate(oplist):
        inport = [] if i == 0 else [[-1, str(i - 1), -1]]
        outport = [] if i == len(oplist) - 1 else [[-1, str(i + 1), -1]]
        nodedesc = {
            "name": str(i),
            "op": op,
            "attrs": [],
            "inport": inport,
            "outport": outport,
        }
        chain.append(nodedesc)
    return chain


def createSerialPattern(oplist: list[str]):
    chain = createSerialOpChain(oplist)
    return FusionPattern(chain)


class AttrExpr:
    def __init__(self, raw: []):
        self.attrname = raw[0]
        self.expr = raw[1]
        if len(raw) == 4:
            self.idx = raw[2]
            self.num = raw[3]
        else:
            self.num = raw[2]
            self.idx = -1

    def __call__(self, x):
        if hasattr(x, self.attrname):
            if self.idx == -1:
                return self.logical(x.__getattribute__(self.attrname))
            else:
                return self.logical(x.__getattribute__(self.attrname)[self.idx])
        return False

    def logical(self, attr):
        if self.expr == "<=":
            return attr <= self.num
        if self.expr == "<":
            return attr < self.num
        if self.expr == ">=":
            return attr >= self.num
        if self.expr == ">":
            return attr > self.num
        if self.expr == "==":
            return attr == self.num


class NodeCondition:
    def __init__(self, attrmap: {}):
        self.op = attrmap["op"]
        self.name = attrmap["name"]
        self.attexprs = []
        for att in attrmap["attrs"]:
            self.attexprs.append(AttrExpr(att))
        self.outport = attrmap["outport"]
        self.inport = attrmap["inport"]

    def is_node(self, node: Node):
        flag = True

        if isinstance(self.op, list):
            flag &= node.op_type in self.op
        else:
            if self.op != "Any":
                flag &= node.op_type == self.op
        for attrexpr in self.attexprs:
            flag &= attrexpr(node)
        return flag


def create_descs_from_nodenames(graph: Graph, nodenames: [str]):
    nodedesc = []
    consumed_by = {}
    produced_by = {}
    for name in nodenames:
        node = graph.nodemap[name]
        desc = {
            "name": node.name,
            "op": node.op_type,
            "attrs": [],
            "inport": [],
            "outport": [],
        }
        nodedesc.append(desc)
        for t in node.output:
            consumed_by[t] = []
            produced_by[t] = name

    for i, name in enumerate(nodenames):
        node = graph.nodemap[name]
        for j, t in enumerate(node.input):
            if t in consumed_by.keys():
                pnode = produced_by[t]
                pidx = nodenames.index(pnode)
                prodnode = graph.nodemap[pnode]
                nodedesc[pidx]["outport"].append([prodnode.output.index(t), name, j])
                nodedesc[i]["inport"].append([j, pnode, prodnode.output.index(t)])
    return nodedesc


class FusionPattern:
    def __init__(self, nodedescs: {}, inplace_fusion=False):
        self.nodedesc = {}
        self.first_key = nodedescs[0]["name"]
        self.inplace_fusion = inplace_fusion
        self.append_fusion = inplace_fusion
        for desc in nodedescs:
            self.nodedesc[desc["name"]] = NodeCondition(desc)

    def search_node(self, nodepair, graph, searched):
        curdescname = nodepair[0]
        curnodename = nodepair[1]

        desc = self.nodedesc[curdescname]
        node = graph.nodemap[curnodename]
        searched.append(curnodename)

        # expand in nodes
        invalid = False if len(desc.inport) > 0 else True
        uppaths = []
        for inset in desc.inport:
            inidx = inset[0]
            indesckey = inset[1]
            prev_outidx = inset[2]
            nextdesc = self.nodedesc[indesckey]
            if inidx == -1:
                for tname in node.input:
                    if tname not in graph.producedby:
                        continue
                    producer_node = graph.producedby[tname]
                    for nodename in producer_node:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            prev_outidx == -1
                            or nodeobject.output.index(tname) == prev_outidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    invalid, uppath = self.search_node(
                                        (indesckey, nodename), graph, searched
                                    )
                                    if invalid:
                                        uppaths.append(uppath)
                                        break
                                else:
                                    invalid = True
            elif inidx < len(node.input):
                tname = node.input[inidx]
                if tname in graph.producedby:
                    producer_node = graph.producedby[tname]
                    for nodename in producer_node:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            prev_outidx == -1
                            or nodeobject.output.index(tname) == prev_outidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    invalid, uppath = self.search_node(
                                        (indesckey, nodename), graph, searched
                                    )
                                    if invalid:
                                        uppaths.append(uppath)
                                        break
                                else:
                                    invalid = True

        if not invalid:
            searched.remove(curnodename)
            return False, None
        outpath = []
        for uppath in uppaths:
            if uppath is not None:
                for v in uppath:
                    outpath.append(v)
        outpath.append(nodepair)

        outvalid = False if len(desc.outport) > 0 else True
        downpaths = []
        for outset in desc.outport:
            outidx = outset[0]
            outdescky = outset[1]
            next_inidx = outset[2]
            nextdesc = self.nodedesc[outdescky]
            if outidx == -1:
                for output in node.output:
                    if outvalid:
                        break
                    if output in graph.consumedby:
                        consumed_nodes = graph.consumedby[output]
                        if self.append_fusion and len(consumed_nodes) > 1:
                            # inpalce_fusion the consumer op will be appended to this op as postop
                            # it requires that the output of this op is consumed by next op only
                            continue
                        for nodename in consumed_nodes:
                            nodeobject = graph.nodemap[nodename]
                            if (
                                next_inidx == -1
                                or nodeobject.input.index(output) == next_inidx
                            ):
                                if nextdesc.is_node(nodeobject):
                                    if nodename not in searched:
                                        outvalid, downpath = self.search_node(
                                            (outdescky, nodename), graph, searched
                                        )
                                        if outvalid:
                                            downpaths.append(downpath)
                                            break
                                    else:
                                        outvalid = True

            elif outidx < len(node.output):
                tname = node.output[outidx]
                if tname in graph.consumedby:
                    consumed_nodes = graph.consumedby[tname]
                    if self.append_fusion and len(consumed_nodes) > 1:
                        # inpalce_fusion the consumer op will be appended to this op as postop
                        # it requires that the output of this op is consumed by next op only
                        continue
                    for nodename in consumed_nodes:
                        nodeobject = graph.nodemap[nodename]
                        if (
                            next_inidx == -1
                            or nodeobject.input.index(tname) == next_inidx
                        ):
                            if nextdesc.is_node(nodeobject):
                                if nodename not in searched:
                                    outvalid, downpath = self.search_node(
                                        (outdescky, nodename), graph, searched
                                    )
                                    if outvalid:
                                        downpaths.append(downpath)
                                        break
                                else:
                                    outvalid = True
                                    break

        if outvalid:
            for downpath in downpaths:
                if downpath is not None:
                    for v in downpath:
                        outpath.append(v)
            return True, outpath
        else:
            searched.remove(curnodename)
            return False, None

    def search_pattern(self, graph: Graph):
        ls_nodes = []
        first_desc = self.nodedesc[self.first_key]
        self.found_node_names = []
        self.tmp_names = []
        for name in graph.nodemap.keys():
            if name in self.found_node_names:
                continue
            node = graph.nodemap[name]
            if first_desc.is_node(node):
                searched = []
                valid, path = self.search_node((self.first_key, name), graph, searched)
                if valid:
                    desckeys = list(self.nodedesc.keys())
                    nodes = ["a"] * len(desckeys)
                    for val in path:
                        idx = desckeys.index(val[0])
                        nodes[idx] = val[1]
                    ls_nodes.append(nodes)
        return ls_nodes


def ConvBNFusion(graph: Graph):
    import math

    pattern = FusionPattern(ConvBN)
    node_sets = pattern.search_pattern(graph)
    for nodes in node_sets:
        conv_node = graph.nodemap[nodes[0]]
        bn_node = graph.nodemap[nodes[1]]
        weight = graph.tensormap[conv_node.input[1]]
        gamma = graph.tensormap[bn_node.input[1]]
        beta = graph.tensormap[bn_node.input[2]]
        mean = graph.tensormap[bn_node.input[3]]
        var = graph.tensormap[bn_node.input[4]]
        wshape = weight.shape
        if len(conv_node.input) == 3:
            hasbias = True
            newbias = graph.nodemap[conv_node.input[2]].numpy
        else:
            hasbias = False
            newbias = numpy.zeros((wshape[0],), dtype=numpy.float32)
        for i in range(wshape[0]):
            sm = gamma.numpy[i] / math.sqrt(var.numpy[i] + bn_node.epsilon)
            sv = beta.numpy[i]
            for j in range(wshape[1]):
                for k in range(wshape[2]):
                    for l in range(wshape[3]):
                        weight.numpy[i, j, k, l] = sm * weight.numpy[i, j, k, l]
            newbias[i] = sm * (newbias[i] - mean.numpy[i]) + sv
        weight.update_proto(weight.numpy)
        newbiasname = conv_node.name + "_bias_"
        graph.add_initial(newbiasname, newbias)
        graph.remove_node(nodes[1])
        if hasbias:
            conv_node.input[2] = newbiasname
        else:
            conv_node.input.append(newbiasname)
        graph.producedby[bn_node.output[0]] = [conv_node.name]
        conv_node.output[0] = bn_node.output[0]
