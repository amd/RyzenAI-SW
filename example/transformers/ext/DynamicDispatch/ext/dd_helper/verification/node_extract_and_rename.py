import onnx

model = onnx.load(
    "C:\\vkjain\\win24\\extracted_subgraphs\\PSF_v1.0\\-tulrv6-encoder-layer.0-attention-output-dense-MatMul-subgraph.onnx"
)

mm_count = 0
for node in model.graph.node:
    if node.op_type == "DequantizeLinear":
        mm_count += 1
        print(
            "Name: {}\n- Inputs: {}\n- Outputs: {}\n".format(
                node.name, node.input, node.output
            )
        )
        # Get weights initializer
        initializers = [
            tensor for tensor in model.graph.initializer if tensor.name in node.input
        ]
        # Create graph
        graph = onnx.helper.make_graph(
            [node], "g_" + node.name, node.input, node.output, initializers
        )
        # Create model
        mm_model = onnx.helper.make_model(graph)

        # Input value info
        ivalinfo = onnx.helper.make_tensor_value_info(
            node.input[0], onnx.TensorProto.UINT16, [1, 512, 768]
        )
        mm_model.graph.input.append(ivalinfo)

        # Input value info
        ovalinfo = onnx.helper.make_tensor_value_info(
            node.output[0], onnx.TensorProto.FLOAT, [1, 512, 768]
        )

        mm_model.graph.output.append(ovalinfo)

        # Opset
        mm_model.opset_import[0].version = 17
        mm_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

        # Check
        try:
            onnx.checker.check_model(mm_model)
            # Shape inference
            mm_model = onnx.shape_inference.infer_shapes(mm_model)
        except:
            continue

        mm_file = "test_dql/dql_{}.onnx".format(node.name.replace("/", "_"))
        # Save
        onnx.save(mm_model, mm_file)

from __future__ import annotations

import argparse
import copy
import importlib
import logging
import os

import onnx
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto

from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import attribute_to_kwarg

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class MatMul4BitsRename:
    """Renames MatMulNBits to AMDMatMulNBits"""

    def __init__(self, model: ModelProto | str, nodes_to_exclude=None):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model = (
            ONNXModel(onnx.load(model)) if isinstance(model, str) else ONNXModel(model)
        )
        self.model_path = model if isinstance(model, str) else None
        self.nodes_to_exclude = set(nodes_to_exclude)

    def _amd_q4_matmul_node(
        self, node: NodeProto, graph_stack: list[GraphProto]
    ) -> NodeProto:
        if node.op_type != "DequantizeLinear":
            return node  # only care about MatMul for now

        logger.info(f"start to rename {node.name} ...")
        if node.name in self.nodes_to_exclude:
            logger.info(
                f"exclude to rename {node.name} as specified by nodes_to_exclude..."
            )
            return node

        amd_matmul_q4_node = onnx.helper.make_node(
            "DQL",
            inputs=node.input,
            outputs=node.output,
            name=node.name,
            domain="com.amd",
        )

        logger.info(f"complete rename of {node.name} ...")

        return amd_matmul_q4_node

    def _process_subgraph(self, graph_stack: list[GraphProto]):
        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            graph_attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH
                or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv = {attr.name: self._process_subgraph(graph_stack)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([self._process_subgraph(graph_stack)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            new_nodes.append(self._amd_q4_matmul_node(node, graph_stack))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    def _generate_q4_node_config(self):
        """Generate weight only quant configuration for nodes."""
        q4_node_config = {}
        template_config_q4 = {
            "bits": 4,
            "group_size": self.block_size,
            "scheme": "sym" if self.is_symmetric else "asym",
        }
        for node in self.model.model.graph.node:
            if node.op_type in ["MatMul"]:
                if not all([self.model.get_initializer(i) is None for i in node.input]):
                    q4_node_config[node.name] = template_config_q4
        return q4_node_config

    def process(self):
        # use a stack to keep track of sub-graphs
        graph_stack = [self.model.graph()]
        opset_import = self.model.opset_import()

        has_amd_domain = False
        for opset in opset_import:
            if opset.domain == "com.amd":
                has_amd_domain = True
        if not has_amd_domain:
            opset_import.extend([onnx.helper.make_opsetid("com.amd", 1)])

        self._process_subgraph(graph_stack)
        self.model.clean_initializers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_input", type=str, required=True, help="path of FP32 onnx model"
    )

    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="path of INT4 quantized onnx model",
    )

    # Parse args
    args = parser.parse_args()

    # Load ONNX Model
    model = onnx.load_model(args.model_input, load_external_data=True)

    # Replace
    replacer = MatMul4BitsRename(model)

    # Process and save the model
    replacer.process()
    replacer.model.save_model_to_file(args.model_output, use_external_data_format=True)
