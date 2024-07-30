import os, sys
import numpy
import argparse
import onnx
import onnxruntime as ort
import json
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)
from colorama import Fore
import abc


def get_domain():
    return "com.amd"


def get_opsets():
    return [
        onnx.helper.make_opsetid("", 17),
        onnx.helper.make_opsetid(get_domain(), 1),
        onnx.helper.make_opsetid("com.microsoft", 1),
    ]


class FunctionSubgraph(metaclass=abc.ABCMeta):
    def __init__(self, input_model, output_model):
        self.input_model = input_model
        self.output_model = output_model
        self.model = self.load_model()
        self.graph_inputs = self.model.graph.input
        self.graph_outputs = self.model.graph.output
        self.graph_initializers = self.model.graph.initializer
        self.graph_value_info = self.model.graph.value_info
        self.graph_nodes = self.model.graph.node
        # New graph properties
        self.fgraph = None
        self.function = None
        self.fnode = None
        self.fgraph_in_value_info = []
        self.fgraph_out_value_info = []
        self.fgraph_input_names = []
        self.fgraph_output_names = []
        # Create model
        self.create_model()

    def get_domain(self):
        return "com.amd"

    def get_opsets(self):
        return [
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid(self.get_domain(), 1),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]

    def load_model(self):
        m = onnx.load(self.input_model)
        return onnx.shape_inference.infer_shapes(m)

    def print_io_names(self):
        print("")
        for meta in self.graph_inputs:
            print(" -> Input Name: {}".format(meta.name))
        for meta in self.graph_outputs:
            print(" -> Output Name: {}".format(meta.name))

    def print_value_info(self):
        for info in self.graph_value_info:
            name = info.name
            etype = info.type.tensor_type.elem_type
            shape = [dim.dim_value for dim in info.type.tensor_type.shape.dim]
            print(
                " -> Value Info Name: {}, Type: {}, Shape: {}".format(
                    name, etype, shape
                )
            )

    def create_value_info(self):
        for info in self.graph_inputs:
            self.fgraph_in_value_info.append(
                onnx.helper.make_tensor_value_info(
                    info.name, TensorProto.UINT16, [1, "M", "K"]
                )
            )
        for info in self.graph_outputs:
            self.fgraph_out_value_info.append(
                onnx.helper.make_tensor_value_info(
                    info.name, TensorProto.UINT16, [1, "M", "N"]
                )
            )

    def update_io_names(self):
        # Input names will be inputs for original graphs + initializer names
        ## Input names
        self.fgraph_input_names = [i.name for i in self.fgraph_in_value_info]
        ## Initializer names
        self.fgraph_input_names.extend([i.name for i in self.graph_initializers])
        # Output names would be same as original output names
        self.fgraph_output_names = [i.name for i in self.fgraph_out_value_info]

    def update_initializers(self):
        if self.fgraph:
            self.fgraph.initializer.extend(self.graph_initializers)

    def create_function(self):
        self.function = onnx.helper.make_function(
            self.get_domain(),
            inputs=self.fgraph_input_names,
            outputs=self.fgraph_output_names,
            nodes=self.graph_nodes,
            fname=self.get_function_name(),
            opset_imports=self.get_opsets(),
        )

    def create_function_node(self):
        # Create a node with function name
        self.fnode = onnx.helper.make_node(
            op_type=self.get_function_name(),
            inputs=self.fgraph_input_names,
            outputs=self.fgraph_output_names,
            name=self.get_function_name() + "_node",
            domain=self.get_domain(),
        )

    def create_graph(self):
        # Create I/O value info for new graph
        self.create_value_info()
        # Update input/output names
        self.update_io_names()
        # Create onnx function
        self.create_function()
        # Create function node
        self.create_function_node()
        # Create Graph & Update initializers
        self.fgraph = onnx.helper.make_graph(
            nodes=[self.fnode],
            inputs=self.fgraph_in_value_info,
            outputs=self.fgraph_out_value_info,
            name=self.get_function_name() + "_subgraph",
        )
        # update graph initializers
        self.update_initializers()

    def create_model(self):
        # Create graph
        self.create_graph()
        # Create model
        self.fmodel = onnx.helper.make_model(
            self.fgraph, functions=[self.function], opset_imports=self.get_opsets()
        )

    def save_model(self):
        self.fmodel = onnx.shape_inference.infer_shapes(self.fmodel)
        onnx.checker.check_model(self.fmodel)
        onnx.save(self.fmodel, self.output_model)

    @abc.abstractmethod
    def get_function_name(self):
        pass


class QMatMul(FunctionSubgraph):
    def __init__(self, input_model, output_model):
        super().__init__(input_model, output_model)

    def get_function_name(self):
        return "QMatMul"


class QMatMulAdd(FunctionSubgraph):
    def __init__(self, input_model, output_model):
        super().__init__(input_model, output_model)

    def get_function_name(self):
        return "QMatMulAdd"


class QMatMulAddGeLU(FunctionSubgraph):
    def __init__(self, input_model, output_model):
        super().__init__(input_model, output_model)

    def get_function_name(self):
        return "QMatMulAddGeLU"


class QLayerNorm(FunctionSubgraph):
    def __init__(self, input_model, output_model):
        super().__init__(input_model, output_model)

    def get_function_name(self):
        return "QLayerNorm"


class QMHAGRPB(FunctionSubgraph):
    def __init__(self, input_model, output_model):
        super().__init__(input_model, output_model)

    def get_function_name(self):
        return "QMHAGRPB"

    def create_value_info(self):
        if self.get_function_name() == "QMHAGRPB":
            if len(self.graph_inputs) > 4:
                del self.graph_inputs[-1]
        for info in self.graph_inputs:
            self.fgraph_in_value_info.append(
                onnx.helper.make_tensor_value_info(info.name, TensorProto.UINT16, [])
            )
        for info in self.graph_outputs:
            self.fgraph_out_value_info.append(
                onnx.helper.make_tensor_value_info(info.name, TensorProto.UINT16, [])
            )


def process_subgraphs(subgraph_dir, output_dir, fname, verbose):
    # Check if input paths exists
    if not os.path.exists(subgraph_dir):
        raise ValueError(
            "- Unable to locate subgraph directory: {}".format(subgraph_dir)
        )
    # Check if output path exists, else create
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all the models and process
    models = os.listdir(subgraph_dir)
    for model in models:
        # if "-" in model:
        imodel = os.path.join(subgraph_dir, model)
        omodel = os.path.join(output_dir, model)
        print(
            "\n"
            + Fore.CYAN
            + "- Input Model: "
            + Fore.RESET
            + "{}\n".format(imodel)
            + Fore.CYAN
            + "- Output Model: "
            + Fore.RESET
            + "{}".format(omodel)
        )
        fmodel = None
        if fname == "QMatMul":
            fmodel = QMatMul(imodel, omodel)
        elif fname == "QMatMulAddGeLU":
            fmodel = QMatMulAddGeLU(imodel, omodel)
        elif fname == "QMatMulAdd":
            fmodel = QMatMulAdd(imodel, omodel)
        elif fname == "QLayerNorm":
            fmodel = QLayerNorm(imodel, omodel)
        elif fname == "QMHAGRPB":
            fmodel = QMHAGRPB(imodel, omodel)
        else:
            raise ValueError("-> Function Name: {} not supported\n".format(fname))

        if verbose:
            fmodel.print_io_names()
        fmodel.save_model()

        # create_function_subgraph(imodel, omodel)
        verify_ort_session(omodel, verbose)
        # break


def verify_ort_session(model_path, verbose):
    # Session options
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.log_verbosity_level = 3
    # Session create
    print("\n -> Create ORT Session to check validity with ORT ...")
    sess = ort.InferenceSession(model_path, session_options)
    print(" -> Create ORT Session: Done!")
    if verbose:
        # Get Inputs/Outputs
        for tensor in sess.get_inputs():
            print(
                " -> Input Name: {}, Type: {}, Shape: {}".format(
                    tensor.name, tensor.type, tensor.shape
                )
            )

        for tensor in sess.get_outputs():
            print(
                " -> Output Name: {}, Type: {}, Shape: {}".format(
                    tensor.name, tensor.type, tensor.shape
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        help="Original ONNX subgraph model directory",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory to store function subgraphs",
        required=True,
    )

    parser.add_argument(
        "--fname",
        help="Function name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--verbose",
        help="Enable more prints",
        required=False,
        default=False,
        action="store_true",
    )
    # Parse
    args = parser.parse_args()
    # Process subgraphs
    process_subgraphs(args.model_dir, args.output_dir, args.fname, args.verbose)
