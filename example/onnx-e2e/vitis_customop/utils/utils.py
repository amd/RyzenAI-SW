import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import onnx
except ImportError:
    install("onnx")
    import onnx


def get_domain():
    return "vitis.customop"


def get_opsets():
    return [onnx.helper.make_opsetid("", 14), onnx.helper.make_opsetid(get_domain(), 1)]


def create_graph(nodes, graph_name, invalue_info, outvalue_info):
    # Make graph
    return onnx.helper.make_graph(
        nodes=nodes, name=graph_name, inputs=invalue_info, outputs=outvalue_info
    )


def create_model(graph, funcs=[], en_opset=True):
    opsets = get_opsets() if en_opset else []
    # Make model
    return onnx.helper.make_model(graph, functions=funcs, opset_imports=opsets)


def check_model(model):
    # Check that it works
    onnx.checker.check_model(model)


def save_model(model, path):
    onnx.save(model, path)


def infer_shapes(model):
    # Infer shapes
    return onnx.shape_inference.infer_shapes(model)


def load_model(model_path):
    print("-- Model Path: {}".format(model_path))
    model = onnx.load(model_path)
    for model_input in model.graph.input:
        print("   Input Name: {}".format(model_input.name))
    for model_output in model.graph.output:
        print("   Output Name: {}".format(model_output.name))
    return model


def print_nodes(graph):
    for node in graph.node:
        print(
            "-- Name: {}\n   - Inputs: {}\n   - Outputs:{}".format(
                node.name, node.input, node.output
            )
        )
