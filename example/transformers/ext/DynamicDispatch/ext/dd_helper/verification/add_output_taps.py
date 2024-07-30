import os, sys, argparse
import onnx


class ONNXModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load()

    def load(self):
        m = onnx.load(self.model_path)
        return onnx.shape_inference.infer_shapes(m)

    def add_output_taps(self, output_names=None):
        value_info = {}  # Name, Shape
        for vi in self.model.graph.value_info:
            value_info[vi.name] = [
                dim.dim_value for dim in vi.type.tensor_type.shape.dim
            ]

        initializer_info = {}  # Name, type
        for vi in self.model.graph.initializer:
            initializer_info[vi.name] = vi.data_type

        output_value_info = []
        if output_names is None:
            # Add all QuantizeLinear outputs as model outputs
            for node in self.model.graph.node:
                if node.op_type == "QuantizeLinear":
                    output_dtype = initializer_info.get(node.input[2], None)
                    output_shape = value_info.get(node.input[0], [])
                    print(
                        "QuantizeLinear -> Name: {}, Output: {}, DType: {}, Shape: {}".format(
                            node.name, node.output, output_dtype, output_shape
                        )
                    )
                    output_value_info.append(
                        onnx.helper.make_tensor_value_info(
                            node.output[0], output_dtype, output_shape
                        )
                    )
        else:
            # Add output names to graph outputs
            for name in output_names:
                output_value_info.append(
                    onnx.helper.make_tensor_value_info(
                        # TODO: Data Type may need to be passed
                        name,
                        onnx.TensorProto.UINT8,
                        [],
                    )
                )

        self.model.graph.output.extend(output_value_info)

        for o in self.model.graph.input:
            print(
                "- Input Name: {}\n  Shape: {}, Type: {}".format(
                    o.name,
                    [dim.dim_value for dim in o.type.tensor_type.shape.dim],
                    o.type,
                )
            )
        for o in self.model.graph.output:
            print(
                "- Output Name: {}\n  Shape: {}".format(
                    o.name, [dim.dim_value for dim in o.type.tensor_type.shape.dim]
                )
            )

        print("- Total Number of outputs added:", len(output_value_info))

    def save_as(self, output_path):
        # onnx.checker.check_model(self.model)
        onnx.save(self.model, output_path)


def main():
    # Create a parser
    parser = argparse.ArgumentParser(description="Adds output nodes to ONNX Model")
    # Add args
    parser.add_argument("--model-in", required=True, help="Input Model path")
    parser.add_argument("--model-out", required=True, help="Output Model path")
    # Parse args
    args = parser.parse_args(sys.argv[1:])

    # Load model
    in_model = ONNXModel(args.model_in)

    # Add node names to output_names if specific outputs are required
    output_names = None
    # Add output info to the model
    in_model.add_output_taps(output_names)

    # Save updated model_path
    in_model.save_as(args.model_out)


if __name__ == "__main__":
    main()
