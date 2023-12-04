#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import sys, argparse
import onnx
import tabulate as tab


def display_graph_info(model, args):
    attr_table = []
    node_table = []

    value_info = {}
    for vi in model.graph.value_info:
        value_info[vi.name.split("/")[-1]] = [
            dim.dim_value for dim in vi.type.tensor_type.shape.dim
        ]

    if not value_info:
        print("- Warning: Model doesn't have value info.")
        print(
            "- Info: Run shape inference on the model to see shape info for inputs/outputs"
        )

    for node in model.graph.node:
        i_names = []
        o_names = []
        i_shapes = []
        o_shapes = []
        attr_names = []
        attr_types = []
        attr_value = []
        if args.filter is not None and node.op_type not in args.filter:
            continue
        for attr in node.attribute:
            attr_val = None
            attr_type = None
            if attr.type == onnx.AttributeProto.INT:
                attr_val = attr.i
                attr_type = "INT"
            elif attr.type == onnx.AttributeProto.FLOAT:
                attr_val = attr.f
                attr_type = "FLOAT"
            elif attr.type == onnx.AttributeProto.STRING:
                attr_val = attr.s
                attr_type = "STRING"
            elif attr.type == onnx.AttributeProto.INTS:
                attr_val = attr.ints
                attr_type = "INTS"
            elif attr.type == onnx.AttributeProto.FLOATS:
                attr_val = attr.floats
                attr_type = "FLOATS"
            elif attr.type == onnx.AttributeProto.STRINGS:
                attr_val = attr.strings
                attr_type = "STRINGS"
            elif attr.type == onnx.AttributeProto.TENSOR:
                attr_val = {"Type": str(attr.t.data_type), "Dims": str(attr.t.dims)}
                attr_type = "TENSOR"
            else:
                attr_val = "Check-Model"
                attr_type = "Type-" + str(attr.type)

            attr_names.append(attr.name)
            attr_value.append(str(attr_val))
            attr_types.append(attr_type)

        i_names = [i.split("/")[-1] for i in node.input]
        o_names = [o.split("/")[-1] for o in node.output]

        i_shapes = [
            str(value_info[name]) if name in value_info else str([]) for name in i_names
        ]
        o_shapes = [
            str(value_info[name]) if name in value_info else str([]) for name in o_names
        ]

        node_table.append(
            [
                node.op_type,
                node.name,
                "\n".join(i_names),
                "\n".join(i_shapes),
                "\n".join(o_names),
                "\n".join(o_shapes),
                "\n".join(attr_names),
                "\n".join(attr_types),
                "\n".join(attr_value),
            ]
        )

    if node_table:
        table_data = tab.tabulate(
            node_table,
            headers=[
                "OP Type",
                "Name",
                "Inputs",
                "In Shape",
                "Outputs",
                "Out Shape",
                "Attributes",
                "Attr Type",
                "Attr Values",
            ],
            tablefmt="grid",
            numalign="right",
            maxcolwidths=[None, 30],
        )
    else:
        print("- Info: No operators found with names: {}".format(str(args.filter)))
        return

    if args.outfile:
        with open(args.outfile, "w") as fp:
            fp.write(table_data)
    else:
        print(table_data)


def display_unique_ops(model, args):
    ops = dict()
    for node in model.graph.node:
        if args.filter is not None and node.op_type not in args.filter:
            continue
        names = ops.get(node.op_type, [])
        names.append(node.name)
        ops[node.op_type] = names
    table = []
    node_count = 0
    for op, names in ops.items():
        table.append([op, "\n".join(names), len(names)])
        node_count = node_count + len(names)

    table.append(["Total Number of Nodes", "--------->", node_count])

    table_str = tab.tabulate(
        table, headers=["Operator", "Nodes", "Count"], tablefmt="grid"
    )

    if args.outfile:
        with open(args.outfile, "w") as fp:
            fp.write(table_str)
    else:
        print(table_str)


def main(args):
    # Arguments
    parser = argparse.ArgumentParser(
        description="Python utility to extract metadata from ONNX model"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model Path")
    parser.add_argument(
        "-f",
        "--filter",
        nargs="+",
        default=None,
        help="Filter information with operator names",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default=None,
        help="Path to output file to dump data",
    )
    parser.add_argument(
        "-g",
        "--graph-info",
        action="store_true",
        default=True,
        help="Graph information",
    )
    parser.add_argument(
        "-u",
        "--uniq-ops",
        action="store_true",
        default=False,
        help="Unique operators information",
    )

    # Parse args
    known_args, unknown_args = parser.parse_known_args(args)
    if len(unknown_args) != 0:
        print("- Error: Unknown Arguments: {}".format(unknown_args))
        return

    # Load model and print stuff
    model_path = known_args.model
    print("- Model: {}".format(model_path))
    model = onnx.load(model_path)

    if known_args.uniq_ops:
        display_unique_ops(model, known_args)
    elif known_args.graph_info:
        display_graph_info(model, known_args)


if __name__ == "__main__":
    main(sys.argv[1:])
