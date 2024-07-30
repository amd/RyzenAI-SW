def create_function_subgraph(input_model, output_model):
    # Load model
    model = onnx.load(input_model)
    model = onnx.shape_inference.infer_shapes(model)

    graph_inputs = model.graph.input
    graph_outputs = model.graph.output
    graph_initializers = model.graph.initializer
    graph_value_info = model.graph.value_info
    graph_nodes = model.graph.node

    # for info in graph_value_info:
    #     name = info.name
    #     etype = info.type.tensor_type.elem_type
    #     shape = [dim.dim_value for dim in info.type.tensor_type.shape.dim]
    #     print(" -> Value Info Name: {}, Type: {}, Shape: {}".format(name, etype, shape))
    # print("")
    for meta in graph_inputs:
        print(" -> Input Name: {}".format(meta.name))
    for meta in graph_outputs:
        print(" -> Output Name: {}".format(meta.name))

    # Add input value info
    model_input = make_tensor_value_info(
        graph_inputs[0].name, TensorProto.UINT16, [1, "M", "K"]
    )
    graph_in_value_info = [model_input]

    # Input names will be inputs for original graphs + initializer names
    ## Input names
    input_names = [i.name for i in graph_in_value_info]
    ## Initializer names
    input_names.extend([i.name for i in graph_initializers])

    # Add output value info
    model_output = make_tensor_value_info(
        graph_outputs[0].name, TensorProto.UINT16, [1, "M", "N"]
    )

    graph_out_value_info = [model_output]
    # Output names must be same as original subgraph's output names
    output_names = [i.name for i in graph_out_value_info]

    # Create function with all subgraph nodes
    function = onnx.helper.make_function(
        "com.amd",
        inputs=input_names,
        outputs=output_names,
        nodes=graph_nodes,
        fname="QMatMul",
        opset_imports=get_opsets(),
    )

    # Create a node with function name
    f_node = onnx.helper.make_node(
        op_type="QMatMul",
        inputs=input_names,
        outputs=output_names,
        name="QMatMul_node",
        domain="com.amd",
    )

    # Create graph with one function node
    graph = onnx.helper.make_graph(
        nodes=[f_node],
        inputs=graph_in_value_info,
        outputs=graph_out_value_info,
        name="QMatMul_subgraph",
    )

    # Add initializers to new graph
    graph.initializer.extend(graph_initializers)

    # Create model
    model = onnx.helper.make_model(
        graph, functions=[function], opset_imports=get_opsets()
    )

    # Shape infer
    model = onnx.shape_inference.infer_shapes(model)
    # Check model for validity
    onnx.checker.check_model(model)
    # Save model
    onnx.save_model(model, output_model)
