import onnx
from onnx import helper, ModelProto

def set_node_as_output(onnx_model_path, node_name, modified_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Find the desired node and create a new output for it
    new_outputs = []
    node_found = False
    for node in graph.node:
        if node.name == node_name:
            for output in node.output:
                # Create a new output ValueInfoProto based on the node's output
                output_value_info = helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, (1,80,8400))
                new_outputs.append(output_value_info)
                node_found = True
                break

    if not node_found:
        print(f"Node '{node_name}' not found in the graph.")
    else:
        # Set new outputs to the graph
        graph.output.extend(new_outputs)

    # Save the modified model
    onnx.save(model, modified_model_path)
    print(f"Model saved with modified outputs to {modified_model_path}")

# Example usage
onnx_model_path = 'yolov8m.onnx'
node_name = '/model.22/Sigmoid'
modified_model_path = 'yolov8m_modified.onnx'

set_node_as_output(onnx_model_path, node_name, modified_model_path)
