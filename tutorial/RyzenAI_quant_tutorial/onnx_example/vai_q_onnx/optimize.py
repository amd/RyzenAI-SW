#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import numpy as np
from math import sqrt

import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel
from .quant_utils import remove_nodes, remove_initializers, get_clip_min_max

logger = logging.getLogger(__name__)


class Optimize(object):
    """
    A class for pre-quantization optimizations on DPU/IPU
    Args:
        model (onnx.ModelProto): The ONNX model to be optimized.
        op_types_to_quantize (list): A list of operation types to be quantized.
        nodes_to_quantize (list): A list of node names to be quantized.
        nodes_to_exclude (list): A list of node names to be excluded from quantization.

    """

    def __init__(self, model, op_types_to_quantize, nodes_to_quantize,
                 nodes_to_exclude):
        self.model = model
        self.op_types_to_quantize = op_types_to_quantize
        self.nodes_to_quantize = nodes_to_quantize
        self.nodes_to_exclude = nodes_to_exclude

    def should_quantize_node(self, node):
        if (self.nodes_to_quantize is not None and
                len(self.nodes_to_quantize) != 0 and
                node.name not in self.nodes_to_quantize):
            return False

        if node.op_type not in self.op_types_to_quantize:
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        return True

    def replace_node_with(self, node, replaced_type):
        new_node = onnx.helper.make_node(replaced_type,
                                         inputs=node.input,
                                         outputs=node.output,
                                         name=node.name)

        self.model.graph.node.append(new_node)
        return new_node

    def convert_bn_to_conv(self):
        """Convert BatchNormalization to Conv.
        """

        def _get_folded_conv_weights(bn_gamma, bn_beta, bn_mm, bn_mv,
                                     bn_epsilon):
            if bn_gamma is not None:
                multiplier = bn_gamma / np.sqrt(bn_mv + bn_epsilon)
            else:
                multiplier = 1 / np.sqrt(bn_mv + bn_epsilon)

            folded_conv_kernel = multiplier
            folded_conv_bias = bn_beta + (-bn_mm) * multiplier
            return folded_conv_kernel, folded_conv_bias

        self.op_types_to_quantize.append("BatchNormalization")
        nodes_to_remove = []
        init_to_remove = []
        for node in self.model.graph.node:

            if node.op_type == 'BatchNormalization' and self.should_quantize_node(
                    node):
                input_name = node.input[0]
                input_shape = []
                for input_info in self.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [
                            dim.dim_value
                            for dim in input_info.type.tensor_type.shape.dim
                        ]
                if len(node.input) == 5 and len(input_shape) == 4:
                    bn_epsilon = next((attr.f
                                       for attr in node.attribute
                                       if attr.name == 'epsilon'), None)
                    for init in self.model.graph.initializer:
                        if init.name == node.input[1]:
                            bn_gamma = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[2]:
                            bn_beta = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[3]:
                            bn_mm = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[4]:
                            bn_mv = onnx.numpy_helper.to_array(init)
                    try:
                        weights, bias = _get_folded_conv_weights(
                            bn_gamma, bn_beta, bn_mm, bn_mv, bn_epsilon)
                        num_channel = bn_mm.shape[0]
                        weights = weights.reshape([num_channel, 1, 1, 1])
                        weights_tensor = onnx.numpy_helper.from_array(
                            weights, name=node.output[0] + "weights")
                        bias_tensor = onnx.numpy_helper.from_array(
                            bias, name=node.output[0] + "bias")
                        self.model.graph.initializer.extend(
                            [weights_tensor, bias_tensor])
                        new_node = onnx.helper.make_node(
                            "Conv",
                            inputs=[
                                node.input[0], node.output[0] + "weights",
                                node.output[0] + "bias"
                            ],
                            outputs=[node.output[0]],
                            group=num_channel,
                            kernel_shape=[1, 1],
                            strides=[1, 1],
                            name=node.name)

                        nodes_to_remove.append(node)
                        init_to_remove.extend([
                            node.input[1], node.input[2], node.input[3],
                            node.input[4]
                        ])
                        self.model.graph.node.append(new_node)
                        logger.info(
                            f"Found BatchNormalization node {node.name}. "
                            f"Replacing with Conv.")
                    except Exception as e:
                        logger.warning(
                            f"Fail to generate conv's weights and bias beacuse of {e}, skip converting bn to conv"
                        )
                else:
                    logger.warning(
                        f"Fail to convert bn {node.name} to conv beacuse BatchNormalization's input or shape does not meet the requirements"
                    )
        self.model = remove_nodes(self.model, nodes_to_remove)
        self.model = remove_initializers(self.model, init_to_remove)

    def convert_reduce_mean_to_global_avg_pool(self):
        """Convert ReduceMean to GlobalAveragePool.
        """

        from .quant_utils import check_reduce_mean_condition
        nodes_to_remove = []
        for node in self.model.graph.node:

            if node.op_type == 'ReduceMean' and check_reduce_mean_condition(
                    self.model, node) and self.should_quantize_node(node):
                if len(node.input) == 1:
                    new_node = self.replace_node_with(node, 'GlobalAveragePool')
                    nodes_to_remove.append(node)
                    logger.info(
                        f"Found ReduceMean node {node.name} with axes=[2, 3]. "
                        f"Replacing with GlobalAveragePool.")
                # Handling opset >= 18 for Reduce Mean
                elif len(node.input) == 2:
                    new_node = onnx.helper.make_node('GlobalAveragePool',
                                                     inputs=[node.input[0]],
                                                     outputs=node.output,
                                                     name=node.name)

                    nodes_to_remove.append(node)
                    self.model.graph.node.append(new_node)
                    logger.info(
                        f"Found ReduceMean node {node.name} with axes=[2, 3]. "
                        f"Replacing with GlobalAveragePool.")
        self.model = remove_nodes(self.model, nodes_to_remove)

    def split_large_kernel_pool(self):
        """
        For pooling with an excessively large kernel size in the onnx model,
        split it into multiple smaller poolings.
        """

        def _get_factors(num):
            factor_1 = int(sqrt(num))
            while (factor_1 > 1):
                if (num % (factor_1) == 0):
                    factor_2 = num / factor_1
                    return int(factor_1), int(factor_2)
                factor_1 = factor_1 - 1
            factor_2 = num
            return int(factor_1), int(factor_2)

        for node in self.model.graph.node:
            if node.op_type == "GlobalAveragePool" and self.should_quantize_node(
                    node):
                input_name = node.input[0]
                kw = None
                kh = None
                for input_info in self.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [
                            dim.dim_value
                            for dim in input_info.type.tensor_type.shape.dim
                        ]
                        if len(input_shape) == 4:
                            shape_to_check = True
                            kh = input_shape[2]
                            kw = input_shape[3]
                        break
                if not kw or not kh:
                    logger.warning(
                        'Failed to get the input shape, skip optimizing for GlobalAveragePool {}.'
                        .format(node.name))
                    continue
                # Only one split is supported.
                # TODO: Support multiple split operations
                elif kw * kh > 512:
                    kh1, kh2 = _get_factors(kh)
                    kw1, kw2 = _get_factors(kw)
                    if kh1 * kw1 > 512 or kh2 * kw2 > 512:
                        logger.warning(
                            "After split, the kernel size is still too large."
                            "Currently, only one split is supported. Skip optimization."
                        )
                    else:
                        split_tensor = node.input[0] + "_Split"
                        pool_node = onnx.helper.make_node(
                            "AveragePool",
                            inputs=[node.input[0]],
                            outputs=[split_tensor],
                            kernel_shape=[kh1, kw1],
                            strides=[kh1, kw1],
                            name=split_tensor)
                        if not node.name:
                            node.name = node.output[0]
                        node.input[0] = split_tensor
                        self.model.graph.node.extend([pool_node])
                        logger.info(
                            f"Found GlobalAveragePool node {node.name} with large kernel size. "
                            f"Split it into multiple AveragePools.")

    def convert_split_to_slice(self):
        """Convert Split to Slice.
        """
        nodes_to_remove = []
        init_to_remove = []
        for node in self.model.graph.node:
            if node.op_type == 'Split' and self.should_quantize_node(node):
                num_input = len(node.input)
                axis_attr = next(
                    (attr for attr in node.attribute if attr.name == 'axis'),
                    None)
                axis = axis_attr.i
                input_name = node.input[0]
                output_names = node.output
                if num_input == 2:
                    splits = None
                    for init in self.model.graph.initializer:
                        if init.name == node.input[1]:
                            splits = onnx.numpy_helper.to_array(init).tolist()
                    if splits is None:
                        logger.warning(
                            f"No split detected of {node.name}, "
                            "failed to convert split to slice, please check the input model."
                        )
                        break
                elif num_input == 1:
                    split_attr = next((attr for attr in node.attribute
                                       if attr.name == 'split'), None)
                    if split_attr is None:
                        logger.warning(
                            f"No split detected of {node.name}, "
                            "failed to convert split to slice, please check the input model."
                        )
                        break
                    splits = split_attr.ints
                else:
                    logger.warning(
                        f"Failed to convert split of {node.name} to slice, "
                        "the number of input nodes is not supported.")
                    break
                starts = [sum(splits[:i]) for i in range(len(splits))]
                ends = [sum(splits[:i + 1]) for i in range(len(splits))]
                for i in range(len(output_names)):
                    starts_node = onnx.helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[output_names[i] + '_starts_' + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + '_starts_' + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[starts[i]]))
                    ends_node = onnx.helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[output_names[i] + '_ends_' + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + '_ends_' + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[ends[i]]))
                    axes_node = onnx.helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[output_names[i] + '_axes_' + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + '_axes_' + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[axis]))
                    steps_node = onnx.helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[output_names[i] + '_steps_' + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + '_steps_' + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[1]))
                    slice_node = onnx.helper.make_node(
                        "Slice",
                        inputs=[
                            input_name, output_names[i] + '_starts_' + str(i),
                            output_names[i] + '_ends_' + str(i),
                            output_names[i] + '_axes_' + str(i),
                            output_names[i] + '_steps_' + str(i)
                        ],
                        outputs=[output_names[i]],
                        name=output_names[i] + '_' + str(i))
                    self.model.graph.node.extend([
                        slice_node, starts_node, ends_node, axes_node,
                        steps_node
                    ])
                nodes_to_remove.append(node)
                if len(node.input) > 1:
                    init_to_remove.append(node.input[1])
                logger.info(f"Found Split node {node.name}. "
                            f"Replacing with Slice.")
        self.model = remove_nodes(self.model, nodes_to_remove)
        self.model = remove_initializers(self.model, init_to_remove)

    def fuse_instance_norm(self):
        '''
        The split instance norm operation will be fused to InstanceNorm operation
        '''
        onnx_model = ONNXModel(self.model)
        tensor_to_producer_dict = {}
        remove_nodes = []
        remove_inits = []
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_producer_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_producer_dict[init.name] = init
        for node in onnx_model.model.graph.node:
            if node.op_type == "Add":
                add0_i0 = node.input[0]
                add0_i1 = node.input[1]
                add0_i0_node = tensor_to_producer_dict[add0_i0]
                add0_i1_node = tensor_to_producer_dict[add0_i1]
                # TODO: Use different dictionaries to distinguish between node and init.
                try:
                    if add0_i0_node.op_type == "Mul" and add0_i1_node.op_type == "Sub":
                        sub0_node = add0_i1_node
                        sub0_i0 = sub0_node.input[0]
                        sub0_i1 = sub0_node.input[1]
                        sub0_i1_node = tensor_to_producer_dict[sub0_i1]
                        if sub0_i1_node.op_type == "Mul":
                            mul0_node = sub0_i1_node
                            mul0_i0 = mul0_node.input[0]
                            mul0_i1 = mul0_node.input[1]
                            mul0_i0_node = tensor_to_producer_dict[mul0_i0]
                            mul0_i1_node = tensor_to_producer_dict[mul0_i1]
                            if mul0_i0_node.op_type == "GlobalAveragePool" and mul0_i1_node.op_type == "Mul":
                                mul1_node = mul0_i1_node
                                mul1_i0 = mul1_node.input[0]
                                mul1_i1 = mul1_node.input[1]
                                mul1_i0_node = tensor_to_producer_dict[mul1_i0]
                                mul1_i1_node = tensor_to_producer_dict[mul1_i1]
                                if mul1_i0_node.op_type == "Reciprocal":
                                    rec0_node = mul1_i0_node
                                    rec0_i0 = rec0_node.input[0]
                                    rec0_i0_node = tensor_to_producer_dict[
                                        rec0_i0]
                                    if rec0_i0_node.op_type == "Sqrt":
                                        sqr0_node = rec0_i0_node
                                        sqr0_i0 = sqr0_node.input[0]
                                        sqr0_i0_node = tensor_to_producer_dict[
                                            sqr0_i0]
                                        if sqr0_i0_node.op_type == "Add":
                                            add1_node = sqr0_i0_node
                                            add1_i0 = add1_node.input[0]
                                            add1_i1 = add1_node.input[1]
                                            add1_i0_node = tensor_to_producer_dict[
                                                add1_i0]
                                            if add1_i0_node.op_type == "GlobalAveragePool":
                                                gap0_node = add1_i0_node
                                                gap0_i0 = gap0_node.input[0]
                                                gap0_i0_node = tensor_to_producer_dict[
                                                    gap0_i0]
                                                if gap0_i0_node.op_type == "Mul":
                                                    mul2_node = gap0_i0_node
                                                    mul2_i0 = mul2_node.input[0]
                                                    mul2_i0_node = tensor_to_producer_dict[
                                                        mul2_i0]
                                                    if mul2_i0_node.op_type == "Sub":
                                                        sub1_node = mul2_i0_node
                                                        sub1_i0 = sub1_node.input[
                                                            0]
                                                        sub1_i1 = sub1_node.input[
                                                            1]
                                                        sub1_i0_node = tensor_to_producer_dict[
                                                            sub1_i0]
                                                        sub1_i1_node = tensor_to_producer_dict[
                                                            sub1_i1]
                                                        if sub1_i1_node.op_type == "GlobalAveragePool":

                                                            # Remove nodes
                                                            remove_node_list = [
                                                                node,
                                                                add0_i0_node,
                                                                add0_i1_node,
                                                                sub0_i1_node,
                                                                mul0_i0_node,
                                                                mul0_i1_node,
                                                                mul1_i0_node,
                                                                rec0_i0_node,
                                                                sqr0_i0_node,
                                                                add1_i0_node,
                                                                gap0_i0_node,
                                                                mul2_i0_node,
                                                            ]

                                                            # Add InstanceNormalization
                                                            bias_init = onnx_model.get_initializer(
                                                                sub0_i0)
                                                            bias_init.dims[:] = [
                                                                bias_init.
                                                                dims[1]
                                                            ]
                                                            weight_init = onnx_model.get_initializer(
                                                                mul1_i1)
                                                            weight_init.dims[:] = [
                                                                weight_init.
                                                                dims[1]
                                                            ]
                                                            eps_init = onnx_model.get_initializer(
                                                                add1_i1)

                                                            instance_norm_node = onnx.helper.make_node(
                                                                "InstanceNormalization",
                                                                [
                                                                    sub1_i0,
                                                                    mul1_i1,
                                                                    sub0_i0
                                                                ],
                                                                node.output,
                                                                node.name,
                                                                epsilon=onnx.
                                                                numpy_helper.
                                                                to_array(
                                                                    eps_init
                                                                ).item())
                                                            logger.info(
                                                                f"Matched Instance Normalization, fuse it into InstanceNormalization {node.name}"
                                                            )
                                                            onnx_model.add_node(
                                                                instance_norm_node
                                                            )

                                                            remove_nodes.extend(
                                                                remove_node_list
                                                            )
                                                            remove_inits.append(
                                                                eps_init)
                except Exception as e:
                    logger.debug(
                        f"FuseInstanceNorm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )
        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializer(remove_inits)
        self.model = onnx_model.model

    def fuse_l2_norm(self):
        """
        convert L2norm ops to LpNormalization
        """
        onnx_model = ONNXModel(self.model)
        tensor_to_producer_dict = {}
        remove_nodes = []
        remove_inits = []
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_producer_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_producer_dict[init.name] = init
        for node in onnx_model.model.graph.node:
            if node.op_type == "Mul":
                inp_0 = node.input[0]
                inp_1 = node.input[1]
                inp_0_node = tensor_to_producer_dict[inp_0]
                inp_1_node = tensor_to_producer_dict[inp_1]
                try:
                    if inp_0_node.op_type == "Unsqueeze" and inp_1_node.op_type == "Reciprocal":
                        rec_node = inp_1_node
                        rec_inp_0 = rec_node.input[0]
                        rec_inp_0_node = tensor_to_producer_dict[rec_inp_0]
                        if rec_inp_0_node.op_type == "Sqrt":
                            sqrt_node = rec_inp_0_node
                            sqrt_inp_0 = sqrt_node.input[0]
                            sqrt_inp_0_node = tensor_to_producer_dict[
                                sqrt_inp_0]
                            if sqrt_inp_0_node.op_type == "Max":
                                max_node = sqrt_inp_0_node
                                max_inp_0 = max_node.input[0]
                                max_inp_1 = max_node.input[1]
                                max_inp_0_node = tensor_to_producer_dict[
                                    max_inp_0]
                                if max_inp_0_node.op_type == "ReduceSum":
                                    red_node = max_inp_0_node
                                    red_inp_0 = red_node.input[0]
                                    red_inp_0_node = tensor_to_producer_dict[
                                        red_inp_0]
                                if red_inp_0_node.op_type == "Mul":
                                    mul_node = red_inp_0_node
                                    mul_inp_0 = mul_node.input[0]
                                    mul_inp_0_node = tensor_to_producer_dict[
                                        mul_inp_0]
                                    if mul_inp_0_node.op_type == "Unsqueeze":
                                        uns_node = mul_inp_0_node
                                        # Remove nodes
                                        logger.info(
                                            f"Found L2norm ops from {node.name}."
                                        )
                                        nodes_to_remove_list = [
                                            node,
                                            rec_node,
                                            sqrt_node,
                                            max_node,
                                            red_node,
                                            mul_node,
                                        ]
                                        remove_nodes.extend(
                                            nodes_to_remove_list)
                                        eps_init = onnx_model.get_initializer(
                                            max_inp_1)
                                        remove_inits.append(eps_init)
                                        # Add LpNormalization
                                        inp = uns_node.output[0]
                                        out = node.output[0]
                                        l2norm_node = onnx.helper.make_node(
                                            "LpNormalization", [inp], [out],
                                            node.name,
                                            p=2)
                                        onnx_model.add_node(l2norm_node)
                                        logger.info(
                                            "Converted L2norm ops from {node.name} to LpNormalization."
                                        )
                except Exception as e:
                    logger.debug(
                        f"FuseL2Norm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )
        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializer(remove_inits)
        self.model = onnx_model.model

    def convert_clip_to_relu(self):
        '''
        Convert Clip to Relu.
        '''
        nodes_to_remove = []
        init_to_remove = []

        for node in self.model.graph.node:
            if node.op_type == 'Clip' and self.should_quantize_node(node):
                min_value, max_value, para_type = get_clip_min_max(
                    self.model, node)

                if min_value is None or min_value < 0:
                    continue  # could not be replaced with Relu

                if para_type == 1:
                    # This Clip node's min and max come from initializers
                    for init in self.model.graph.initializer:
                        if len(node.input) > 1 and init.name == node.input[1]:
                            init_to_remove.append(init)
                        if len(node.input) > 2 and init.name == node.input[2]:
                            init_to_remove.append(init)

                elif para_type == 2:
                    # This Clip node's min and max come from other nodes
                    for nd in self.model.graph.node:
                        if ((len(node.input) > 1 and node.input[1] in nd.output)
                                or (len(node.input) > 2 and
                                    node.input[2] in nd.output)) == False:
                            continue

                        if nd.op_type == 'Identity':
                            for init in self.model.graph.initializer:
                                if len(nd.input
                                      ) > 1 and init.name == nd.input[1]:
                                    init_to_remove.append(init)
                                if len(nd.input
                                      ) > 2 and init.name == nd.input[2]:
                                    init_to_remove.append(init)
                            nodes_to_remove.append(nd)

                        elif nd.op_type == 'Constant':
                            nodes_to_remove.append(nd)

                logger.info(
                    f"Convert Clip node {node.name} to Relu, "
                    f"its min is {min_value}, max is {max_value} and type is {para_type}"
                )
                relu_node = onnx.helper.make_node("Relu", [node.input[0]],
                                                  node.output, node.name)
                self.model.graph.node.extend([relu_node])  # insert a Relu node
                nodes_to_remove.append(node)  # to remove this Clip node

        self.model = remove_nodes(self.model, nodes_to_remove)
        self.model = remove_initializers(self.model, init_to_remove)


def optimize(
    model,
    op_types_to_quantize,
    nodes_to_quantize,
    nodes_to_exclude,
    convert_bn_to_conv=True,
    convert_reduce_mean_to_global_avg_pool=True,
    split_large_kernel_pool=True,
    convert_split_to_slice=True,
    fuse_instance_norm=True,
    fuse_l2_norm=True,
    convert_clip_to_relu=True,
):
    """optimizing models to meet the DPU/IPU constraints."""

    optimizer = Optimize(
        model,
        op_types_to_quantize,
        nodes_to_quantize,
        nodes_to_exclude,
    )

    if fuse_instance_norm:
        optimizer.fuse_instance_norm()

    if convert_bn_to_conv:
        optimizer.convert_bn_to_conv()

    if convert_reduce_mean_to_global_avg_pool:
        optimizer.convert_reduce_mean_to_global_avg_pool()

    if split_large_kernel_pool:
        optimizer.split_large_kernel_pool()

    if convert_split_to_slice:
        optimizer.convert_split_to_slice()

    if fuse_l2_norm:
        optimizer.fuse_l2_norm()

    if convert_clip_to_relu:
        optimizer.convert_clip_to_relu()

    return optimizer.model
