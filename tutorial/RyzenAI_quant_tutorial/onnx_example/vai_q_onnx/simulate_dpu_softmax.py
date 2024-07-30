#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Simulated softmax for DPU."""

import onnx
from onnx import onnx_pb as onnx_proto


class SimulateDPUSoftmax(object):

    def __init__(self, opset_version=11):
        """
        This is a softmax that simulates DPU behavior
        :param opset_version: ONNX model opset version
        """

        self._opset_version = opset_version

    def simulate(self, node):
        """
        This simulated softmax is compatiable with opset7 or higher
        :param node: the Softmax node to simulate for DPU
        :return: the new nodes used to replace Softmax node
        """

        new_nodes = []

        if self._opset_version < 7:
            return new_nodes

        reduce_axes = -1
        for attr in node.attribute:
            if attr.name == 'axis':
                reduce_axes = attr.i

        softmax_name = node.name

        data_type = onnx_proto.TensorProto.BFLOAT16

        def _exp_poly_approximation(input_node_name):
            nonlocal new_nodes

            exp_poly_namescope = softmax_name + "/exp_poly"

            # exp poly - round
            round_namescope = exp_poly_namescope + "/round"

            cast_0_node_name = round_namescope + "/cast"
            cast_0_node = onnx.helper.make_node("Cast", [input_node_name],
                                                [cast_0_node_name + "_output"],
                                                name=cast_0_node_name,
                                                to=data_type)
            new_nodes.append(cast_0_node)

            rcp_ln2_node_name = round_namescope + "/rcp_ln2"
            rcp_ln2_node = onnx.helper.make_node(
                'Constant', [], [rcp_ln2_node_name + '_output'],
                name=rcp_ln2_node_name,
                value=onnx.helper.make_tensor(rcp_ln2_node_name + "_value",
                                              data_type, [],
                                              [1.4426950408889634]))
            new_nodes.append(rcp_ln2_node)

            mul_0_node_name = round_namescope + "/mul"
            mul_0_node = onnx.helper.make_node(
                "Mul",
                [cast_0_node_name + '_output', rcp_ln2_node_name + '_output'],
                [mul_0_node_name + '_output'],
                name=mul_0_node_name)
            new_nodes.append(mul_0_node)

            round_0_node_name = round_namescope + "/round"
            round_0_node = onnx.helper.make_node(
                "Floor", [mul_0_node_name + '_output'],
                [round_0_node_name + '_output'],
                name=round_0_node_name)
            new_nodes.append(round_0_node)

            # exp poly - modulo
            modulo_namescope = exp_poly_namescope + "/modulo"

            ln2_node_name = modulo_namescope + "/ln2"
            ln2_node = onnx.helper.make_node('Constant', [],
                                             [ln2_node_name + '_output'],
                                             name=ln2_node_name,
                                             value=onnx.helper.make_tensor(
                                                 ln2_node_name + "_value",
                                                 data_type, [],
                                                 [0.6931471805599453]))
            new_nodes.append(ln2_node)

            mul_1_node_name = modulo_namescope + "/mul"
            mul_1_node = onnx.helper.make_node(
                "Mul",
                [round_0_node_name + '_output', ln2_node_name + '_output'],
                [mul_1_node_name + '_output'],
                name=mul_1_node_name)
            new_nodes.append(mul_1_node)

            sub_1_node_name = modulo_namescope + "/sub"
            sub_1_node = onnx.helper.make_node(
                "Sub",
                [cast_0_node_name + '_output', mul_1_node_name + '_output'],
                [sub_1_node_name + '_output'],
                name=sub_1_node_name)
            new_nodes.append(sub_1_node)

            # exp poly - approx
            poly_approx_namescope = exp_poly_namescope + "/poly_approx"

            cast_1_node_name = poly_approx_namescope + "/cast_1"
            cast_1_node = onnx.helper.make_node("Cast",
                                                [sub_1_node_name + "_output"],
                                                [cast_1_node_name + "_output"],
                                                name=cast_1_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_1_node)

            alpha_3_node_name = poly_approx_namescope + "/alpha_3"
            alpha_3_node = onnx.helper.make_node(
                'Constant', [], [alpha_3_node_name + '_output'],
                name=alpha_3_node_name,
                value=onnx.helper.make_tensor(alpha_3_node_name + "_value",
                                              data_type, [], [0.21875]))
            new_nodes.append(alpha_3_node)

            cast_2_node_name = poly_approx_namescope + "/cast_2"
            cast_2_node = onnx.helper.make_node("Cast",
                                                [alpha_3_node_name + "_output"],
                                                [cast_2_node_name + "_output"],
                                                name=cast_2_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_2_node)

            mul_2_node_name = poly_approx_namescope + "/mul_2"
            mul_2_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + '_output', cast_2_node_name + '_output'],
                [mul_2_node_name + '_output'],
                name=mul_2_node_name)
            new_nodes.append(mul_2_node)

            alpha_2_node_name = poly_approx_namescope + "/alpha_2"
            alpha_2_node = onnx.helper.make_node(
                'Constant', [], [alpha_2_node_name + '_output'],
                name=alpha_2_node_name,
                value=onnx.helper.make_tensor(alpha_2_node_name + "_value",
                                              data_type, [], [0.486328125]))
            new_nodes.append(alpha_2_node)

            cast_3_node_name = poly_approx_namescope + "/cast_3"
            cast_3_node = onnx.helper.make_node("Cast",
                                                [alpha_2_node_name + "_output"],
                                                [cast_3_node_name + "_output"],
                                                name=cast_3_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_3_node)

            add_0_node_name = poly_approx_namescope + "/add"
            add_0_node = onnx.helper.make_node(
                "Add",
                [mul_2_node_name + '_output', cast_3_node_name + '_output'],
                [add_0_node_name + '_output'],
                name=add_0_node_name)
            new_nodes.append(add_0_node)

            cast_4_node_name = poly_approx_namescope + "/cast_4"
            cast_4_node = onnx.helper.make_node("Cast",
                                                [add_0_node_name + "_output"],
                                                [cast_4_node_name + "_output"],
                                                name=cast_4_node_name,
                                                to=data_type)
            new_nodes.append(cast_4_node)

            cast_5_node_name = poly_approx_namescope + "/cast_5"
            cast_5_node = onnx.helper.make_node("Cast",
                                                [cast_4_node_name + "_output"],
                                                [cast_5_node_name + "_output"],
                                                name=cast_5_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_5_node)

            mul_3_node_name = poly_approx_namescope + "/mul_3"
            mul_3_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + '_output', cast_5_node_name + '_output'],
                [mul_3_node_name + '_output'],
                name=mul_3_node_name)
            new_nodes.append(mul_3_node)

            alpha_1_node_name = poly_approx_namescope + "/alpha_1"
            alpha_1_node = onnx.helper.make_node(
                'Constant', [], [alpha_1_node_name + '_output'],
                name=alpha_1_node_name,
                value=onnx.helper.make_tensor(alpha_1_node_name + "_value",
                                              data_type, [], [1.]))
            new_nodes.append(alpha_1_node)

            cast_6_node_name = poly_approx_namescope + "/cast_6"
            cast_6_node = onnx.helper.make_node("Cast",
                                                [alpha_1_node_name + "_output"],
                                                [cast_6_node_name + "_output"],
                                                name=cast_6_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_6_node)

            add_1_node_name = poly_approx_namescope + "/add_1"
            add_1_node = onnx.helper.make_node(
                "Add",
                [mul_3_node_name + '_output', cast_6_node_name + '_output'],
                [add_1_node_name + '_output'],
                name=add_1_node_name)
            new_nodes.append(add_1_node)

            cast_7_node_name = poly_approx_namescope + "/cast_7"
            cast_7_node = onnx.helper.make_node("Cast",
                                                [add_1_node_name + "_output"],
                                                [cast_7_node_name + "_output"],
                                                name=cast_7_node_name,
                                                to=data_type)
            new_nodes.append(cast_7_node)

            cast_8_node_name = poly_approx_namescope + "/cast_8"
            cast_8_node = onnx.helper.make_node("Cast",
                                                [cast_7_node_name + "_output"],
                                                [cast_8_node_name + "_output"],
                                                name=cast_8_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_8_node)

            mul_4_node_name = poly_approx_namescope + "/mul_4"
            mul_4_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + '_output', cast_8_node_name + '_output'],
                [mul_4_node_name + '_output'],
                name=mul_4_node_name)
            new_nodes.append(mul_4_node)

            alpha_0_node_name = poly_approx_namescope + "/alpha_0"
            alpha_0_node = onnx.helper.make_node(
                'Constant', [], [alpha_0_node_name + '_output'],
                name=alpha_0_node_name,
                value=onnx.helper.make_tensor(alpha_0_node_name + "_value",
                                              data_type, [], [1.]))
            new_nodes.append(alpha_0_node)

            cast_9_node_name = poly_approx_namescope + "/cast_9"
            cast_9_node = onnx.helper.make_node("Cast",
                                                [alpha_0_node_name + "_output"],
                                                [cast_9_node_name + "_output"],
                                                name=cast_9_node_name,
                                                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_9_node)

            add_2_node_name = poly_approx_namescope + "/add_2"
            add_2_node = onnx.helper.make_node(
                "Add",
                [mul_4_node_name + '_output', cast_9_node_name + '_output'],
                [add_2_node_name + '_output'],
                name=add_2_node_name)
            new_nodes.append(add_2_node)

            cast_10_node_name = poly_approx_namescope + "/cast_10"
            cast_10_node = onnx.helper.make_node(
                "Cast", [add_2_node_name + "_output"],
                [cast_10_node_name + "_output"],
                name=cast_10_node_name,
                to=data_type)
            new_nodes.append(cast_10_node)

            # exp poly - power
            pow_namescope = exp_poly_namescope + "/pow"

            pow_x_node_name = pow_namescope + "/pow/x"
            pow_x_node = onnx.helper.make_node('Constant', [],
                                               [pow_x_node_name + '_output'],
                                               name=pow_x_node_name,
                                               value=onnx.helper.make_tensor(
                                                   pow_x_node_name + "_value",
                                                   data_type, [], [2.0]))
            new_nodes.append(pow_x_node)

            pow_node_name = pow_namescope + "/pow"
            pow_node = onnx.helper.make_node(
                "Pow",
                [pow_x_node_name + '_output', round_0_node_name + '_output'],
                [pow_node_name + '_output'],
                name=pow_node_name)
            new_nodes.append(pow_node)

            # exp poly
            exp_x_node_name = exp_poly_namescope + "/exp_x"
            exp_x_node = onnx.helper.make_node(
                "Mul",
                [pow_node_name + '_output', cast_10_node_name + '_output'],
                [exp_x_node_name + '_output'],
                name=exp_x_node_name)
            new_nodes.append(exp_x_node)

            return exp_x_node_name

        def _exp_sum(exp_x_node_name):
            nonlocal new_nodes

            exp_sum_namescope = softmax_name + "/exp_sum"

            sum_node_name = exp_sum_namescope + "/sum"
            if self._opset_version <= 12:
                sum_node = onnx.helper.make_node("ReduceSum",
                                                 [exp_x_node_name + '_output'],
                                                 [sum_node_name + '_output'],
                                                 name=sum_node_name,
                                                 axes=[reduce_axes],
                                                 keepdims=1)
                new_nodes.append(sum_node)
            else:
                sum_axis_node_name = exp_sum_namescope + "/sum/reduction_indices"
                sum_axis_node = onnx.helper.make_node(
                    'Constant', [], [sum_axis_node_name + '_output'],
                    name=sum_axis_node_name,
                    value=onnx.helper.make_tensor(sum_axis_node_name + "_value",
                                                  onnx_proto.TensorProto.INT64,
                                                  [1], [reduce_axes]))
                new_nodes.append(sum_axis_node)

                sum_node = onnx.helper.make_node("ReduceSum", [
                    exp_x_node_name + '_output', sum_axis_node_name + '_output'
                ], [sum_node_name + '_output'],
                                                 name=sum_node_name,
                                                 keepdims=1)
                new_nodes.append(sum_node)

            cast_sum_out_16_name = exp_sum_namescope + "/cast_reduce_sum_out_16"
            cast_sum_out_16 = onnx.helper.make_node(
                "Cast", [sum_node_name + "_output"],
                [cast_sum_out_16_name + "_output"],
                name=cast_sum_out_16_name,
                to=data_type)
            new_nodes.append(cast_sum_out_16)

            return cast_sum_out_16_name

        def _reciprocal_approximation(exp_x_node_name, cast_sum_out_16_name):
            nonlocal new_nodes

            reciprocal_namescope = softmax_name + "/reciprocal"

            to_int_node_name = reciprocal_namescope + "/to_int"
            to_int_node = onnx.helper.make_node(
                "Bitcast", [cast_sum_out_16_name + "_output"],
                [to_int_node_name + "_output"],
                name=to_int_node_name,
                type=onnx_proto.TensorProto.INT16)
            new_nodes.append(to_int_node)

            complement_node_name = reciprocal_namescope + "/complement"
            complement_node = onnx.helper.make_node(
                'Constant', [], [complement_node_name + '_output'],
                name=complement_node_name,
                value=onnx.helper.make_tensor(complement_node_name + "_value",
                                              onnx_proto.TensorProto.INT16, [],
                                              [0x7eb5]))
            new_nodes.append(complement_node)

            sub_2_node_name = reciprocal_namescope + "/sub_2"
            sub_2_node = onnx.helper.make_node("Sub", [
                complement_node_name + '_output', to_int_node_name + '_output'
            ], [sub_2_node_name + '_output'],
                                               name=sub_2_node_name)
            new_nodes.append(sub_2_node)

            y0_node_name = reciprocal_namescope + "/y0"
            y0_node = onnx.helper.make_node("Bitcast",
                                            [sub_2_node_name + "_output"],
                                            [y0_node_name + "_output"],
                                            name=y0_node_name,
                                            type=data_type)
            new_nodes.append(y0_node)

            newton_k1_name = reciprocal_namescope + "/mul_6/k1"
            newton_k1 = onnx.helper.make_node('Constant', [],
                                              [newton_k1_name + '_output'],
                                              name=newton_k1_name,
                                              value=onnx.helper.make_tensor(
                                                  newton_k1_name + "_value",
                                                  data_type, [], [1.9395974]))
            new_nodes.append(newton_k1)

            mul_6_node_name = reciprocal_namescope + "/mul_6"
            mul_6_node = onnx.helper.make_node(
                "Mul", [y0_node_name + '_output', newton_k1_name + '_output'],
                [mul_6_node_name + '_output'],
                name=mul_6_node_name)
            new_nodes.append(mul_6_node)

            cast_11_node_name = reciprocal_namescope + "/cast_11"
            cast_11_node = onnx.helper.make_node(
                "Cast", [cast_sum_out_16_name + "_output"],
                [cast_11_node_name + "_output"],
                name=cast_11_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_11_node)

            cast_12_node_name = reciprocal_namescope + "/cast_12"
            cast_12_node = onnx.helper.make_node(
                "Cast", [y0_node_name + "_output"],
                [cast_12_node_name + "_output"],
                name=cast_12_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_12_node)

            mul_7_node_name = reciprocal_namescope + "/mul_7"
            mul_7_node = onnx.helper.make_node(
                "Mul",
                [cast_12_node_name + '_output', cast_11_node_name + '_output'],
                [mul_7_node_name + '_output'],
                name=mul_7_node_name)
            new_nodes.append(mul_7_node)

            newton_k2_name = reciprocal_namescope + "/sub_3/k2"
            newton_k2 = onnx.helper.make_node('Constant', [],
                                              [newton_k2_name + '_output'],
                                              name=newton_k2_name,
                                              value=onnx.helper.make_tensor(
                                                  newton_k2_name + "_value",
                                                  data_type, [], [1.436142]))
            new_nodes.append(newton_k2)

            cast_13_node_name = reciprocal_namescope + "/cast_13"
            cast_13_node = onnx.helper.make_node(
                "Cast", [newton_k2_name + "_output"],
                [cast_13_node_name + "_output"],
                name=cast_13_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_13_node)

            sub_3_node_name = reciprocal_namescope + "/sub_3"
            sub_3_node = onnx.helper.make_node(
                "Sub",
                [cast_13_node_name + '_output', mul_7_node_name + '_output'],
                [sub_3_node_name + '_output'],
                name=sub_3_node_name)
            new_nodes.append(sub_3_node)

            cast_14_node_name = reciprocal_namescope + "/cast_14"
            cast_14_node = onnx.helper.make_node(
                "Cast", [sub_3_node_name + "_output"],
                [cast_14_node_name + "_output"],
                name=cast_14_node_name,
                to=data_type)
            new_nodes.append(cast_14_node)

            y1_node_name = reciprocal_namescope + "/y1"
            y1_node = onnx.helper.make_node(
                "Mul",
                [mul_6_node_name + '_output', cast_14_node_name + '_output'],
                [y1_node_name + '_output'],
                name=y1_node_name)
            new_nodes.append(y1_node)

            cast_y1_node_name = reciprocal_namescope + "/cast_15"
            cast_y1_node = onnx.helper.make_node(
                "Cast", [y1_node_name + "_output"],
                [cast_y1_node_name + "_output"],
                name=cast_y1_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_y1_node)

            mul_9_node_name = reciprocal_namescope + "/mul_9"
            mul_9_node = onnx.helper.make_node(
                "Mul",
                [cast_y1_node_name + '_output', cast_11_node_name + '_output'],
                [mul_9_node_name + '_output'],
                name=mul_9_node_name)
            new_nodes.append(mul_9_node)

            newton_ones_name = reciprocal_namescope + "/add/ones"
            newton_ones = onnx.helper.make_node('Constant', [],
                                                [newton_ones_name + '_output'],
                                                name=newton_ones_name,
                                                value=onnx.helper.make_tensor(
                                                    newton_ones_name + "_value",
                                                    data_type, [], [1.0]))
            new_nodes.append(newton_ones)

            cast_16_node_name = reciprocal_namescope + "/cast_16"
            cast_16_node = onnx.helper.make_node(
                "Cast", [newton_ones_name + "_output"],
                [cast_16_node_name + "_output"],
                name=cast_16_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_16_node)

            sub_4_node_name = reciprocal_namescope + "/sub_4"
            sub_4_node = onnx.helper.make_node(
                "Sub",
                [cast_16_node_name + '_output', mul_9_node_name + '_output'],
                [sub_4_node_name + '_output'],
                name=sub_4_node_name)
            new_nodes.append(sub_4_node)

            cast_17_node_name = reciprocal_namescope + "/cast_17"
            cast_17_node = onnx.helper.make_node(
                "Cast", [sub_4_node_name + "_output"],
                [cast_17_node_name + "_output"],
                name=cast_17_node_name,
                to=data_type)
            new_nodes.append(cast_17_node)

            cast_18_node_name = reciprocal_namescope + "/cast_18"
            cast_18_node = onnx.helper.make_node(
                "Cast", [cast_17_node_name + "_output"],
                [cast_18_node_name + "_output"],
                name=cast_18_node_name,
                to=onnx_proto.TensorProto.FLOAT)
            new_nodes.append(cast_18_node)

            mul_10_node_name = reciprocal_namescope + "/mul_10"
            mul_10_node = onnx.helper.make_node(
                "Mul",
                [cast_18_node_name + '_output', cast_y1_node_name + '_output'],
                [mul_10_node_name + '_output'],
                name=mul_10_node_name)
            new_nodes.append(mul_10_node)

            add_4_node_name = reciprocal_namescope + "/add_4"
            add_4_node = onnx.helper.make_node(
                "Add",
                [mul_10_node_name + '_output', cast_y1_node_name + '_output'],
                [add_4_node_name + '_output'],
                name=add_4_node_name)
            new_nodes.append(add_4_node)

            cast_19_node_name = reciprocal_namescope + "/cast_19"
            cast_19_node = onnx.helper.make_node(
                "Cast", [add_4_node_name + "_output"],
                [cast_19_node_name + "_output"],
                name=cast_19_node_name,
                to=data_type)
            new_nodes.append(cast_19_node)

            y2_node_name = softmax_name + "/y2"
            y2_node = onnx.helper.make_node(
                "Mul",
                [exp_x_node_name + '_output', cast_19_node_name + '_output'],
                [y2_node_name + '_output'],
                name=y2_node_name)
            new_nodes.append(y2_node)

            return y2_node_name

        def _exp_div(exp_x_node_name, cast_sum_out_16_name):
            nonlocal new_nodes

            y2_node_name = softmax_name + "/div"
            y2_node = onnx.helper.make_node(
                "Div",
                [exp_x_node_name + '_output', cast_sum_out_16_name + '_output'],
                [y2_node_name + '_output'],
                name=y2_node_name)
            new_nodes.append(y2_node)

            return y2_node_name

        input_node_name = node.input[0]
        output_node_name = node.output[0]

        # there are 3 steps for the softmax approximation,
        # note that each step uses bfloat16 datatype as DPU does
        exp_x_node_name = _exp_poly_approximation(input_node_name)

        # ONNX has no Op just like 'tf.UnsortedSegmentSum' for parallel summary
        #cast_sum_out_16_name = _exp_sum_parallel(exp_x_node_name)
        cast_sum_out_16_name = _exp_sum(exp_x_node_name)

        # ONNX has no Op just like 'np.frexp' or 'tf.Bitcast' for reciprocal approximation
        #y2_node_name = _reciprocal_approximation(exp_x_node_name, cast_sum_out_16_name)
        y2_node_name = _exp_div(exp_x_node_name, cast_sum_out_16_name)

        # at the end, cast to float for output
        cast_y2_node_name = softmax_name + "/output_cast"
        cast_y2_node = onnx.helper.make_node("Cast", [y2_node_name + "_output"],
                                             [output_node_name],
                                             name=cast_y2_node_name,
                                             to=onnx_proto.TensorProto.FLOAT)
        new_nodes.append(cast_y2_node)

        return new_nodes
