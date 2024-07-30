#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import numpy as np

import logging
from enum import Enum

import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto

from onnxruntime.quantization.qdq_quantizer import QDQQuantizer, QDQQuantTensorType, QDQTensorQuantInfo
from onnxruntime.quantization.quant_utils import (
    QuantType,
    QuantFormat,
    TENSOR_NAME_QUANT_SUFFIX,
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
)

from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer, tensor_proto_to_array
from .quant_utils import (
    VitisQuantFormat,
    __producer__,
    __version__,
    FIX_OP_NAME,
    # add BFP
    BFPFIX_OP_NAME,
    VAI_DOMAIN,
    COP_DOMAIN,
    COP_QUANT_OP_NAME,
    COP_DEQUANT_OP_NAME,
    ONNX_WBIT_QTYPES_LIST,
    get_annotate_tensors,
    get_qdq_to_remove,
    remove_nodes,
    modified_annotate_input,
    PowerOfTwoMethod,
    is_approximately_equal,
)
from .registry import CreateQDQQuantizer
from .refine import adjust_quantize_info
from .simulate_dpu import simulate_transforms
from .onnx_quantizer import VitisONNXQuantizer

logger = logging.getLogger(__name__)


class VitisQDQQuantizer(VitisONNXQuantizer):

    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        extra_options=None,
    ):
        self.calibrate_method = calibrate_method
        VitisONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.bias_to_quantize = []

        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = (
            [] if "OpTypesToExcludeOutputQuantization" not in extra_options else
            extra_options["OpTypesToExcludeOutputQuantization"])

        # Some scenarios do not need the bias quantized. For example, in the case of Quantization Aware Training,
        # quantizing the bias is not needed. This is because in QAT, all model parameters are expected to be in
        # floating point format. To that end, we can use the FakeQuant operator for weights and activations that
        # can always have QDQ pairs (by using AddQDQPairToWeight). But for biases in a quantized model, we can't use
        # FakeQuant because it only ever appears before a DQ (since it is quantized as int32).
        self.quantize_bias = True if "QuantizeBias" not in extra_options else extra_options[
            "QuantizeBias"]

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (False if "AddQDQPairToWeight"
                                       not in extra_options else
                                       extra_options["AddQDQPairToWeight"])

        # Some scenarios do not need the bias quantized. For example, in the case of Quantization Aware Training,
        # quantizing the bias is not needed. This is because in QAT, all model parameters are expected to be in
        # floating point format. To that end, we can use the FakeQuant operator for weights and activations that
        # can always have QDQ pairs (by using AddQDQPairToWeight). But for biases in a quantized model, we can't use
        # FakeQuant because it only ever appears before a DQ (since it is quantized as int32).
        self.quantize_bias = True if "QuantizeBias" not in extra_options else extra_options[
            "QuantizeBias"]

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (False
                                   if "DedicatedQDQPair" not in extra_options
                                   else extra_options["DedicatedQDQPair"])
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {} if "QDQOpTypePerChannelSupportToAxis" not in extra_options else
            extra_options["QDQOpTypePerChannelSupportToAxis"])

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField(
                    "tensor_type"
            ) and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                return True
        else:
            logger.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected."
                .format(tensor_name))

        return False

    def __quantize_tensor(self,
                          tensor_name,
                          quant_sharing_param=None,
                          tensor_type=QDQQuantTensorType.ACTIVATION):
        """
        Quantize tensors. If quant_param_tensor is not None, tensor with name tensor_name will be quantized with same
        quantization parameters as tensor quant_param_tensor

        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
            tensor_type: QDQQuantTensorType default ACTIVATION
        """
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_param:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type,
                    quant_para_provider=quant_sharing_param)
            elif tensor_name not in self.tensors_to_quantize:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type)

    def quantize_activation_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Activation Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param,
                                      QDQQuantTensorType.ACTIVATION)

    def quantize_weight_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Weight Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param,
                                      QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis)
        else:
            logger.warning(
                "only support per-channel quantization on weight. Tensor: {} is not quantized."
                .format(tensor_name))

    def quantize_bias_tensor(self,
                             bias_name,
                             input_name,
                             weight_name,
                             beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append(
                    (bias_name, input_name, weight_name, beta))
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(
                            node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias:
            self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (output_name in self.quantization_params.keys() and len(
                self.model.input_name_to_nodes()[upstream_output_name]) == 1 and
                not self.model.is_graph_output(upstream_output_name) and
                not self.model.is_graph_input(upstream_output_name)):
            self.model.replace_output_of_all_nodes(upstream_output_name,
                                                   output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_qdq_nodes(self,
                          q_input,
                          q_output,
                          quant_node_name,
                          dq_input,
                          dq_output,
                          dequant_node_name,
                          scale_name,
                          zp_name,
                          axis=None):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self,
                                      weight_proto,
                                      tensor_type,
                                      axis=None):
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError(
                    "Per-Channel support with QDQ format requires onnx opset version 13 or above."
                )
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                onnx_proto.TensorProto.INT8,
                axis,
                self.calibrate_method,
                keep_float_weight=self.add_qdq_pair_to_weight)
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT
                else self.activation_qType,
                self.calibrate_method,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name,
                                              weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (self.dedicated_qdq_pair and
                tensor_name in self.tensor_to_its_receiving_nodes and
                len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1):
            num_dedicated_qdq_pair = len(
                self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(
                    tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(
                    tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(
                    tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(
                    tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(
                    node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name,
                                           self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer,
                                                       tensor_info.tensor_type,
                                                       tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp)

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_qdq_pair_for_activation(tensor_name, scale_name,
                                                      zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy(
            ).items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[
                        tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name,
                                               self.model.initializer())
                    if initializer is not None:
                        raise ValueError(
                            "Quantization parameter shared mode is not supported for weight yet"
                        )
                    self._add_qdq_pair_for_activation(
                        tensor_name, quantized_value.scale_name,
                        quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(
                find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [
                quant_value.q_name, quant_value.scale_name, quant_value.zp_name
            ]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                )
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize


class VitisQDQDPUQuantizer(VitisQDQQuantizer):

    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        extra_options=None,
    ):
        self.calibrate_method = calibrate_method
        VitisQDQQuantizer.__init__(
            self,
            model,
            False,
            False,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
        self.tensors_to_quantize = {}

        if per_channel:
            logger.error(
                "Only per-tensor quantization is supported when enable_dpu=True, `per_channel` must be set to False."
            )

        if reduce_range:
            logger.error(
                "reduce_range is not supported when enable_dpu=True, `reduce_range` must be set to False."
            )

        if weight_qType != QuantType.QInt8:
            logger.error(
                "Only QuantType.QInt8 weight_type is supported when enable_dpu=True."
            )

        # If using nable_dpu, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options[
                "WeightSymmetric"]:
            logger.error(
                "When enable_dpu=True, WeightSymmetric must be set to true.")
        self.is_weight_symmetric = True

        # If using enable_dpu, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options[
                "ActivationSymmetric"]:
            logger.error(
                "When enable_dpu=True, ActivationSymmetric must be set to true."
            )
        self.is_activation_symmetric = True

    def quantize_model(self):
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(
                            node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model,
                                          dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info(
                "Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu)."
            )
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )
        if "SimulateDPU" not in self.extra_options or self.extra_options[
                "SimulateDPU"] == True:
            self._simulate_transforms()

        if "IPULimitationCheck" not in self.extra_options or self.extra_options[
                "IPULimitationCheck"] == True:
            self._quantize_refine()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def _add_qdq_pair_for_initializer(self,
                                      weight_proto,
                                      tensor_type,
                                      axis=None):
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.quantize_initializer(
            weight_proto,
            self.weight_qType,
            self.calibrate_method,
            keep_float_weight=self.add_qdq_pair_to_weight,
        )
        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name,
                                              weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )

            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self,
                             bias_name,
                             input_name,
                             weight_name,
                             beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self):
        adjust_shift_cut = True
        if "AdjustShiftCut" in self.extra_options:
            adjust_shift_cut = self.extra_options["AdjustShiftCut"]
        adjust_shift_bias = True
        if "AdjustShiftBias" in self.extra_options:
            adjust_shift_bias = self.extra_options["AdjustShiftBias"]
        adjust_shift_read = True
        if "AdjustShiftRead" in self.extra_options:
            adjust_shift_read = self.extra_options["AdjustShiftRead"]
        adjust_shift_write = True
        if "AdjustShiftWrite" in self.extra_options:
            adjust_shift_write = self.extra_options["AdjustShiftWrite"]
        adjust_hard_sigmoid = True
        if "AdjustHardSigmoid" in self.extra_options:
            adjust_hard_sigmoid = self.extra_options["AdjustHardSigmoid"]
        adjust_shift_swish = True
        if "AdjustShiftSwish" in self.extra_options:
            adjust_shift_swish = self.extra_options["AdjustShiftSwish"]
        align_concat = True
        if "AlignConcat" in self.extra_options:
            align_concat = self.extra_options["AlignConcat"]
        align_pool = True
        if "AlignPool" in self.extra_options:
            align_pool = self.extra_options["AlignPool"]

        self.model = adjust_quantize_info(
            self.model,
            adjust_shift_cut=adjust_shift_cut,
            adjust_shift_bias=adjust_shift_bias,
            adjust_shift_read=adjust_shift_read,
            adjust_shift_write=adjust_shift_write,
            adjust_hard_sigmoid=adjust_hard_sigmoid,
            adjust_shift_swish=adjust_shift_swish,
            align_concat=align_concat,
            align_pool=align_pool,
        )

    def _simulate_transforms(self):
        convert_leaky_relu_to_dpu_version = True
        if "ConvertLeakyReluToDPUVersion" in self.extra_options:
            convert_leaky_relu_to_dpu_version = self.extra_options[
                "ConvertLeakyReluToDPUVersion"]
        convert_sigmoid_to_hard_sigmoid = True
        if "ConvertSigmoidToHardSigmoid" in self.extra_options:
            convert_sigmoid_to_hard_sigmoid = self.extra_options[
                "ConvertSigmoidToHardSigmoid"]
        convert_hard_sigmoid_to_dpu_version = True
        if "ConvertHardSigmoidToDPUVersion" in self.extra_options:
            convert_hard_sigmoid_to_dpu_version = self.extra_options[
                "ConvertHardSigmoidToDPUVersion"]
        convert_avg_pool_to_dpu_version = True
        if "ConvertAvgPoolToDPUVersion" in self.extra_options:
            convert_avg_pool_to_dpu_version = self.extra_options[
                "ConvertAvgPoolToDPUVersion"]
        convert_reduce_mean_to_dpu_version = True
        if "ConvertReduceMeanToDPUVersion" in self.extra_options:
            convert_reduce_mean_to_dpu_version = self.extra_options[
                "ConvertReduceMeanToDPUVersion"]
        convert_softmax_to_dpu_version = False
        if "ConvertSoftmaxToDPUVersion" in self.extra_options:
            convert_softmax_to_dpu_version = self.extra_options[
                "ConvertSoftmaxToDPUVersion"]

        self.model.model, self.nodes_to_exclude = simulate_transforms(
            self.model.model,
            self.should_quantize_node,
            self.nodes_to_quantize,
            self.nodes_to_exclude,
            convert_leaky_relu_to_dpu_version=convert_leaky_relu_to_dpu_version,
            convert_sigmoid_to_hard_sigmoid=convert_sigmoid_to_hard_sigmoid,
            convert_hard_sigmoid_to_dpu_version=
            convert_hard_sigmoid_to_dpu_version,
            convert_avg_pool_to_dpu_version=convert_avg_pool_to_dpu_version,
            convert_reduce_mean_to_dpu_version=
            convert_reduce_mean_to_dpu_version,
            convert_softmax_to_dpu_version=convert_softmax_to_dpu_version,
        )


class VitisExtendedQuantizer(VitisQDQQuantizer):

    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        quant_format,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        extra_options=None,
    ):
        self.calibrate_method = calibrate_method
        VitisQDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
        self.tensors_to_quantize = {}

        self.quant_format = quant_format
        self.add_qdq_pair_to_weight = True

        if per_channel:
            logger.error(
                "Only per-tensor quantization is supported when using VitisQuantFormat, `per_channel` must be set to False."
            )

        if reduce_range:
            logger.error(
                "reduce_range is not supported when using VitisQuantFormat, `reduce_range` must be set to False."
            )

    def quantize_model(self):

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(
                            node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        self._quantize_refine()

        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        '''
        if (output_name in self.quantization_params.keys() and len(
                self.model.input_name_to_nodes()[upstream_output_name]) == 1 and
                not self.model.is_graph_output(upstream_output_name) and
                not self.model.is_graph_input(upstream_output_name)):
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        '''
        # TODO : Understand the principle here and fix the issue caused by QDQRemovableActivation.
        # As showed at onnxruntime/quantization/operators/activation.py, if activation uses asymmetric,
        # the QDQRemovableActivation remove nodes, which caused the graph broken.

        return False

    def _create_fn_nodes(self,
                         q_input,
                         dq_output,
                         dequant_node_name,
                         scale_name,
                         zp_name,
                         axis=None):
        """
        create fix_neuron node
        """
        fix_neuron_node = onnx.helper.make_node(
            FIX_OP_NAME,
            [q_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")
        fix_neuron_node.attribute.append(bit_width)

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))
        fix_neuron_node.attribute.append(pos_attr)

        self.model.add_nodes([fix_neuron_node])

    def _create_qdq_nodes(self,
                          q_input,
                          q_output,
                          quant_node_name,
                          dq_input,
                          dq_output,
                          dequant_node_name,
                          scale_name,
                          zp_name,
                          axis=None):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))

        qlinear_node.attribute.append(bit_width)
        qlinear_node.attribute.append(pos_attr)
        dequant_node.attribute.append(bit_width)
        dequant_node.attribute.append(pos_attr)
        self.model.add_nodes([qlinear_node, dequant_node])

    def _create_customqdq_nodes(self,
                                q_input,
                                q_output,
                                quant_node_name,
                                dq_input,
                                dq_output,
                                dequant_node_name,
                                scale_name,
                                zp_name,
                                axis=None):
        qlinear_node = onnx.helper.make_node(
            COP_QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=COP_DOMAIN,
        )
        dequant_node = onnx.helper.make_node(
            COP_DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=COP_DOMAIN,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_fn_pair_for_weight(self, weight_proto, axis=None):
        weight_name = weight_proto.name

        q_weight_name, zp_name, scale_name = self.quantize_initializer(
            weight_proto,
            self.weight_qType,
            self.calibrate_method,
            keep_float_weight=self.add_qdq_pair_to_weight,
        )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name,
                                              weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            if self.quant_format == VitisQuantFormat.FixNeuron:
                self._create_fn_nodes(
                    weight_name,
                    weight_dequant_output,
                    add_dequant_suffix(weight_name),
                    scale_name,
                    zp_name,
                    axis,
                )
            elif self.quant_format == VitisQuantFormat.QDQ:
                if self.weight_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                    self._create_customqdq_nodes(
                        weight_name,
                        weight_quant_output,
                        add_quant_suffix(weight_name),
                        weight_quant_output,
                        weight_dequant_output,
                        add_dequant_suffix(weight_name),
                        scale_name,
                        zp_name,
                        axis,
                    )
                else:
                    self._create_qdq_nodes(
                        weight_name,
                        weight_quant_output,
                        add_quant_suffix(weight_name),
                        weight_quant_output,
                        weight_dequant_output,
                        add_dequant_suffix(weight_name),
                        scale_name,
                        zp_name,
                        axis,
                    )
        else:
            if self.weight_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                dequant_node = onnx.helper.make_node(
                    COP_DEQUANT_OP_NAME,
                    [q_weight_name, scale_name, zp_name],
                    [weight_dequant_output],
                    add_dequant_suffix(weight_name),
                    axis=axis,
                    domain=COP_DOMAIN,
                )
                self.model.add_node(dequant_node)
            else:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    [q_weight_name, scale_name, zp_name],
                    [weight_dequant_output],
                    add_dequant_suffix(weight_name),
                    axis=axis,
                    domain=VAI_DOMAIN,
                )
                bit_width = onnx.helper.make_attribute("bit_width", "8")
                dequant_node.attribute.append(bit_width)
                self.model.add_node(dequant_node)

    def _add_fn_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (self.dedicated_qdq_pair and
                tensor_name in self.tensor_to_its_receiving_nodes and
                len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1):
            num_dedicated_qdq_pair = len(
                self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(
                    tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(
                    tensor_name) + postfix
                if self.quant_format == VitisQuantFormat.FixNeuron:
                    self._create_fn_nodes(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )
                elif self.quant_format == VitisQuantFormat.QDQ:
                    if self.activation_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                        self._create_customqdq_nodes(
                            weight_name,
                            weight_quant_output,
                            add_quant_suffix(weight_name),
                            weight_quant_output,
                            weight_dequant_output,
                            add_dequant_suffix(weight_name),
                            scale_name,
                            zp_name,
                            axis,
                        )
                    else:
                        self._create_qdq_nodes(
                            weight_name,
                            weight_quant_output,
                            add_quant_suffix(weight_name),
                            weight_quant_output,
                            weight_dequant_output,
                            add_dequant_suffix(weight_name),
                            scale_name,
                            zp_name,
                            axis,
                        )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(
                    node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            if self.quant_format == VitisQuantFormat.FixNeuron:
                self._create_fn_nodes(
                    q_input,
                    dq_output,
                    add_dequant_suffix(tensor_name),
                    scale_name,
                    zp_name,
                )
            elif self.quant_format == VitisQuantFormat.QDQ:
                if self.activation_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                    self._create_customqdq_nodes(
                        q_input,
                        add_quant_output_suffix(tensor_name),
                        add_quant_suffix(tensor_name),
                        add_quant_output_suffix(tensor_name),
                        dq_output,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )
                else:
                    self._create_qdq_nodes(
                        q_input,
                        add_quant_output_suffix(tensor_name),
                        add_quant_suffix(tensor_name),
                        add_quant_output_suffix(tensor_name),
                        dq_output,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name,
                                           self.model.initializer())
                if initializer:
                    self._add_fn_pair_for_weight(initializer, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp)

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_fn_pair_for_activation(tensor_name, scale_name,
                                                     zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy(
            ).items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[
                        tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name,
                                               self.model.initializer())
                    if initializer is not None:
                        raise ValueError(
                            "Quantization parameter shared mode is not supported for weight yet"
                        )
                    self._add_fn_pair_for_activation(tensor_name,
                                                     quantized_value.scale_name,
                                                     quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(
                find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [
                quant_value.q_name, quant_value.scale_name, quant_value.zp_name
            ]
            node_name = add_dequant_suffix(bias_name)

            if self.weight_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        COP_DEQUANT_OP_NAME,
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=COP_DOMAIN,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        COP_DEQUANT_OP_NAME,
                        inputs,
                        [bias_name],
                        node_name,
                        domain=COP_DOMAIN,
                    )
                self.model.add_node(dequant_node)
                continue

            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                    domain=VAI_DOMAIN,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    domain=VAI_DOMAIN,
                )
            bit_width = onnx.helper.make_attribute("bit_width", "8")
            dequant_node.attribute.append(bit_width)
            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self,
                             bias_name,
                             input_name,
                             weight_name,
                             beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT and self.quantize_bias:
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def _adjust_model_scale(self):
        for node in self.model.model.graph.node:
            if node.op_type == "DequantizeLinear" or node.op_type == "QuantizeLinear":
                pos = None
                for attr in node.attribute:
                    if attr.name == "pos":
                        pos = int(attr.s)
                if pos is None:
                    continue
                new_scale = float(np.power(2., -pos))
                for i in self.model.model.graph.initializer:
                    if i.name == node.input[1]:
                        if i.float_data[0] != new_scale:
                            i.float_data[0] = new_scale

    def _quantize_refine(self):
        if ((self.activation_qType in ONNX_WBIT_QTYPES_LIST) or
            (self.weight_qType in ONNX_WBIT_QTYPES_LIST)):
            logger.info(
                "Skiped refinement becuase the quant_type has wide bits")
            return
        elif self.use_qdq_vitis_custom_ops:
            logger.info("Skiped refinement becuase using custom qdq")
            return

        self.model = adjust_quantize_info(self.model,
                                          adjust_hard_sigmoid=True,
                                          adjust_shift_cut=True,
                                          adjust_shift_bias=True,
                                          adjust_shift_read=True,
                                          adjust_shift_write=True,
                                          align_concat=True,
                                          align_pool=True)
        if self.quant_format == VitisQuantFormat.QDQ:
            self._adjust_model_scale()


class VitisBFPQuantizer(VitisQDQQuantizer):

    def __init__(self,
                 model,
                 per_channel,
                 reduce_range,
                 mode,
                 static,
                 weight_qType,
                 activation_qType,
                 tensors_range,
                 nodes_to_quantize,
                 nodes_to_exclude,
                 op_types_to_quantize,
                 calibrate_method,
                 extra_options=None):
        super().__init__(model, per_channel, reduce_range, mode, static,
                         weight_qType, activation_qType, tensors_range,
                         nodes_to_quantize, nodes_to_exclude,
                         op_types_to_quantize, calibrate_method, extra_options)
        self._extra_options = extra_options

    def _create_fn_nodes(self,
                         q_input,
                         dq_output,
                         dequant_node_name,
                         axis=None):
        """
        create fix_neuron node
        """
        bfp_neuron_node = onnx.helper.make_node(
            BFPFIX_OP_NAME,
            [q_input],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain="com.vai.quantize",
        )

        for k, v in self._extra_options.items():
            bfp_neuron_node.attribute.append(onnx.helper.make_attribute(k, v))
        self.model.add_nodes([bfp_neuron_node])

    def _add_fn_pair_for_weight(self, weight_proto, axis=None):
        weight_name = weight_proto.name
        quantized_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        self.model.replace_input_of_all_nodes(weight_name,
                                              quantized_weight_name)
        self._create_fn_nodes(weight_name, quantized_weight_name,
                              add_dequant_suffix(weight_name), axis)

    def _add_fn_pair_for_activation(self, tensor_name):
        q_input = tensor_name
        dq_output = add_dequant_output_suffix(tensor_name)
        if self.model.is_graph_output(tensor_name):
            q_input = add_quant_input_suffix(tensor_name)
            dq_output = tensor_name
            self.model.replace_output_of_all_nodes(tensor_name, q_input)
        else:
            self.model.replace_input_of_all_nodes(tensor_name, dq_output)

        self._create_fn_nodes(q_input, dq_output,
                              add_dequant_suffix(tensor_name))

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name,
                                           self.model.initializer())
                if initializer:
                    self._add_fn_pair_for_weight(initializer, tensor_info.axis)
                else:
                    self._add_fn_pair_for_activation(tensor_name)

                del self.tensors_to_quantize[tensor_name]

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

        self._quantize_normal_tensors()
        self.remove_nodes()
        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model
