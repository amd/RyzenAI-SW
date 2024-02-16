#!/usr/bin/env python
# coding: utf-8
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import abc
import itertools
import uuid
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from tqdm import tqdm

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto

import onnxruntime
from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataCollector, CalibrationDataReader, MinMaxCalibrater
from onnxruntime.quantization.quant_utils import QuantType
from .quant_utils import PowerOfTwoMethod, get_tensor_type_from_qType, quantize_data_pof2s, is_ort_version_below_1_16, VitisQuantType

logger = logging.getLogger(__name__)


class PowOfTwoCalibrater(CalibraterBase):

    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        activation_type=QuantType.QInt8,
        method=PowerOfTwoMethod.MinMSE,
        symmetric=True,
        minmse_mode="All",
    ):

        super(PowOfTwoCalibrater,
              self).__init__(model, op_types_to_calibrate, augmented_model_path,
                             symmetric, use_external_data_format)
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(
            output.name for output in self.model.graph.output)
        self.collector = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None
        self.activation_type = activation_type
        self.use_external_data_format = use_external_data_format
        self.minmse_mode = minmse_mode

    def augment_graph(self):
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        if is_ort_version_below_1_16():
            from onnxruntime.quantization.quant_utils import clone_model_with_shape_infer
            model = clone_model_with_shape_infer(self.model)
        else:
            model = self.model

        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(
            model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                model.graph.output.append(value_infos[tensor])
        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(
                self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [
            self.infer_session.get_outputs()[i].name
            for i in range(len(self.intermediate_outputs[0]))
        ]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output))
            for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i])
                                 for i in merged_dict
                                 if i in self.tensors_to_calibrate)

        if not self.collector:
            self.collector = PowOfTwoCollector(
                activation_type=self.activation_type,
                method=self.method,
                symmetric=self.symmetric,
                minmse_mode=self.minmse_mode)
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self):
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError(
                "No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()


class PowOfTwoCollector(CalibrationDataCollector):
    """
    Collecting PowOfTwoCollector quantize for each tensor. Support MinMSE method.

    """

    def __init__(self,
                 activation_type=QuantType.QInt8,
                 method=PowerOfTwoMethod.MinMSE,
                 symmetric=True,
                 minmse_mode="All"):
        self.name_to_arr = {}
        self.method = method
        self.symmetric = symmetric
        self.minmse_mode = minmse_mode
        self.activation_qType = get_tensor_type_from_qType(activation_type)

    def collect(self, name_to_arr):

        self.name_to_arr = name_to_arr

        return

    def compute_collection_result(self):
        if not self.name_to_arr or len(self.name_to_arr) == 0:
            raise ValueError(
                "PowerOfTwoMethod has not been collected. Please run collect() first."
            )
        logger.info(
            "Finding optimal threshold for each tensor using {} algorithm ...".
            format(self.method))

        if self.method == PowerOfTwoMethod.MinMSE:
            return self.compute_minmse_range()
        else:
            raise ValueError("Only 'MinMSE' method are supported")

    def compute_minmse_range(self):
        thresholds_dict = {}
        if self.minmse_mode == "MostCommon" and self.symmetric:
            logger.info("Use the most common min mse from each batch")
            for tensor, data_arr in tqdm(self.name_to_arr.items(),
                                         desc="Computing range",
                                         unit="tensor"):
                scale_list = []
                scale2threshold = {}
                for d in data_arr:
                    rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse = quantize_data_pof2s(
                        d,
                        self.activation_qType,
                        self.symmetric,
                        method=self.method)
                    scale2threshold[scale_mse] = (rmin_mse, rmax_mse)
                    scale_list.append(scale_mse)
                # get most common pos
                u, indices = np.unique(scale_list, return_inverse=True)
                scale = u[np.argmax(np.bincount(indices))]
                thresholds_dict[tensor] = scale2threshold[scale]

        else:
            if self.minmse_mode == "MostCommon":
                logger.warning(
                    "Activation asymmetric does not support using 'most common' to calculate min mse"
                )
            if self.minmse_mode != "All":
                logger.warning(
                    "Currently MinMSEMode only supports 'All' and 'MostCommon'."
                    f"Does not support {self.minmse_mode}")
            logger.info("Use all calibration data to calculate min mse")
            for tensor, data_arr in tqdm(self.name_to_arr.items(),
                                         desc="Computing range",
                                         unit="tensor"):
                d = np.array(data_arr).flatten()
                rmin_mse, rmax_mse, _, _, _ = quantize_data_pof2s(
                    d,
                    self.activation_qType,
                    self.symmetric,
                    method=self.method)
                thresholds_dict[tensor] = (rmin_mse, rmax_mse)

        return thresholds_dict


def create_calibrator_power_of_two(
    model,
    op_types_to_calibrate: Optional[Sequence[str]] = None,
    augmented_model_path="augmented_model.onnx",
    activation_type=QuantType.QInt8,
    method=PowerOfTwoMethod.NonOverflow,
    use_external_data_format=False,
    execution_providers=['CPUExecutionProvider'],
    extra_options={},
):

    calibrator = None

    # default settings for min-max algorithm
    method = method
    symmetric = True if "symmetric" not in extra_options else extra_options[
        "symmetric"]
    moving_average = False if "moving_average" not in extra_options else extra_options[
        "moving_average"]
    averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options[
        "averaging_constant"]
    minmse_mode = 'All' if "minmse_mode" not in extra_options else extra_options[
        "minmse_mode"]
    calib_quant_type = [
        QuantType.QInt8, QuantType.QUInt8, VitisQuantType.QInt16,
        VitisQuantType.QUInt16
    ]
    activation_type = QuantType.QInt8 if activation_type not in calib_quant_type else activation_type
    if method == PowerOfTwoMethod.NonOverflow:
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
    elif method == PowerOfTwoMethod.MinMSE:
        calibrator = PowOfTwoCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            activation_type=activation_type,
            method=method,
            symmetric=symmetric,
            minmse_mode=minmse_mode,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator
