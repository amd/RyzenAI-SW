/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#pragma once
#include <stdio.h>
#include <nlohmann/json.hpp>
#include <adf/adf_types.hpp>

using json = nlohmann::json;

namespace aiectrl
{

inline void json_to_gmioconfig(const json* gmios, const int NUM_GMIO, GMIOConfig* gmio_cfg) {
  int i = 0;
  for (auto temp : gmios->items()) {
    auto gmio_json = temp.value();

    gmio_cfg[i].id = gmio_json["id"].get<int>();
    gmio_cfg[i].name = gmio_json["name"].get<std::string>();
    gmio_cfg[i].logicalName = gmio_json["logical_name"].get<std::string>();
    gmio_cfg[i].type = gmio_json["type"].get<gmio_config::gmio_type>();
    gmio_cfg[i].shimColumn = gmio_json["shim_column"].get<short>();
    gmio_cfg[i].channelNum = gmio_json["channel_number"].get<short>();
    gmio_cfg[i].streamId = gmio_json["stream_id"].get<short>();
    gmio_cfg[i].burstLength = gmio_json["burst_length_in_16byte"].get<short>();
    gmio_cfg[i].plKernelInstanceName = gmio_json["pl_kernel_instance_name"].get<std::string>();
    gmio_cfg[i].plParameterIndex = gmio_json["pl_parameter_index"].get<int>();
    gmio_cfg[i].plId = 0;
    gmio_cfg[i].plDriverSetAxiMMAddr = nullptr;
    i++;
  }
  return;
}

inline void json_to_dma_ch_config(const json* dma_channels, const int NUM_MEMTILE_CHANNELS, DMAChannelConfig* dma_ch_cfg) {
  int i = 0;
  for (auto temp : dma_channels->items()) {
    auto dma_ch_json = temp.value();

    dma_ch_cfg[i].portId = dma_ch_json["port_id"].get<int>();
    dma_ch_cfg[i].portName = dma_ch_json["port_name"].get<std::string>();
    dma_ch_cfg[i].parentId = dma_ch_json["parent_id"].get<int>();
    dma_ch_cfg[i].tileType = dma_ch_json["tile_type"].get<int>();
    dma_ch_cfg[i].column = dma_ch_json["column"].get<short>();
    dma_ch_cfg[i].row = dma_ch_json["row"].get<short>();
    dma_ch_cfg[i].S2MMOrMM2S = dma_ch_json["s2mm_or_mm2s"].get<int>();
    dma_ch_cfg[i].channel = dma_ch_json["channel"].get<short>();
    i++;
  }
  return;
}

};
