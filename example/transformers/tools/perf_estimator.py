#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

#!/bin/env python3
import numpy as np
import logging

import torch 
from torch.fx.passes.shape_prop import ShapeProp

from typing import Tuple

import os 
import sys 


class OpsProfile:
    
    @classmethod
    def fxtrace(cls, model:torch.nn.Module, inputs:tuple) -> None :
        """ does not support trace through conditional 
        keep for documenation/legacy """
        gm: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
        g = gm.graph
        # does not work on on nn.Module structures
        ShapeProp(gm).propagate(input)
        #import pdb; pdb.set_trace()
        for node in gm.graph.nodes:
            print( f"NodeOP:{node.op},\tTarget:{node.target},\tNodeName:{node.name},\tNodeArgs:{node.args}")        

    @classmethod
    def profile(cls, model:torch.nn.Module, inputs:tuple) -> None :
        # https://github.com/pytorch/pytorch/blob/main/torch/profiler/profiler.py#L508
        with torch.profiler.profile( activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, with_flops= True ) as prof:
            with torch.profiler.record_function("model_inference"):
                outputs = model(*inputs)              
        profile_data = prof.key_averages(group_by_input_shape=True)#.table(sort_by="cpu_time_total", row_limit=-1)
        
        # reference: # https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L1019
        op_counts = {}
        for event_id in range(len(profile_data)):
            #print(event_id, profile_data[event_id]) 
            event = profile_data[event_id]
            #print(type(event)) #class 'torch.autograd.profiler_util.FunctionEventAvg'
            # reference: https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L656
            flops = event.flops
            op_name = event.key
            op_count = event.count 
            inp_shapes = event.input_shapes
            if flops != 0:
                #print(f"event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes:{inp_shapes}")
                if op_counts.get(op_name) == None:
                    op_counts[op_name] = flops
                else:
                    op_counts[op_name] += flops
        
        total_ops = 0
        for op in op_counts.keys():
            print(f"{op}: {op_counts[op]}")
            total_ops += op_counts[op]
        print(f"Total GOPs: {total_ops*1e-9}")
        return total_ops
