#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import numpy as np
import onnx

from onnx import numpy_helper
from onnx_graph import ONNXGraph

class GroupMatMulInteger:
    index = 0
    op_type = "MatMulInteger"
    def __init__(self, model):   
        self.onnx_graph = ONNXGraph(onnx.load(model))

    def gen_new_name(prefix):
        str = prefix + f"/GMatMulInteger_{GroupMatMulInteger.index}"
        GroupMatMulInteger.index += 1
        return str

    def replace(self, dynamicquant, matmulint_list):
        weight_concat = None
        weight_concat_name = ""
        split_dim = []
        outv = []
        for i in range(len(matmulint_list)):
            m = matmulint_list[i]
            weight = self.onnx_graph.initializer_info_cache[m.input[1]]
            weight_np = numpy_helper.to_array(weight)
            split_dim.append(weight_np.shape[1])
            if i == 0:
                weight_concat = weight_np    
            else:
                weight_concat = np.hstack((weight_concat,weight_np))
            weight_concat_name += weight.name
            weight_concat_name += "_"
            outv.append(m.output[0])

        weight_concat_tensor = numpy_helper.from_array(weight_concat,name=weight_concat_name)
        new_inp = [dynamicquant.output[0], weight_concat_tensor.name, dynamicquant.output[2], matmulint_list[0].input[3]]

        op_name = GroupMatMulInteger.gen_new_name(dynamicquant.name)
        op_out_name = op_name + "/Output"
        new_matmul = onnx.helper.make_node(name=op_name, op_type=GroupMatMulInteger.op_type, inputs=new_inp, outputs=[op_out_name])

        split_op_name = op_out_name + "_Split"
        split_dim_tensor = numpy_helper.from_array(np.array(split_dim).astype(np.int64), name=split_op_name+"/Dim")
        split_op = onnx.helper.make_node(name=split_op_name, op_type="Split", inputs=[op_out_name,split_dim_tensor.name], outputs=outv, axis=-1)

        for m in matmulint_list:
            self.onnx_graph.model.graph.node.remove(m)

        self.onnx_graph.model.graph.node.extend([new_matmul, split_op])
        self.onnx_graph.model.graph.initializer.extend([weight_concat_tensor, split_dim_tensor])

    def group(self):
        count = 0
        dynamicquantlinear = [node for node in self.onnx_graph.nodev.values() if (node.op != None) and (node.op.op_type == "DynamicQuantizeLinear")]
        for d in dynamicquantlinear:
            x = self.onnx_graph.nodev.get(d.op.output[0])
            zp = self.onnx_graph.nodev.get(d.op.output[2])
            if(x == None):
                continue
            if(zp == None):
                continue

            if len(x.outputs) != 3:
                continue

            if len(zp.outputs) != 3:
                continue

            mvec = []
            status = True
            for i in range(3):
                if x.outputs[i].name != zp.outputs[i].name:
                    status = False
                    break

                matmul = self.onnx_graph.nodev.get(x.outputs[i].name)
                if ((matmul != None) and (matmul.op != None) and (matmul.op.op_type == "MatMulInteger")):
                    mvec.append(matmul.op)
                else:
                    status = False
                    break
            
            if status == False:
                continue

            self.replace(d.op, mvec)
            count += 1
        
        if count > 0:
            print(f"{count} patterns found") 
            return True
        else:
            print("No patterns found")    
        return False
    
    def save(self, output):
        self.onnx_graph.model.opset_import.append(onnx.helper.make_opsetid("ryzenai.customop", 1))
        onnx.save(self.onnx_graph.model, output)