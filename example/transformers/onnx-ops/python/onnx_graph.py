#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

class GraphNode:
    def __init__(self, name, node = None):
        self.name = name;
        self.op = node
        self.inputs = []
        self.outputs = []

    def addInput(self, node):
        self.inputs.append(node)

    def addOutput(self, node):
        self.outputs.append(node)

class ONNXGraph:
    def __init__(self, model):
        self.nodev = {}
        self.model = model
        ##Exclude list contains initializer items + tensors which are driven by constant operator
        self.exclude_list = {}
        self.value_info_cache = {}
        self.initializer_info_cache = {}

        for vi in model.graph.value_info:
            self.value_info_cache[vi.name] = vi

        for ci in model.graph.initializer:
            self.exclude_list[ci.name] = ci
            self.initializer_info_cache[ci.name] = ci

        for node in model.graph.node:
            if node.op_type == "Constant":
                for out in node.output:
                    self.exclude_list[out] = node
                self.exclude_list[node.name] = node

        for node in model.graph.node:
            if self.is_excluded(node.name):
                continue

            graph_node = self.addNode(node.name, node)
            for inp in node.input:
                if self.is_excluded(inp):
                    continue
            
                inp_node = self.addNode(inp)
                inp_node.addOutput(graph_node)
                graph_node.addInput(inp_node)

            for outp in node.output:
                out_node = self.addNode(outp)
                out_node.addInput(graph_node)
                graph_node.addOutput(out_node)

    def is_excluded(self, name):
        if self.exclude_list.get(name) != None:
            return True
        return False
    
    def addNode(self, name, onnx_node = None):
        node = self.nodev.get(name)
        if node == None:
            node = GraphNode(name, onnx_node)
            self.nodev[name] = node
        return node
    
    def getPrimaryInputs(self):
        ret = []
        for n in self.nodev.values():
            if not n.inputs:
                ret.append(n)
        return ret
    
    def getValueInfo(self, name):
        return self.value_info_cache.get(name)
    
    def getInitializerInfo(self, name):
        return self.initializer_info_cache.get(name)
    
    def getPredOp(self, name):
        node = self.nodev.get(name)
        ret = {}
        if(node != None):
            for inp in node.inputs:
                if inp.op != None:
                    ret[inp.name] = inp
                else:
                    ret.update(self.getPredOp(inp.name))
        return ret
    
    def getSuccOp(self, name):
        node = self.nodev.get(name)
        ret = {}
        if(node != None):
            for outp in node.outputs:
                if outp.op != None:
                    ret[outp.name] = outp
                else:
                    ret.update(self.getSuccOp(outp.name))
        return ret
    
    def showOpPred_Succ(self, name):
        print(f"Operator - {name}\n")
        pv = self.getPredOp(name)
        nv = self.getSuccOp(name)
        print("Inputs")
        for p in pv.values():
            print(p.op)

        print("Outputs")
        for p in nv.values():
            print(p.op)

    def showTensorInfo(self, name):
        print("Tensor Info")
        print(self.value_info_cache.get(name))