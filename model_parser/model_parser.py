from typing import Any, List, Set, Optional, Union
import os

import json
import networkx as nx 
from domino.graph_pass import GraphVisitor
from domino.graph_ir import Op, Graph, Tensor
from domino.base import AccTask
from domino.utils import ONNXConvertor
import torch.onnx
import numpy as np

from model_parser.model_definition import ModelShape, model_dict


def export_onnx_graph(model_type: str, model_shape: ModelShape, output_dir: str):
    assert model_type in model_dict.keys(), f"We now only support these models: {list(model_dict.keys())}."

    model = model_dict[model_type](model_shape)

    inputs_np = np.random.uniform(-1, 1, [4, 1024, model_shape.dim])
    inputs_torch = torch.tensor(inputs_np, dtype=torch.float16)
    if torch.cuda.is_available():
        inputs_torch = inputs_torch.cuda()
        model = model.cuda()
    with torch.inference_mode():
        output = model(inputs_torch)

    torch.onnx.export(
        model,
        inputs_torch,
        f"{output_dir}/{model_type}.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    del model
    del inputs_torch
    del output
    torch.cuda.empty_cache()
    return f"{output_dir}/{model_type}.onnx"


class GraphIRConverter(GraphVisitor):
    def __init__(self):
        self.op2index = {}
        self.g = nx.DiGraph()
        super(GraphIRConverter, self).__init__()
        
    def get_id(self, op: Op.NamedOp):
        if op not in self.op2index:
            self.op2index[op] = len(self.op2index)
        return self.op2index[op]

    def visit_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        if self.has_visited_op(op):
            return self.get_op_visited(op)
        id = self.get_id(op)
        self.g.add_node(id, op=op, task = AccTask(id), start = 0.0, end = 0.0, acc = (None, None))
        for name, input_tensor in op.inputs.items():
            if input_tensor in boundary_tensors:
                # subgraph inputs
                pass
            elif input_tensor.produce_op is not None:
                # compute op
                input_id = self.get_id(input_tensor.produce_op)      
                self.g.add_edge(input_id, id)
                visitor = self.get_visitor(input_tensor.produce_op)
                visitor(input_tensor.produce_op, boundary_tensors)
            else:
                # producer op
                pass
        return self.record_visited_op(op, None)
    
    def __call__(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True) -> Any:
        self.visit_graph(graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self.g 
    
    def postprocess(self):
        self.g = nx.transitive_closure(self.g)
        considered_ops = [Op.OpName.MatrixOp.Gemm, Op.OpName.MatrixOp.MatMul]
        self.g = self.g.subgraph([id for id in self.g.nodes if self.g.nodes[id]['op'].name in considered_ops])
        G = nx.transitive_reduction(self.g)
        G.add_nodes_from(self.g.nodes(data = True))
        self.g = G
        
        for id in self.g.nodes:
            node = self.g.nodes[id]
            task = node['task']
            task.name = f'T{id}'
            task.depend_tasks = [self.g.nodes[i]['task'] for i in self.g.pred[id]]
            task.params = node['op'].get_config()
            if node['op'].name == Op.OpName.MatrixOp.Gemm or node['op'].name == Op.OpName.MatrixOp.MatMul:
                task.task_kind = "Gemm"
            else:
                raise RuntimeError()
        return self.g
    

def visualize(graph, name, output_dir):
    s = 'digraph G{\n'
    node2name = {x:i+1 for i,x in enumerate(graph.nodes)}
    for u,v in graph.edges:
        s += f'{node2name[u]} -> {node2name[v]}\n'
    s += '}\n'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    with open(f'{output_dir}/{name}.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng {output_dir}/{name}.gv -o {output_dir}/{name}.png'
    os.system(cmd)


def parse_graph(model_type: str, model_shape: ModelShape, parser_output_dir: str):
    onnx_path = export_onnx_graph(model_type, model_shape, parser_output_dir)
    irConverter = GraphIRConverter()
    convertor = ONNXConvertor(onnx_path, inference=True)
    graph  = convertor.parse()
    irConverter(graph)
    graph = irConverter.postprocess()
    visualize(graph, model_type, parser_output_dir)

    if model_type == 'opt':
        operator_graph = {
            "node": {
                "q": 6, "k": 7, "v": 8, "qk": 5, "sv": 4, "o": 3, "f1": 2, "f2": 1
            },
            "edge": [
                (6, 5), (7, 5), (5, 4), (8, 4), (4, 3), (3, 2), (2, 1)
            ]
        }
        with open(f"{parser_output_dir}/operator_graph.json", "w") as f:
            json.dump(operator_graph, f, indent=4)
    elif model_type == 'llama':
        operator_graph = {
            "node": {
                "q": 6, "k": 7, "v": 8, "qk": 5, "sv": 4, "o": 3, "f1": 2, "f3": 9, "f2": 1
            },
            "edge": [
                (6, 5), (7, 5), (5, 4), (8, 4), (4, 3), (3, 2), (3, 9), (2, 1), (9, 1)
            ]
        }
        with open(f"{parser_output_dir}/operator_graph.json", "w") as f:
            json.dump(operator_graph, f, indent=4)
    elif model_type == 'palm':
        operator_graph = {
            "node": {
                "q": 4, "k": 5, "v": 6, "qk": 3, "sv": 2, "o": 1, "f1": 8, "f3": 9, "f2": 7
            },
            "edge": [
                (4, 3), (5, 3), (3, 2), (6, 2), (2, 1), (8, 7), (9, 7)
            ]
        }
        with open(f"{parser_output_dir}/operator_graph.json", "w") as f:
            json.dump(operator_graph, f, indent=4)
