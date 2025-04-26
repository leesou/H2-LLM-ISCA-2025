from typing import Optional, Dict
from dataclasses import dataclass
import json

import networkx as nx


@dataclass
class ModelArgs:
    model_type: str = 'opt'

    layer_num: int = 32
    dim: int = 4096
    intermediate_size: Optional[int] = None
    n_heads: int = 32
    n_kv_heads: Optional[int] = None

    max_batch_size: int = 32
    input_seq_len: int = 1024
    max_gen_len: int = 1024
    max_seq_len: int = 2048

    @classmethod
    def init_from_config(cls, args) -> "ModelArgs":
        with open(args.model_shape, 'r') as f:
            model_shape_config = json.load(f)
        return cls(
            model_type=args.model_type,
            layer_num=model_shape_config["layer_num"],
            dim=model_shape_config["hidden_dim"],
            intermediate_size=model_shape_config["intermediate_dim"],
            n_heads=model_shape_config["q_head_num"],
            n_kv_heads=model_shape_config["kv_head_num"],
            max_batch_size=args.max_batch_size,
            input_seq_len=args.input_seq_len,
            max_gen_len=args.max_gen_len,
            max_seq_len=args.max_seq_len,
        )


class Model:
    precision_bit_dict = {
        "fp16" : 16,
        "int8" : 8,
        "int4" : 4
    }

    def __init__(
        self,
        args: ModelArgs,
        operator_graph: Dict,
        precision: str="fp16"
    ) -> None:
        self.model_args = args
        self.operator_graph = operator_graph
        self.precision = precision
        assert precision in ["fp16", "int8", "int4"], f"Wrong precision {precision}"

        self.model_type = args.model_type
        self.layer_num = args.layer_num
        self.dim = args.dim
        self.head_num = args.n_heads
        self.kv_head_num = args.n_kv_heads
        self.hidden_dim = args.intermediate_size
        self.head_dim = self.dim // self.head_num

        self.input_seq_len = args.input_seq_len
        self.batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.max_gen_len = args.max_gen_len
        self.capacity_estimation = self._estimate_capacity()

        self._extract_operator_graph()
        self.op_graph = self.generate_operator_graph(self.model_type)
        self.simplified_op_graph = self.op_graph.copy()
        self._check_legality()

    def _extract_operator_graph(self):
        self.node_name = {self.model_type: {}}
        for op_name, op_node_id in self.operator_graph["node"].items():
            self.node_name[self.model_type][op_node_id] = op_name
        
        self.edge_list = {self.model_type: []}
        for edge in self.operator_graph["edge"]:
            in_node, out_node = edge
            self.edge_list[self.model_type].append((in_node, out_node))
        
        self.qk_node_id = {self.model_type: self.operator_graph["node"]["qk"]}
        self.sv_node_id = {self.model_type: self.operator_graph["node"]["sv"]}
        self.kv_fc_node_id = {self.model_type: [
            self.operator_graph["node"]["k"],
            self.operator_graph["node"]["v"]
        ]}

    def get_qk_node_id(self):
        return self.qk_node_id[self.model_type]
    
    def get_sv_node_id(self):
        return self.sv_node_id[self.model_type]

    def get_kv_fc_node_id(self):
        return self.kv_fc_node_id[self.model_type]

    def generate_operator_graph(self, model_type: str):
        op_graph = nx.DiGraph()
        op_graph.add_nodes_from(list(self.node_name[model_type].keys()))
        op_graph.add_edges_from(self.edge_list[model_type])

        for op_id in op_graph.nodes:
            tmp_node = op_graph.nodes[op_id]
            tmp_node['name'] = self.node_name[self.model_type][op_id]
            if op_id == self.get_qk_node_id():
                tmp_node['N'] = self.input_seq_len
                tmp_node['K'] = self.head_dim
                if self.model_type == 'opt':
                    tmp_node['B'] = self.head_num * self.batch_size
                    tmp_node['M'] = self.input_seq_len
                elif self.model_type == 'llama' or self.model_type == 'palm':
                    tmp_node['B'] = self.kv_head_num * self.batch_size
                    tmp_node['M'] = self.input_seq_len * self.head_num // self.kv_head_num
            elif op_id == self.get_sv_node_id():
                tmp_node['N'] = self.head_dim
                tmp_node['K'] = self.input_seq_len
                if self.model_type == 'opt':
                    tmp_node['B'] = self.head_num * self.batch_size
                    tmp_node['M'] = self.input_seq_len
                elif self.model_type == 'llama' or self.model_type == 'palm':
                    tmp_node['B'] = self.kv_head_num * self.batch_size
                    tmp_node['M'] = self.input_seq_len * self.head_num // self.kv_head_num
            else:
                tmp_node['B'] = 1
                tmp_node['M'] = self.batch_size * self.input_seq_len
                if tmp_node['name'] in ['q', 'o']:
                    tmp_node['N'] = self.dim
                    tmp_node['K'] = self.dim
                elif tmp_node['name'] in ['k', 'v']:
                    tmp_node['N'] = self.head_dim * self.kv_head_num
                    tmp_node['K'] = self.dim
                elif tmp_node['name'] in ['f1', 'f3']:
                    tmp_node['N'] = self.hidden_dim
                    tmp_node['K'] = self.dim
                else:
                    tmp_node['N'] = self.dim
                    tmp_node['K'] = self.hidden_dim

        return op_graph
    
    def _check_legality(self):
        for node_id in self.simplified_op_graph.nodes:
            if self.node_name[self.model_type][node_id] == "q" or self.node_name[self.model_type][node_id] == "o":
                assert self.simplified_op_graph.nodes[node_id]['B'] == 1, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {1} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['N'] == self.dim, \
                    f"Wrong dimension size of N for node {self.node_name[self.model_type][node_id]}, expect {self.dim} but get {self.simplified_op_graph.nodes[node_id]['N']}"
                assert self.simplified_op_graph.nodes[node_id]['K'] == self.dim, \
                    f"Wrong dimension size of K for node {self.node_name[self.model_type][node_id]}, expect {self.dim} but get {self.simplified_op_graph.nodes[node_id]['K']}"
            elif self.node_name[self.model_type][node_id] == "k" or self.node_name[self.model_type][node_id] == "v":
                assert self.simplified_op_graph.nodes[node_id]['B'] == 1, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {1} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['N'] == self.head_dim * self.kv_head_num, \
                    f"Wrong dimension size of N for node {self.node_name[self.model_type][node_id]}, expect {self.head_dim * self.kv_head_num} but get {self.simplified_op_graph.nodes[node_id]['N']}"
                assert self.simplified_op_graph.nodes[node_id]['K'] == self.dim, \
                    f"Wrong dimension size of K for node {self.node_name[self.model_type][node_id]}, expect {self.dim} but get {self.simplified_op_graph.nodes[node_id]['K']}"
            elif self.node_name[self.model_type][node_id] == "f1" or self.node_name[self.model_type][node_id] == "f3":
                assert self.simplified_op_graph.nodes[node_id]['B'] == 1, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {1} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['N'] == self.hidden_dim, \
                    f"Wrong dimension size of N for node {self.node_name[self.model_type][node_id]}, expect {self.hidden_dim} but get {self.simplified_op_graph.nodes[node_id]['N']}"
                assert self.simplified_op_graph.nodes[node_id]['K'] == self.dim, \
                    f"Wrong dimension size of K for node {self.node_name[self.model_type][node_id]}, expect {self.dim} but get {self.simplified_op_graph.nodes[node_id]['K']}"
            elif self.node_name[self.model_type][node_id] == "f2":
                assert self.simplified_op_graph.nodes[node_id]['B'] == 1, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {1} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['N'] == self.dim, \
                    f"Wrong dimension size of N for node {self.node_name[self.model_type][node_id]}, expect {self.dim} but get {self.simplified_op_graph.nodes[node_id]['N']}"
                assert self.simplified_op_graph.nodes[node_id]['K'] == self.hidden_dim, \
                    f"Wrong dimension size of K for node {self.node_name[self.model_type][node_id]}, expect {self.hidden_dim} but get {self.simplified_op_graph.nodes[node_id]['K']}"
            elif self.node_name[self.model_type][node_id] == "qk":
                assert self.simplified_op_graph.nodes[node_id]['B'] == self.batch_size * self.kv_head_num, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {self.batch_size * self.kv_head_num} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['K'] == self.head_dim, \
                    f"Wrong dimension size of K for node {self.node_name[self.model_type][node_id]}, expect {self.head_dim} but get {self.simplified_op_graph.nodes[node_id]['K']}"
            elif self.node_name[self.model_type][node_id] == "sv":
                assert self.simplified_op_graph.nodes[node_id]['B'] == self.batch_size * self.kv_head_num, \
                    f"Wrong dimension size of B for node {self.node_name[self.model_type][node_id]}, expect {self.batch_size * self.kv_head_num} but get {self.simplified_op_graph.nodes[node_id]['B']}"
                assert self.simplified_op_graph.nodes[node_id]['N'] == self.head_dim, \
                    f"Wrong dimension size of N for node {self.node_name[self.model_type][node_id]}, expect {self.head_dim} but get {self.simplified_op_graph.nodes[node_id]['N']}"

    def _estimate_capacity(self):
        _MB = 2 ** 20
        per_layer_qo_weight_size = 2 * self.dim * self.dim * (self.precision_bit_dict[self.precision] / 8) / _MB
        per_layer_kv_weight_size = 2 * self.dim * self.head_dim * self.kv_head_num * (self.precision_bit_dict[self.precision] / 8) / _MB
        if self.model_type == "llama" or self.model_type == "palm":
            per_layer_ffn_weight_size = 3 * self.dim * self.hidden_dim * (self.precision_bit_dict[self.precision] / 8) / _MB
        else:
            per_layer_ffn_weight_size = 2 * self.dim * self.hidden_dim * (self.precision_bit_dict[self.precision] / 8) / _MB
        per_layer_kv_cache_size = 2 * self.max_seq_len * self.head_dim * self.kv_head_num * (self.precision_bit_dict[self.precision] / 8) / _MB
        per_layer_size = per_layer_qo_weight_size + per_layer_kv_weight_size + per_layer_ffn_weight_size + per_layer_kv_cache_size
        return per_layer_size * self.layer_num
