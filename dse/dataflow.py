from enum import IntEnum
from typing import List, Dict, Optional
import random
import multiprocessing

import networkx as nx

from .hardware import *
from .utils import *


OPEN_SPECPIM = multiprocessing.Value('b', False)


class ExecutionMapping(IntEnum):
    NPU = 0
    NMP = 1
    MIX = 2


class OperatorType(IntEnum):
    FC = 0
    QK = 1
    SV = 2


def generate_legal_execution_mappings(nmp_channels: List[int], normal_channels: List[Tuple]) -> List[ExecutionMapping]:
    assert len(nmp_channels) > 0 or len(normal_channels) > 0, "Each operator must have at least one normal/NMP channel"
    if len(nmp_channels) == 0:
        return [ExecutionMapping.NPU]
    elif len(normal_channels) == 0:
        return [ExecutionMapping.NMP]
    else:
        if OPEN_SPECPIM.value == True:
            assert False, "In SpecPIM dataflow, each operator only takes up one kind of channel (normal or nmp)"
        return [ExecutionMapping.MIX]


def find_closest_value(input_list, target_value):
    closest_value = min(input_list, key=lambda x: abs(x - target_value))
    return closest_value


def find_top_k_closest(input_list, target_value, k=10):
    k = min(k, len(input_list))
    sorted_list = sorted(input_list, key=lambda x: abs(x - target_value))
    return sorted_list[:k]


possible_nmp_comp_ratios = {
    1  : [i/16 for i in range(1, 16)],
    8  : [i/8 for i in range(1, 8)],
    32 : [i/32 for i in range(1, 32)],
    40 : [i/40 for i in range(1, 40)],
    56 : [i/56 for i in range(1, 56)]
}


class OperatorMapping:
    def __init__(
        self, operator_id, B, M, N, K,
        nmp_channels, normal_channels, nmp_storing_ratio,
        prefill_execution_mapping, decoding_execution_mapping
    ) -> None:
        self.operator_id: int = operator_id
        self.B: int = B
        self.M: int = M
        self.N: int = N
        self.K: int = K

        self.nmp_channels: List[int] = nmp_channels
        self.normal_channels: List[Tuple] = normal_channels
        self.nmp_storing_ratio: float = nmp_storing_ratio

        self.prefill_execution_mapping: ExecutionMapping = prefill_execution_mapping
        self.decoding_execution_mapping: ExecutionMapping = decoding_execution_mapping

        self._check_legality()
        
    def _check_legality(self):
        assert len(self.normal_channels) + len(self.nmp_channels) > 0, "No channel is assigned to this operator"
        assert len(self.normal_channels) <= 1, "Normal channel only has one element, since NPU need to access all normal channels."

        assert self.nmp_storing_ratio >= 0 and self.nmp_storing_ratio <= 1, f"illegal nmp_storing_ratio {self.nmp_storing_ratio}"
        legal_execution_mappings = generate_legal_execution_mappings(nmp_channels=self.nmp_channels, normal_channels=self.normal_channels)
        assert self.prefill_execution_mapping == ExecutionMapping.NPU, "illegal prefill execution mapping"
        assert self.decoding_execution_mapping in legal_execution_mappings, "illegal decoding execution mapping"

    def update_operator_shape(self, B=None, M=None, N=None, K=None):
        if B is not None:
            self.B = B
        if M is not None:
            self.M = M
        if N is not None:
            self.N = N
        if K is not None:
            self.K = K

    @classmethod
    def random_sample_operator_mapping(
        cls, 
        operator_id: int, 
        B: int, 
        M: int, 
        N: int, 
        K: int,
        kv_head_num: int,
        nmp_channels: List[int], 
        normal_channels: List[Tuple],
        per_nmp_channel_bw: float, per_normal_channel_bw: float,
        per_nmp_channel_comp: float, npu_comp: float
    ) -> "OperatorMapping":
        legal_execution_mappings = generate_legal_execution_mappings(nmp_channels, normal_channels)
        prefill_execution_mapping = ExecutionMapping.NPU
        decoding_execution_mapping = random.choice(legal_execution_mappings)

        if len(normal_channels) > 0 and len(nmp_channels) > 0:
            if prefill_execution_mapping == ExecutionMapping.MIX or decoding_execution_mapping == ExecutionMapping.MIX:
                nmp_npu_bw_ratio = (per_nmp_channel_bw * len(nmp_channels)) / ((per_nmp_channel_bw * len(nmp_channels)) + (2 * per_normal_channel_bw * len(normal_channels[0])))
                nmp_npu_comp_ratio = (per_nmp_channel_comp * len(nmp_channels)) / ((per_nmp_channel_comp * len(nmp_channels)) + (2 * per_normal_channel_bw * len(normal_channels[0]) * M))
                original_ratio = random.choices([nmp_npu_bw_ratio, nmp_npu_comp_ratio, -1], weights=[0.7, 0.1, 0.0], k=1)[0]
                if original_ratio != -1:
                    if B > 1:
                        if kv_head_num > 1:
                            nmp_storing_ratio = find_closest_value(possible_nmp_comp_ratios[kv_head_num], original_ratio)
                        else:
                            nmp_storing_ratio = find_closest_value([i/B for i in range(1, B)], original_ratio)
                    else:
                        nmp_storing_ratio = find_closest_value([i/N for i in range(1, N)], original_ratio)
                else:
                    if B > 1:
                        if kv_head_num > 1:
                            nmp_storing_ratio = random.choice(find_top_k_closest(possible_nmp_comp_ratios[kv_head_num], nmp_npu_bw_ratio))
                        else:
                            nmp_storing_ratio = random.choice(find_top_k_closest([i/B for i in range(1, B)], nmp_npu_bw_ratio))
                    else:
                        nmp_storing_ratio = random.choice(find_top_k_closest([i/N for i in range(1, N)], nmp_npu_bw_ratio))
            else:
                nmp_storing_ratio = len(nmp_channels) / (len(nmp_channels) + len(normal_channels[0]))
        elif len(normal_channels) == 0 and len(nmp_channels) > 0:
            nmp_storing_ratio = 1
        elif len(normal_channels) > 0 and len(nmp_channels) == 0:
            nmp_storing_ratio = 0
        else:
            assert False, "Each operator should have at least one normal/nmp channel"

        return cls(
            operator_id=operator_id,
            B=B, M=M, N=N, K=K,
            nmp_channels=nmp_channels,
            normal_channels=normal_channels,
            nmp_storing_ratio=nmp_storing_ratio,
            prefill_execution_mapping=prefill_execution_mapping,
            decoding_execution_mapping=decoding_execution_mapping
        )


class MPGChannelMapping:
    def __init__(
        self, mpg_operator_graph, nmp_channels, normal_channels,
        operator_stratifications, operator_mappings
    ) -> None:
        self.mpg_operator_graph: nx.DiGraph = mpg_operator_graph
        self.nmp_channels: List[int] = nmp_channels
        self.normal_channels: List[Tuple] = normal_channels

        self.operator_stratifications: List[nx.DiGraph] = operator_stratifications
        self.operator_mappings: Dict[int, OperatorMapping] = operator_mappings
        self._check_legality()

    def _check_legality(self):
        assert len(self.normal_channels) + len(self.nmp_channels) > 0, "No channel is assigned to this MPG"
        assert len(self.normal_channels) <= 1, "Normal channel only has one element, since NPU need to access all normal channels."
        
        stratification_node_groups = [list(stratification.nodes) for stratification in self.operator_stratifications]
        assert check_stratification_legality(stratification_node_groups, self.mpg_operator_graph), "Wrong stratification parition"

        assert set(self.operator_mappings.keys()) == set(self.mpg_operator_graph.nodes), "Operator mappings' node set mismatches with MPG graph's operator set."
        for stratification in self.operator_stratifications:
            nmp_channel_sets = []
            normal_channel_sets = []

            for node_id in list(stratification.nodes):
                nmp_channel_sets.append(set(self.operator_mappings[node_id].nmp_channels))
                normal_channel_sets.append(set(self.operator_mappings[node_id].normal_channels))

            nmp_channel_set_union = set.union(*nmp_channel_sets)
            normal_channel_set_union = set.union(*normal_channel_sets)
            assert nmp_channel_set_union == set(self.nmp_channels), f"Wrong assignment of nmp channel set {nmp_channel_sets} {self.nmp_channels}"
            assert normal_channel_set_union == set(self.normal_channels), f"Wrong assignment of normal channel set {normal_channel_sets} {self.normal_channels}"

            assert are_sets_disjoint(nmp_channel_sets), f"NMP channel assignment has overlap: {nmp_channel_sets}"

    @classmethod
    def random_sample_from_operator_stratifications(
        cls, mpg_operator_graph: nx.DiGraph, kv_head_num: int, nmp_channels: List[int], normal_channels: List[Tuple],
        per_nmp_channel_bw: float, per_normal_channel_bw: float,
        per_nmp_channel_comp: float, npu_comp: float
    ) -> "MPGChannelMapping":
        if len(normal_channels) == 0:
            assert len(nmp_channels) > 0, "This MPG is not allocated with any NMP/normal channel."
            max_node_num_per_stratification = len(nmp_channels)
        else:
            assert len(normal_channels) == 1, "We fuse all normal channels into one channel, since NPU will access all normal channels."
            max_node_num_per_stratification = -1
        stratification_node_ids, operator_stratifications = random_generate_stratifications(mpg_operator_graph, max_node_num_per_stratification)
        num_stratifications = len(operator_stratifications)
 
        operator_mappings = {}
        for i in range(num_stratifications):
            tmp_node_ids = stratification_node_ids[i]
            tmp_stratification = operator_stratifications[i]
            tmp_node_num = len(tmp_node_ids)
    
            if len(normal_channels) == 0:
                assert len(nmp_channels) > 0, "This MPG is not allocated with any NMP/normal channel."
                tmp_nmp_channel_split = split_list_random_lengths(nmp_channels, tmp_node_num, False)
            else:
                tmp_nmp_channel_split = split_list_random_lengths(nmp_channels, tmp_node_num, True)

            normal_channel_assigned = False
            for j in range(tmp_node_num):
                tmp_node_id = tmp_node_ids[j]

                if len(tmp_nmp_channel_split[j]) == 0:
                    assign_normal_channel = True
                    normal_channel_assigned = True
                else:
                    if not normal_channel_assigned and j == tmp_node_num - 1:
                        assign_normal_channel = True
                        normal_channel_assigned = True
                    else:
                        assign_normal_channel = bool(random.getrandbits(1))
                        if assign_normal_channel:
                            normal_channel_assigned = True

                tmp_operator_mapping = OperatorMapping.random_sample_operator_mapping(
                    operator_id=tmp_node_id,
                    B=tmp_stratification.nodes[tmp_node_id]['B'],
                    M=tmp_stratification.nodes[tmp_node_id]['M'],
                    N=tmp_stratification.nodes[tmp_node_id]['N'],
                    K=tmp_stratification.nodes[tmp_node_id]['K'],
                    kv_head_num=kv_head_num,
                    nmp_channels=tmp_nmp_channel_split[j],
                    normal_channels=normal_channels if assign_normal_channel else [],
                    per_nmp_channel_bw=per_nmp_channel_bw, per_normal_channel_bw=per_normal_channel_bw,
                    per_nmp_channel_comp=per_nmp_channel_comp, npu_comp=npu_comp
                )
                operator_mappings[tmp_node_id] = tmp_operator_mapping

        return cls(
            mpg_operator_graph=mpg_operator_graph, 
            nmp_channels=nmp_channels, 
            normal_channels=normal_channels,
            operator_stratifications=operator_stratifications, 
            operator_mappings=operator_mappings
        )

    @classmethod
    def random_sample_from_operator_mappings(
        cls, mpg_operator_graph: nx.DiGraph, kv_head_num: int, nmp_channels: List[int], normal_channels: List[Tuple], operator_stratifications: List[nx.DiGraph],
        per_nmp_channel_bw: float, per_normal_channel_bw: float,
        per_nmp_channel_comp: float, npu_comp: float
    ) -> "MPGChannelMapping":
        stratification_node_ids = [list(stratification.nodes) for stratification in operator_stratifications]
        num_stratifications = len(operator_stratifications)

        operator_mappings = {}
        for i in range(num_stratifications):
            tmp_node_ids = stratification_node_ids[i]
            tmp_stratification = operator_stratifications[i]
            tmp_node_num = len(tmp_node_ids)

            if len(normal_channels) == 0:
                tmp_nmp_channel_split = split_list_random_lengths(nmp_channels, tmp_node_num, False)
            else:
                tmp_nmp_channel_split = split_list_random_lengths(nmp_channels, tmp_node_num, True)
            
            normal_channel_assigned = False
            for j in range(tmp_node_num):
                tmp_node_id = tmp_node_ids[j]

                if len(tmp_nmp_channel_split[j]) == 0:
                    assign_normal_channel = True
                    normal_channel_assigned = True
                else:
                    if not normal_channel_assigned and j == tmp_node_num - 1:
                        assign_normal_channel = True
                        normal_channel_assigned = True
                    else:
                        assign_normal_channel = bool(random.getrandbits(1))
                        if assign_normal_channel:
                            normal_channel_assigned = True

                tmp_operator_mapping = OperatorMapping.random_sample_operator_mapping(
                    operator_id=tmp_node_id,
                    B=tmp_stratification.nodes[tmp_node_id]['B'],
                    M=tmp_stratification.nodes[tmp_node_id]['M'],
                    N=tmp_stratification.nodes[tmp_node_id]['N'],
                    K=tmp_stratification.nodes[tmp_node_id]['K'],
                    kv_head_num=kv_head_num,
                    nmp_channels=tmp_nmp_channel_split[j],
                    normal_channels=normal_channels if assign_normal_channel else [],
                    per_nmp_channel_bw=per_nmp_channel_bw, per_normal_channel_bw=per_normal_channel_bw,
                    per_nmp_channel_comp=per_nmp_channel_comp, npu_comp=npu_comp
                )
                operator_mappings[tmp_node_id] = tmp_operator_mapping

        return cls(
            mpg_operator_graph=mpg_operator_graph, 
            nmp_channels=nmp_channels, 
            normal_channels=normal_channels,
            operator_stratifications=operator_stratifications, 
            operator_mappings=operator_mappings
        )


class MemoryAccessGroup:
    def __init__(self, mag_operator_graph, nmp_channels, normal_channels, mpg_channel_mappings) -> None:
        self.mag_operator_graph: nx.DiGraph = mag_operator_graph
        self.nmp_channels: List[int] = nmp_channels
        self.normal_channels: List[Tuple] = normal_channels
        self.mpg_channel_mappings: List["MPGChannelMapping"] = mpg_channel_mappings
        self._check_legality()

    def _check_legality(self):
        assert len(self.normal_channels) + len(self.nmp_channels) > 0, "No channel is assigned to this operator"
        assert len(self.normal_channels) <= 1, "Normal channel only has one element, since NPU need to access all normal channels."

        for mpg_channel_mapping in self.mpg_channel_mappings:
            mpg_channel_mapping._check_legality()

        mpg_node_groups = [list(mpg_channel_mapping.mpg_operator_graph.nodes) for mpg_channel_mapping in self.mpg_channel_mappings]
        check_memory_partition_group_legality(mpg_node_groups, self.mag_operator_graph)

        nmp_channel_partition = [set(mpg_channel_mapping.nmp_channels) for mpg_channel_mapping in self.mpg_channel_mappings]
        normal_channel_partition = [set(mpg_channel_mapping.normal_channels) for mpg_channel_mapping in self.mpg_channel_mappings]
        if OPEN_SPECPIM.value == False:
            nmp_channel_partition_union = set.union(*nmp_channel_partition)
            assert nmp_channel_partition_union == set(self.nmp_channels), f"Wrong assignment of nmp channels {nmp_channel_partition} {self.nmp_channels}."
            normal_channel_partition_union = set.union(*normal_channel_partition)
            assert normal_channel_partition_union == set(self.normal_channels), f"Wrong assignment of normal channels {normal_channel_partition} {self.normal_channels}"
        assert are_sets_disjoint(nmp_channel_partition), f"NMP channel assignment has overlap {nmp_channel_partition}"
        assert are_sets_disjoint(normal_channel_partition), f"Normal channel assignment has overlap {normal_channel_partition}"

    @classmethod
    def random_sample_from_mpg_channel_mapping(
        cls, mag_operator_graph: nx.DiGraph, kv_head_num: int, nmp_channels: List[int], normal_channels: List[Tuple],
        per_nmp_channel_bw: float, per_normal_channel_bw: float,
        per_nmp_channel_comp: float, npu_comp: float
    ) -> Optional[List["MemoryAccessGroup"]]:
        mpg_node_lists, mpg_operator_graphs = generate_memory_partition_groups(mag_operator_graph)
        num_mpgs = len(mpg_operator_graphs)

        mpg_groups = []
        tmp_group_id = 0
        max_mpg_num_per_group = len(nmp_channels) + len(normal_channels)
        keep_original_partition = random.randint(0, 1)
        if num_mpgs <= max_mpg_num_per_group and keep_original_partition == 1:
            mpg_groups.append(mpg_operator_graphs)
        else:
            while tmp_group_id < num_mpgs:
                tmp_mpg_num = random.randint(1, max_mpg_num_per_group)
                mpg_groups.append(mpg_operator_graphs[tmp_group_id:tmp_group_id+tmp_mpg_num])
                tmp_group_id += tmp_mpg_num
        
        mag_list = []
        for mpg_group in mpg_groups:
            if OPEN_SPECPIM.value == False:
                mpg_channel_partition = split_list_random_lengths(nmp_channels + normal_channels, len(mpg_group), False)
            else:
                if len(mpg_group) == 1:
                    mpg_channel_partition = random.choice([[random.sample(nmp_channels, random.randint(1, len(nmp_channels)))], [normal_channels]])
                else:
                    mpg_channel_partition = split_list_random_lengths(nmp_channels, len(mpg_group)-1, False) + [normal_channels]
                    random.shuffle(mpg_channel_partition)
            assert len(mpg_channel_partition) == len(mpg_group)
            mpg_channel_mappings = []
            for i in range(len(mpg_group)):
                tmp_nmp_channels = []
                tmp_normal_channels = []
                for channel_id in mpg_channel_partition[i]:
                    if channel_id in nmp_channels:
                        tmp_nmp_channels.append(channel_id)
                    else:
                        tmp_normal_channels.append(channel_id)

                tmp_mpg_channel_mapping = MPGChannelMapping.random_sample_from_operator_stratifications(
                    mpg_operator_graph=mpg_group[i],
                    kv_head_num=kv_head_num,
                    nmp_channels=tmp_nmp_channels,
                    normal_channels=tmp_normal_channels,
                    per_nmp_channel_bw=per_nmp_channel_bw, per_normal_channel_bw=per_normal_channel_bw,
                    per_nmp_channel_comp=per_nmp_channel_comp, npu_comp=npu_comp
                )
                mpg_channel_mappings.append(tmp_mpg_channel_mapping)

            mag_list.append(cls(
                mag_operator_graph=mag_operator_graph, 
                nmp_channels=nmp_channels, 
                normal_channels=normal_channels, 
                mpg_channel_mappings=mpg_channel_mappings
            ))
        return mag_list


class Dataflow:
    def __init__(
        self,
        operator_graph,
        node_name,
        device,
        kv_head_num,
        memory_access_groups
    ) -> None:
        self.operator_graph: nx.DiGraph = operator_graph
        self.node_name = node_name
        self.device: Device = device
        self.kv_head_num = kv_head_num

        self.memory_access_groups: List["MemoryAccessGroup"] = memory_access_groups
        self._check_legality()

        self.mag_node_lists: List[List[int]] = []
        self.mpg_node_lists: List[List[List[int]]] = []
        self.stratification_node_lists: List[List[List[List[int]]]] = []
        self.op_channel_mappings: Dict[int, OperatorMapping] = {}
        self.simplified_op_channel_mappings: Dict[int, Tuple] = {}
        for mag in self.memory_access_groups:
            self.mag_node_lists.append(list(mag.mag_operator_graph.nodes))
            tmp_mpg_list = []
            tmp_stratification_list = []
            for mpg_channel_mapping in mag.mpg_channel_mappings:
                tmp_mpg_list.append(list(mpg_channel_mapping.mpg_operator_graph.nodes))
                tmp_mpg_stratification_list = []
                for stratification in mpg_channel_mapping.operator_stratifications:
                    tmp_mpg_stratification_list.append(list(stratification.nodes))
                tmp_stratification_list.append(tmp_mpg_stratification_list)
                for op in mpg_channel_mapping.operator_mappings.keys():
                    self.op_channel_mappings[op] = mpg_channel_mapping.operator_mappings[op]
                    self.simplified_op_channel_mappings[op] = (mpg_channel_mapping.operator_mappings[op].normal_channels, mpg_channel_mapping.operator_mappings[op].nmp_channels)
            self.mpg_node_lists.append(tmp_mpg_list)
            self.stratification_node_lists.append(tmp_stratification_list)

        self.edp = 0.
        self.latency = 0.
        self.energy = 0.
        self.prefill_latency = 0.
        self.prefill_energy = 0.
        self.decoding_latency = []
        self.decoding_energy = []

    def __lt__(self, other: "Dataflow"):
        if self.latency < other.latency:
            return True
        elif self.latency == other.latency:
            if self.energy < other.energy:
                return True
        return False

    def _check_legality(self):
        for mag in self.memory_access_groups:
            mag._check_legality()
            assert(len(mag.nmp_channels)) == self.device.nmp_channel_num, "Each MAG should have all nmp channels."
            if self.device.normal_channel_num > 0:
                assert len(mag.normal_channels) == 1, "Each MAG should have all normal channels (1 abstract channel)"
                assert len(mag.normal_channels[0]) == self.device.normal_channel_num, "The element in normal channel should be a tuple with #normal_channel_num elements"
            else:
                assert len(mag.normal_channels) == 0, "This design has no normal channel but some MAG contains normal channel."
        
        mag_node_groups = [list(mag.mag_operator_graph.nodes) for mag in self.memory_access_groups]
        check_memory_access_group_legality(mag_node_groups, self.operator_graph)

    def set_operator_shapes(
        self, 
        model_name: str, 
        q_num_per_kv: int,
        batch_size: int,
        input_seq_len: int,
        context_len: int
    ):
        for op_id in self.op_channel_mappings.keys():
            if self.node_name[model_name][op_id] in ['q', 'k', 'v', 'o', 'f1', 'f3', 'f2']:
                self.op_channel_mappings[op_id].M = batch_size * input_seq_len
            elif self.node_name[model_name][op_id] == 'qk':
                self.op_channel_mappings[op_id].M = input_seq_len * q_num_per_kv
                self.op_channel_mappings[op_id].N = context_len
            elif self.node_name[model_name][op_id] == 'sv':
                self.op_channel_mappings[op_id].M = input_seq_len * q_num_per_kv
                self.op_channel_mappings[op_id].K = context_len
            else:
                assert False, "Wrong node name"

    @classmethod
    def random_generate_from_mags(
        cls,
        operator_graph: nx.DiGraph, 
        node_name: Dict[str, Dict[int, str]],
        device: Device,
        kv_head_num: int
    ) -> "Dataflow":
        nmp_channels = list(range(device.nmp_channel_num))
        if device.normal_channel_num > 0:
            normal_channels = [tuple(range(device.nmp_channel_num, device.total_channel_num))]
        else:
            normal_channels = []

        meet_legal_design = False
        while not meet_legal_design:
            mag_node_groups, mag_operator_graphs = random_generate_memory_access_groups(operator_graph)
            memory_access_groups = []
            for mag in mag_operator_graphs:
                memory_access_groups.extend(list(MemoryAccessGroup.random_sample_from_mpg_channel_mapping(
                    mag_operator_graph=mag,
                    kv_head_num=kv_head_num,
                    nmp_channels=nmp_channels,
                    normal_channels=normal_channels,
                    per_nmp_channel_bw=device.pe_bandwidth*device.pe_num_per_channel, 
                    per_normal_channel_bw=12.8*0.7,
                    per_nmp_channel_comp=device.pe_computation_capacity*device.pe_num_per_channel, 
                    npu_comp=128*1000
                )))
            if not None in memory_access_groups:
                meet_legal_design = True
                break

        return cls(
            operator_graph=operator_graph,
            node_name=node_name,
            device=device,
            kv_head_num=kv_head_num,
            memory_access_groups=memory_access_groups
        )

    def random_generate_from_mpg_channel_mappings(self) -> "Dataflow":
        meet_legal_design = False
        while not meet_legal_design:
            new_memory_access_groups = []
            for mag in self.memory_access_groups:
                new_memory_access_groups.extend(list(MemoryAccessGroup.random_sample_from_mpg_channel_mapping(
                    mag_operator_graph=mag.mag_operator_graph,
                    kv_head_num=self.kv_head_num,
                    nmp_channels=mag.nmp_channels,
                    normal_channels=mag.normal_channels,
                    per_nmp_channel_bw=self.device.pe_bandwidth*self.device.pe_num_per_channel, 
                    per_normal_channel_bw=12.8*0.7,
                    per_nmp_channel_comp=self.device.pe_computation_capacity*self.device.pe_num_per_channel, 
                    npu_comp=128*1000
                )))
            if not None in new_memory_access_groups:
                meet_legal_design = True
                break
        
        new_dataflow = Dataflow(
            operator_graph=self.operator_graph,
            node_name=self.node_name,
            device=self.device,
            kv_head_num=self.kv_head_num,
            memory_access_groups=new_memory_access_groups
        )
        return new_dataflow

    def random_generate_from_stratifications(self) -> "Dataflow":
        new_memory_access_groups = []
        for mag in self.memory_access_groups:
            new_mpg_channel_mappings = []
            for mpg_channel_mapping in mag.mpg_channel_mappings:
                new_mpg_channel_mappings.append(MPGChannelMapping.random_sample_from_operator_stratifications(
                    mpg_operator_graph=mpg_channel_mapping.mpg_operator_graph,
                    kv_head_num=self.kv_head_num,
                    nmp_channels=mpg_channel_mapping.nmp_channels,
                    normal_channels=mpg_channel_mapping.normal_channels,
                    per_nmp_channel_bw=self.device.pe_bandwidth*self.device.pe_num_per_channel, 
                    per_normal_channel_bw=12.8*0.7,
                    per_nmp_channel_comp=self.device.pe_computation_capacity*self.device.pe_num_per_channel, 
                    npu_comp=128*1000
                ))
            new_memory_access_groups.append(MemoryAccessGroup(
                mag_operator_graph=mag.mag_operator_graph,
                nmp_channels=mag.nmp_channels,
                normal_channels=mag.normal_channels,
                mpg_channel_mappings=new_mpg_channel_mappings
            ))
        
        new_dataflow = Dataflow(
            operator_graph=self.operator_graph,
            node_name=self.node_name,
            device=self.device,
            kv_head_num=self.kv_head_num,
            memory_access_groups=new_memory_access_groups
        )
        return new_dataflow

    def random_generate_from_operator_mappings(self) -> "Dataflow":
        new_memory_access_groups = []
        for mag in self.memory_access_groups:
            new_mpg_channel_mappings = []
            for mpg_channel_mapping in mag.mpg_channel_mappings:
                new_mpg_channel_mappings.append(MPGChannelMapping.random_sample_from_operator_mappings(
                    mpg_operator_graph=mpg_channel_mapping.mpg_operator_graph,
                    kv_head_num=self.kv_head_num,
                    nmp_channels=mpg_channel_mapping.nmp_channels,
                    normal_channels=mpg_channel_mapping.normal_channels,
                    operator_stratifications=mpg_channel_mapping.operator_stratifications,
                    per_nmp_channel_bw=self.device.pe_bandwidth*self.device.pe_num_per_channel, 
                    per_normal_channel_bw=12.8*0.7,
                    per_nmp_channel_comp=self.device.pe_computation_capacity*self.device.pe_num_per_channel, 
                    npu_comp=128*1000
                ))
            new_memory_access_groups.append(MemoryAccessGroup(
                mag_operator_graph=mag.mag_operator_graph,
                nmp_channels=mag.nmp_channels,
                normal_channels=mag.normal_channels,
                mpg_channel_mappings=new_mpg_channel_mappings
            ))

        new_dataflow = Dataflow(
            operator_graph=self.operator_graph,
            node_name=self.node_name,
            device=self.device,
            kv_head_num=self.kv_head_num,
            memory_access_groups=new_memory_access_groups
        )
        return new_dataflow
