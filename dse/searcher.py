from dataclasses import dataclass
from typing import List, Optional, Union
from multiprocessing import Process, Queue
from itertools import product
import os
import random
import heapq
import sys

import tqdm

from .hardware import *
from .model import *
from .dataflow import *
from .evaluator import *


@dataclass
class DSEParams:
    population_num_per_generation: int = 100000
    num_generations: int = 200
    mutate_ratio: float = 1.0
    topk: int = 50

    process_num: int = 50
    seed: int = 1919810
    log_dir: str = "./decaparated/tmp"
    dump_intermediate_samples: bool = False


@dataclass
class HardwareHyperParams:
    pe_precision: DataPrecision = DataPrecision.FP16
    is_device_fixed: bool = False
    fixed_device: Device = None


@dataclass
class HardwareDesignSpace:
    total_channel_num: int

    bank_num_per_channel: int

    bank_max_capacity: int

    fpu_simd_width: int

    nmp_channel_num_space: List[int]

    fpu_pe_bw_space: List[Tuple[int, float, float]]

    input_global_buffer_size_space: List[int]

    output_global_buffer_size_space: List[int]

    weight_buffer_size_space: List[int]


class SingleTaskSearcher:
    def __init__(
        self, 
        dse_params: DSEParams,
        hardware_hyper_params: HardwareHyperParams,
        hardware_design_space: HardwareDesignSpace,
        model: Model
    ) -> None:
        self.dse_params = dse_params
        self.new_individual_per_topk = int(dse_params.population_num_per_generation * dse_params.mutate_ratio) // dse_params.topk
        self.new_crossover_individuals = dse_params.population_num_per_generation - int(dse_params.population_num_per_generation * dse_params.mutate_ratio)
        self.num_dataflow_per_worker = dse_params.population_num_per_generation // dse_params.process_num + 1
        random.seed(dse_params.seed)

        self.model = model
        self.model_graph = model.simplified_op_graph
        self.q_head_num = model.head_num
        self.kv_head_num = model.kv_head_num
        self.batch_size = model.batch_size
        self.prompt_len = model.input_seq_len
        self.generation_len = model.max_gen_len
        self.context_len = model.max_seq_len
        
        self.hardware_hyper_params = hardware_hyper_params
        self.precision_bit = precision_bit_dict[hardware_hyper_params.pe_precision]
        self.hardware_design_space = hardware_design_space
        if hardware_hyper_params.is_device_fixed:
            assert hardware_hyper_params.fixed_device is not None, "fixed_device should not be None when is_device_fixed is open"
            self.legal_hardware_designs = [hardware_hyper_params.fixed_device]
        else:
            self.legal_hardware_designs = self.init_legal_hardware_designs()

        assert self.q_head_num % self.kv_head_num == 0, "Query head number should be divisible by Key/Value head number"
        self.evaluator = Evaluator(
            model_name=self.model.model_args.model_type,
            q_head_num=self.q_head_num,
            kv_head_num=self.kv_head_num,
            element_size=self.precision_bit / 8
        )

    def dataflow_capacity_checker(self, dataflow: Dataflow):
        dataflow.set_operator_shapes(
            model_name=self.model.model_args.model_type,
            q_num_per_kv=(self.q_head_num // self.kv_head_num),
            batch_size=self.batch_size,
            input_seq_len=1,
            context_len=self.context_len
        )
        nmp_channel_occupied_capacity_list = [0 for _ in range(dataflow.device.nmp_channel_num)]
        normal_channel_occupied_capacity = 0.
        for op_id in dataflow.op_channel_mappings.keys():
            B = dataflow.op_channel_mappings[op_id].B
            M = dataflow.op_channel_mappings[op_id].M
            N = dataflow.op_channel_mappings[op_id].N
            K = dataflow.op_channel_mappings[op_id].K
            nmp_storing_ratio = dataflow.op_channel_mappings[op_id].nmp_storing_ratio

            if self.model.node_name[self.model.model_type][op_id] in ['sv']:
                input_size = self.batch_size * self.context_len * self.q_head_num * self.precision_bit / 8 / (2**20)
                weight_size = self.batch_size * self.kv_head_num * self.context_len * self.model.head_dim * self.precision_bit / 8 / (2**20)
                output_size = self.batch_size * self.model.dim * self.precision_bit / 8 / (2**20)
            elif self.model.node_name[self.model.model_type][op_id] in ['qk']:
                input_size = self.batch_size * self.model.dim * self.precision_bit / 8 / (2**20)
                weight_size = self.batch_size * self.kv_head_num * self.context_len * self.model.head_dim * self.precision_bit / 8 / (2**20)
                output_size = self.batch_size * self.context_len * self.q_head_num * self.precision_bit / 8 / (2**20)
            else:
                input_size = B * M * K * self.precision_bit / 8 / (2**20)
                weight_size = B * K * N * self.precision_bit / 8 / (2**20)
                output_size = B * M * N * self.precision_bit / 8 / (2**20)
            total_size = self.model.layer_num * weight_size
            
            for nmp_channel_id in dataflow.op_channel_mappings[op_id].nmp_channels:
                nmp_channel_occupied_capacity_list[nmp_channel_id] += total_size * nmp_storing_ratio / len(dataflow.op_channel_mappings[op_id].nmp_channels)
            if len(dataflow.op_channel_mappings[op_id].normal_channels) > 0:
                assert len(dataflow.op_channel_mappings[op_id].normal_channels) == 1
                normal_channel_occupied_capacity += total_size * (1 - nmp_storing_ratio)
        
        for nmp_channel_occupied_capacity in nmp_channel_occupied_capacity_list:
            if nmp_channel_occupied_capacity > dataflow.device.nmp_channel_capacity:
                return False
        if normal_channel_occupied_capacity > dataflow.device.normal_channel_capacity * dataflow.device.normal_channel_num:
            return False
        total_size = sum(nmp_channel_occupied_capacity_list) + normal_channel_occupied_capacity
        if total_size > dataflow.device.total_capacity:
            return False
        return True

    def init_legal_hardware_designs(self) -> List[Device]:
        legal_hardware_designs = []

        hardware_parameter_combinations = list(product(
            self.hardware_design_space.nmp_channel_num_space,
            self.hardware_design_space.fpu_pe_bw_space,
            self.hardware_design_space.input_global_buffer_size_space,
            self.hardware_design_space.output_global_buffer_size_space,
            self.hardware_design_space.weight_buffer_size_space
        ))
        for hardware_parameter_combination in hardware_parameter_combinations:
            (
                nmp_channel_num,
                fpu_pe_bw,
                input_global_buffer_size,
                output_global_buffer_size,
                weight_buffer_size
            ) = hardware_parameter_combination

            (
                fpu_num_per_pe,
                pe_frequency,
                pe_bandwidth
            ) = fpu_pe_bw

            tmp_hardware_design = Device.init_from_hardware_design_space(
                pe_precision=self.hardware_hyper_params.pe_precision,
                total_channel_num=self.hardware_design_space.total_channel_num,
                bank_num_per_channel=self.hardware_design_space.bank_num_per_channel,
                bank_max_capacity=self.hardware_design_space.bank_max_capacity,
                fpu_simd_width=self.hardware_design_space.fpu_simd_width,
                nmp_channel_num=nmp_channel_num,
                fpu_num_per_pe=fpu_num_per_pe,
                pe_frequency=pe_frequency,
                pe_bandwidth=pe_bandwidth,
                input_global_buffer_size=input_global_buffer_size,
                output_global_buffer_size=output_global_buffer_size,
                weight_buffer_size=weight_buffer_size,
            )
            legal_hardware_designs.append(tmp_hardware_design)
        
        return legal_hardware_designs

    def mutate(self, father_dataflow: Dataflow, operation_type: int) -> Dataflow:
        if operation_type == 0:
            new_device = random.choice(self.legal_hardware_designs)
            return Dataflow.random_generate_from_mags(
                operator_graph=self.model_graph,
                node_name=self.model.node_name,
                device=new_device,
                kv_head_num=self.kv_head_num
            )
        elif operation_type == 1:
            return Dataflow.random_generate_from_mags(
                operator_graph=self.model_graph,
                node_name=self.model.node_name,
                device=father_dataflow.device,
                kv_head_num=self.kv_head_num
            )
        elif operation_type == 2:
            return father_dataflow.random_generate_from_mpg_channel_mappings()
        elif operation_type == 3:
            return father_dataflow.random_generate_from_stratifications()
        elif operation_type == 4:
            return father_dataflow.random_generate_from_operator_mappings()
        else:
            assert False, f"Wrong mutate operation type {operation_type}"

    def crossover(self, father_dataflow: Dataflow, mother_dataflow: Dataflow, operation_type: int) -> Optional[Dataflow]:
        if operation_type == 0:
            child_device = random.choice([father_dataflow.device, mother_dataflow.device])
            child_nmp_channels = list(range(child_device.nmp_channel_num))
            if child_device.total_channel_num - child_device.nmp_channel_num > 0:
                child_normal_channels = [tuple(range(child_device.nmp_channel_num, child_device.total_channel_num))]
            else:
                child_normal_channels = []
            
            meet_legal_design = False
            retry_count = 0
            while not meet_legal_design:
                father_mag_graphs = [mag.mag_operator_graph for mag in father_dataflow.memory_access_groups]
                mother_mag_graphs = [mag.mag_operator_graph for mag in mother_dataflow.memory_access_groups]
                selector = random.choice([0, 1])
                if selector == 0:
                    front_mag_graphs = father_mag_graphs
                    back_mag_graphs = mother_mag_graphs
                else:
                    front_mag_graphs = mother_mag_graphs
                    back_mag_graphs = father_mag_graphs
                
                front_selected_mag_graphs = []
                front_selected_node_set = set()
                back_selected_mag_graphs = []
                back_selected_node_set = set()
                front_idx = 0
                back_idx = len(back_mag_graphs) - 1
                while front_idx < len(front_mag_graphs) and back_idx >= 0:
                    if set(front_mag_graphs[front_idx].nodes).isdisjoint(back_selected_node_set):
                        front_selected_mag_graphs.append(front_mag_graphs[front_idx])
                        front_selected_node_set.update(set(front_mag_graphs[front_idx].nodes))
                        front_idx += 1
                    else:
                        break
                    if set(back_mag_graphs[back_idx].nodes).isdisjoint(front_selected_node_set):
                        back_selected_mag_graphs.append(back_mag_graphs[back_idx])
                        back_selected_node_set.update(set(back_mag_graphs[back_idx].nodes))
                        back_idx -= 1
                    else:
                        break

                remained_graph = self.model_graph.copy()
                nodes_to_remove = set()
                for mag in (front_selected_mag_graphs + back_selected_mag_graphs):
                    nodes_to_remove.update(list(mag.nodes))
                remained_graph.remove_nodes_from(nodes_to_remove)
                _, remained_mag_graphs = random_generate_memory_access_groups(remained_graph)

                new_mag_graphs = front_selected_mag_graphs + remained_mag_graphs + list(reversed(back_selected_mag_graphs))
                new_memory_access_groups = []
                for new_mag_graph in new_mag_graphs:
                    new_memory_access_groups.extend(list(MemoryAccessGroup.random_sample_from_mpg_channel_mapping(
                        mag_operator_graph=new_mag_graph,
                        kv_head_num=self.kv_head_num,
                        nmp_channels=child_nmp_channels, 
                        normal_channels=child_normal_channels,
                        per_nmp_channel_bw=child_device.pe_bandwidth*child_device.pe_num_per_channel, 
                        per_normal_channel_bw=12.8,
                        per_nmp_channel_comp=child_device.pe_computation_capacity*child_device.pe_num_per_channel, 
                        npu_comp=128*1000
                    )))
                if not None in new_memory_access_groups:
                    meet_legal_design = True
                    break
                retry_count += 1
                if retry_count > 10:
                    break
            
            if not meet_legal_design:
                return None
            return Dataflow(
                operator_graph=self.model_graph,
                node_name=self.model.node_name,
                device=child_device,
                kv_head_num=self.kv_head_num,
                memory_access_groups=new_memory_access_groups
            )
        else:
            assert False,  f"Wrong crossover operation type {operation_type}"

    def init_population(self) -> List[Dataflow]:
        population = []
        individual_num = 0
        while individual_num < self.dse_params.population_num_per_generation:
            tmp_device = random.choice(self.legal_hardware_designs)
            tmp_dataflow = Dataflow.random_generate_from_mags(
                operator_graph=self.model_graph,
                node_name=self.model.node_name,
                device = tmp_device,
                kv_head_num=self.kv_head_num
            )
            if self.dataflow_capacity_checker(tmp_dataflow):
                population.append(tmp_dataflow)
                individual_num += 1
        return population

    def evolve_from_prev_population(self, prev_topk: List[Dataflow]) -> List[Dataflow]:
        population = []
        for dataflow in prev_topk:
            tmp_mutate_dataflow_num = 0
            while tmp_mutate_dataflow_num < self.new_individual_per_topk:
                mutate_op = random.randint(0, 4)
                new_dataflow = self.mutate(dataflow, mutate_op)
                if self.dataflow_capacity_checker(new_dataflow):
                    population.append(new_dataflow)
                    tmp_mutate_dataflow_num += 1
        tmp_crossover_dataflow_num = 0
        while tmp_crossover_dataflow_num < self.new_crossover_individuals:
            crossover_op = 0
            father_dataflow, mother_dataflow = random.sample(prev_topk, 2)
            new_dataflow = self.crossover(father_dataflow, mother_dataflow, crossover_op)
            if new_dataflow is not None and self.dataflow_capacity_checker(new_dataflow):
                population.append(new_dataflow)
                tmp_crossover_dataflow_num += 1
        return population
    
    def genetic_evolve_search(self):
        os.makedirs(self.dse_params.log_dir, exist_ok=True)
        with open(self.dse_params.log_dir+"/exploration.log", "w") as f:
            f.write("")

        per_generation_best_idv: List[Dataflow] = []
        for generation in tqdm.trange(self.dse_params.num_generations):
            if generation == 0:
                tmp_population = self.init_population()
            else:
                tmp_population = self.evolve_from_prev_population(tmp_topk_individuals)

            tmp_best_individual: Dataflow = None
            for individual in tmp_population:
                self.evaluator.evaluate_single_dataflow(
                    dataflow=individual, 
                    batch_size=self.batch_size, 
                    prompt_len=self.prompt_len, 
                    generation_len=self.generation_len,
                    context_len=self.context_len
                )
                if tmp_best_individual is None or individual < tmp_best_individual:
                    tmp_best_individual = individual
            tmp_topk_individuals = heapq.nsmallest(self.dse_params.topk, tmp_population)
            per_generation_best_idv.append(tmp_best_individual) 

        best_idv = None
        for tmp_generation_best_idv in per_generation_best_idv:
            if best_idv is None or tmp_generation_best_idv < best_idv:
                best_idv = tmp_generation_best_idv
        with open(self.dse_params.log_dir+"/exploration.log", "a") as f:
            f.write(f"Best idv EDP is {best_idv.edp}, latency is {best_idv.latency}, energy is {best_idv.energy}\n")

    def init_population_multiprocessing(self, generation: int) -> Union[Dataflow, List[Dataflow]]:
        def init_worker(tmp_worker_total_individual_num: int, pid: int, generation: int, data_queue: Queue):
            tmp_worker_population = []
            tmp_worker_best_dataflow = None

            tmp_individual_num = 0
            oom_individual_num = 0
            total_individual_num = 0
            while tmp_individual_num < tmp_worker_total_individual_num:
                tmp_device = random.choice(self.legal_hardware_designs)
                tmp_dataflow = Dataflow.random_generate_from_mags(
                    operator_graph=self.model_graph,
                    node_name=self.model.node_name,
                    device = tmp_device,
                    kv_head_num=self.kv_head_num
                )

                total_individual_num += 1
                if self.dataflow_capacity_checker(tmp_dataflow):
                    tmp_individual_num += 1
                    tmp_worker_population.append(tmp_dataflow)
                    self.evaluator.evaluate_single_dataflow(
                        dataflow=tmp_dataflow, 
                        batch_size=self.batch_size, 
                        prompt_len=self.prompt_len, 
                        generation_len=self.generation_len,
                        context_len=self.context_len
                    )
                    if tmp_worker_best_dataflow is None or tmp_dataflow < tmp_worker_best_dataflow:
                        tmp_worker_best_dataflow = tmp_dataflow
                else:
                    oom_individual_num += 1
            
            tmp_worker_topk_dataflows = heapq.nsmallest(self.dse_params.topk, tmp_worker_population)
            self.evaluator.evaluate_single_dataflow(
                dataflow=tmp_worker_topk_dataflows[0], 
                batch_size=self.batch_size, 
                prompt_len=self.prompt_len, 
                generation_len=self.generation_len,
                context_len=self.context_len
            )
            data_queue.put((pid, tmp_worker_best_dataflow, tmp_worker_topk_dataflows))

        data_queue = Queue()
        process_list = []
        for pid in range(self.dse_params.process_num):
            p = Process(target=init_worker, args=(self.num_dataflow_per_worker, pid, generation, data_queue))
            process_list.append(p)
        for p in process_list:
            p.start()

        population = []
        best_dataflow = None
        collect_data_count = 0
        while collect_data_count < self.dse_params.process_num:
            pid, tmp_worker_best_dataflow, tmp_worker_topk_dataflows = data_queue.get()
            if best_dataflow is None or tmp_worker_best_dataflow < best_dataflow:
                best_dataflow = tmp_worker_best_dataflow
            population.extend(tmp_worker_topk_dataflows)
            collect_data_count += 1
        for p in process_list:
            p.join()
        return best_dataflow, population

    def evolve_population_multi_processing(self, generation: int, prev_topk: List[Dataflow]) -> Union[Dataflow, List[Dataflow]]:        
        def evolve_worker(tmp_worker_total_individual_num: int, pid: int, generation: int, data_queue: Queue):
            tmp_worker_population = []
            tmp_worker_best_dataflow = None

            tmp_individual_num = 0
            oom_individual_num = 0
            total_individual_num = 0
            while tmp_individual_num < tmp_worker_total_individual_num:
                evolve_type = random.choices([0, 1], weights=[self.dse_params.mutate_ratio, 1-self.dse_params.mutate_ratio], k=1)[0]
                if evolve_type == 0:
                    mutate_op = random.randint(0, 4)
                    father_dataflow = random.choice(prev_topk)
                    tmp_dataflow = self.mutate(father_dataflow, mutate_op)
                else:
                    crossover_op = 0
                    father_dataflow, mother_dataflow = random.sample(prev_topk, 2)
                    tmp_dataflow = self.crossover(father_dataflow, mother_dataflow, crossover_op)

                total_individual_num += 1
                if tmp_dataflow is not None and self.dataflow_capacity_checker(tmp_dataflow):
                    tmp_individual_num += 1
                    tmp_worker_population.append(tmp_dataflow)
                    self.evaluator.evaluate_single_dataflow(
                        dataflow=tmp_dataflow, 
                        batch_size=self.batch_size, 
                        prompt_len=self.prompt_len, 
                        generation_len=self.generation_len,
                        context_len=self.context_len
                    )
                    if tmp_worker_best_dataflow is None or tmp_dataflow < tmp_worker_best_dataflow:
                        tmp_worker_best_dataflow = tmp_dataflow
                else:
                    oom_individual_num += 1

            tmp_worker_topk_dataflows = heapq.nsmallest(self.dse_params.topk, tmp_worker_population)
            data_queue.put((pid, tmp_worker_best_dataflow, tmp_worker_topk_dataflows))

        data_queue = Queue()
        process_list = []
        for pid in range(self.dse_params.process_num):
            p = Process(target=evolve_worker, args=(self.num_dataflow_per_worker, pid, generation, data_queue))
            process_list.append(p)
        for p in process_list:
            p.start()

        population = []
        best_dataflow = None
        collect_data_count = 0
        while collect_data_count < self.dse_params.process_num:
            pid, tmp_worker_best_dataflow, tmp_worker_topk_dataflows = data_queue.get()
            if best_dataflow is None or tmp_worker_best_dataflow < best_dataflow:
                best_dataflow = tmp_worker_best_dataflow
            population.extend(tmp_worker_topk_dataflows)
            collect_data_count += 1
        for p in process_list:
            p.join()
        return best_dataflow, population
    
    def genetic_evolve_search_multi_processing(self):
        os.makedirs(self.dse_params.log_dir, exist_ok=True)

        per_generation_best_idv: List[Dataflow] = []
        for generation in tqdm.trange(self.dse_params.num_generations, file=sys.stdout, disable=False, dynamic_ncols=False):
            if generation == 0:
                tmp_best_individual, tmp_population = self.init_population_multiprocessing(generation)
            else:
                tmp_best_individual, tmp_population = self.evolve_population_multi_processing(generation, tmp_topk_individuals)

            tmp_topk_individuals = heapq.nsmallest(self.dse_params.topk, tmp_population)
            per_generation_best_idv.append(tmp_best_individual)
            if generation % 5 == 0:
                print(f"{generation} rounds of DSE have finished.", flush=True)

        best_idv: Dataflow = None
        for tmp_generation_best_idv in per_generation_best_idv:
            if best_idv is None or tmp_generation_best_idv < best_idv:
                best_idv = tmp_generation_best_idv
        with open(self.dse_params.log_dir+"/result.log", "w") as f:
            f.write(f"Best idv latency is {best_idv.latency} (per layer).\n")
            f.write(f"Prefill latency is {best_idv.prefill_latency} (per layer).\n")
            f.write(f"Decoding latency is {sum(best_idv.decoding_latency)} (per layer).\n")
            f.write(f"PE config {best_idv.device.fpu_num_per_pe} fpus @ {best_idv.device.pe_frequency*1e-3:.1f}GHz, hybrid bonding bandwidth {best_idv.device.pe_bandwidth}GB/s.\n")
            f.write(f"Input buffer size {best_idv.device.input_global_buffer_size}KB, weight buffer size {best_idv.device.weight_buffer_size}KB, "
                    f"output buffer size {best_idv.device.output_global_buffer_size*1.0/best_idv.device.bank_num_per_channel:.2f}KB, NMP channel num {best_idv.device.nmp_channel_num}.\n")
