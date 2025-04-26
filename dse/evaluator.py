from typing import Tuple

from .dataflow import *
from .npu_evaluator import *
from .nmp_evaluator import *


def evaluate_npu_performance(
    npu_evaluator,
    B, M, N, K,
    nmp_channel_num, normal_channel_num
) -> Tuple[float, float]:
    return npu_evaluator.evaluate_performance(
        B=B, M=M, N=N, K=K,
        nmp_channel_num=nmp_channel_num, 
        normal_channel_num=normal_channel_num
    )


def evaluate_nmp_performance(
    nmp_evaluator,
    B, M, N, K,
    nmp_channel_num, 
    device: Device
) -> Tuple[float, float]:
    return nmp_evaluator.evaluate_performance(
        B=B, M=M, N=N, K=K,
        nmp_channel_num=nmp_channel_num,
        fpu_simd_width=device.fpu_simd_width,
        fpu_num_per_pe=device.fpu_num_per_pe,
        pe_frequency=device.pe_frequency, 
        pe_bandwidth=device.pe_bandwidth, 
        pe_num_per_channel=device.pe_num_per_channel,
        input_global_buffer_size=device.input_global_buffer_size, 
        output_global_buffer_size=device.output_global_buffer_size, 
        weight_buffer_size=device.weight_buffer_size
    )


class FixedMappingEvaluator:
    fixed_mapping_type_list = [
        'npu_only',
        '2x_npu_only',
        'fc_offloading',
        'attention_offloading',
        'attention_offloading_with_ffn_splitting'
    ]

    npu_computation_capacity_dict = {
        'default' : 128, 
        '2x_npu_only' : 256,
    }

    npu_bw_per_normal_channel_dict = {
        'default' : 12.8,
        '2x_npu_only' : 12.8,
    }

    npu_bw_per_nmp_channel_dict = {
        'default' : 12.8, 
        '2x_npu_only' : 12.8,
    }

    npu_type_dict = {
        'default' : 'normal', 
        '2x_npu_only' : 'normal_2x',
    }

    def __init__(
        self,
        fixed_mapping_type: str,
        q_head_num: int,
        kv_head_num: int,
        nmp_channel_num: int,
        normal_channel_num: int,
        element_size: float, # Byte per element
    ) -> None:
        self.fixed_mapping_type = fixed_mapping_type
        assert fixed_mapping_type in self.fixed_mapping_type_list
        
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        assert q_head_num % kv_head_num == 0
        self.q_num_per_kv = q_head_num // kv_head_num
        
        self.nmp_channel_num = nmp_channel_num
        self.normal_channel_num = normal_channel_num
        if self.fixed_mapping_type in ['npu_only', '2x_npu_only']:
            assert self.nmp_channel_num == 0

        self.element_size = element_size

        if fixed_mapping_type not in ['2x_npu_only']:
            self.npu_computation_capacity = self.npu_computation_capacity_dict['default']
            self.npu_bandwidth_per_normal_channel = self.npu_bw_per_normal_channel_dict['default']
            self.npu_bandwidth_per_nmp_channel = self.npu_bw_per_nmp_channel_dict['default']
            self.npu_type = self.npu_type_dict['default']
        else:
            self.npu_computation_capacity = self.npu_computation_capacity_dict[fixed_mapping_type]
            self.npu_bandwidth_per_normal_channel = self.npu_bw_per_normal_channel_dict[fixed_mapping_type]
            self.npu_bandwidth_per_nmp_channel = self.npu_bw_per_nmp_channel_dict[fixed_mapping_type]
            self.npu_type = self.npu_type_dict[fixed_mapping_type]
        self.npu_evaluator = NPURoofline(
            element_size=self.element_size,
            npu_computation_capacity=self.npu_computation_capacity,
            bandwidth_per_normal_channel=self.npu_bandwidth_per_normal_channel,
            bandwidth_per_nmp_channel=self.npu_bandwidth_per_nmp_channel,
        )

        self.nmp_evaluator = NMPRoofline(
            element_size=self.element_size,
            bandwidth_per_nmp_channel=12.8
        )

    def set_operator_shapes(
        self,
        operator_shape_dict: Dict[str, Dict[str, int]],
        batch_size: int,
        input_seq_len: int,
        context_len: int
    ):
        for op_name in operator_shape_dict.keys():
            if op_name in ['qkv', 'q', 'k', 'v', 'o', 'f1/3', 'f1', 'f3', 'f2']:
                operator_shape_dict[op_name]['M'] = batch_size * input_seq_len
            elif op_name in ['qk']:
                operator_shape_dict[op_name]['M'] = input_seq_len * self.q_num_per_kv
                operator_shape_dict[op_name]['N'] = context_len
            elif op_name in ['sv']:
                operator_shape_dict[op_name]['M'] = input_seq_len * self.q_num_per_kv
                operator_shape_dict[op_name]['K'] = context_len

    def _evaluate_npu_only(self, operator_shape_dict: Dict[str, Tuple]):
        total_latency = 0.
        total_energy = 0.
        for op_name in operator_shape_dict.keys():
            op_shape = operator_shape_dict[op_name]
            tmp_latency, tmp_energy = evaluate_npu_performance(
                npu_evaluator=self.npu_evaluator,
                B=op_shape['B'], M=op_shape['M'], 
                N=op_shape['N'], K=op_shape['K'],
                nmp_channel_num=self.nmp_channel_num,
                normal_channel_num=self.normal_channel_num
            )
            total_latency += tmp_latency
            total_energy += tmp_energy
        
        return total_latency, total_energy

    def _evaluate_fc_offloading(self, operator_shape_dict: Dict[str, Tuple], is_prefill, device: Device = None):
        total_latency = 0.
        total_energy = 0.
        for op_name in operator_shape_dict.keys():
            op_shape = operator_shape_dict[op_name]
            if is_prefill or op_name in ['qk', 'sv']:
                tmp_latency, tmp_energy = evaluate_npu_performance(
                    npu_evaluator=self.npu_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num, # Non-NMP operators are placed into normal channels only
                    normal_channel_num=0 if op_name not in ['qk', 'sv'] else self.normal_channel_num 
                )
            else:
                tmp_latency, tmp_energy = evaluate_nmp_performance(
                    nmp_evaluator=self.nmp_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    device=device
                )
            total_latency += tmp_latency
            total_energy += tmp_energy
        
        return total_latency, total_energy

    def _evaluate_attention_offloading(self, operator_shape_dict: Dict[str, Tuple], is_prefill, device: Device = None):
        total_latency = 0.
        total_energy = 0.
        for op_name in operator_shape_dict.keys():
            op_shape = operator_shape_dict[op_name]
            if is_prefill or op_name in ['qkv', 'q', 'k', 'v', 'o', 'f1', 'f3', 'f1/3', 'f2']:
                tmp_latency, tmp_energy = evaluate_npu_performance(
                    npu_evaluator=self.npu_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num, # Non-NMP operators are placed into normal channels only
                    normal_channel_num=self.normal_channel_num if op_name in ['qkv', 'q', 'k', 'v', 'o', 'f1', 'f3', 'f1/3', 'f2'] else 0
                )
            else:
                tmp_latency, tmp_energy = evaluate_nmp_performance(
                    nmp_evaluator=self.nmp_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    device=device
                )
            total_latency += tmp_latency
            total_energy += tmp_energy
        return total_latency, total_energy

    def _evaluate_attention_offloading_with_ffn_splitting(self, operator_shape_dict: Dict[str, Tuple], is_prefill, device: Device = None):
        total_latency = 0.
        total_energy = 0.
        for op_name in operator_shape_dict.keys():
            op_shape = operator_shape_dict[op_name]
            if is_prefill or op_name in ['qkv', 'q', 'k', 'v', 'o']:
                tmp_latency, tmp_energy = evaluate_npu_performance(
                    npu_evaluator=self.npu_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    normal_channel_num=self.normal_channel_num if op_name not in ['qk', 'sv'] else 0
                )
            elif op_name in ['f1/3', 'f1', 'f3', 'f2']:
                ratio = int((self.nmp_channel_num * device.pe_bandwidth) / (self.normal_channel_num * 12.8))
                npu_ratio = ratio / (ratio+1)
                nmp_ratio = 1 / (ratio+1)
                tmp_npu_latency, tmp_npu_energy = evaluate_npu_performance(
                    npu_evaluator=self.npu_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=math.ceil(op_shape['N']*npu_ratio) , K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    normal_channel_num=self.normal_channel_num
                )
                tmp_nmp_latency, tmp_nmp_energy = evaluate_nmp_performance(
                    nmp_evaluator=self.nmp_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=math.ceil(op_shape['N']*nmp_ratio), K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    device=device
                )
                tmp_latency = max(tmp_nmp_latency, tmp_npu_latency)
                tmp_energy = tmp_nmp_energy + tmp_npu_energy
            else:
                tmp_latency, tmp_energy = evaluate_nmp_performance(
                    nmp_evaluator=self.nmp_evaluator,
                    B=op_shape['B'], M=op_shape['M'], 
                    N=op_shape['N'], K=op_shape['K'],
                    nmp_channel_num=self.nmp_channel_num,
                    device=device
                )
            total_latency += tmp_latency
            total_energy += tmp_energy
        return total_latency, total_energy

    def evaluate_single_forward_pass(self, operator_shape_dict: Dict[str, Tuple], is_prefill, device: Device = None):
        if self.fixed_mapping_type in ['npu_only', '2x_npu_only']:
            return self._evaluate_npu_only(operator_shape_dict)
        elif self.fixed_mapping_type == 'fc_offloading':
            return self._evaluate_fc_offloading(operator_shape_dict, is_prefill, device)
        elif self.fixed_mapping_type == 'attention_offloading':
            return self._evaluate_attention_offloading(operator_shape_dict, is_prefill, device)
        elif self.fixed_mapping_type == 'attention_offloading_with_ffn_splitting':
            return self._evaluate_attention_offloading_with_ffn_splitting(operator_shape_dict, is_prefill, device)
        else:
            assert False, "wrong fixed mapping type"

    def evaluate_single_dataflow(
        self, 
        operator_shape_dict: Dict[str, Tuple],
        batch_size: int, prompt_len: int, 
        generation_len: int, context_len: int,
        device: Device = None
    ):
        tmp_context_len = 0

        # First estimate prefill latency
        tmp_context_len = prompt_len
        self.set_operator_shapes(
            operator_shape_dict=operator_shape_dict,
            batch_size=batch_size,
            input_seq_len=prompt_len,
            context_len=tmp_context_len
        )
        prefill_latency, prefill_energy = self.evaluate_single_forward_pass(operator_shape_dict, True, device)
        
        # Then estimate decoding latency
        tmp_context_len = min(context_len, prompt_len+generation_len)
        self.set_operator_shapes(
            operator_shape_dict=operator_shape_dict,
            batch_size=batch_size,
            input_seq_len=1,
            context_len=tmp_context_len
        )
        tmp_decoding_latency, tmp_decoding_energy = self.evaluate_single_forward_pass(operator_shape_dict, False, device)
        decoding_latency = [tmp_decoding_latency] * generation_len
        decoding_energy = [tmp_decoding_energy * generation_len]
        
        total_latency = prefill_latency + sum(decoding_latency)
        total_energy = prefill_energy + sum(decoding_energy)
        total_edp = total_latency * total_energy
        return ((total_latency, total_energy, total_edp),
                (prefill_latency, prefill_energy),
                (decoding_latency, decoding_energy))


class Evaluator:
    def __init__(
        self, 
        model_name: str, 
        q_head_num: int,
        kv_head_num: int,
        element_size: float,
    ) -> None:
        self.model_name = model_name
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        assert q_head_num % kv_head_num == 0
        self.q_num_per_kv = q_head_num // kv_head_num
        self.element_size = element_size

        self.npu_evaluator = NPURoofline(
            element_size=self.element_size,
            npu_computation_capacity=128,
            bandwidth_per_normal_channel=12.8,
            bandwidth_per_nmp_channel=12.8,
        )
        
        self.nmp_evaluator = NMPRoofline(
            element_size=self.element_size,
            bandwidth_per_nmp_channel=12.8
        )

    def evaluate_single_prefill_pass(self, dataflow: Dataflow):
        mag_latency_list = []
        mag_energy_list = []
        for mag in dataflow.memory_access_groups:
            mpg_latency_list = []
            mpg_energy_list = []
            for mpg_channel_mapping in mag.mpg_channel_mappings:
                stratification_latency_list = []
                stratification_energy_list = []
                for stratification in mpg_channel_mapping.operator_stratifications:
                    operator_npu_latency_list = []
                    operator_energy_list = []
                    for operator_id in list(stratification.nodes):
                        operator_mapping = mpg_channel_mapping.operator_mappings[operator_id]
                        B = operator_mapping.B
                        M = operator_mapping.M
                        N = operator_mapping.N
                        K = operator_mapping.K
                        
                        nmp_channel_num = len(operator_mapping.nmp_channels)
                        if len(operator_mapping.normal_channels) > 0:
                            assert len(operator_mapping.normal_channels) == 1
                            normal_channel_num = len(operator_mapping.normal_channels[0])
                        else:
                            normal_channel_num = 0

                        assert operator_mapping.prefill_execution_mapping == ExecutionMapping.NPU
                        npu_latency, npu_energy = evaluate_npu_performance(
                            self.npu_evaluator,
                            B=B, M=M, N=N, K=K,
                            nmp_channel_num=nmp_channel_num, 
                            normal_channel_num=normal_channel_num
                        )

                        operator_npu_latency_list.append(npu_latency)
                        operator_energy_list.append(npu_energy)
                    stratification_latency_list.append(sum(operator_npu_latency_list))
                    stratification_energy_list.append(sum(operator_energy_list))
                mpg_latency_list.append(sum(stratification_latency_list))
                mpg_energy_list.append(sum(stratification_energy_list))
            mag_latency_list.append(sum(mpg_latency_list))
            mag_energy_list.append(sum(mpg_energy_list))

        dataflow.prefill_latency = sum(mag_latency_list)
        dataflow.prefill_energy = sum(mag_energy_list)

    def evaluate_single_decoding_pass(self, dataflow: Dataflow):
        mag_latency_list = []
        mag_energy_list = []
        for mag in dataflow.memory_access_groups:
            mpg_latency_list = []
            mpg_energy_list = []
            for mpg_channel_mapping in mag.mpg_channel_mappings:
                stratification_latency_list = []
                stratification_energy_list = []
                for stratification in mpg_channel_mapping.operator_stratifications:
                    operator_npu_latency_list = []
                    operator_nmp_latency_list = []
                    operator_energy_list = []
                    for operator_id in list(stratification.nodes):
                        operator_mapping = mpg_channel_mapping.operator_mappings[operator_id]
                        B = operator_mapping.B
                        M = operator_mapping.M
                        N = operator_mapping.N
                        K = operator_mapping.K
                        nmp_storing_ratio = operator_mapping.nmp_storing_ratio

                        nmp_channel_num = len(operator_mapping.nmp_channels)
                        if len(operator_mapping.normal_channels) > 0:
                            assert len(operator_mapping.normal_channels) == 1
                            normal_channel_num = len(operator_mapping.normal_channels[0])
                        else:
                            normal_channel_num = 0
                        
                        execution_mapping = operator_mapping.decoding_execution_mapping
                        if execution_mapping == ExecutionMapping.NPU:
                            npu_latency, npu_energy = evaluate_npu_performance(
                                self.npu_evaluator,
                                B=B, M=M, N=N, K=K,
                                nmp_channel_num=nmp_channel_num, 
                                normal_channel_num=normal_channel_num
                            )
                            nmp_latency, nmp_energy = 0., 0.
                        elif execution_mapping == ExecutionMapping.NMP:
                            npu_latency, npu_energy = 0., 0.
                            nmp_latency, nmp_energy = evaluate_nmp_performance(
                                self.nmp_evaluator,
                                B=B, M=M, N=N, K=K,
                                nmp_channel_num=nmp_channel_num,
                                device=dataflow.device
                            )
                        else:
                            if B>1:
                                npu_B = B - int(B*nmp_storing_ratio)
                                npu_N = N
                                nmp_B = int(B*nmp_storing_ratio)
                                nmp_N = N
                            else:
                                npu_B = B
                                npu_N = N - int(N*nmp_storing_ratio)
                                nmp_B = B
                                nmp_N = int(N*nmp_storing_ratio)
                            npu_latency, npu_energy = evaluate_npu_performance(
                                self.npu_evaluator,
                                B=npu_B, M=M, N=npu_N, K=K,
                                nmp_channel_num=nmp_channel_num, 
                                normal_channel_num=normal_channel_num
                            )
                            nmp_latency, nmp_energy = evaluate_nmp_performance(
                                self.nmp_evaluator,
                                B=nmp_B, M=M, N=nmp_N, K=K,
                                nmp_channel_num=nmp_channel_num,
                                device=dataflow.device
                            )

                        operator_npu_latency_list.append(npu_latency)
                        operator_nmp_latency_list.append(nmp_latency)
                        operator_energy_list.append(npu_energy + nmp_energy)
                    stratification_latency_list.append(max(max(operator_nmp_latency_list), sum(operator_npu_latency_list)))
                    stratification_energy_list.append(sum(operator_energy_list))
                mpg_latency_list.append(sum(stratification_latency_list))
                mpg_energy_list.append(sum(stratification_energy_list))
            mag_latency_list.append(max(mpg_latency_list))
            mag_energy_list.append(sum(mpg_energy_list))
        
        dataflow.decoding_latency.append(sum(mag_latency_list))
        dataflow.decoding_energy.append(sum(mag_energy_list))

    def evaluate_single_dataflow(
        self, 
        dataflow: Dataflow, 
        batch_size: int, prompt_len: int, 
        generation_len: int, context_len: int
    ):
        self.nmp_evaluator.set_utilization(batch_size, dataflow.device)
        tmp_context_len = 0

        # First estimate prefill latency
        tmp_context_len = prompt_len
        dataflow.set_operator_shapes(
            model_name=self.model_name,
            q_num_per_kv=self.q_num_per_kv,
            batch_size=batch_size,
            input_seq_len=prompt_len,
            context_len=tmp_context_len
        )
        self.evaluate_single_prefill_pass(dataflow)
        
        # Then estimate decoding latency
        tmp_context_len = min(context_len, prompt_len+generation_len)
        dataflow.set_operator_shapes(
            model_name=self.model_name,
            q_num_per_kv=self.q_num_per_kv,
            batch_size=batch_size,
            input_seq_len=1,
            context_len=tmp_context_len
        )
        self.evaluate_single_decoding_pass(dataflow)
        dataflow.decoding_latency = dataflow.decoding_latency * generation_len
        dataflow.decoding_energy = dataflow.decoding_energy * generation_len
        
        dataflow.latency = dataflow.prefill_latency + sum(dataflow.decoding_latency)
        dataflow.energy = dataflow.prefill_energy + sum(dataflow.decoding_energy)
        dataflow.edp = dataflow.latency * dataflow.energy

