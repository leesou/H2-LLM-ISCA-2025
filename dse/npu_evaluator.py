from typing import Tuple


class NPURoofline:
    def __init__(
        self,
        element_size,
        npu_computation_capacity,
        bandwidth_per_normal_channel, # GB/s
        bandwidth_per_nmp_channel, # GB/s
    ) -> None:
        self.element_size = element_size
        self.npu_computation_capacity = npu_computation_capacity
        self.bandwidth_per_normal_channel = bandwidth_per_normal_channel
        self.bandwidth_per_nmp_channel = bandwidth_per_nmp_channel

    def evaluate_performance(
        self,
        # operator shape
        B, M, N, K,
        # channel number
        nmp_channel_num, normal_channel_num
    ) -> Tuple[float, float]:
        total_bandwidth = nmp_channel_num * self.bandwidth_per_nmp_channel + \
                          normal_channel_num * self.bandwidth_per_normal_channel
        # From LLM-Analysis, assume util is 70%
        npu_arithmetic_intensity = (self.npu_computation_capacity * 1e3 * 0.7) / (total_bandwidth * 0.7)
        
        operator_mac_count = B*M*K*N / 1e9 # GMAC
        operator_arithmetic_intensity = (M*K*N) / (self.element_size * (M*K + K*N))

        if npu_arithmetic_intensity < operator_arithmetic_intensity:
            real_computation_capacity = self.npu_computation_capacity * 1e3 * 0.7 # GMACS
        else:
            real_computation_capacity = (total_bandwidth * 0.7) * operator_arithmetic_intensity # GMACS
        
        computation_latency = operator_mac_count / real_computation_capacity
        computation_energy = 0.
        return computation_latency, computation_energy
