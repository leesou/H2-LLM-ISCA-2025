from typing import Tuple
import math

from .hardware import *


def get_factors(n):
    factors = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)


def find_closest(num_list, target):
    return min(num_list, key=lambda x: abs(x - target))


class NMPRoofline:
    def __init__(
        self,
        element_size, # Byte
        bandwidth_per_nmp_channel, # GB/s
    ) -> None:
        self.element_size = element_size
        self.bandwidth_per_nmp_channel = bandwidth_per_nmp_channel
        self.utilization = 1.0
    
    def set_utilization(self, batch_size: int, device: Device):
        if device.fpu_num_per_pe == 1 and device.pe_frequency == 1000 and device.pe_bandwidth == 6.4:
            if batch_size == 1:
                self.utilization = 0.71
            elif batch_size == 4:
                self.utilization = 0.45
            elif batch_size == 16:
                self.utilization = 0.52
        if device.fpu_num_per_pe == 1 and device.pe_frequency == 200 and device.pe_bandwidth == 6.4:
            if batch_size == 1:
                self.utilization = 0.46
            elif batch_size == 4:
                self.utilization = 0.71
            elif batch_size == 16:
                self.utilization = 0.83
    
    def evaluate_performance(
        self,
        B, M, N, K,
        nmp_channel_num, 
        fpu_simd_width, fpu_num_per_pe,
        pe_frequency, pe_bandwidth, pe_num_per_channel,
        input_global_buffer_size, output_global_buffer_size, weight_buffer_size
    ) -> Tuple[float, float]:
        if B > nmp_channel_num:
            per_gemm_channel_num = 1
            per_channel_gemm_num = math.ceil(B/nmp_channel_num)
        elif B > 1:
            group_num = math.gcd(B, nmp_channel_num)
            per_gemm_channel_num = nmp_channel_num // group_num
            per_channel_gemm_num = B // group_num
        elif B == 1:
            per_gemm_channel_num = nmp_channel_num
            per_channel_gemm_num = 1
        else:
            assert False, "B should > 0"

        optimal_K_dim_divisor = find_closest(get_factors(per_gemm_channel_num), math.sqrt(K*per_gemm_channel_num/N))
        optimal_N_dim_divisor = per_gemm_channel_num // optimal_K_dim_divisor
        assert optimal_K_dim_divisor * optimal_N_dim_divisor == per_gemm_channel_num, "Wrong dim divisor allocation"
        per_channel_M = M
        per_channel_K = math.ceil(K/optimal_K_dim_divisor)
        per_channel_N = math.ceil(N/optimal_N_dim_divisor)

        per_channel_gemm_input_transfer_latency = self.element_size * per_channel_M * per_channel_K / (2 ** 30) / self.bandwidth_per_nmp_channel
        per_channel_gemm_output_transfer_latency = self.element_size * per_channel_M * per_channel_N / (2 ** 30) / self.bandwidth_per_nmp_channel

        per_channel_computation_capacity = fpu_simd_width * fpu_num_per_pe * pe_frequency * pe_num_per_channel / 1e3
        per_channel_internal_bandwidth = pe_bandwidth * pe_num_per_channel
        per_channel_pe_arithmetic_intensity = per_channel_computation_capacity / per_channel_internal_bandwidth
        
        per_channel_gemm_arithmetic_intensity = (per_channel_M*per_channel_K*per_channel_N) / (self.element_size * (per_channel_M*per_channel_K + per_channel_K*per_channel_N))
        per_channel_gemm_mac_count = (per_channel_M*per_channel_K*per_channel_N) / 1e9
        
        if per_channel_pe_arithmetic_intensity < per_channel_gemm_arithmetic_intensity:
            real_computation_capacity = per_channel_computation_capacity
        else:
            real_computation_capacity = per_channel_internal_bandwidth * per_channel_gemm_arithmetic_intensity
        per_channel_gemm_computation_latency = per_channel_gemm_mac_count / real_computation_capacity

        per_gemm_latency = per_channel_gemm_input_transfer_latency + per_channel_gemm_computation_latency + per_channel_gemm_output_transfer_latency
        per_gemm_energy = 0.

        all_gemm_latency = per_gemm_latency * per_channel_gemm_num / self.utilization
        all_gemm_energy = per_gemm_energy * B
        return all_gemm_latency, all_gemm_energy
