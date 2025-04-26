from dataclasses import dataclass
from enum import Enum, IntEnum


class DataPrecision(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not invalid for {cls.__name__}")


precision_bit_dict = {
    DataPrecision.FP16 : 16,
    DataPrecision.INT8 : 8,
    DataPrecision.INT4 : 4
}


@dataclass
class Device:
    # fixed parameters
    # Part 1: precision and integration type
    pe_precision: DataPrecision # determined by DSE's user input
    # Part 2: memory spec
    total_channel_num: int
    bank_num_per_channel: int
    bank_max_capacity: int # The unit of capacity is MB
    # Part 3: NMP PE
    fpu_simd_width: int

    # DSE parameters related with PE design
    # These parameters are still variable under different integration technologies
    nmp_channel_num: int
    fpu_num_per_pe: int
    pe_frequency: int # MHz
    pe_bandwidth: float # GB/s
    input_global_buffer_size: int # The unit is KB
    output_global_buffer_size: int # The unit is KB
    weight_buffer_size: int # The unit is KB

    # Derived parameters
    memory_capacity_per_normal_bank: int # determined by memory type, MB as the unit
    memory_capacity_per_nmp_bank: int # determined by NMP design, MB as the unit
    memory_capacity_per_pe: int # memory_capacity_per_bank * bank_num_per_pe
    bank_num_per_pe: int # determined by NMP placement & memory type
    pe_num_per_channel: int

    @property
    def normal_channel_num(self):
        return self.total_channel_num - self.nmp_channel_num

    @property
    def normal_channel_capacity(self):
        return self.memory_capacity_per_normal_bank * self.bank_num_per_channel

    @property
    def nmp_channel_capacity(self):
        return self.memory_capacity_per_nmp_bank * self.bank_num_per_channel

    @property
    def total_capacity(self):
        return self.normal_channel_capacity * self.normal_channel_num + \
               self.nmp_channel_capacity * self.nmp_channel_num

    @property
    def pe_computation_capacity(self):
        # The unit is GOPS, recording one PE's computation capacity
        return self.fpu_num_per_pe * self.fpu_simd_width * self.pe_frequency / 1e3

    @classmethod
    def init_from_hardware_design_space(
        self,
        # fixed architecture parameters
        pe_precision: DataPrecision,
        total_channel_num: int,
        bank_num_per_channel: int,
        bank_max_capacity: int,
        fpu_simd_width: float,
        # DSE parameters related with PE design
        # These parameters are still variable under different integration technologies
        nmp_channel_num: int,
        fpu_num_per_pe: int,
        pe_frequency: int, # MHz
        pe_bandwidth: float, # GB/s
        input_global_buffer_size: int,
        output_global_buffer_size: int,
        weight_buffer_size: int,
    ):  
        bank_num_per_pe = 1
        pe_num_per_channel = bank_num_per_channel // bank_num_per_pe
        memory_capacity_per_normal_bank = bank_max_capacity
        memory_capacity_per_nmp_bank = bank_max_capacity
        memory_capacity_per_pe = memory_capacity_per_nmp_bank * bank_num_per_pe
        
        return Device(
            # fixed parameters
            pe_precision=pe_precision,
            total_channel_num=total_channel_num,
            bank_num_per_channel=bank_num_per_channel,
            bank_max_capacity=bank_max_capacity,
            fpu_simd_width=fpu_simd_width,
            # DSE parameters related with PE design
            # These parameters are still variable under different integration technologies
            nmp_channel_num=nmp_channel_num,
            fpu_num_per_pe=fpu_num_per_pe,
            pe_frequency=pe_frequency,
            pe_bandwidth=pe_bandwidth,
            input_global_buffer_size=input_global_buffer_size,
            output_global_buffer_size=output_global_buffer_size,
            weight_buffer_size=weight_buffer_size,
            # Derived parameters
            memory_capacity_per_normal_bank=memory_capacity_per_normal_bank,
            memory_capacity_per_nmp_bank=memory_capacity_per_nmp_bank,
            memory_capacity_per_pe=memory_capacity_per_pe,
            bank_num_per_pe=bank_num_per_pe,
            pe_num_per_channel=pe_num_per_channel
        )
