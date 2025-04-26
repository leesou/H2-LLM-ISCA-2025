import argparse
import json
import ast

from dse.searcher import *
from model_parser.model_parser import *


def parse_args():
    def parse_int_list(string: str):
        return list(map(int, string.split(",")))

    parser = argparse.ArgumentParser(description="H2-LLM")
    parser.add_argument("--execution-type", type=str, default="dse", choices=["parse", "dse"])
    # Model config arguments
    parser.add_argument("--model-type", type=str, default="opt", choices=["opt", "llama", "palm"])
    parser.add_argument("--model-shape", type=str, default="config/opt-6.7b/shape.json")
    # Model parser arguments
    parser.add_argument("--parser-output-dir", type=str, default="config/opt-6.7b")
    # Workload arguments
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--input-seq-len", type=int, default=1024)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "int8", "int4"])
    # Architecture arguments
    parser.add_argument("--total-channel-num", type=int, default=8)
    parser.add_argument("--bank-num-per-channel", type=int, default=16)
    parser.add_argument("--bank-memory-capacity", type=int, default=256)
    parser.add_argument("--fpu-simd-width", type=int, default=16)
    parser.add_argument("--nmp-channel-num-space", type=parse_int_list, default="2,4,6,8")
    parser.add_argument("--fpu-pe-bw-space", type=str,
                        default="[(8, 400, 6.4),  (8, 600, 6.4),  (8, 800, 6.4),  (8, 1000, 6.4), "
                                +"(8, 400, 12.8), (8, 600, 12.8), (8, 800, 12.8), (4, 1000, 12.8), "
                                +"(8, 400, 25.6), (8, 600, 25.6), (4, 800, 25.6), (4, 1000, 25.6), "
                                +"(8, 400, 51.2), (4, 600, 51.2), (4, 800, 51.2), (2, 1000, 51.2)]")
    parser.add_argument("--input-buffer-size-space", type=parse_int_list, default="4,8,16,32,64,128")
    parser.add_argument("--weight-buffer-size-space", type=parse_int_list, default="4,8,16,32,64,128")
    parser.add_argument("--output-buffer-total-size-space", type=parse_int_list, default="4,8,16,32,64,128")
    parser.add_argument("--is-device-fixed", action="store_true")
    # DSE arguments
    parser.add_argument("--dse-output-dir", type=str, default=f"./kick_the_tires/")
    parser.add_argument("--process-num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--population-num-per-generation", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=10),
    parser.add_argument("--mutate-ratio", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--is-dataflow-fixed", type=str, default="no",
                        choices=[
                            "no",
                            "specpim",
                            "npu_only",
                            "2x_npu_only",
                            "fc_offloading",
                            "attention_offloading",
                            "attention_offloading_with_ffn_splitting"])

    args = parser.parse_args()
    return args


def parse_multiple_tuples(value: str):
    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Parse multiple tuple {value} error: {e}")
    
    if not isinstance(result, list):
        raise ValueError(f"Expected as list but get: {type(result).__name__}")
    
    final_list = []
    for item in result:
        if not isinstance(item, tuple):
            raise ValueError(f"Expect elements are tuple but get: {item}")
        if len(item) != 3:
            raise ValueError(f"Expect elements are (FPU num, FPU frequency, Bandwidth) but get: {item}")
        final_list.append(item)

    return final_list


def conduct_dse(args):
    with open(args.model_shape, "r") as f:
        model_shape_config = json.load(f)
    model_shape = ModelShape(
        dim=model_shape_config["hidden_dim"],
        ffn_dim=model_shape_config["intermediate_dim"],
        n_heads=model_shape_config["q_head_num"],
        n_kv_heads=model_shape_config["kv_head_num"]
    )

    if args.execution_type == "parse":
        parse_graph(
            model_type=args.model_type,
            model_shape=model_shape,
            parser_output_dir=args.parser_output_dir
        )
        return
    if not os.path.exists(f"{args.parser_output_dir}/operator_graph.json"):
        parse_graph(
            model_type=args.model_type,
            model_shape=model_shape,
            parser_output_dir=args.parser_output_dir
        )

    # init model
    model_args = ModelArgs.init_from_config(args)
    with open(f"{args.parser_output_dir}/operator_graph.json", "r") as f:
        operator_graph = json.load(f)
    precision = args.precision
    model = Model(
        args=model_args,
        operator_graph=operator_graph,
        precision=precision
    )

    os.makedirs(args.dse_output_dir, exist_ok=True)
    # init DSE algorithm params
    dse_params = DSEParams(
        log_dir=f"{args.dse_output_dir}",
        process_num=args.process_num,
        seed=args.seed,
        population_num_per_generation=args.population_num_per_generation,
        num_generations=args.num_generations,
        mutate_ratio=args.mutate_ratio,
        topk=args.topk
    )
    
    # init hardware space
    nmp_channel_num_space = args.nmp_channel_num_space
    fpu_pe_bw_space = parse_multiple_tuples(args.fpu_pe_bw_space)
    input_buffer_size_space = args.input_buffer_size_space
    weight_buffer_size_space = args.weight_buffer_size_space
    output_buffer_total_size_space = args.output_buffer_total_size_space
    fixed_device = None
    if args.is_device_fixed:
        assert len(nmp_channel_num_space) == 1
        assert len(fpu_pe_bw_space) == 1
        assert len(input_buffer_size_space) == 1
        assert len(weight_buffer_size_space) == 1
        assert len(output_buffer_total_size_space) == 1
        fixed_device = Device.init_from_hardware_design_space(
            pe_precision=DataPrecision.from_string(precision),
            total_channel_num=args.total_channel_num,
            bank_num_per_channel=args.bank_num_per_channel,
            bank_max_capacity=args.bank_memory_capacity,
            fpu_simd_width=args.fpu_simd_width,
            nmp_channel_num=nmp_channel_num_space[0],
            fpu_num_per_pe=fpu_pe_bw_space[0][0],
            pe_frequency=fpu_pe_bw_space[0][1],
            pe_bandwidth=fpu_pe_bw_space[0][2],
            input_global_buffer_size=input_buffer_size_space[0],
            output_global_buffer_size=output_buffer_total_size_space[0],
            weight_buffer_size=weight_buffer_size_space[0]
        )
    hardware_hyper_params = HardwareHyperParams(
        pe_precision=DataPrecision.from_string(precision),
        is_device_fixed=args.is_device_fixed,
        fixed_device=fixed_device
    )
    hardware_design_space = HardwareDesignSpace(
        total_channel_num=args.total_channel_num,
        bank_num_per_channel=args.bank_num_per_channel,
        bank_max_capacity=args.bank_memory_capacity,
        fpu_simd_width=args.fpu_simd_width,
        nmp_channel_num_space=nmp_channel_num_space,
        fpu_pe_bw_space=fpu_pe_bw_space,
        input_global_buffer_size_space=input_buffer_size_space,
        weight_buffer_size_space=weight_buffer_size_space,
        output_global_buffer_size_space=output_buffer_total_size_space
    )

    # quick evaluation for fixed dataflow
    if args.is_dataflow_fixed not in ["no", "specpim"]:
        assert fixed_device is not None
        operator_shape_dict = {}
        for op_id in model.simplified_op_graph.nodes:
            node_name = model.node_name[model.model_type][op_id]
            operator_shape_dict[node_name] = {'B':0, 'M':0, 'N':0, 'K':0}
            operator_shape_dict[node_name]['B'] = model.simplified_op_graph.nodes[op_id]['B']
            operator_shape_dict[node_name]['M'] = model.simplified_op_graph.nodes[op_id]['M']
            operator_shape_dict[node_name]['N'] = model.simplified_op_graph.nodes[op_id]['N']
            operator_shape_dict[node_name]['K'] = model.simplified_op_graph.nodes[op_id]['K']
        
        nmp_channel_num_space = args.nmp_channel_num_space
        assert len(nmp_channel_num_space) == 1
        nmp_channel_num = nmp_channel_num_space[0]
        normal_channel_num = args.total_channel_num - nmp_channel_num
        fixed_mapping_evaluator = FixedMappingEvaluator(
            fixed_mapping_type=args.is_dataflow_fixed,
            q_head_num=model.head_num,
            kv_head_num=model.kv_head_num,
            nmp_channel_num=nmp_channel_num,
            normal_channel_num=normal_channel_num,
            element_size=model.precision_bit_dict[model.precision] // 8,
        )

        e2e_performance, prefill_performance, decoding_performance = fixed_mapping_evaluator.evaluate_single_dataflow(
            operator_shape_dict=operator_shape_dict,
            batch_size=model.batch_size,
            prompt_len=model.input_seq_len,
            generation_len=model.max_gen_len,
            context_len=model.max_seq_len,
            device=fixed_device
        )
        with open(args.dse_output_dir+'/result.log', 'w') as f:
            f.write(f"Best idv latency is {e2e_performance[0]} (per layer).\n")
            f.write(f"Prefill latency is {prefill_performance[0]} (per layer).\n")
            f.write(f"Decoding latency is {sum(decoding_performance[0])} (per layer).\n")
        return
    
    # init DSE searcher
    searcher = SingleTaskSearcher(
        dse_params=dse_params, 
        hardware_hyper_params=hardware_hyper_params,
        hardware_design_space=hardware_design_space,
        model=model
    )

    # conduct DSE
    if args.is_dataflow_fixed == "specpim":
        global OPEN_SPECPIM
        OPEN_SPECPIM.value = True
    searcher.genetic_evolve_search_multi_processing()


if __name__ == "__main__":
    args = parse_args()
    conduct_dse(args)
