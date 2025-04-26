import os
import argparse
import subprocess
import time


cmd = (
    "conda run --name {env} python -u main.py "
    "--model-type {model_type} "
    "--model-shape {model_shape} "
    "--parser-output-dir {parser_output_dir} "
    "--max-batch-size {max_batch_size} "
    "--input-seq-len {input_seq_len} "
    "--max-gen-len {max_gen_len} "
    "--nmp-channel-num-space {nmp_channel_num_space} "
    "--fpu-pe-bw-space {fpu_pe_bw_space} "
    "--input-buffer-size-space 32 "
    "--weight-buffer-size-space 32 "
    "--output-buffer-total-size-space 64 "
    "{is_device_fixed} "
    "--dse-output-dir {dse_output_dir} "
    "--process-num {process_num} "
    "--population-num-per-generation 2500 "
    "--num-generations 50 "
    "--mutate-ratio {mutate_ratio} "
    "--topk 50 "
    "--is-dataflow-fixed {is_dataflow_fixed} "
)


model_configs_for_ae = {
    "opt": {"model_shape": "config/opt-6.7b/shape.json", "parser_output_dir": "config/opt-6.7b"},
    "llama": {"model_shape": "config/llama3-8b/shape.json", "parser_output_dir": "config/llama3-8b"},
    "palm": {"model_shape": "config/palm-8b/shape.json", "parser_output_dir": "config/palm-8b"}
}


dataset_config_for_ae = {
    "he": (157, 67),
    "sg": (783, 209),
    "lb": (1886, 97),
    "lg": (1971, 17)
}


def parse_args():
    parser = argparse.ArgumentParser("ISCA 2025 AE")
    parser.add_argument("--experiment-type", type=str, default="all", choices=["all", "e2e", "dataflow", "arch"])
    parser.add_argument("--python-env", type=str, default="h2llm")
    parser.add_argument("--dse-process-num", type=int, default=50)
    args = parser.parse_args()
    return args


def run_e2e(args):
    # Run CP
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[E2E] CP's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/CP/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[E2E] CP's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="0",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="2x_npu_only"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[E2E] CP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run ID-NMP
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[E2E] ID-NMP's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/ID-NMP/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[E2E] ID-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="8",
                    fpu_pe_bw_space="\"[(1, 200, 6.4)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[E2E] ID-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run ID-NMP+
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[E2E] ID-NMP+'s model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/ID-NMP+/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[E2E] ID-NMP+'s model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="8",
                    fpu_pe_bw_space="\"[(1, 1000, 6.4)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[E2E] ID-NMP+'s model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run H2-LLM
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[E2E] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/H2-LLM/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[E2E] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[E2E] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")


def run_dataflow_comparison(args):
    # Run Attn-NMP
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Dataflow] Attn-NMP's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/Attn-NMP/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Dataflow] Attn-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="attention_offloading"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Dataflow] Attn-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run Attn-NMP-Split
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Dataflow] Attn-NMP-Split's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/Attn-NMP-Split/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Dataflow] Attn-NMP-Split's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="attention_offloading_with_ffn_splitting"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Dataflow] Attn-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run FC-NMP
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Dataflow] FC-NMP's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/FC-NMP/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Dataflow] FC-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="fc_offloading"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Dataflow] Attn-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run CC-NMP
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Dataflow] CC-NMP's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/CC-NMP/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Dataflow] CC-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="specpim"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Dataflow] CC-NMP's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run H2-LLM
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Dataflow] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/H2-LLM/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Dataflow] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Dataflow] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")


def run_arch_dse(args):
    # Run H2-LLM
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Arch] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/H2-LLM/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Arch] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="6",
                    fpu_pe_bw_space="\"[(8, 600, 25.6)]\"",
                    is_device_fixed="--is-device-fixed",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Arch] H2-LLM's model {model_type} dataset {dataset} batch size {batch_size} ends.")

    # Run H2-LLM with full arch space
    for model_type in model_configs_for_ae.keys():
        for dataset in dataset_config_for_ae.keys():
            for batch_size in [1, 4, 16]:
                print(f"[Arch] H2-LLM-Full's model {model_type} dataset {dataset} batch size {batch_size} begins.")
                dse_output_dir = f"ae/results/H2-LLM-Full/{model_type}_{dataset}_{batch_size}"
                if os.path.exists(f"{dse_output_dir}/result.log"):
                    print(f"[Arch] H2-LLM-Full's model {model_type} dataset {dataset} batch size {batch_size} ends.")
                    continue
                cur_experiment_cmd = cmd.format(
                    env=args.python_env,
                    model_type=model_type,
                    model_shape=model_configs_for_ae[model_type]["model_shape"],
                    parser_output_dir=model_configs_for_ae[model_type]["parser_output_dir"],
                    max_batch_size=batch_size,
                    input_seq_len=dataset_config_for_ae[dataset][0],
                    max_gen_len=dataset_config_for_ae[dataset][1],
                    nmp_channel_num_space="2,4,6,8",
                    fpu_pe_bw_space="\"[(8, 400, 6.4),  (8, 600, 6.4),  (8, 800, 6.4),  (8, 1000, 6.4), "
                                   +"(8, 400, 12.8), (8, 600, 12.8), (8, 800, 12.8), (4, 1000, 12.8), "
                                   +"(8, 400, 25.6), (8, 600, 25.6), (4, 800, 25.6), (4, 1000, 25.6), "
                                   +"(8, 400, 51.2), (4, 600, 51.2), (4, 800, 51.2), (2, 1000, 51.2)]\"",
                    is_device_fixed="",
                    dse_output_dir=dse_output_dir,
                    process_num=args.dse_process_num,
                    mutate_ratio=1.0 if model_type=="palm" else 0.8,
                    is_dataflow_fixed="no"
                )
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(["bash", "-c", cur_experiment_cmd], env=env)
                print(f"[Arch] H2-LLM-Full's model {model_type} dataset {dataset} batch size {batch_size} ends.")


if __name__ == '__main__':
    args = parse_args()
    if args.experiment_type == "all":
        start_time = time.time()
        run_e2e(args)
        run_dataflow_comparison(args)
        run_arch_dse(args)
        end_time = time.time()
        print(f"time consumption {end_time-start_time}")
    elif args.experiment_type == "e2e":
        run_e2e(args)
    elif args.experiment_type == "dataflow":
        run_dataflow_comparison(args)
    elif args.experiment_type == "arch":
        run_arch_dse(args)
    else:
        raise ValueError(
            f"Wrong experiment type {args.experiment_type}, "
            "choices are: [\"all\", \"e2e\", \"dataflow\", \"arch\"]"
        )
