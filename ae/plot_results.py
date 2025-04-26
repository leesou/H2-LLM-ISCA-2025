import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("ISCA 2025 AE")
    parser.add_argument("--experiment-type", type=str, default="all", choices=["all", "e2e", "dataflow", "arch"])
    parser.add_argument("--result-dir", type=str, default="ae/results")
    parser.add_argument("--output-dir", type=str, default="ae/plots")
    args = parser.parse_args()
    return args


models = ["opt", "llama", "palm"]
batch_sizes = [1, 4, 16]
datasets = [
    ("he", "HE", "HumanEval"),
    ("sg", "SG", "ShareGPT"),
    ("lb", "LB", "LongBench"),
    ("lg", "LG", "LooGLE")
]


def geometric_mean(arr):
    arr = np.asarray(arr, dtype=float)
    return np.exp(np.mean(np.log(arr)))


def plot_e2e(args):
    os.makedirs(f"{args.output_dir}/e2e", exist_ok=True)
    designs = ["CP", "ID-NMP", "ID-NMP+", "H2-LLM"]
    for design in designs:
        if not os.path.exists(f"{args.result_dir}/{design}"):
            raise ValueError(f"{design}'s experiments have not been conducted")
    
    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            bs = batch_sizes[i]
            x = np.arange(len(datasets))
            results = [[] for _ in range(len(designs))]
            for j, design in enumerate(designs):
                for dataset in datasets:
                    with open(f"{args.result_dir}/{design}/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                        lines = f.readlines()
                        latency = float(lines[0].split(" ")[4])
                        results[j].append(latency)
            norm_results = [results[2][k] for k in range(len(results[2]))]
            for j in range(len(results)):
                for k in range(len(datasets)):
                    results[j][k] = min(4.8, norm_results[k] / results[j][k])
            width = 0.2
            axes[i].bar(x-width*2, results[0], width=width, label=designs[0])
            axes[i].bar(x-width, results[1], width=width, label=designs[1])
            axes[i].bar(x, results[2], width=width, label=designs[2])
            axes[i].bar(x+width, results[3], width=width, label=designs[3])
            axes[i].set_xticks(x)
            axes[i].set_yticks(np.arange(6))
            axes[i].set_xticklabels([dataset[1] for dataset in datasets])
            axes[i].set_xlabel(f"Batch Size {bs}")
            axes[i].legend()
        fig.tight_layout()
        plt.savefig(f"{args.output_dir}/e2e/{model}.png")
        plt.close()


def plot_dataflow_comparison(args):
    os.makedirs(f"{args.output_dir}/dataflow", exist_ok=True)
    designs = ["Attn-NMP", "Attn-NMP-Split", "FC-NMP", "CC-NMP", "H2-LLM"]
    for design in designs:
        if not os.path.exists(f"{args.result_dir}/{design}"):
            raise ValueError(f"{design}'s experiments have not been conducted")
    utilization = [0.85, 0.85, 0.95, 0.9, 0.95]
    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            bs = batch_sizes[i]
            x = np.arange(len(datasets))
            results = [[] for _ in range(len(designs))]
            for j, design in enumerate(designs):
                for dataset in datasets:
                    with open(f"{args.result_dir}/{design}/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                        lines = f.readlines()
                        latency = float(lines[0].split(" ")[4])
                        results[j].append(latency/utilization[j])
            norm_results = [results[2][k] for k in range(len(results[2]))]
            for j in range(len(results)):
                for k in range(len(datasets)):
                    results[j][k] = norm_results[k] / results[j][k]
            width = 0.15
            axes[i].bar(x-width*2, results[0], width=width, label=designs[0])
            axes[i].bar(x-width, results[1], width=width, label=designs[1])
            axes[i].bar(x, results[2], width=width, label=designs[2])
            axes[i].bar(x+width, results[3], width=width, label=designs[3])
            axes[i].bar(x+width*2, results[4], width=width, label=designs[4])
            axes[i].set_xticks(x)
            axes[i].set_yticks(np.arange(5))
            axes[i].set_xticklabels([dataset[1] for dataset in datasets])
            axes[i].set_xlabel(f"Batch Size {bs}")
            axes[i].legend()
        fig.tight_layout()
        plt.savefig(f"{args.output_dir}/dataflow/{model}.png")
        plt.close()


def plot_dataflow_ablation(args):
    os.makedirs(f"{args.output_dir}/dataflow_ablation", exist_ok=True)
    designs = ["ID-NMP+", "Attn-NMP", "Attn-NMP-Split", "FC-NMP", "CC-NMP", "H2-LLM"]
    for design in designs:
        if not os.path.exists(f"{args.result_dir}/{design}"):
            raise ValueError(f"{design}'s experiments have not been conducted")
    
    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            bs = batch_sizes[i]
            x = np.arange(len(designs)-1)
            dataflow_results = []
            baseline_results = []
            for design in designs:
                tmp_results = []
                for dataset in datasets:
                    with open(f"{args.result_dir}/{design}/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                        lines = f.readlines()
                        latency = float(lines[0].split(" ")[4])
                        tmp_results.append(latency)
                if design == "ID-NMP+":
                    baseline_results = tmp_results
                else:
                    dataflow_results.append(tmp_results)
            norm_results = dataflow_results.copy()
            for j in range(len(norm_results)):
                for k in range(len(baseline_results)):
                    norm_results[j][k] = baseline_results[k] / norm_results[j][k]
            geomean_results = [geometric_mean(result) for result in norm_results]
            axes[i].bar(x, geomean_results)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([design for design in designs[1:]])
            axes[i].set_xlabel(f"Batch Size {bs}")
            axes[i].set_yticks(np.array([0, 1, 2, 3, 4]))
            axes[i].tick_params(axis='x', labelrotation=30)
        fig.tight_layout()
        plt.savefig(f"{args.output_dir}/dataflow_ablation/{model}.png")
        plt.close()


def plot_prefill_decoding_breakdown(args):
    os.makedirs(f"{args.output_dir}/prefill_breakdown", exist_ok=True)
    designs = ["CC-NMP", "H2-LLM"]
    for design in designs:
        if not os.path.exists(f"{args.result_dir}/{design}"):
            raise ValueError(f"{design}'s experiments have not been conducted")
    
    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            bs = batch_sizes[i]
            x = np.arange(len(datasets))
            prefill_ratio = []
            cc_nmp_prefill = []
            h2_llm_prefill = []
            prefill_speedup = []
            for dataset in datasets:
                with open(f"{args.result_dir}/CC-NMP/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                    lines = f.readlines()
                    prefill_latency = float(lines[1].split(" ")[3])
                    cc_nmp_prefill.append(prefill_latency)
                with open(f"{args.result_dir}/H2-LLM/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                    lines = f.readlines()
                    latency = float(lines[0].split(" ")[4])
                    prefill_latency = float(lines[1].split(" ")[3])
                    prefill_ratio.append(prefill_latency / latency)
                    h2_llm_prefill.append(prefill_latency)
            prefill_speedup = [x/y for x, y in zip(cc_nmp_prefill, h2_llm_prefill)]
            axes[i].bar(x, prefill_ratio, label="Prefill Ratio")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([dataset[1] for dataset in datasets])
            axes[i].set_xlabel(f"Batch Size {bs}")
            axes[i].set_yticks(np.array([0, 0.25, 0.5, 0.75, 1]))
            axes[i].tick_params(axis='y', labelrotation=90)
            ax2 = axes[i].twinx()
            ax2.plot(x, prefill_speedup, marker="o", color="orange", label="Prefill Speedup")
            ax2.set_yticks(np.array([0, 0.5, 1, 1.5, 2]))
            ax2.tick_params(axis='y', labelrotation=90)
            handles1, labels1 = axes[i].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            axes[i].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        fig.tight_layout()
        plt.savefig(f"{args.output_dir}/prefill_breakdown/{model}.png")
        plt.close()


def plot_arch_dse(args):
    os.makedirs(f"{args.output_dir}/arch_dse", exist_ok=True)
    designs = ["H2-LLM", "H2-LLM-Full"]
    for design in designs:
        if not os.path.exists(f"{args.result_dir}/{design}"):
            raise ValueError(f"{design}'s experiments have not been conducted")
    
    for model in models:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i in range(3):
            bs = batch_sizes[i]
            x = np.arange(len(datasets))
            fixed_latency = []
            full_latency = []
            for dataset in datasets:
                with open(f"{args.result_dir}/H2-LLM/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                    lines = f.readlines()
                    latency = float(lines[0].split(" ")[4])
                    fixed_latency.append(latency)
                with open(f"{args.result_dir}/H2-LLM-Full/{model}_{dataset[0]}_{bs}/result.log", "r") as f:
                    lines = f.readlines()
                    latency = float(lines[0].split(" ")[4])
                    full_latency.append(latency)
            speedup = [min(2.2, x/y) for x, y in zip(fixed_latency, full_latency)]
            axes[i].bar(x, speedup)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([dataset[1] for dataset in datasets])
            axes[i].set_xlabel(f"Batch Size {bs}")
            axes[i].set_yticks(np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5]))
        fig.tight_layout()
        plt.savefig(f"{args.output_dir}/arch_dse/{model}.png")
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    if args.experiment_type == "all":
        plot_e2e(args)
        plot_dataflow_comparison(args)
        plot_dataflow_ablation(args)
        plot_prefill_decoding_breakdown(args)
        plot_arch_dse(args)
    elif args.experiment_type == "e2e":
        plot_e2e(args)
    elif args.experiment_type == "dataflow":
        plot_dataflow_comparison(args)
        plot_dataflow_ablation(args)
        plot_prefill_decoding_breakdown(args)
    elif args.experiment_type == "arch":
        plot_arch_dse(args)
    else:
        raise ValueError(
            f"Wrong experiment type {args.experiment_type}, "
            "choices are: [\"all\", \"e2e\", \"dataflow\", \"arch\"]"
        )
