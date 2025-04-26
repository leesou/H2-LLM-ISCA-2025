# Instructions for ISCA-2025 Artifact Evaluation

## 1. Getting Started

Before running experiments, please follow the instructions in project [README.md](../README.md) (Section 1) to install all required packages. The experiments are conducted on a server with the following configurations. Note that these experiments do not depend on specific GPU/GPUs. However, We recommend to utilize a CPU server with more than 50 cores for the subsequent experiments to guarantee their completion within the estimated timeframe.

> * OS: Ubuntu 20.04
> * CPU: Intel(R) Xeon(R) Gold 6226R CPU (dual-socket, 64 cores)
> * Memory: 384GB DDR4 memory


## 2. Run Experiments 

### 2.1. Main Results All-in-one (~9 Hours)

Please run the following commands to run all experiments:

```bash
# in project folder
python ae/run_experiments.py --experiment-type all

# in project folder
python ae/plot_results.py --experiment-type all
```

After finishing, all plotted results are saved in the ```ae/plots``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

The following description introduces how to run each experiment separately, which can be skipped if you have run the all-in-one commands. Note that some experiments share the same DSE results. Therefore, by reusing these results, the all-in-one script's time consumption is shorter than the sum of each experiment's time consumption.

### 2.2. End-to-end Latency Comparison (~6 Hours)

Please run the following commands to run all experiments:

```bash
# in project folder
python ae/run_experiments.py --experiment-type e2e

# in project folder
python ae/plot_results.py --experiment-type e2e
```

After finishing, all plotted results are saved in the ```ae/plots``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

### 2.3. Dataflow DSE Analysis (~3.5 Hours)

Please run the following commands to run all experiments:

```bash
# in project folder
python ae/run_experiments.py --experiment-type dataflow

# in project folder
python ae/plot_results.py --experiment-type dataflow
```

After finishing, all plotted results are saved in the ```ae/plots``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

### 2.4. Architecture Exploration Analysis (~3.5 Hours)

Please run the following commands to run all experiments:

```bash
# in project folder
python ae/run_experiments.py --experiment-type arch

# in project folder
python ae/plot_results.py --experiment-type arch
```

After finishing, all plotted results are saved in the ```ae/plots``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

## 3. Validate Results

Due to the privacy issue of our industry collaborator, we cannot directly provide the simulator, so we adopt a rooflined-based evaluator for result reproduction. Therefore, we only conduct artifact evaluation for latency comparision, and the reproduced results can be slightly different from that in the paper. However, the difference does not affect these reproduced results to prove H$^2$-LLM's superior performance against all baselines. For reference, we also provide pre-run results in the ```ae/plots_ref``` folder, which has the same organization as the ```ae/plots``` folder.

### 3.1 End-to-end Latency Comparison

Each model's latency comparison result is depicted in ```ae/plots/e2e/{model-name}.png```, which corresponds to the upper-half of Figure 10 (end to end latency comparison).

### 3.2 Dataflow Design Comparison

Each model's dataflow comparison results on H$^2$-LLM are stored in ```ae/plots/dataflow/{model-name}.png```, which corresponds to the results in Figure 12 (comparison against existing dataflow designs). Each model's dataflow design ablation study against ID-NMP+ are stored in ```ae/plots/dataflow_ablation/{model-name}.png```, which corresponds to the results in Figure 13 (dataflow designs on H$^2$-LLM v.s. ID-NMP+). Each model's prefill latency ratio and prefill speedup results are stored in ```ae/plots/dataflow/{prefill-breakdown}.png```, which corresponds to the results in Figure 14 (prefill latency ratio and prefill speedup).

### 3.3 Architecture Exploration Analysis

Each model's architecture DSE results are stored in ```ae/plots/arch_dse/{model-name}.png```, which corresponds to the results in Figure 15 (performance analysis of architecture DSE).
