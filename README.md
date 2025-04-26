# Artifact of H$^2$-LLM (ISCA '25)

This is the artifact of H$^2$-LLM's dataflow exploration framwork, which contains the implementation of both **compute-centric** dataflow exploration in SpecPIM (ASPLOS '24) and **data-centric** dataflow exploration in H$^2$-LLM (ISCA '25).

## Overview

[1. Installation](#1-installation)  
[2. Kick-the-Tires](#2-kick-the-tires)  
[3. Instructions for Artifact Evaluation](#3-instructions-for-artifact-evaluation)    
[4. Usage and Framework Extension](#4-usage-and-framework-extension)  
[5. Citation](#5-citation)

## 1. Installation

Please run the following commands to create a python environment and install required packages.

```bash
# Create environment
# We recommend using conda to manage python environment.
# If you have not been installed conda, please refer to https://www.anaconda.com/docs/getting-started/miniconda/main for more instructions.
conda create -n h2llm python=3.10 -y
conda activate h2llm
pip install -r requirements.txt

# Install domino
# domino adopts git SSH to fetch its submodules. Please refer to https://docs.github.com/en/authentication/connecting-to-github-with-ssh for more instructions with git SSH.
git submodule init
git submodule update --init --recursive
cd 3rd/domino
mkdir -p build && cd build
cmake ..
make
cd ../
pip install -r requirements.txt
cd python
python setup.py develop --no-deps
```

## 2. Kick-the-Tires

Then, you can run ```python main.py``` under the project folder to try our DSE framework, where we provide a toy example to conduct exploration on OPT model. After finishing exploration, the exemplified results can be found in ```kick_the_tires/``` folder.


## 3. Instructions for Artifact Evaluation

We have provided click-to-run scripts in the ```ae``` folder for ISCA '25 artifact evaluation. Please refer to the [README.md](./ae/README.md) in the `ae` folder for detailed instructions. 

The following section will introduce how to reuse this repository beyond the artifact evaluation.

## 4. Usage and Framework Extenstion

### Argument Configurations

You can run ```main.py``` and set arguments to set the simulation you want to run:

```
python main.py [--execution-type {parse,dse}] [--model-type {opt,llama,palm}] [--model-shape MODEL_SHAPE] 
               [--parser-output-dir PARSER_OUTPUT_DIR] [--max-batch-size MAX_BATCH_SIZE] [--input-seq-len INPUT_SEQ_LEN] 
               [--max-gen-len MAX_GEN_LEN] [--max-seq-len MAX_SEQ_LEN] [--precision {fp16,int8,int4}] 
               [--total-channel-num TOTAL_CHANNEL_NUM] [--bank-num-per-channel BANK_NUM_PER_CHANNEL] 
               [--bank-memory-capacity BANK_MEMORY_CAPACITY] [--fpu-simd-width FPU_SIMD_WIDTH] 
               [--nmp-channel-num-space NMP_CHANNEL_NUM_SPACE] [--fpu-pe-bw-space FPU_PE_BW_SPACE] 
               [--input-buffer-size-space INPUT_BUFFER_SIZE_SPACE] [--weight-buffer-size-space WEIGHT_BUFFER_SIZE_SPACE] 
               [--output-buffer-total-size-space OUTPUT_BUFFER_TOTAL_SIZE_SPACE] [--is-device-fixed] 
               [--dse-output-dir DSE_OUTPUT_DIR] [--process-num PROCESS_NUM] [--seed SEED] 
               [--population-num-per-generation POPULATION_NUM_PER_GENERATION] [--num-generations NUM_GENERATIONS] 
               [--mutate-ratio MUTATE_RATIO] [--topk TOPK] [--is-dataflow-fixed IS_DATAFLOW_FIXED]
```

- ```--execution-type```: ```parse``` means only executing model parsing. ```dse``` means executing dataflow DSE with the following settings.
- ```--model-type```: LLM layer operator graph type. Currently we support ```opt```, ```llama```, ```palm``` (parallel transformer). 
- ```--model-shape```: Model shape description json file. You can refer to ```config/*/shape.json``` for format example.
- ```--parser-output-dir```: Model parser's output directory.
- ```--max-batch-size```: Batch size for DSE evaluation.
- ```--input-seq-len```: Prompt length for DSE evaluation.
- ```--max-gen-len```: Decoding length for DSE evaluation.
- ```--max-seq-len```: Model's max context length.
- ```precision```: Data precision. Currently we support ```fp16```, ```int8```, ```int4```.
- ```--total-channel-num```: Total number of Normal & NMP channels.
- ```--bank-num-per-channel```: Bank number of each Normal/NMP channel.
- ```--bank-memory-capacity```: Each bank's memory capacity (MB).
- ```--fpu-simd-width```: Each FPU's MAC number.
- ```--nmp-channel-num-space```: A list of comma-separated integers, each of which represents one candidate of NMP channel number (should be no more than ```--total-channel-num```).
- ```--fpu-pe-bw-space```: A list of comma-separated tuples. Each tuple contains (NMP PE FPU number, NMP PE FPU frequency (MHz), NMP PE bandwidth (GB/s)).
- ```--input-buffer-size-space```: A list of comma-separated integers, each of which represents one candidate of input global buffer size (KB).
- ```--weight-buffer-size-space```: A list of comma-separated integers, each of which represents one candidate of PE's weight buffer size (KB).
- ```--output-buffer-total-size-space```: A list of comma-separated integers, each of which represents one candidate of output buffer's total size in one channel (KB).
- ```--is-device-fixed```: If set, the design space should only contain one device (i.e., each  space only has one element).
- ```--dse-output-dir```: DSE result's output directory.
- ```--process-num```: CPU process number used during DSE.
- ```--seed```: Random seed used during DSE.
- ```--population-num-per-generation```: Sampled population number in each DSE generation.
- ```--num-generations```: Evolution generation number during DSE.
- ```--mutate-ratio```: Mutate ratio during population generation.
- ```--topk```: Top-k number during population generation.
- ```--is-dataflow-fixed```: ```no``` means using non-constrained data-centric dataflow exploration. ```specpim``` means using compute-centric dataflow exploration.

### Adding New Models

There are two ways to adopt our DSE framework for other models:

- If your model operator graph is simple, you can directly edit a json file following the format of ```config/*/operator_graph.json``` for your own model. Please specify correct node id for attention operators (i.e., ```qk``` and  ```sv```).
- We also provide a simple onnx-based model parser to extract FC-GEMM and Attention-GEMM for complicated operator graphs. Please add your model definition in ```model_parser/model_definition.py``` and run ```main.py``` with the ```--execution-type``` of ```parse```, which will generate a ```.gv``` and a ```.png``` file for extracted operator graph. Then you can use these information to edit the ```operator_graph.json``` file. It is recommended to conduct parsing on a GPU server since torch.onnx needs to conduct model inference for graph extraction.

Apart from operator graph description, model shape description (e.g., head number, hidden dim, etc.) is also required to conduct DSE. Please refer to ```config/*/shape.json``` for examplified file format.

### Integrating New Centralized Processor or NMP Architectures 

The DSE framework can be extended to evaluate other centralized processor or NMP architectures. Please refer to the ```evaluate_performance``` function interface in ```nmp_evaluator.py``` and ```npu_evaluator.py``` to integrate your own simulator to the DSE framework. Specifically, the DSE framework provides the operator shape and channel number for performance evaluation. For centralized processor, channel number can be used to provide off-chip DRAM bandwidth. For NMP, channel number determines total computation resources for the NMP operator. You can adjust the inter-channel operator mapping and intra-channel execution flow according to your own design. You can also adjust the architecture parameters of these function interfaces for your own use.

## 5. Citation

If you use this project in your research, please cite our paper:

```bibtex
@inproceedings{specpim,
  title={SpecPIM: Accelerating speculative inference on PIM-enabled system via architecture-dataflow co-exploration},
  author={Li, Cong and Zhou, Zhe and Zheng, Size and Zhang, Jiaxi and Liang, Yun and Sun, Guangyu},
  booktitle={Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
  pages={950--965},
  year={2024}
}
```

```bibtex
@inproceedings{h2llm,
  title={H$^2$-LLM: Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference},
  author={Li, Cong and Yin, Yihan and Wu, Xintong and Zhu, Jingchen and Gao, Zhutianya and Niu, Dimin and Wu, Qiang and Si, Xin and Xie, Yuan and Zhang, Chen and Sun, Guangyu},
  booktitle={2025 ACM/IEEE 52st Annual International Symposium on Computer Architecture (ISCA)},
  year={2025}
}
```

