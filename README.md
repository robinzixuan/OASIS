<div align="center">


# OASIS: Null-Aware Attention Residuals for Outlier-Efficient Large Language Models

<!-- TODO: replace the paper link once available -->
[![arXiv](https://img.shields.io/badge/arXiv-OASIS-ff0000.svg?style=for-the-badge)](#)  [![Github](https://img.shields.io/badge/OASIS-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/robinzixuan/OASIS)
</div>



The repo contains:
1. **Official Implementation**
   The official implementation of OASIS and AoS, which couples the per-token *null posterior* of `Softmax_1` attention with depth-wise **Attention Residuals**.

2. **Supported Backbones**
   OASIS decoder layers for `Llama`, `Qwen3` and `Phi-4`.

3. **Training Code**
   Causal LM continued pre-training with OASIS (`run_clm_oasis.py`) and vanilla/outlier-free baselines (`run_clm.py`).

4. **Quantization & Outlier Testing Code**
   Post-training W*n*A*n* quantization and outlier statistics (activation ∞-norm and kurtosis) via `validate_*.py`.

## Contents

- [OASIS](#oasis-null-aware-attention-residuals-for-outlier-efficient-large-language-models)
  - [1. Introduction](#1-introduction)
  - [2. Environment setup](#2-environment-setup)
  - [3. Repository structure](#3-repository-structure)
  - [4. Training](#4-training)
    - [4.1 OASIS training](#41-oasis-training)
    - [4.2 Baselines](#42-baselines)
  - [5. Evaluation & Quantization](#5-evaluation--quantization)
  - [6. Key arguments](#6-key-arguments)
  - [7. Citation](#7-citation)
  - [8. Acknowledgement](#8-acknowledgement)

## 1. Introduction

Large Transformers develop massive activation outliers because attention heads that "want to do nothing" dump their probability mass on sink tokens. Outlier-efficient attention (`Softmax_1`, from OutEffHop) fixes this by adding a null slot to the softmax denominator:

$$\mathrm{Softmax}_1(x)_i = \frac{\exp(x_i)}{1 + \sum_j \exp(x_j)}$$

The mass routed to the null slot, $1 - \sum_j a_{ij}$, is a per-token signal of *how much a layer abstains*. OASIS turns this signal into a routing prior across depth:

1. **Null posterior.** Each attention layer computes the per-head null mass and averages it over heads into a branch-level statistic $\psi_{l,t}$.
2. **Attention Residuals.** The fixed residual $h_l = h_{l-1} + f_l(h_{l-1})$ is replaced by a learned attention over *all* previous layer outputs (including the embedding), applied independently per token.
3. **OASIS coupling.** The depth-routing logits are shifted by the centered null statistic,
   $$g^{\text{new}}_{l,t} = g^{\text{old}}_{l,t} - \beta\,(\psi_{l,t} - \bar\psi_t), \qquad \beta = \mathrm{softplus}(\beta_{\text{raw}}),$$
   so layers that abstained on a token receive less weight when the hidden state is aggregated. $\beta$ is initialized near zero, which makes OASIS a near no-op at the start of fine-tuning.

## 2. Environment setup

    # create and activate virtual python environment
    conda create -n oasis python=3.10
    conda activate oasis

    # install required packages
    cd source_code
    pip install -r requirement.txt

> **Note:** the OASIS modules (`*_oasis_attention.py`) import `transformers.masking_utils` and `GradientCheckpointingLayer`, so they need a recent `transformers` release, newer than the version pinned in `requirement.txt`. `vutils/softmax_1.py` imports `triton`, so a CUDA machine is required.

## 3. Repository structure

```
OASIS/
├── source_code/
│   ├── run_clm_oasis.py              # OASIS training (Llama / Qwen3 / Phi-4)
│   ├── run_clm.py, run_clm_ddp.py    # causal LM baselines
│   ├── validate_clm.py               # evaluation + quantization
│   ├── transformers_language/
│   │   ├── models/
│   │   │   ├── llama_oasis_attention.py
│   │   │   ├── qwen_oasis_attention.py
│   │   │   ├── phi4_oasis_attention.py
│   │   │   ├── softmax.py            # SOFTMAX_MAPPING (vanilla, softmax1, clipped, ...)
│   │   │   └── quantized_*.py
│   │   ├── args.py
│   │   └── dataset_setups.py         # wikitext_2 | wikitext_103 | bookcorpus_and_wiki
│   ├── quantization/                 # fake-quant utilities
│   ├── vutils/softmax_1.py           # Softmax_1 (+ Triton flash-attention kernel)
│   ├── accelerate_configs/
│   └── model_configs/
└── scripts/                          # Slurm submission scripts
```

## 4. Training

All commands below are run from `source_code/`.

### 4.1 OASIS training

OASIS requires `--attn_softmax softmax1`. With `vanilla` softmax the null posterior is identically zero, so the model reduces to plain Attention Residuals.

```bash
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_oasis.py \
  --pad_to_max_length \
  --wd_LN_gamma \
  --with_tracking \
  --report_to wandb \
  --run_name oasis_llama3_1b \
  --seed 1000 \
  --dataset_setup bookcorpus_and_wiki \
  --preprocessing_num_workers 10 \
  --data_cache_dir your/path/to/.hf_data \
  --model_cache_dir your/path/to/.hf_cache \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --tokenizer_name meta-llama/Llama-3.2-1B \
  --max_seq_length 2048 \
  --block_size 512 \
  --learning_rate 0.0004 \
  --lr_scheduler_type linear \
  --max_train_steps 2000 \
  --num_warmup_steps 2000 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 6 \
  --gradient_accumulation_steps 32 \
  --max_grad_norm 1.0 \
  --weight_decay 0.1 \
  --checkpointing_steps 500 \
  --attn_softmax softmax1 \
  --attn_res_softmax_fn vanilla \
  --max_checkpointing_number 2 \
  --output_dir your/path/to/save
```

Replace the model with `Qwen/Qwen3-0.6B` or `microsoft/Phi-4-mini-instruct` to train the other backbones. If no model is given, the script defaults to `microsoft/Phi-4-mini-instruct`.

On a Slurm cluster, adjust the account, partition and paths in the scripts and submit from the repo root:

```bash
sbatch scripts/submit_phi4_oasis_train.sh     # Phi-4 + OASIS
sbatch scripts/submit_phi4_vanilla_train.sh   # Phi-4 vanilla baseline
```

### 4.2 Baselines

```bash
# Causal LM (Llama / Qwen3) without OASIS coupling: same arguments as 4.1, using run_clm_ddp.py
sh run_qwen.sh

# OPT outlier experiments
sh ../scripts/submit_outlier_opt.sh
```

Choose the attention variant with `--attn_softmax` (`vanilla`, `softmax1`, `clipped(...)`, `clippedsoftmax1(...)`; see `transformers_language/models/softmax.py`).

## 5. Evaluation & Quantization

To evaluate perplexity and outlier statistics (max activation ∞-norm and kurtosis), run:

```bash
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
  --seed 1000 \
  --dataset_setup wikitext_2 \
  --block_size 512 \
  --per_device_eval_batch_size 4 \
  --attn_softmax softmax1 \
  --attn_res_softmax_fn vanilla \
  --data_cache_dir your/path/to/.hf_data \
  --model_cache_dir your/path/to/.hf_cache \
  --model_name_or_path your/path/to/checkpoint \
  --output_dir your/path/to/metrics
```

To run W**n**A**n** (Weights-**n**bit, Activations-**n**bit) post-training quantization, add:

```bash
--quantize \
--n_bits n \
--n_bits_act n
```

On Slurm: `sbatch scripts/submit_phi4_validate.sh` (Phi-4 quantization is not yet implemented, so this script runs without `--quantize`). For OPT, use `scripts/submit_outlier_valid_opt.sh`.

## 6. Key arguments

| Argument | Description |
| --- | --- |
| `--attn_softmax` | Softmax used inside token attention. Must be `softmax1` for OASIS to have a non-zero null posterior. |
| `--attn_res_softmax_fn` | Softmax used for the depth-wise Attention Residual aggregation. |
| `--dataset_setup` | `wikitext_2`, `wikitext_103` or `bookcorpus_and_wiki`. |
| `--block_size` | Training / evaluation sequence length after grouping. |
| `--quantize`, `--n_bits`, `--n_bits_act` | Enable post-training quantization and set its bit-widths. |

## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use OASIS in your work, please kindly cite our paper:

**OASIS**

```
TODO: add BibTeX once the paper is public.
```

**OutEffHop**

```
@inproceedings{hu2024outlier,
  title={Outlier-Efficient Hopfield Layers for Large Transformer-Based Models},
  author={Jerry Yao-Chieh Hu and Pei-Hsuan Chang and Haozheng Luo and Hong-Yu Chen and Weijian Li and Wei-Po Wang and Han Liu},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

## 8. Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
- OutEffHop (https://github.com/MAGICS-LAB/OutEffHop)
- Outlier-free Transformers (https://github.com/Qualcomm-AI-research/outlier-free-transformers)
- GERM (https://github.com/MAGICS-LAB/GERM)
- Hugging Face Transformers (https://github.com/huggingface/transformers)
