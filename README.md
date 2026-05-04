# Attention Sinks and Outliers in Attention Residuals

This repository contains the implementation for the NeurIPS submission "Attention Sinks and Outliers in Attention Residuals". It includes training and evaluation code for language models and vision transformers, along with the attention-residual and quantization modules used in the experiments.

## repository layout

- `OutEffHop/`: Python package and runnable scripts. The main language-model entry points are `run_mlm.py`, `run_clm.py`, `run_clm_oasis.py`, and the matching validation scripts. `run_vit.py` and `validate_vit.py` cover the vision experiments.
- `OutEffHop/transformers_language/models/`: modified attention layers, OASIS variants, and quantized model wrappers.
- `OutEffHop/quantization/`: quantizers, range estimators, and wrappers used for post-training quantization experiments.
- `OutEffHop/model_configs/` and `OutEffHop/accelerate_configs/`: model and distributed training configs.
- `OutEffHop_script/`: SLURM scripts for reproducing the submitted experiments.

## setup

Create a Python environment with PyTorch built for your CUDA version, then install the package in editable mode:

```bash
cd OutEffHop
pip install -r requirement.txt
pip install -e .
```

The pinned environment is in `OutEffHop/requirement.txt`. If you install packages manually, the main dependencies are `torch`, `transformers`, `datasets`, `accelerate`, `timm`, `numpy`, `pandas`, `scipy`, `pyyaml`, `tqdm`, `wandb`, `einops`, and `triton`.

## running experiments

For cluster runs, start from the scripts in `OutEffHop_script/`. For local runs, use the Python entry points in `OutEffHop/` with the configs under `model_configs/` and `accelerate_configs/`.

Example:

```bash
cd OutEffHop
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
  --config_path model_configs/opt-350m.yaml \
  --dataset_setup wikitext_103 \
  --output_dir output/example
```

Checkpoints, logs, cached datasets, and metrics are not included in the repository. Point the scripts to your local data and model cache directories, or let Hugging Face download them during the first run.
