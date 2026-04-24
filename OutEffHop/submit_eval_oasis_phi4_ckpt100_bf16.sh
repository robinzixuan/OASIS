#!/bin/bash
#SBATCH --account=p32013
#SBATCH --job-name=eval-o-phi4-c100-bf16
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/p32013/residual/OutEffHop/slurm_eval_oasis_phi4_ckpt100_bf16_%j.out
#SBATCH --error=/gpfs/projects/p32013/residual/OutEffHop/slurm_eval_oasis_phi4_ckpt100_bf16_%j.err

cd /gpfs/projects/p32013/residual/OutEffHop
module purge 2>/dev/null || true
module load python-miniconda3/4.12.0
module load cuda/12.6.2-gcc-12.4.0 2>/dev/null || module load cuda/12.4.1-gcc-12.3.0 2>/dev/null || module load cuda/12.1.0-gcc-11.2.0 || true
eval "$(conda shell.bash hook)"
conda activate outlier

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 "${CONDA_PREFIX}/bin/python" -m accelerate.commands.launch \
  --mixed_precision bf16 \
  --config_file accelerate_configs/1gpu_fp16.yaml validate_clm.py \
  --dataset_setup wikitext_2 \
  --preprocessing_num_workers 8 \
  --per_device_eval_batch_size 1 \
  --block_size 256 \
  --model_name_or_path /scratch/$USER/residual/output/oasis_phi4_bf16_128_s200_20260414_214508/checkpoints/checkpoint_100 \
  --tokenizer_name microsoft/Phi-4-mini-instruct \
  --config_name microsoft/Phi-4-mini-instruct \
  --output_dir /scratch/$USER/residual/eval/oasis_phi4_ckpt100_bf16_$(date +%Y%m%d_%H%M%S)
