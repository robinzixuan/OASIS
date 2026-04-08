#!/bin/bash
#===============================================================================
# Phi-3 / Phi-4 文本模型 — validate_clm（走 phi4_attention + replace_attention_modules）
#
# 你可能还要改的地方：
#   1) 下面 module load：已与 Quest 上 module avail 里「全名」对齐；若 PyTorch 是 cu118 请改用 cuda/11.8 那行
#   2) PHI4_MODEL、SCRATCH_ROOT：见下方「用户配置」
#
# 若每次 module avail 都出现 luac: ... spiderT... unexpected symbol：
#   退出登录后执行  rm -rf ~/.cache/lmod  再重新 ssh（损坏的是本机 Lmod 缓存，不是集群）
#
# 模型用 Hub 还是本地？脚本里 PHI4_MODEL 默认是 Hugging Face Hub ID。
#   - 能联网且接受默认小模型：不用改。
#   - 本地已有权重目录：export PHI4_MODEL=/projects/p32013/.../your_model_dir
#     （该目录里应有 config.json、tokenizer 与权重；与 from_pretrained 要求一致）
#
# 提交（在任意目录均可）：
#   sbatch /path/to/OASIS/OutEffHop_script/submit_phi4_validate.sh
#===============================================================================

#SBATCH -A p32013
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=phi4-validate
#SBATCH --output=slurm_phi4_validate_%j.out
#SBATCH --error=slurm_phi4_validate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onionzsy@umich.edu

set -euo pipefail

#--------------- 用户配置（至少改 PHI4_MODEL；建议改 SCRATCH_ROOT）---------------
PHI4_MODEL="${PHI4_MODEL:-microsoft/Phi-3-mini-4k-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
# 试跑可把数据集换成 wikitext_2、block_size 改小
DATASET_SETUP="${DATASET_SETUP:-bookcorpus_and_wiki}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
EVAL_BS="${EVAL_BS:-4}"
ATTN_SOFTMAX="${ATTN_SOFTMAX:-vanilla}"
ATTN_RES_SOFTMAX="${ATTN_RES_SOFTMAX:-vanilla}"
SEED="${SEED:-5678}"
#------------------------------------------------------------------------------

module purge 2>/dev/null || true

# Conda 初始化（与 mentor 脚本一致）。若你登录后已有 conda，可注释掉下一行避免重复。
module load python-miniconda3/4.12.0

# CUDA toolkit：你的环境为 torch 2.7+cu128（自带 CUDA 12.8 运行时），以下 module 主要提供
# CUDA_HOME / nvcc；与 12.8 不必逐位相同。按 Quest 上 module avail 选「最新的 12.x」即可。
# 顺序：先试 rhel8 的 12.6，再 12.4，再 rhel7 spack 的 12.1（哪个成功用哪个）。
if ! module load cuda/12.6.2-gcc-12.4.0 2>/dev/null; then
  if ! module load cuda/12.4.1-gcc-12.3.0 2>/dev/null; then
    module load cuda/12.1.0-gcc-11.2.0 || true
  fi
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate outlier

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/OutEffHop" || { echo "Cannot cd to ${REPO_ROOT}/OutEffHop"; exit 1; }

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"

DATA_CACHE="${SCRATCH_ROOT}/.hf_data"
MODEL_CACHE="${SCRATCH_ROOT}/.hf_cache"
OUT_DIR="${SCRATCH_ROOT}/output_metrics/phi4_validate_${SLURM_JOB_ID:-local}"
mkdir -p "${DATA_CACHE}" "${MODEL_CACHE}" "$(dirname "${OUT_DIR}")"

# 评测：单卡、无混合精度（与 mentor 的 validate 脚本一致，数值更稳）
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
  --seed "${SEED}" \
  --dataset_setup "${DATASET_SETUP}" \
  --preprocessing_num_workers 8 \
  --block_size "${BLOCK_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --attn_softmax "${ATTN_SOFTMAX}" \
  --attn_res_softmax_fn "${ATTN_RES_SOFTMAX}" \
  --data_cache_dir "${DATA_CACHE}" \
  --model_cache_dir "${MODEL_CACHE}" \
  --model_name_or_path "${PHI4_MODEL}" \
  --output_dir "${OUT_DIR}"

echo "Done. Metrics under: ${OUT_DIR}"
