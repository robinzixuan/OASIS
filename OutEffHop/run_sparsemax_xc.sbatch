#!/bin/bash
#SBATCH --account=p32013
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --job-name=sparsemax
#SBATCH --output=log_sparsemax.out

module load cuda/12.6.2-gcc-12.4.0

source /software/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate outlier

cd /projects/p32013/residual/OutEffHop

bash run_sparsemax_xc.sh

