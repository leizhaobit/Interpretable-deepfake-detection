#!/bin/bash

#SBATCH --account=cuixiaohui
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00
module load nvidia/cuda/10.2


cd $SLURM_SUBMIT_DIR

python forge.py