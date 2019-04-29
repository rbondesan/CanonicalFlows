#!/bin/bash
#SBATCH -A LAMACRAFT-SL3-GPU
#SBATCH -p pascal
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3

module purge
module load rhel7/default-peta4
module load cuda/10.0 cudnn/7.4_cuda-10.0
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz

source activate tensorflow
python ./train.py