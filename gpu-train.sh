#!/bin/bash
#SBATCH -A LAMACRAFT-SL3-GPU
#SBATCH -p pascal
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

module purge
module load rhel7/default-peta4
module load cuda/10.0 cudnn/7.4_cuda-10.0
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz

source activate tensorflow
python ./circle_train.py --hamiltonian="kepler" --d=3 --num_particles=1 --trajectory_duration=256 --resample_trajectories=False --hparams=minibatch_size=32