#!/bin/bash
#SBATCH -A siller
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

module load apptainer pytorch/2.9.0
apptainer run --nv $CONTAINERDIR/pytorch-2.9.0.sif ready_cnn.py
