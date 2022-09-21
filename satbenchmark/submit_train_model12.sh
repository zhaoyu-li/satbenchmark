#!/bin/bash
#SBATCH --job-name=train_model12
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

python train_model.py satisfiability ~/scratch/satbenchmark/easy/sr/train --valid_dir ~/scratch/satbenchmark/easy/sr/valid --label satisfiability --graph lcg --updater mlp1 --aggregator sum --init_emb learned --n_iterations 16 --scheduler ReduceLROnPlateau