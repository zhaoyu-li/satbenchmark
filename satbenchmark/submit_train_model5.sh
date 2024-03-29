#!/bin/bash
#SBATCH --job-name=train_model5
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

python train_model.py satisfiability ~/scratch/satbenchmark/easy/3-sat/train --valid_dir ~/scratch/satbenchmark/easy/3-sat/valid --label satisfiability --graph vcg --updater gru --aggregator sum --init_emb learned --n_iterations 32