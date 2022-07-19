#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

bash scripts/gen_data.sh