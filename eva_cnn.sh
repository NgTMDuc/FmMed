#!/bin/bash
#SBATCH --job-name=ducntm       # Job name
#SBATCH --output=../log/results.txt      # Output file
#SBATCH --error=../log/error_training.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=40G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd /home/user01/aiotlab/ducntm/FmMed/
python3 evaluate.py --config config/CtViT.yaml