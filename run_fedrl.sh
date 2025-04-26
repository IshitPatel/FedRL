#!/bin/bash
#SBATCH --job-name=fedrl_resnet
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1                 # Request 1 GPU (keep this minimal)
#SBATCH --cpus-per-task=2            # Reduce CPU to increase schedulability
#SBATCH --time=02:00:00              # Shorter time helps in scheduling
#SBATCH --output=logs/fedrl_%j.out
#SBATCH --error=logs/fedrl_%j.err

# Activate your virtual environment
source /Users/922917772/FCCL/.venv/bin/activate

# Navigate to your project directory
cd /Users/922917772/FCCL/FedRL

# Run your Python script
python main.py

