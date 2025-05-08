#!/bin/bash

#SBATCH --job-name=fedrl_cpu
#SBATCH --partition=cpucluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/fedRLcifar_cpu_%j.out
#SBATCH --error=logs/fedRLcifar_cpu_%j.err

# Activate virtual environment
source /Users/922917772/FCCL/.venv/bin/activate

# Navigate to project directory
cd /Users/922917772/FCCL/FedRL

# Run the Python script (CPU-only)
python main.py
