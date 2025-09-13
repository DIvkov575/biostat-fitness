#!/bin/bash
#SBATCH --job-name=R0.4.1_array
#SBATCH --output=logs/gp_array_%A_%a.out
#SBATCH --error=logs/gp_array_%A_%a.err
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ../venv/bin/activate

# Define the grid
variance_values=(0.1 1.0 10.0)
lengthscale_values=(0.1 1.0 10.0)

# Map SLURM_ARRAY_TASK_ID to grid index
idx=$SLURM_ARRAY_TASK_ID
var_idx=$(( idx / 3 ))
ls_idx=$(( idx % 3 ))

var=${variance_values[$var_idx]}
ls=${lengthscale_values[$ls_idx]}

python R0.4.1.py --variance $var --lengthscale $ls

