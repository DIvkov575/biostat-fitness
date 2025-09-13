#!/bin/bash
#SBATCH --job-name=R0.5.1
#SBATCH --output=logs/R0.5.1_%A_%a.out
#SBATCH --error=logs/R0.5.1_%A_%a.err
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ../venv/bin/activate

variance_values=(0.1 )
lengthscale_values=(0.1 )

# Map SLURM_ARRAY_TASK_ID to grid index
idx=$SLURM_ARRAY_TASK_ID
var_idx=$(( idx / 3 ))
ls_idx=$(( idx % 3 ))

var=${variance_values[$var_idx]}
ls=${lengthscale_values[$ls_idx]}

#/usr/bin/time -v python R0.5.1.py --variance $var --lengthscale $ls > resource_log_R0.5.1_%j.txt



