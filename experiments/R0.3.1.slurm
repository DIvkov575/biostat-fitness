#!/bin/bash
#SBATCH --job-name=R0.3.1 # Job name
#SBATCH --output=logs/%x_%j.out       # Output file (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err        # Error file
#SBATCH --time=02:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=main# Partition/queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=1             # CPUs per task
#SBATCH --mem=40G                      # Memory per node

# Load Python (depends on your cluster)
#module load python/3.10

# Run Python script
srun python R0.3.1.py

