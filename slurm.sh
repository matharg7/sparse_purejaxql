#!/bin/bash
#SBATCH --account=def-yani
#SBATCH --job-name=wandb_sweep_agent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=09:30:00
#SBATCH --array=1-5
#SBATCH --output=logs/def_yani_wandb_agent_%A_%a.log # Log file for each agent

# --- Environment Setup ---
echo "Starting Slurm job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# 1. Load necessary modules
module purge
module load python/3.10
module load cuda/12.2
echo "Modules loaded."

# 2. Set up virtual environment in temporary Slurm directory
# This is a good practice as $SLURM_TMPDIR is a fast, local scratch space
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Virtual environment activated."

# 3. Install project dependencies from requirements.txt
# This file is now the single source of truth for third-party packages.
pip install -r requirements.txt
echo "Dependencies installed."

# 4. Install your project in editable mode WITHOUT its dependencies
# The --no-deps flag prevents pip from re-installing dependencies listed
# in pyproject.toml, avoiding conflicts with requirements.txt.
pip install -e . --no-deps
echo "Project installed in editable mode."

# --- W&B Sweep Execution ---
# IMPORTANT: Replace YOUR_USERNAME/YOUR_PROJECT/YOUR_SWEEP_ID with your actual sweep ID.
# You can get this from the wandb UI.
export SWEEP_ID="ucalgary/pqn_atari_sweep_atari5/aik6cg5h"

echo "Starting wandb agent for sweep: $SWEEP_ID"

# The `wandb agent` command will run continuously, executing runs for the sweep.
# The Slurm job will end when the time limit is reached or the agent is stopped.
wandb agent $SWEEP_ID

echo "Job finished."

