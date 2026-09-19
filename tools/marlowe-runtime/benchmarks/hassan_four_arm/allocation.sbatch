#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00

# Supply --partition, --job-name and a fresh --output path on the sbatch CLI.
# Run reproduce.py from the login shell after this explicit job is RUNNING.
set -euo pipefail
sleep infinity
