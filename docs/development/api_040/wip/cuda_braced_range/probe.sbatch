#!/bin/bash -l
#SBATCH --job-name=pops-cuda-braced-pair-6d898599
#SBATCH --account=r250127
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --constraint=armgpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1/slurm-%j.out
#SBATCH --error=/gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1/slurm-%j.err
set -euo pipefail
umask 077
module purge
module load cuda/12.6
export TMPDIR=/gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1/tmp
export XDG_CACHE_HOME=/gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec python3 /gpfs/scratch/rmdraux/PoPS-api040-cuda-braced-probe-6d898599-20261008-v1/probe.py
