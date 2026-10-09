#!/bin/bash
#SBATCH --account=r250127
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
# Scheduler actual short24h limit; no extra per-command wall cap.
set -euo pipefail
umask 077
: "${FUTURE_GPU_ROOT:?Root supplies admitted exact private namespace}"
[[ "$SLURM_JOB_ACCOUNT" == r250127 ]]
cd "$FUTURE_GPU_ROOT"
source "$FUTURE_GPU_ROOT/private_runtime_context.sh"
# R5 resumes ONLY existing compiled probe, then Native; setup/envclone/Kokkos build never rerun.
bash "$FUTURE_GPU_ROOT/resume_probe_r5.sh"
bash "$FUTURE_GPU_ROOT/build_native_r5.sh"
[[ -f "$FUTURE_GPU_ROOT/results/gpu-native-admission-r5.json" ]]
