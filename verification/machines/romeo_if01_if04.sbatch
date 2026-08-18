#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=00:50:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-if14
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true
ROOT=/scratch_p/rmdraux/pops-verify
export POPS_NATIVE_DIM=1
export POPS_KOKKOS_ROOT=/scratch_p/rmdraux/kokkos-x64-pic
export POPS_INCLUDE=$ROOT/include
export POPS_CACHE_DIR=/scratch_p/rmdraux/pops-cache
export POPS_CODEGEN_DIR=/scratch_p/rmdraux/pops-cache/codegen
export XDG_CACHE_HOME=/scratch_p/rmdraux/xdg-cache
export TMPDIR=/scratch_p/rmdraux/tmp
export OMP_NUM_THREADS=4
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python
echo "host=$(hostname) suite=if01-if04"
"$POPS_PYTHON" "$ROOT/verification/machines/run_if01_if04.py"
echo DONE
