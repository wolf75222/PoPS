#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-tm2o
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
export OMP_NUM_THREADS=8
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python
echo "host=$(hostname) suite=tm02-order"
"$POPS_PYTHON" "$ROOT/verification/machines/run_tm02_order.py"
echo DONE
