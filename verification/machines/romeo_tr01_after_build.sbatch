#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-tr01b
#SBATCH --output=/scratch_p/rmdraux/pops-verify/tr01b.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/tr01b.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

ROOT=/scratch_p/rmdraux/pops-verify
export POPS_NATIVE_DIM=1
export POPS_KOKKOS_ROOT="${KOKKOS_CPU_ROOT:-$HOME/kokkos-x64}"
export POPS_INCLUDE=$ROOT/include
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python:${PYTHONPATH}
export OMP_NUM_THREADS=8

echo "host=$(hostname)"
ls -l "$ROOT/python/pops/_native" "$ROOT/build-dim1/python/pops/_native" 2>/dev/null || true
"$POPS_PYTHON" "$ROOT/verification/machines/run_tr01_native.py"
echo DONE
