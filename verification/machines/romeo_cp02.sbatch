#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-cp02
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
export XDG_CACHE_HOME=/scratch_p/rmdraux/xdg-cache
export TMPDIR=/scratch_p/rmdraux/tmp
export OMP_NUM_THREADS=8
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python
mkdir -p "$POPS_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
echo "host=$(hostname)"
"$POPS_PYTHON" - <<'PY'
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/scratch_p/rmdraux/pops-verify")
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

cp02 = load_sibling_module(ROOT / "verification/cases/euler_poisson/langmuir_cold/run.py")
exact = load_sibling_module(ROOT / "verification/cases/euler_poisson/langmuir_cold/exact.py")
cons = cp02.run_native(64, t_end=0.25)
centers, volumes = exact.uniform_cell_centers(64)
err = reference_errors(cons[0], exact.n_e(centers, 0.25), volumes)
print("CP-02", cons.shape, "n_L2", err.l2, "n_Linf", err.linf)
print("DONE")
PY
echo DONE
