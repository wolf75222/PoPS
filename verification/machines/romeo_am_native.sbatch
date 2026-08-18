#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-amn
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# Native smoke: AM-01 static CF wave + AM-10 composite Poisson on Dim1.

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
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python:${PYTHONPATH:-}
mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

echo "host=$(hostname) dim=1 suite=am-native"
"$POPS_PYTHON" - <<'PY'
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/scratch_p/rmdraux/pops-verify")
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

def load(rel):
    return load_sibling_module(ROOT / rel)

print("=== AM-01 resolve ===")
am01 = load("verification/cases/amr/static_cf_wave/run.py")
plan = am01.resolve_plan(16)
print("AM-01 resolve", type(plan).__name__)

print("=== AM-01 native ===")
field = np.asarray(am01.run_native(16, t_end=0.25), dtype=np.float64)
print("AM-01", field.shape, "min", float(field.min()), "max", float(field.max()), "finite", bool(np.isfinite(field).all()))

print("=== AM-10 resolve ===")
am10 = load("verification/cases/amr/composite_poisson/run.py")
plan = am10.resolve_plan(16)
print("AM-10 resolve", type(plan).__name__)

print("=== AM-10 native ===")
phi = np.asarray(am10.run_native(16), dtype=np.float64)
print("AM-10", phi.shape, "phi_min", float(np.min(phi)), "phi_max", float(np.max(phi)), "finite", bool(np.isfinite(phi).all()))
print("DONE")
PY
echo DONE
