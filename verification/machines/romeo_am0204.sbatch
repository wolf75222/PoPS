#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-am24
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# Native smoke: AM-02 prescribed patch + AM-04 subcycling ratios 1/2/4 on Dim1.

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

echo "host=$(hostname) dim=1 suite=am02-am04"
"$POPS_PYTHON" - <<'PY'
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/scratch_p/rmdraux/pops-verify")
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from verification.pops_verify.case_authoring import load_sibling_module

def load(rel):
    return load_sibling_module(ROOT / rel)

print("=== AM-02 resolve ===")
am02 = load("verification/cases/amr/prescribed_patch/run.py")
plan = am02.resolve_plan(32)
print("AM-02 resolve", type(plan).__name__)

print("=== AM-02 native ===")
field = np.asarray(am02.run_native(32, t_end=0.25), dtype=np.float64)
print("AM-02", field.shape, "min", float(field.min()), "max", float(field.max()), "finite", bool(np.isfinite(field).all()))

am04 = load("verification/cases/amr/subcycling/run.py")
for ratio in (1, 2, 4):
    print(f"=== AM-04 ratio={ratio} resolve ===")
    plan = am04.resolve_plan(16, ratio=ratio)
    print("AM-04 resolve", ratio, type(plan).__name__)
    print(f"=== AM-04 ratio={ratio} native ===")
    field = np.asarray(am04.run_native(16, t_end=0.25, ratio=ratio), dtype=np.float64)
    print("AM-04", ratio, field.shape, "min", float(field.min()), "max", float(field.max()), "finite", bool(np.isfinite(field).all()))
print("DONE")
PY
echo DONE
