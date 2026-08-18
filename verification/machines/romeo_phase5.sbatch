#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-ph5
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
echo "host=$(hostname) suite=phase5-rb1d"
"$POPS_PYTHON" - <<'PY'
from pathlib import Path
import sys, numpy as np
ROOT = Path("/scratch_p/rmdraux/pops-verify")
sys.path[:0] = [str(ROOT/"python"), str(ROOT)]
from verification.pops_verify.case_authoring import load_sibling_module

def load(rel):
    return load_sibling_module(ROOT/rel)

def run_one(name, rel, n, t):
    print(f"=== {name} ===")
    try:
        run = load(rel)
        field = np.asarray(run.run_native(n, t_end=t), dtype=np.float64)
        print(name, field.shape, float(np.min(field)), float(np.max(field)),
              bool(np.isfinite(field).all()))
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)

run_one("RB-01", "verification/cases/robustness/sod/run.py", 32, 0.05)
run_one("RB-02", "verification/cases/robustness/double_rarefaction/run.py", 32, 0.05)
run_one("RB-03", "verification/cases/robustness/strong_shock/run.py", 32, 0.004)
run_one("RB-04", "verification/cases/robustness/shu_osher/run.py", 32, 0.2)
run_one("RB-09", "verification/cases/robustness/blast_waves/run.py", 64, 0.01)
run_one("RB-06", "verification/cases/robustness/noh/run.py", 32, 0.1)
print("DONE")
PY
echo DONE
