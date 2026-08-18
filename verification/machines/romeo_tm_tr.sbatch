#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-tmtr
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# TM-02 retry + TM-04..08 + TR-04/TR-05 Dim1 smoke.

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

echo "host=$(hostname) dim=1 suite=tm-tr"
"$POPS_PYTHON" - <<'PY'
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/scratch_p/rmdraux/pops-verify")
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from verification.pops_verify.case_authoring import load_sibling_module

def load(rel):
    return load_sibling_module(ROOT / rel)

def report(name, field):
    arr = np.asarray(field, dtype=np.float64)
    print(name, arr.shape, "min", float(np.min(arr)), "max", float(np.max(arr)),
          "finite", bool(np.isfinite(arr).all()))

def run_one(name, fn):
    print(f"=== {name} ===")
    try:
        fn()
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)

def do_tm02():
    tm = load("verification/cases/time/noncommuting_strang/run.py")
    print("resolve", type(tm.resolve_plan(0.1)).__name__)
    report("TM-02", tm.run_native(0.1, t_end=0.2))

def do_tm04():
    tm = load("verification/cases/time/larmor/run.py")
    print("resolve", type(tm.resolve_plan(0.05)).__name__)
    report("TM-04", tm.run_native(0.05, t_end=0.25))

def do_tm05():
    tm = load("verification/cases/time/ap_limit/run.py")
    print("resolve", type(tm.resolve_plan(0.1, eps=0.1)).__name__)
    report("TM-05", tm.run_native(0.1, t_end=0.2, eps=0.1))

def do_tm06():
    tm = load("verification/cases/time/multirate/run.py")
    print("resolve", type(tm.resolve_plan(0.25, r=2)).__name__)
    report("TM-06", tm.run_native(0.25, r=2, t_end=0.25))

def do_tm08():
    tm = load("verification/cases/time/reversible_strang/run.py")
    print("resolve", type(tm.resolve_plan(0.05, n_cells=32)).__name__)
    report("TM-08", tm.run_native(0.05, t_end=0.25, n_cells=32))

def do_tr04():
    tr = load("verification/cases/transport/block_crossings/run.py")
    print("resolve", type(tr.resolve_plan(32)).__name__)
    report("TR-04", tr.run_native(32, t_end=0.25))

def do_tr05():
    tr = load("verification/cases/transport/block_shift/run.py")
    print("resolve", type(tr.resolve_plan(32)).__name__)
    report("TR-05", tr.run_native(32, t_end=0.25))

for name, fn in (
    ("TM-02", do_tm02),
    ("TM-04", do_tm04),
    ("TM-05", do_tm05),
    ("TM-06", do_tm06),
    ("TM-08", do_tm08),
    ("TR-04", do_tr04),
    ("TR-05", do_tr05),
):
    run_one(name, fn)
print("DONE")
PY
echo DONE
