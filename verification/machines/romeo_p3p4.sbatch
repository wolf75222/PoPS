#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-p34
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# AM-11, AM-12, PO-06 + Phase 4 CP-05 / TM-02 / TM-03 on Dim1.

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

echo "host=$(hostname) dim=1 suite=p3p4"
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

def do_am11():
    am11 = load("verification/cases/amr/ep_sync/run.py")
    print("resolve", type(am11.resolve_plan(16)).__name__)
    report("AM-11", am11.run_native(16, t_end=0.05))

def do_am12():
    am12 = load("verification/cases/amr/patch_symmetry/run.py")
    print("resolve", type(am12.resolve_plan(16)).__name__)
    report("AM-12", am12.run_native(16, t_end=0.10))

def do_po06():
    po06 = load("verification/cases/poisson/cf_gradient/run.py")
    print("resolve", type(po06.resolve_plan(16)).__name__)
    report("PO-06", po06.run_native(16))

def do_cp05():
    cp05 = load("verification/cases/euler_poisson/multifluid_modes/run.py")
    print("resolve", type(cp05.resolve_plan(16, mode="plus")).__name__)
    report("CP-05", cp05.run_native(16, t_end=0.10, mode="plus"))

def do_tm02():
    tm02 = load("verification/cases/time/noncommuting_strang/run.py")
    print("resolve", type(tm02.resolve_plan(0.1)).__name__)
    report("TM-02", tm02.run_native(0.1, t_end=0.2))

def do_tm03():
    tm03 = load("verification/cases/time/collision_relax/run.py")
    print("resolve", type(tm03.resolve_plan(0.05)).__name__)
    report("TM-03", tm03.run_native(0.05, t_end=0.2))

for name, fn in (
    ("AM-11", do_am11),
    ("AM-12", do_am12),
    ("PO-06", do_po06),
    ("CP-05", do_cp05),
    ("TM-02", do_tm02),
    ("TM-03", do_tm03),
):
    run_one(name, fn)
print("DONE")
PY
echo DONE
