#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-ph2
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# Phase 1 leftovers + Phase 2 entry: TR-02, TM-01, PO-01, CP-02 on Dim1.

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

echo "host=$(hostname) dim=1"
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

print("=== TR-02 ===")
tr02 = load("verification/cases/transport/gaussian_pulse/run.py")
field = tr02.run_native(64, t_end=0.25)
print("TR-02", field.shape, float(field.min()), float(field.max()))

print("=== TM-01 ===")
tm01 = load("verification/cases/time/pure_temporal/run.py")
field = tm01.run_native(1.0 / 64.0, t_end=0.25, n_cells=64)
print("TM-01", field.shape, float(np.min(field)), float(np.max(field)))

print("=== PO-01 ===")
po01 = load("verification/cases/poisson/periodic_trig/run.py")
exact = load("verification/cases/poisson/periodic_trig/exact.py")
phi = po01.run_native(64)
centers, volumes = exact.uniform_cell_grid(64)
oracle = exact.phi_exact(centers)
# FFT gauge: subtract means before the norm
phi = np.asarray(phi, dtype=np.float64)
phi = phi - float(np.mean(phi))
oracle = oracle - float(np.mean(oracle))
err = reference_errors(phi, oracle, volumes)
print("PO-01", phi.shape, "L2", err.l2, "Linf", err.linf)

print("=== CP-02 ===")
cp02 = load("verification/cases/euler_poisson/langmuir_cold/run.py")
cp_exact = load("verification/cases/euler_poisson/langmuir_cold/exact.py")
cons = cp02.run_native(64, t_end=0.25)
n = cons[0]
centers, volumes = cp_exact.uniform_cell_centers(64)
n_ref = cp_exact.n_e(centers, 0.25)
err = reference_errors(n, n_ref, volumes)
print("CP-02", cons.shape, "n_L2", err.l2, "n_Linf", err.linf)
print("DONE")
PY
echo DONE
