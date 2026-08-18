#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-cpn
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

# Native smoke: CP-03 / CP-07 / CP-08 / CP-12 on Dim1.

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

echo "host=$(hostname) dim=1 suite=cp-native"
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

print("=== CP-03 ===")
cp03 = load("verification/cases/euler_poisson/langmuir_warm/run.py")
exact03 = load("verification/cases/euler_poisson/langmuir_warm/exact.py")
cons = cp03.run_native(32, t_end=0.25, cycles=1)
centers, volumes = exact03.uniform_cell_centers(32)
n_ref = exact03.exact_fields(centers, 0.25, k=exact03.wavenumber(1))["n_e"]
err = reference_errors(cons[0], n_ref, volumes)
print("CP-03", cons.shape, "n_L2", err.l2, "n_Linf", err.linf)

print("=== CP-07 ===")
cp07 = load("verification/cases/euler_poisson/pressure_balance/run.py")
cons = cp07.run_native(32, t_end=0.05)
u_max = float(np.max(np.abs(cons[1] / cons[0])))
print("CP-07", cons.shape, "u_max", u_max)

print("=== CP-08 ===")
cp08 = load("verification/cases/euler_poisson/uniform_e_accel/run.py")
exact08 = load("verification/cases/euler_poisson/uniform_e_accel/exact.py")
cons = cp08.run_native(32, t_end=0.25)
u_num = cons[1] / cons[0]
u_ref = exact08.velocity(0.25)
print("CP-08", cons.shape, "u_mean", float(np.mean(u_num)), "u_ref", float(u_ref))

print("=== CP-12 ===")
cp12 = load("verification/cases/euler_poisson/charge_cancel/run.py")
phi = cp12.run_native(32, delta=0.1)
phi = np.asarray(phi, dtype=np.float64)
phi = phi - float(np.mean(phi))
print("CP-12", phi.shape, "phi_Linf", float(np.max(np.abs(phi))))
print("DONE")
PY
echo DONE
