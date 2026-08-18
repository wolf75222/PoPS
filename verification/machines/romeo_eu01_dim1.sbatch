#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-eu01
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
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python:${PYTHONPATH:-}
mkdir -p "$POPS_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

echo "host=$(hostname) dim=1"
"$POPS_PYTHON" - <<'PY'
import sys
from pathlib import Path
sys.path[:0] = [str(Path("/scratch_p/rmdraux/pops-verify") / p) for p in ("python", "")]
from verification.pops_verify.case_authoring import load_sibling_module
run = load_sibling_module(
    Path("/scratch_p/rmdraux/pops-verify/verification/cases/euler/linear_waves/run.py")
)
field = run.run_native(32, t_end=0.05, mode="entropy")
print("OK", field.shape, float(field[0].min()), float(field[0].max()))
PY
echo DONE
