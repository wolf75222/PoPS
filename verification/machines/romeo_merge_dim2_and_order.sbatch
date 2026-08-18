#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-ord-fix
#SBATCH --output=/scratch_p/rmdraux/pops-verify/order.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/order.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

ROOT=/scratch_p/rmdraux/pops-verify
export POPS_KOKKOS_ROOT=/scratch_p/rmdraux/kokkos-x64-pic
export POPS_INCLUDE=$ROOT/include
export POPS_CACHE_DIR=/scratch_p/rmdraux/pops-cache
export POPS_CODEGEN_DIR=/scratch_p/rmdraux/pops-cache/codegen
export XDG_CACHE_HOME=/scratch_p/rmdraux/xdg-cache
export TMPDIR=/scratch_p/rmdraux/tmp
export POPS_ORDER_OUT=$ROOT/verification/out/native_order
export POPS_ORDER_OMP=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$POPS_ORDER_OUT"

# Merge Dim2 into variants.json next to Dim1
SO2=$(ls "$ROOT/build-dim2/python/pops/_native/dim2/"_pops.cpython-311-*.so | head -1)
"$POPS_PYTHON" "$ROOT/scripts/write_native_variant_manifest.py" \
  --extension "$SO2" \
  --manifest "$ROOT/python/pops/_native/variants.json" \
  --dimension 2 --version 1.0.0
rsync -a "$ROOT/build-dim2/python/pops/_native/dim2/" "$ROOT/python/pops/_native/dim2/"
cat "$ROOT/python/pops/_native/variants.json"

export POPS_NATIVE_DIM=1
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim1/python:${PYTHONPATH:-}
echo "=== DIM1 ORDER Serial/OpenMP ==="
"$POPS_PYTHON" "$ROOT/verification/machines/run_native_order.py"

export POPS_NATIVE_DIM=2
export PYTHONPATH=$ROOT/python:$ROOT:$ROOT/build-dim2/python:${PYTHONPATH:-}
echo "=== DIM2 DOCTOR ==="
"$POPS_PYTHON" -c "import pops; print(pops.__file__); print(pops.doctor())"
echo DONE
