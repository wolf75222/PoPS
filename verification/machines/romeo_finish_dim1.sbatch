#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-dim1-fin
#SBATCH --output=/scratch_p/rmdraux/pops-verify/build.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/build.%J.err

# Incremental finish after PIC Kokkos + Dim1 link. Needs
# scripts/write_native_variant_manifest.py on the scratch tree.

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

ROOT=/scratch_p/rmdraux/pops-verify
BUILD=$ROOT/build-dim1
export POPS_NATIVE_DIM=1
export POPS_KOKKOS_ROOT=/scratch_p/rmdraux/kokkos-x64-pic
export POPS_INCLUDE=$ROOT/include
export OMP_NUM_THREADS=8
export CMAKE_PREFIX_PATH="$POPS_KOKKOS_ROOT:${CMAKE_PREFIX_PATH:-}"

echo "host=$(hostname) cxx=$CXX"
ls -l "$BUILD/python/pops/_native/dim1" || true
cmake --build "$BUILD" -j 8
echo BUILD_OK
if [ -f "$BUILD/python/pops/_native/variants.json" ]; then
  mkdir -p "$ROOT/python/pops/_native"
  rsync -a "$BUILD/python/pops/_native/" "$ROOT/python/pops/_native/"
fi
ls -l "$ROOT/python/pops/_native" "$ROOT/python/pops/_native/dim1"
cat "$ROOT/python/pops/_native/variants.json"
export PYTHONPATH=$ROOT/python:$ROOT:$BUILD/python:${PYTHONPATH:-}
"$POPS_PYTHON" "$ROOT/verification/machines/run_tr01_native.py"
echo DONE
