#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-dim2
#SBATCH --output=/scratch_p/rmdraux/pops-verify/build.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/build.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

ROOT=/scratch_p/rmdraux/pops-verify
BUILD=$ROOT/build-dim2
export POPS_NATIVE_DIM=2
export POPS_KOKKOS_ROOT=/scratch_p/rmdraux/kokkos-x64-pic
export POPS_INCLUDE=$ROOT/include
export POPS_CACHE_DIR=/scratch_p/rmdraux/pops-cache
export TMPDIR=/scratch_p/rmdraux/tmp
export CMAKE_PREFIX_PATH="$POPS_KOKKOS_ROOT:${CMAKE_PREFIX_PATH:-}"
mkdir -p "$POPS_CACHE_DIR" "$TMPDIR"

echo "host=$(hostname) cxx=$CXX dim=2"
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPOPS_NATIVE_DIM=2 \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI=OFF \
  -DKokkos_ROOT="$POPS_KOKKOS_ROOT" \
  -DPython_EXECUTABLE="$POPS_PYTHON" \
  -DPython3_EXECUTABLE="$POPS_PYTHON"
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-16}"
echo BUILD_OK
if [ -f "$BUILD/python/pops/_native/variants.json" ]; then
  # Merge dim2 variant next to dim1
  mkdir -p "$ROOT/python/pops/_native/dim2"
  rsync -a "$BUILD/python/pops/_native/dim2/" "$ROOT/python/pops/_native/dim2/" || true
  "$POPS_PYTHON" "$ROOT/scripts/write_native_variant_manifest.py" \
    --extension "$BUILD/python/pops/_native/dim2/"_pops.cpython-311-x86_64-linux-gnu.so \
    --manifest "$ROOT/python/pops/_native/variants.json" \
    --dimension 2 --version 1.0.0 || true
fi
ls -l "$ROOT/python/pops/_native" "$ROOT/python/pops/_native/dim2" 2>/dev/null || true
cat "$ROOT/python/pops/_native/variants.json"
echo DONE
