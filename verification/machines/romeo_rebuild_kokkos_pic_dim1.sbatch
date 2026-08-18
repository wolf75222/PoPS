#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-kokkos-pic
#SBATCH --output=/scratch_p/rmdraux/pops-verify/build.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/build.%J.err

# Rebuild Kokkos with -fPIC on scratch, then Dim1 _pops.so + TR-01 smoke.
# $HOME/kokkos-x64 is a non-PIC static archive and cannot link a Python module.

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

SCRATCH=/scratch_p/rmdraux
ROOT=$SCRATCH/pops-verify
KOKKOS_SRC=$SCRATCH/src/kokkos
KOKKOS_PREFIX=$SCRATCH/kokkos-x64-pic
BUILD=$ROOT/build-dim1
NPROC="${SLURM_CPUS_PER_TASK:-16}"

export POPS_NATIVE_DIM=1
export POPS_KOKKOS_ROOT="$KOKKOS_PREFIX"
export POPS_INCLUDE=$ROOT/include
export OMP_NUM_THREADS=8
export CMAKE_PREFIX_PATH="$KOKKOS_PREFIX:${CMAKE_PREFIX_PATH:-}"

echo "host=$(hostname) cxx=$CXX python=$POPS_PYTHON"
echo "kokkos_src=$KOKKOS_SRC prefix=$KOKKOS_PREFIX"

mkdir -p "$SCRATCH/src"
if [ ! -d "$KOKKOS_SRC/.git" ]; then
  git clone --depth 1 --branch 4.5.01 https://github.com/kokkos/kokkos.git "$KOKKOS_SRC"
fi

cmake -S "$KOKKOS_SRC" -B "$SCRATCH/build-kokkos-pic" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ARCH_ZEN4=ON
cmake --build "$SCRATCH/build-kokkos-pic" -j "$NPROC"
cmake --install "$SCRATCH/build-kokkos-pic"
echo KOKKOS_PIC_OK
ls -l "$KOKKOS_PREFIX/lib64" "$KOKKOS_PREFIX/lib" 2>/dev/null | head

rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPOPS_NATIVE_DIM=1 \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI=OFF \
  -DKokkos_ROOT="$POPS_KOKKOS_ROOT" \
  -DPython_EXECUTABLE="$POPS_PYTHON" \
  -DPython3_EXECUTABLE="$POPS_PYTHON"
cmake --build "$BUILD" -j "$NPROC"
echo BUILD_OK

# Prefer the in-tree module next to variants.json
if [ -f "$BUILD/python/pops/_native/variants.json" ]; then
  mkdir -p "$ROOT/python/pops/_native"
  rsync -a "$BUILD/python/pops/_native/" "$ROOT/python/pops/_native/"
fi
ls -l "$ROOT/python/pops/_native" "$ROOT/python/pops/_native/dim1" 2>/dev/null || true
cat "$ROOT/python/pops/_native/variants.json" 2>/dev/null || \
  cat "$BUILD/python/pops/_native/variants.json"

export PYTHONPATH=$ROOT/python:$ROOT:$BUILD/python:${PYTHONPATH:-}
"$POPS_PYTHON" "$ROOT/verification/machines/run_tr01_native.py"
echo DONE
