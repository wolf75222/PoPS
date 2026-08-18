#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-dim1
#SBATCH --output=/scratch_p/rmdraux/pops-verify/build.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/build.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

ROOT=/scratch_p/rmdraux/pops-verify
BUILD=$ROOT/build-dim1
export POPS_NATIVE_DIM=1
export POPS_KOKKOS_ROOT="${KOKKOS_CPU_ROOT:-$HOME/kokkos-x64}"
export POPS_INCLUDE=$ROOT/include
export OMP_NUM_THREADS=8

echo "host=$(hostname) cxx=$CXX kokkos=$POPS_KOKKOS_ROOT"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DPOPS_NATIVE_DIM=1 \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI=OFF \
  -DKokkos_ROOT="$POPS_KOKKOS_ROOT" \
  -DPython_EXECUTABLE="$POPS_PYTHON" \
  -DPython3_EXECUTABLE="$POPS_PYTHON"
cmake --build "$BUILD" -j 16
echo BUILD_OK
ls -l "$ROOT/python/pops/_native" 2>/dev/null || ls -l "$BUILD/python/pops/_native" 2>/dev/null || true

export PYTHONPATH=$ROOT/python:$ROOT:${BUILD}/python:${PYTHONPATH}
"$POPS_PYTHON" "$ROOT/verification/machines/run_tr01_native.py"
echo DONE
