#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops676-b
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# Isolated Serial or MPI+collective-HDF5 build. No exclusive node. no GPU.
echo "starting job=${SLURM_JOB_ID:-local} host=$(hostname)"
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-676/src/verification/machines/romeo_676_env.sh
set -eo pipefail
pops676_require_pic_kokkos

DIM="${POPS676_DIM:?set POPS676_DIM to 1 or 2}"
MPI="${POPS676_MPI:-off}"
case "$DIM" in
  1|2) ;;
  *) echo "ERROR: POPS676_DIM must be 1 or 2" >&2; exit 1 ;;
esac
case "$MPI" in
  off|on) ;;
  *) echo "ERROR: POPS676_MPI must be off or on" >&2; exit 1 ;;
esac
if [ "$MPI" = "on" ] && [ "$DIM" != "1" ]; then
  echo "ERROR: Step 4 MPI artifact is Dim1 only" >&2
  exit 1
fi

KIND=serial
CMAKE_MPI=OFF
CMAKE_HDF5=OFF
EXPECT_BACKEND=(--expect-serial)
CMAKE_MPI_FLAGS=()
CMAKE_C="${CC:-cc}"
CMAKE_CXX="${CXX:-c++}"
if [ "$MPI" = "on" ]; then
  KIND=mpi
  CMAKE_MPI=ON
  CMAKE_HDF5=ON
  EXPECT_BACKEND=(--expect-mpi --expect-parallel-hdf5)
  pops676_load_mpi_hdf5
  # FindMPI leaves include/lib identity empty when the wrapper is already
  # CMAKE_CXX_COMPILER. Interrogate mpicc/mpicxx and compile with gcc/g++.
  CMAKE_C="$(mpicc -show | awk '{print $1}')"
  CMAKE_CXX="$(mpicxx -show | awk '{print $1}')"
  CMAKE_MPI_FLAGS=(
    -DMPI_CXX_COMPILER="$(command -v mpicxx)"
    -DMPI_C_COMPILER="$(command -v mpicc)"
  )
  if [ ! -f "${MPI_HOME}/include/mpi.h" ] || [ ! -f "${MPI_HOME}/lib/libmpi.so" ]; then
    echo "ERROR: MPICH include/libmpi.so missing under ${MPI_HOME}" >&2
    exit 1
  fi
fi

ROOT="$POPS676_SRC"
BUILD="$POPS676_ROOT/build-dim${DIM}-${KIND}"
export POPS_NATIVE_DIM="$DIM"
pops676_pythonpath "$BUILD"
NPROC="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname) dim=$DIM kind=$KIND cmake_cxx=$CMAKE_CXX kokkos=$POPS_KOKKOS_ROOT"
echo "build=$BUILD python=$POPS_PYTHON sha=$(git -C "$ROOT" rev-parse HEAD)"

if [ "$MPI" = "on" ] && [ ! -f "$BUILD/python/pops/_native/variants.json" ]; then
  rm -rf "$BUILD"
fi
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CMAKE_C}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPOPS_NATIVE_DIM="$DIM" \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI="$CMAKE_MPI" \
  -DPOPS_USE_HDF5="$CMAKE_HDF5" \
  -DKokkos_ROOT="$POPS_KOKKOS_ROOT" \
  -DPython_EXECUTABLE="$POPS_PYTHON" \
  -DPython3_EXECUTABLE="$POPS_PYTHON" \
  "${CMAKE_MPI_FLAGS[@]}"
cmake --build "$BUILD" -j "$NPROC"
echo BUILD_OK

test -f "$BUILD/python/pops/_native/variants.json"
"$POPS_PYTHON" "$ROOT/scripts/verify_installed_native.py" \
  --expect-dim "$DIM" \
  "${EXPECT_BACKEND[@]}" \
  --json
echo AUTH_OK
echo DONE
