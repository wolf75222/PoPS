#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --job-name=popscp02-w
#SBATCH --output=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.err

set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-cp02-v15/src/verification/machines/romeo_cp02_v15_env.sh
set -eo pipefail

pops_cp02_select_kokkos KokkosSerial
pops_cp02_load_mpi_hdf5
export FI_PROVIDER=tcp
MPICC="${MPI_HOME}/bin/mpicc"
MPICXX="${MPI_HOME}/bin/mpicxx"
CMAKE_C="$("$MPICC" -show | awk '{print $1}')"
CMAKE_CXX="$("$MPICXX" -show | awk '{print $1}')"

ROOT="$POPS_CP02_SRC"
BUILD="$POPS_CP02_ROOT/build-dim1-mpi-weno-halo"
export POPS_NATIVE_DIM=1
pops_cp02_pythonpath "$BUILD"
NPROC="${SLURM_CPUS_ON_NODE:-8}"
echo "host=$(hostname) sha=$(git -C "$ROOT" rev-parse HEAD) tests=ON target=test_mpi_generated_weno_halo"

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CMAKE_C}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPOPS_NATIVE_DIM=1 \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=ON \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI=ON \
  -DPOPS_USE_HDF5=ON \
  -DKokkos_ROOT="$POPS_KOKKOS_ROOT" \
  -DPython_EXECUTABLE="$POPS_PYTHON" \
  -DPython3_EXECUTABLE="$POPS_PYTHON" \
  -DMPI_CXX_COMPILER="$MPICXX" \
  -DMPI_C_COMPILER="$MPICC" \
  -DMPI_CXX_INCLUDE_DIRS="${MPI_HOME}/include" \
  -DMPI_CXX_HEADER_DIR="${MPI_HOME}/include" \
  -DMPI_C_INCLUDE_PATH="${MPI_HOME}/include" \
  -DMPI_CXX_LIBRARIES="${MPI_HOME}/lib/libmpi.so" \
  -DMPI_C_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
cmake --build "$BUILD" --target test_mpi_generated_weno_halo -j "$NPROC"

BIN="$(find "$BUILD" -type f -name test_mpi_generated_weno_halo -perm -111 | head -n 1)"
test -n "$BIN"
echo "binary=$BIN"
OUT="$POPS_CP02_ROOT/evidence/weno-halo"
mkdir -p "$OUT"
unset FI_PROVIDER
set +e
srun --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$BIN" --gtest_filter='*generated_weno_halo*'
NP1=$?
srun --mpi=pmi2 --ntasks=2 --cpus-per-task=1 --cpu-bind=cores "$BIN" --gtest_filter='*generated_weno_halo*'
NP2=$?
set -e
{
  echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "binary=$BIN"
  echo "np1_exit=$NP1"
  echo "np2_exit=$NP2"
} | tee "$OUT/result.txt"
echo WENO_HALO_NP1=$NP1
echo WENO_HALO_NP2=$NP2
test "$NP1" -eq 0
test "$NP2" -eq 0
echo DONE
