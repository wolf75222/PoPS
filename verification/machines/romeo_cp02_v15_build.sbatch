#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=popscp02-b
#SBATCH --output=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.err

set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-cp02-v15/src/verification/machines/romeo_cp02_v15_env.sh
set -eo pipefail

SPACE="${POPS_CP02_SPACE:-KokkosSerial}"
MPI="${POPS_CP02_MPI:-off}"
pops_cp02_select_kokkos "$SPACE"
KIND=serial
CMAKE_MPI=OFF
CMAKE_HDF5=OFF
EXPECT_BACKEND=(--expect-serial)
CMAKE_C="${CC:-cc}"
CMAKE_CXX="${CXX:-c++}"
CMAKE_MPI_FLAGS=()
if [ "$MPI" = "on" ]; then
  KIND=mpi
  CMAKE_MPI=ON
  CMAKE_HDF5=ON
  EXPECT_BACKEND=(--expect-mpi)
  pops_cp02_load_mpi_hdf5
  export FI_PROVIDER=tcp
  MPICC="${MPI_HOME}/bin/mpicc"
  MPICXX="${MPI_HOME}/bin/mpicxx"
  CMAKE_C="$("$MPICC" -show | awk '{print $1}')"
  CMAKE_CXX="$("$MPICXX" -show | awk '{print $1}')"
  CMAKE_MPI_FLAGS=(
    -DMPI_CXX_COMPILER="$MPICXX"
    -DMPI_C_COMPILER="$MPICC"
    -DMPI_CXX_INCLUDE_DIRS="${MPI_HOME}/include"
    -DMPI_CXX_HEADER_DIR="${MPI_HOME}/include"
    -DMPI_C_INCLUDE_PATH="${MPI_HOME}/include"
    -DMPI_CXX_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
    -DMPI_C_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
  )
elif [ "$SPACE" = "KokkosOpenMP" ]; then
  KIND=openmp
fi

ROOT="$POPS_CP02_SRC"
BUILD="$POPS_CP02_ROOT/build-dim1-${KIND}"
export POPS_NATIVE_DIM=1
pops_cp02_pythonpath "$BUILD"
NPROC="${SLURM_CPUS_PER_TASK:-8}"
echo "host=$(hostname) kind=$KIND kokkos=$POPS_KOKKOS_ROOT sha=$(git -C "$ROOT" rev-parse HEAD)"
if [ "$MPI" = "on" ] && [ ! -f "$BUILD/build.ninja" ]; then
  rm -rf "$BUILD"
fi
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CMAKE_C}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPOPS_NATIVE_DIM=1 \
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
if [ ! -f "$BUILD/python/pops/_native/variants.json" ]; then
  leaf="$BUILD/python/pops/_native/dim1/_pops$(
    "$POPS_PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  )"
  "$POPS_PYTHON" "$ROOT/scripts/write_native_variant_manifest.py" \
    --extension "$leaf" \
    --manifest "$BUILD/python/pops/_native/variants.json" \
    --dimension 1 \
    --version 1.0.0
fi
test -f "$BUILD/python/pops/_native/variants.json"
"$POPS_PYTHON" "$ROOT/scripts/verify_installed_native.py" \
  --expect-dim 1 \
  "${EXPECT_BACKEND[@]}" \
  --json
echo AUTH_OK
# Header-only finalize exception contract. Does not enable the full C++ suite.
cat > "$TMPDIR/test_finalize_inner.cpp" <<'EOF'
#include <pops/runtime/native_package_finalize.hpp>
#include <exception>
#include <stdexcept>
#include <string>
int main() {
  try {
    pops::rethrow_native_package_finalize_failure(
        std::make_exception_ptr(std::runtime_error("inner bind/fft")));
    return 2;
  } catch (const std::runtime_error& error) {
    if (error.what() != std::string("inner bind/fft"))
      return 3;
  }
  try {
    pops::rethrow_native_package_finalize_failure({});
    return 4;
  } catch (const std::runtime_error& error) {
    if (error.what() != std::string(pops::kNativePackageFinalizeCollectiveMessage))
      return 5;
  }
  return 0;
}
EOF
"${CMAKE_CXX}" -std=c++20 -I "$ROOT/include" "$TMPDIR/test_finalize_inner.cpp" -o "$TMPDIR/test_finalize_inner"
"$TMPDIR/test_finalize_inner"
echo FINALIZE_TEST_OK
echo DONE
