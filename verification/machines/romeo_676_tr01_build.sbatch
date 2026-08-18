#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops676-t1b
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# Isolated Serial, OpenMP, or MPI TR-01 leaf. Dim 1/2/3. No exclusive node. no GPU.
echo "starting job=${SLURM_JOB_ID:-local} host=$(hostname)"
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-676/src/verification/machines/romeo_676_env.sh
set -eo pipefail

DIM="${POPS676_DIM:?set POPS676_DIM to 1, 2, or 3}"
MPI="${POPS676_MPI:-off}"
SPACE="${POPS676_SPACE:-KokkosSerial}"
case "$DIM" in
  1|2|3) ;;
  *) echo "ERROR: POPS676_DIM must be 1, 2, or 3" >&2; exit 1 ;;
esac
case "$MPI" in
  off|on) ;;
  *) echo "ERROR: POPS676_MPI must be off or on" >&2; exit 1 ;;
esac
case "$SPACE" in
  KokkosSerial|KokkosOpenMP) ;;
  *) echo "ERROR: POPS676_SPACE must be KokkosSerial or KokkosOpenMP" >&2; exit 1 ;;
esac
if [ "$MPI" = "on" ] && [ "$SPACE" != "KokkosSerial" ]; then
  echo "ERROR: MPI TR-01 leaf is Serial-space only" >&2
  exit 1
fi

KIND=serial
CMAKE_MPI=OFF
CMAKE_HDF5=OFF
EXPECT_BACKEND=(--expect-serial)
CMAKE_MPI_FLAGS=()
CMAKE_C="${CC:-cc}"
CMAKE_CXX="${CXX:-c++}"
pops676_select_kokkos "$SPACE"
pops676_require_pic_kokkos
if [ "$MPI" = "on" ]; then
  KIND=mpi
  CMAKE_MPI=ON
  CMAKE_HDF5=ON
  EXPECT_BACKEND=(--expect-mpi --expect-parallel-hdf5)
  pops676_load_mpi_hdf5
  export FI_PROVIDER=tcp
  MPICC="${MPI_HOME}/bin/mpicc"
  MPICXX="${MPI_HOME}/bin/mpicxx"
  if [ ! -x "$MPICC" ] || [ ! -x "$MPICXX" ]; then
    echo "ERROR: mpicc/mpicxx missing under ${MPI_HOME}/bin" >&2
    exit 1
  fi
  CMAKE_C="$("$MPICC" -show | awk '{print $1}')"
  CMAKE_CXX="$("$MPICXX" -show | awk '{print $1}')"
  CMAKE_MPI_FLAGS=(
    -DMPI_CXX_COMPILER="$MPICXX"
    -DMPI_C_COMPILER="$MPICC"
    -DMPI_CXX_INCLUDE_DIRS="${MPI_HOME}/include"
    -DMPI_C_INCLUDE_PATH="${MPI_HOME}/include"
    -DMPI_CXX_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
    -DMPI_C_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
  )
elif [ "$SPACE" = "KokkosOpenMP" ]; then
  KIND=openmp
fi

ROOT="$POPS676_SRC"
BUILD="$POPS676_ROOT/build-dim${DIM}-${KIND}"
export POPS_NATIVE_DIM="$DIM"
pops676_pythonpath "$BUILD"
NPROC="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname) dim=$DIM kind=$KIND cmake_cxx=$CMAKE_CXX kokkos=$POPS_KOKKOS_ROOT"
echo "build=$BUILD python=$POPS_PYTHON sha=$(git -C "$ROOT" rev-parse HEAD)"

if [ "$MPI" = "on" ] && [ ! -f "$BUILD/build.ninja" ]; then
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

if [ ! -f "$BUILD/python/pops/_native/variants.json" ]; then
  leaf="$BUILD/python/pops/_native/dim${DIM}/_pops$(
    "$POPS_PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  )"
  echo "retrying native variant authentication for $leaf"
  "$POPS_PYTHON" "$ROOT/scripts/write_native_variant_manifest.py" \
    --extension "$leaf" \
    --manifest "$BUILD/python/pops/_native/variants.json" \
    --dimension "$DIM" \
    --version 1.0.0
fi
test -f "$BUILD/python/pops/_native/variants.json"
"$POPS_PYTHON" "$ROOT/scripts/verify_installed_native.py" \
  --expect-dim "$DIM" \
  "${EXPECT_BACKEND[@]}" \
  --json
echo AUTH_OK
echo DONE
