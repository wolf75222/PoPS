#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops676-kser
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# Pure KokkosSerial PIC prefix. OpenMP stays off so DefaultExecutionSpace is Serial.
set -eo pipefail
set +e
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

SCRATCH=/scratch_p/rmdraux
KOKKOS_SRC=$SCRATCH/src/kokkos
KOKKOS_PREFIX=$SCRATCH/kokkos-x64-pic-serial
BUILD=$SCRATCH/build-kokkos-pic-serial
NPROC="${SLURM_CPUS_PER_TASK:-16}"

echo "host=$(hostname) cxx=$CXX prefix=$KOKKOS_PREFIX"
mkdir -p "$SCRATCH/src" "$SCRATCH/pops-676/logs"
if [ ! -d "$KOKKOS_SRC/.git" ]; then
  git clone --depth 1 --branch 4.5.01 https://github.com/kokkos/kokkos.git "$KOKKOS_SRC"
fi

cmake -S "$KOKKOS_SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ARCH_ZEN4=ON
cmake --build "$BUILD" -j "$NPROC"
cmake --install "$BUILD"
test -f "$KOKKOS_PREFIX/lib64/libkokkoscore.a"
echo KOKKOS_SERIAL_PIC_OK
ls -l "$KOKKOS_PREFIX/lib64" "$KOKKOS_PREFIX/lib" 2>/dev/null | head
echo DONE
