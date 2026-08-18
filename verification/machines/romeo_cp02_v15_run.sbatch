#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=popscp02-r
#SBATCH --output=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-cp02-v15/logs/%x.%j.err

set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-cp02-v15/src/verification/machines/romeo_cp02_v15_env.sh
set -eo pipefail

GATE="${POPS_CP02_GATE:-serial}"
SPACE=KokkosSerial
MPI=off
KIND=serial
NTASKS=1
OUT_GATE="$GATE"
if [ "$GATE" = "analyze" ]; then
  SPACE=KokkosSerial
  KIND=serial
  OUT_GATE=serial
elif [ "$GATE" = "temporal" ] || [ "$GATE" = "spatial" ] || [ "$GATE" = "parity" ]; then
  SPACE=KokkosSerial
  KIND=serial
elif [ "$GATE" = "openmp" ]; then
  SPACE=KokkosOpenMP
  KIND=openmp
  export OMP_NUM_THREADS=4
  export POPS_THREADS=4
elif [ "$GATE" = "mpi" ]; then
  MPI=on
  KIND=mpi
  NTASKS=2
  pops_cp02_load_mpi_hdf5
  export FI_PROVIDER=tcp
fi
pops_cp02_select_kokkos "$SPACE"
BUILD="$POPS_CP02_ROOT/build-dim1-${KIND}"
export POPS_NATIVE_DIM=1
export CMAKE_BUILD_TYPE=Release
pops_cp02_pythonpath "$BUILD"
pops_cp02_ensure_pydeps
OUT="$POPS_CP02_ROOT/evidence/${OUT_GATE}"
mkdir -p "$OUT"
cd "$POPS_CP02_SRC"
if [ "$GATE" = "mpi" ]; then
  unset FI_PROVIDER
  srun --mpi=pmi2 --ntasks="$NTASKS" --cpus-per-task=1 --cpu-bind=cores \
    "$POPS_PYTHON" verification/machines/run_cp02_v15.py --output "$OUT" --gate mpi
else
  "$POPS_PYTHON" verification/machines/run_cp02_v15.py --output "$OUT" --gate "$GATE"
fi
echo DONE
