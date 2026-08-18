#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pops-eu02-r
#SBATCH --output=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.err

# EU-02 v1.5 campaign. STAGE=series|temporal|spatial|tvd|smoke|dump|parity|analyze
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-v15-eu02/src/verification/machines/romeo_v15_eu02_env.sh
set -eo pipefail

STAGE="${POPS_EU02_STAGE:?set POPS_EU02_STAGE}"
SPACE="${POPS_EU02_SPACE:-KokkosSerial}"
MPI="${POPS_EU02_MPI:-off}"
DIM=2
case "$STAGE" in
  series|temporal|spatial|tvd|smoke|dump|parity|analyze) ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 1 ;;
esac

KIND=serial
if [ "$MPI" = "on" ]; then
  KIND=mpi
  pops_eu02_load_mpi_hdf5
  export FI_PROVIDER=tcp
elif [ "$SPACE" = "KokkosOpenMP" ]; then
  KIND=openmp
fi
pops_eu02_select_kokkos "$SPACE"
pops_eu02_require_pic_kokkos
BUILD="$POPS_EU02_ROOT/build-dim${DIM}-${KIND}"
export POPS_NATIVE_DIM="$DIM"
export POPS_NATIVE_VARIANTS_ROOT="$BUILD/python/pops/_native"
if [ "$SPACE" = "KokkosOpenMP" ]; then
  export OMP_NUM_THREADS=4
  export POPS_THREADS=4
else
  export OMP_NUM_THREADS=1
  export POPS_THREADS=1
fi
pops_eu02_pythonpath "$BUILD"
GATE="EU02_D${DIM}_${SPACE}_${MPI}_${STAGE}"
OUT="$POPS_EU02_ROOT/evidence/${GATE}"
mkdir -p "$OUT"
echo "host=$(hostname) stage=$STAGE space=$SPACE mpi=$MPI sha=$(git -C "$POPS_EU02_SRC" rev-parse HEAD)"
echo "build=$BUILD out=$OUT"

if [ "$MPI" = "on" ]; then
  srun --mpi=pmi2 --ntasks="${SLURM_NTASKS:-2}" --cpu-bind=cores \
    "$POPS_PYTHON" "$POPS_EU02_SRC/verification/machines/run_eu02_v15.py" \
      --stage "$STAGE" --space "$SPACE" --mpi "$MPI" --output "$OUT"
else
  "$POPS_PYTHON" "$POPS_EU02_SRC/verification/machines/run_eu02_v15.py" \
    --stage "$STAGE" --space "$SPACE" --mpi "$MPI" --output "$OUT"
fi
echo DONE
