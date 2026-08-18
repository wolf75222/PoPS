#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:10:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --job-name=pops-eu02-m
#SBATCH --output=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.err

echo "starting job=${SLURM_JOB_ID:-local} host=$(hostname)"
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-v15-eu02/src/verification/machines/romeo_v15_eu02_env.sh
set -eo pipefail

export POPS_EU02_DIM=2
export POPS_EU02_SPACE=KokkosSerial
export POPS_EU02_MPI=on
pops_eu02_select_kokkos KokkosSerial
pops_eu02_load_mpi_hdf5
export FI_PROVIDER=tcp
pops_eu02_maybe_build mpi
BUILD="$POPS_EU02_ROOT/build-dim2-mpi"
export POPS_NATIVE_DIM=2
export POPS_NATIVE_VARIANTS_ROOT="$BUILD/python/pops/_native"
export OMP_NUM_THREADS=1
export POPS_THREADS=1
pops_eu02_pythonpath "$BUILD"
OUT="$POPS_EU02_ROOT/evidence/EU02_D2_KokkosSerial_on_smoke"
mkdir -p "$OUT"
srun --mpi=pmi2 --ntasks=2 --cpu-bind=cores \
  "$POPS_PYTHON" "$POPS_EU02_SRC/verification/machines/run_eu02_v15.py" \
    --stage smoke --space KokkosSerial --mpi on --output "$OUT"
echo MPI_SMOKE_DONE
