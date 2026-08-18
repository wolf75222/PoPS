#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pops-eu02-o
#SBATCH --output=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.err

echo "starting job=${SLURM_JOB_ID:-local} host=$(hostname)"
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-v15-eu02/src/verification/machines/romeo_v15_eu02_env.sh
set -eo pipefail

export POPS_EU02_DIM=2
export POPS_EU02_SPACE=KokkosOpenMP
export POPS_EU02_MPI=off
bash "$POPS_EU02_SRC/verification/machines/romeo_v15_eu02_build.sbatch"

pops_eu02_select_kokkos KokkosOpenMP
BUILD="$POPS_EU02_ROOT/build-dim2-openmp"
export POPS_NATIVE_DIM=2
export POPS_NATIVE_VARIANTS_ROOT="$BUILD/python/pops/_native"
export OMP_NUM_THREADS=4
export POPS_THREADS=4
pops_eu02_pythonpath "$BUILD"
OUT="$POPS_EU02_ROOT/evidence/EU02_D2_KokkosOpenMP_off_smoke"
mkdir -p "$OUT"
"$POPS_PYTHON" "$POPS_EU02_SRC/verification/machines/run_eu02_v15.py" \
  --stage smoke --space KokkosOpenMP --mpi off --output "$OUT"
echo OPENMP_SMOKE_DONE
