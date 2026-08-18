#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pops-eu02-s
#SBATCH --output=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-v15-eu02/logs/%x.%j.err

# Isolated Serial leaf + global/temporal/spatial/dump/analyze. One node.
echo "starting job=${SLURM_JOB_ID:-local} host=$(hostname)"
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-v15-eu02/src/verification/machines/romeo_v15_eu02_env.sh
set -eo pipefail

export POPS_EU02_DIM=2
export POPS_EU02_SPACE=KokkosSerial
export POPS_EU02_MPI=off
bash "$POPS_EU02_SRC/verification/machines/romeo_v15_eu02_build.sbatch"

pops_eu02_select_kokkos KokkosSerial
BUILD="$POPS_EU02_ROOT/build-dim2-serial"
export POPS_NATIVE_DIM=2
export POPS_NATIVE_VARIANTS_ROOT="$BUILD/python/pops/_native"
export OMP_NUM_THREADS=1
export POPS_THREADS=1
pops_eu02_pythonpath "$BUILD"

DRIVER="$POPS_EU02_SRC/verification/machines/run_eu02_v15.py"
for STAGE in series temporal spatial dump; do
  OUT="$POPS_EU02_ROOT/evidence/EU02_D2_KokkosSerial_off_${STAGE}"
  mkdir -p "$OUT"
  echo "=== STAGE $STAGE ==="
  "$POPS_PYTHON" "$DRIVER" --stage "$STAGE" --space KokkosSerial --mpi off --output "$OUT"
done

# Analyze the official four-resolution global series; overlays dump snapshots if present.
ANALYZE_OUT="$POPS_EU02_ROOT/evidence/EU02_D2_KokkosSerial_off_series"
if [ -f "$POPS_EU02_ROOT/evidence/EU02_D2_KokkosSerial_off_dump/n064/snapshots.npz" ]; then
  cp -f "$POPS_EU02_ROOT/evidence/EU02_D2_KokkosSerial_off_dump/n064/snapshots.npz" \
    "$ANALYZE_OUT/n064/snapshots.npz" 2>/dev/null || true
fi
"$POPS_PYTHON" "$DRIVER" --stage analyze --space KokkosSerial --mpi off --output "$ANALYZE_OUT"
"$POPS_PYTHON" "$POPS_EU02_SRC/verification/cases/euler/isentropic_vortex/plot_eu02.py" \
  "$ANALYZE_OUT" "$POPS_EU02_ROOT/build/verification/EU-02/real_plots" || true
echo SERIAL_CAMPAIGN_DONE
