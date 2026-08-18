#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pops676-m
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# R3: Dim1 MPI IF-01, 2 ranks on one node or two nodes via srun.
# Override with: sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1
# No exclusive node. no GPU.
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-676/src/verification/machines/romeo_676_env.sh
set -eo pipefail
pops676_require_pic_kokkos
pops676_load_mpi_hdf5
# verify_installed_native and the campaign both dlopen the MPI leaf.
# Site FI_PROVIDER=shm fails OFI getinfo on this MPICH 4.2.3 stack.
export FI_PROVIDER=tcp

GATE="${POPS676_GATE:-R3_1N}"
case "$GATE" in
  R3_1N|R3_2N) ;;
  *) echo "ERROR: POPS676_GATE must be R3_1N or R3_2N" >&2; exit 1 ;;
esac

DIM=1
CASE=IF-01
BUILD="$POPS676_ROOT/build-dim1-mpi"
export POPS_NATIVE_DIM="$DIM"
pops676_pythonpath "$BUILD"
OUT="$POPS676_ROOT/out/${GATE}"
mkdir -p "$OUT"
SUITE=pr
MAX_NODES=1
if [ "$GATE" = "R3_2N" ]; then
  SUITE=two_node
  MAX_NODES=2
fi

echo "host=$(hostname) gate=$GATE nodes=${SLURM_JOB_NUM_NODES:-1} ntasks=${SLURM_NTASKS:-}"
echo "sha=$(git -C "$POPS676_SRC" rev-parse HEAD) build=$BUILD"
test -f "$BUILD/python/pops/_native/variants.json"

"$POPS_PYTHON" "$POPS676_SRC/scripts/verify_installed_native.py" \
  --expect-dim "$DIM" --expect-mpi --expect-parallel-hdf5 --json

SRUN_ARGS=(--ntasks=2 --cpu-bind=cores)
if [ "$GATE" = "R3_2N" ]; then
  SRUN_ARGS+=(--ntasks-per-node=1)
fi

srun "${SRUN_ARGS[@]}" \
  "$POPS_PYTHON" "$POPS676_SRC/scripts/run_verification.py" \
  --suite "$SUITE" \
  --dimensions "$DIM" \
  --max-nodes "$MAX_NODES" \
  --pops-native-dim "$DIM" \
  --cases "$CASE" \
  --mpi-mode on \
  --execution-space KokkosSerial \
  --execute \
  --output "$OUT"

"$POPS_PYTHON" "$POPS676_SRC/verification/machines/romeo_676_validate.py" \
  --gate "$GATE" \
  --output "$OUT" \
  --case "$CASE" \
  --dim "$DIM" \
  --mpi-mode on \
  --expect-ranks 2
echo DONE
