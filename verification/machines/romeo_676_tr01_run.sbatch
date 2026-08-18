#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=01:20:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pops676-t1r
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# TR-01 exact-rank campaign. Serial first. OpenMP reuses the serial leaf.
# MPI jobs must set POPS676_MPI=on and launch with srun --mpi=pmi2.
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-676/src/verification/machines/romeo_676_env.sh
set -eo pipefail
pops676_require_pic_kokkos

DIM="${POPS676_DIM:?set POPS676_DIM to 1, 2, or 3}"
STAGE="${POPS676_TR01_STAGE:?set POPS676_TR01_STAGE to smoke, pair, or series}"
SPACE="${POPS676_SPACE:-KokkosSerial}"
MPI="${POPS676_MPI:-off}"
CONFIG="${POPS676_TR01_CONFIG:-}"
case "$DIM" in
  1|2|3) ;;
  *) echo "ERROR: POPS676_DIM must be 1, 2, or 3" >&2; exit 1 ;;
esac
case "$STAGE" in
  smoke) RESOLUTIONS=16 ;;
  pair) RESOLUTIONS=16,32 ;;
  series) RESOLUTIONS=16,32,64,128 ;;
  *) echo "ERROR: stage must be smoke, pair, or series" >&2; exit 1 ;;
esac
case "$SPACE" in
  KokkosSerial|KokkosOpenMP) ;;
  *) echo "ERROR: POPS676_SPACE must be KokkosSerial or KokkosOpenMP" >&2; exit 1 ;;
esac
case "$MPI" in
  off|on) ;;
  *) echo "ERROR: POPS676_MPI must be off or on" >&2; exit 1 ;;
esac
if [ "$MPI" = "on" ] && [ "$SPACE" != "KokkosSerial" ]; then
  echo "ERROR: MPI TR-01 is Serial-space only in this step" >&2
  exit 1
fi
if [ "$MPI" = "on" ] && [ "${SLURM_NTASKS:-1}" -lt 2 ]; then
  echo "ERROR: MPI TR-01 needs sbatch --ntasks=2" >&2
  exit 1
fi

KIND=serial
if [ "$MPI" = "on" ]; then
  KIND=mpi
  pops676_load_mpi_hdf5
  export FI_PROVIDER=tcp
fi
BUILD="$POPS676_ROOT/build-dim${DIM}-${KIND}"
export POPS_NATIVE_DIM="$DIM"
export POPS_TR01_SPACE="$SPACE"
export POPS_TR01_MPI="$MPI"
# Lexical path only. Dereferencing the scratch symlink publishes a second prefix.
export POPS_NATIVE_VARIANTS_ROOT="$BUILD/python/pops/_native"
if [ "$SPACE" = "KokkosOpenMP" ]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi
pops676_pythonpath "$BUILD"
GATE="TR01_D${DIM}_${SPACE}_${MPI}_${STAGE}"
OUT="$POPS676_ROOT/out/${GATE}"
mkdir -p "$OUT"

echo "host=$(hostname) gate=$GATE dim=$DIM space=$SPACE mpi=$MPI stage=$STAGE"
echo "sha=$(git -C "$POPS676_SRC" rev-parse HEAD) build=$BUILD"
test -f "$BUILD/python/pops/_native/variants.json"

EXPECT_BACKEND=(--expect-serial)
if [ "$MPI" = "on" ]; then
  EXPECT_BACKEND=(--expect-mpi --expect-parallel-hdf5)
fi
"$POPS_PYTHON" "$POPS676_SRC/scripts/verify_installed_native.py" \
  --expect-dim "$DIM" \
  "${EXPECT_BACKEND[@]}" \
  --json

if [ "$STAGE" = "smoke" ]; then
  if [ "$MPI" = "on" ]; then
    srun --mpi=pmi2 --ntasks=2 --cpu-bind=cores \
      "$POPS_PYTHON" "$POPS676_SRC/scripts/run_verification.py" \
        --suite pr \
        --dimensions "$DIM" \
        --max-nodes 1 \
        --pops-native-dim "$DIM" \
        --cases TR-01 \
        --mpi-mode on \
        --execution-space "$SPACE" \
        --execute \
        --output "$OUT/campaign"
  else
    "$POPS_PYTHON" "$POPS676_SRC/scripts/run_verification.py" \
      --suite pr \
      --dimensions "$DIM" \
      --max-nodes 1 \
      --pops-native-dim "$DIM" \
      --cases TR-01 \
      --mpi-mode off \
      --execution-space "$SPACE" \
      --execute \
      --output "$OUT/campaign"
  fi
fi

SMOKE_FLAG=()
if [ "$STAGE" != "series" ]; then
  SMOKE_FLAG=(--smoke)
fi
CONFIG_FLAG=()
if [ -n "$CONFIG" ]; then
  CONFIG_FLAG=(--config-id "$CONFIG")
fi
if [ "$MPI" = "on" ]; then
  srun --mpi=pmi2 --ntasks=2 --cpu-bind=cores \
    "$POPS_PYTHON" "$POPS676_SRC/verification/machines/run_tr01_complement.py" \
      --dim "$DIM" \
      --out "$OUT/native" \
      --resolutions "$RESOLUTIONS" \
      "${CONFIG_FLAG[@]}" \
      "${SMOKE_FLAG[@]}"
else
  "$POPS_PYTHON" "$POPS676_SRC/verification/machines/run_tr01_complement.py" \
    --dim "$DIM" \
    --out "$OUT/native" \
    --resolutions "$RESOLUTIONS" \
    "${CONFIG_FLAG[@]}" \
    "${SMOKE_FLAG[@]}"
fi

"$POPS_PYTHON" "$POPS676_SRC/verification/machines/romeo_676_tr01_validate.py" \
  --gate "$GATE" \
  --output "$OUT" \
  --dim "$DIM" \
  --stage "$STAGE" \
  --space "$SPACE" \
  --mpi-mode "$MPI"
echo DONE
