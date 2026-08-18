#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=pops676-s
#SBATCH --output=/scratch_p/rmdraux/pops-676/logs/%x.%j.out
#SBATCH --error=/scratch_p/rmdraux/pops-676/logs/%x.%j.err

# R1: Dim2 Serial IF-08. R2: Dim1 Serial EU-01. No exclusive node. no GPU.
set -eo pipefail
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-676/src/verification/machines/romeo_676_env.sh
set -eo pipefail
pops676_require_pic_kokkos

GATE="${POPS676_GATE:?set POPS676_GATE to R1 or R2}"
case "$GATE" in
  R1)
    DIM=2
    CASE=IF-08
    BUILD="$POPS676_ROOT/build-dim2-serial"
    ;;
  R2)
    DIM=1
    CASE=EU-01
    BUILD="$POPS676_ROOT/build-dim1-serial"
    ;;
  *)
    echo "ERROR: POPS676_GATE must be R1 or R2" >&2
    exit 1
    ;;
esac

export POPS_NATIVE_DIM="$DIM"
pops676_pythonpath "$BUILD"
OUT="$POPS676_ROOT/out/${GATE}"
mkdir -p "$OUT"

echo "host=$(hostname) gate=$GATE dim=$DIM case=$CASE"
echo "sha=$(git -C "$POPS676_SRC" rev-parse HEAD) build=$BUILD"
test -f "$BUILD/python/pops/_native/variants.json"

"$POPS_PYTHON" "$POPS676_SRC/scripts/verify_installed_native.py" \
  --expect-dim "$DIM" --expect-serial --json
"$POPS_PYTHON" "$POPS676_SRC/scripts/run_verification.py" \
  --suite pr \
  --dimensions "$DIM" \
  --max-nodes 1 \
  --pops-native-dim "$DIM" \
  --cases "$CASE" \
  --mpi-mode off \
  --execution-space KokkosSerial \
  --execute \
  --output "$OUT"
"$POPS_PYTHON" "$POPS676_SRC/verification/machines/romeo_676_validate.py" \
  --gate "$GATE" \
  --output "$OUT" \
  --case "$CASE" \
  --dim "$DIM" \
  --mpi-mode off \
  --expect-ranks 1
echo DONE
