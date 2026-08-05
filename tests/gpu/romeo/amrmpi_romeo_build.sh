#!/usr/bin/env bash
# VALIDATION INTEGREE AmrSystem + MPI + GPU + STRONG-SCALING sur ROMEO (GH200) : MPI + Kokkos (backend
# Cuda) + OpenMPI CUDA-aware. Batit amrmpi_integrated et le lance en np=1/2/4 (un GH200 par rang).
# Le binaire prouve aussi que le ProgramGraph consomme B_z sur les niveaux grossier et fin avant la
# mesure de strong-scaling.
# CHAQUE run mesure DEUX modes dans le meme binaire : grossier REPLIQUE (defaut, ne scale pas) puis
# REPARTI (distribute_coarse=true, 2x2, le mode strong-scaling). Le script :
#   - verifie cmax bit-identique cross-rang dans les deux modes (max insensible a l'ordre) ;
#   - reporte per_step_ms np=1/2/4 pour replique ET reparti -> montre (ou non) le strong-scaling.
# Lancer depuis un checkout PoPS complet (ou definir POPS_SOURCE_ROOT). Le job copie dans son espace
# temporaire le CMake racine, cmake/, include/ et src/ : src/CMakeLists.txt reste ainsi l'autorite
# unique des sources runtime exactes. Kokkos (Cuda+Serial, Hopper90) reste installe dans
# $HOME/pops_gpu_p1/kinstall. Soumettre : sbatch tests/gpu/romeo/amrmpi_romeo_build.sh
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:60:00
#SBATCH --job-name=amrmpi
set -euo pipefail

module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda          # OpenMPI 4.1.7 CUDA-aware (UCX)
echo "noeud=$(hostname) arch=$(uname -m)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POPS_SOURCE_ROOT="${POPS_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
[[ -f "$POPS_SOURCE_ROOT/CMakeLists.txt" \
   && -f "$POPS_SOURCE_ROOT/src/CMakeLists.txt" ]] || {
  echo "POPS_SOURCE_ROOT is not a complete PoPS checkout: $POPS_SOURCE_ROOT" >&2
  exit 3
}

PERSIST_ROOT="${POPS_AMRMPI_ROOT:-$HOME/pops_gpu_p1}"
WORK_ROOT="${SLURM_TMPDIR:-$PERSIST_ROOT/tmp-${SLURM_JOB_ID:-manual}}/pops-amrmpi"
STAGED_POPS="$WORK_ROOT/pops"
HARNESS_SOURCE="$WORK_ROOT/harness"
BUILD_DIR="$WORK_ROOT/build"
NW="$PERSIST_ROOT/kinstall/bin/nvcc_wrapper"
[[ -x "$NW" ]] || { echo "missing Kokkos nvcc_wrapper: $NW" >&2; exit 3; }

# The target is job-scoped and explicit. Copy the complete canonical runtime inputs, not a private
# hand-maintained list of implementation TUs.
cmake -E rm -rf "$WORK_ROOT"
cmake -E make_directory "$STAGED_POPS" "$HARNESS_SOURCE"
cmake -E copy "$POPS_SOURCE_ROOT/CMakeLists.txt" "$STAGED_POPS/CMakeLists.txt"
cmake -E copy_directory "$POPS_SOURCE_ROOT/cmake" "$STAGED_POPS/cmake"
cmake -E copy_directory "$POPS_SOURCE_ROOT/include" "$STAGED_POPS/include"
cmake -E copy_directory "$POPS_SOURCE_ROOT/src" "$STAGED_POPS/src"
cmake -E copy "$SCRIPT_DIR/amrmpi_CMakeLists.txt" "$HARNESS_SOURCE/CMakeLists.txt"
cmake -E copy "$SCRIPT_DIR/amrmpi_integrated.cpp" "$HARNESS_SOURCE/amrmpi_integrated.cpp"

CFG_LOG="$PERSIST_ROOT/amrmpi_cfg.log"
BUILD_LOG="$PERSIST_ROOT/amrmpi_build.log"
cmake -S "$HARNESS_SOURCE" -B "$BUILD_DIR" -DCMAKE_CXX_COMPILER="$NW" \
  -DKokkos_ROOT="$PERSIST_ROOT/kinstall" -DPOPS_ROOT="$STAGED_POPS" \
  -DCMAKE_BUILD_TYPE=Release \
  > "$CFG_LOG" 2>&1 || { echo CFG_FAIL; tail -40 "$CFG_LOG"; exit 1; }
cmake --build "$BUILD_DIR" --target amrmpi_integrated -j 8 \
  > "$BUILD_LOG" 2>&1 || { echo BUILD_FAIL; tail -60 "$BUILD_LOG"; exit 1; }
echo AMRMPI_BUILD_OK
OUT="$PERSIST_ROOT/amrmpi_out.txt"
: > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n "$NP" --gpus-per-task=1 "$BUILD_DIR/amrmpi_integrated" 2>&1 | tee -a "$OUT"
done
echo "=== PARITE cmax + STRONG-SCALING per_step_ms (replique vs reparti, np=1/2/4) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
# AMRMPI[tag] np=N ... cmax=... cmax_crossrank_spread=...
sig = {}   # (tag, np) -> (cmax, spread)
tms = {}   # (tag, np) -> per_step_ms
for line in open(sys.argv[1]):
    m = re.search(r'AMRMPI\[(\w+)\] np=(\d+).*cmax=([-\d.eE+]+) \| cmax_crossrank_spread=([-\d.eE+]+)', line)
    if m:
        sig[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    # ligne per_step_ms : "AMRMPI[tag] exec=... per_step_ms=X (max over ranks, n=N, measured=M)".
    # le np n'y figure pas ; on le retient depuis la derniere ligne signature du meme tag.
    m = re.search(r'AMRMPI\[(\w+)\].*per_step_ms=([-\d.eE+]+)', line)
    if m:
        tag = m.group(1)
        nps_tag = [n for (t, n) in sig if t == tag]
        if nps_tag:
            tms[(tag, max(nps_tag))] = float(m.group(2))
# cmax bit-identique cross-rang (les deux modes) + cmax identique entre np (max insensible a l'ordre)
worst = 0.0
for tag in ("replique", "reparti"):
    nps = sorted(n for (t, n) in sig if t == tag)
    if not nps: continue
    ref = sig[(tag, nps[0])][0]
    for n in nps:
        cmax, spread = sig[(tag, n)]
        worst = max(worst, abs(cmax - ref), spread)
        print(f"[{tag}] np={n} cmax={cmax:.17e} dcmax_vs_np1={abs(cmax-ref):.3e} crossrank_spread={spread:.3e}")
print(f"PARITE cmax dmax (tous tags/np) = {worst:.3e}")
print("PARITE cmax OK (bit-identique)" if worst == 0.0 else "PARITE cmax NON BIT-IDENTIQUE")
print("--- STRONG-SCALING (per_step_ms, max sur les rangs) ---")
def scaling(tag):
    nps = sorted(n for (t, n) in tms if t == tag)
    if not nps: return
    base = tms[(tag, nps[0])]
    for n in nps:
        ms = tms[(tag, n)]
        sp = base / ms if ms > 0 else float('nan')
        eff = sp / n * 100.0
        print(f"[{tag}] np={n} per_step_ms={ms:.4f} speedup={sp:.2f}x efficiency={eff:.1f}%")
scaling("replique")
scaling("reparti")
PY
echo AMRMPI_DONE
