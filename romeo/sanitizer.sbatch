#!/usr/bin/env bash
# Porte de CORRECTION GPU sur UN GH200 d'armgpu : passe les EXEMPLES Kokkos/CUDA sous
# compute-sanitizer (NVIDIA) + rejoue la validation bit-identique CPU vs GPU. But :
# attraper la classe de bug la plus subtile (cf. docs/CHOICES.md : un device_fence()
# oublie = lecture hote de memoire unifiee encore ecrite par un kernel, silencieux en
# CI CPU).
#
# Deux filets COMPLEMENTAIRES :
#   1. compute-sanitizer (memcheck/initcheck/synccheck) : erreurs memoire GPU (acces
#      hors borne / mal alignes, lectures non initialisees, synchro). On l'applique aux
#      EXEMPLES coupled_kokkos / diocotron_amr_kokkos, qui appellent Kokkos::initialize
#      via un ScopeGuard. PAS aux tests unitaires : ecrits pour le seam CPU, ils n'ont
#      pas de ScopeGuard et echouent "device not initialized" sous backend Cuda (rien a
#      verifier). Pas de --leak-check : cudaMallocManaged signale des fuites benignes en
#      sortie (memoire unifiee liberee a la fin du process), ce n'est pas le sujet.
#   2. checksum CPU vs GPU : c'est LUI qui attrape un fence manquant. diocotron_amr_kokkos
#      imprime un checksum compare a la valeur CPU connue 4394594.404318 ; un fence oublie
#      ferait diverger ce nombre. C'est le detecteur reel des fences (compute-sanitizer ne
#      voit pas la course hote/device sur memoire managee).
#
# Pre-requis : kokkos-src clone sous /scratch_p/$USER (les noeuds n'ont pas internet),
# cf. romeo/README.md. Soumettre depuis la racine du depot sur ROMEO :
#   sbatch romeo/sanitizer.sbatch

#SBATCH --account=r250127
#SBATCH --job-name=adc_sanitizer
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --error=adc_sanitizer.%J.err
#SBATCH --output=adc_sanitizer.%J.out

set -uo pipefail
module load cuda/12.6
echo "== uname=$(uname -m) ; nvcc=$(which nvcc) ; sanitizer=$(which compute-sanitizer) =="
nvcc --version | tail -2
compute-sanitizer --version 2>/dev/null | head -2 || echo "compute-sanitizer introuvable (verifier le module cuda)"

ROOT=/scratch_p/$USER
KSRC=$ROOT/kokkos-src
KINST=$ROOT/kokkos-install

# 1. Kokkos depuis les sources (cache : on saute si deja installe). IDENTIQUE a
#    diocotron_amr_gpu.sbatch (Serial + CUDA, Hopper sm_90, nvcc_wrapper).
if [ ! -f "$KINST/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KINST/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "== build Kokkos (Serial+CUDA, Hopper90) =="
  cmake -S "$KSRC" -B "$ROOT/kokkos-build" -DCMAKE_INSTALL_PREFIX="$KINST" \
    -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DCMAKE_CXX_COMPILER="$KSRC/bin/nvcc_wrapper" -DCMAKE_CXX_STANDARD=17 \
    || { echo "KOKKOS_CFG_KO"; exit 1; }
  cmake --build "$ROOT/kokkos-build" -j 24 --target install || { echo "KOKKOS_BUILD_KO"; exit 1; }
fi
echo "== Kokkos installe: $KINST =="

# 2. le projet, backend Kokkos/CUDA. Seuls les 2 exemples GPU (ScopeGuard Kokkos) sont
#    sanitizes ; ils exercent le pas couple complet + l'AMR multi-patch, dont mass()
#    route maintenant par for_each_cell_reduce_sum.
SRC=$SLURM_SUBMIT_DIR
BUILD=$SRC/build-gh200
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KINST/bin/nvcc_wrapper" -DKokkos_ROOT="$KINST" \
  || { echo "ADC_CFG_KO"; exit 1; }
cmake --build "$BUILD" -j 10 --target coupled_kokkos diocotron_amr_kokkos \
  || { echo "ADC_BUILD_KO"; exit 1; }

# --- helper : un binaire sous un outil compute-sanitizer, verdict honnete ---
run_san () {  # $1=tool  $2=binaire  $3...=args
  local tool="$1"; shift
  local bin="$1"; shift
  echo "----- compute-sanitizer --tool $tool  $(basename "$bin") -----"
  local log; log=$(compute-sanitizer --tool "$tool" "$bin" "$@" 2>&1)
  echo "$log" | tail -20
  if echo "$log" | grep -qE "not initialized|didn't terminate successfully|returned an error"; then
    echo "  >>> $tool $(basename "$bin") : ECHEC (l'appli a plante avant le check)"
  elif echo "$log" | grep -q "ERROR SUMMARY: 0 errors"; then
    echo "  >>> $tool $(basename "$bin") : OK"
  else
    echo "  >>> $tool $(basename "$bin") : ERREURS MEMOIRE (voir ci-dessus)"
  fi
}

echo "===== compute-sanitizer (memcheck / initcheck / synccheck) ====="
# coupled_kokkos est petit (N=128, 20 pas) : les 3 outils passent vite. diocotron_amr_kokkos
# est plus lourd (AMR, ~18 s nominal -> minutes sous sanitizer) : memcheck seul.
for t in memcheck initcheck synccheck; do
  run_san "$t" "$BUILD/bin/coupled_kokkos"
done
run_san memcheck "$BUILD/bin/diocotron_amr_kokkos"

echo "===== detecteur reel des fences manquants : checksum bit-identique CPU vs GPU ====="
"$BUILD/bin/coupled_kokkos"
echo "== diocotron_amr_kokkos : checksum doit valoir EXACTEMENT 4394594.404318 (= CPU) =="
"$BUILD/bin/diocotron_amr_kokkos"
echo "== FIN sanitizer =="
