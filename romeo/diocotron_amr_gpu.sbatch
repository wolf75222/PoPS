#!/usr/bin/env bash
# Smoke test GPU du pas AMR sur UN seul GH200 d'armgpu (etape 1 reelle du hero-run AMR,
# cf. docs/HERO_RUN_AMR.md). Kokkos n'etant pas fourni par spack/module sur ROMEO, on le
# compile depuis les sources (Serial + CUDA, Hopper sm_90, nvcc_wrapper) une fois, puis on
# build l'exemple AMR sous le seam Kokkos, espace d'execution Cuda. But : verifier que tout
# le pas AMR (Poisson grossier + injection d'aux + reflux multi-patch + regrid) tourne sur
# GPU et que le resultat est BIT-IDENTIQUE au CPU (checksum serie/Kokkos-OpenMP = 4394594.404318).
# coupled_kokkos sert de reference GPU connue (uniforme, auto-valide CPU vs GPU).
#
# Pre-requis : kokkos-src deja clone sous /scratch_p/$USER (les noeuds de calcul n'ont pas
# internet). Soumettre depuis la racine du depot sur ROMEO :  sbatch romeo/diocotron_amr_gpu.sbatch

#SBATCH --account=r250127
#SBATCH --job-name=adc_amr_gpu
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --error=adc_amr_gpu.%J.err
#SBATCH --output=adc_amr_gpu.%J.out

set -uo pipefail
module load cuda/12.6
echo "== uname=$(uname -m) ; nvcc=$(which nvcc) =="
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

ROOT=/scratch_p/$USER
KSRC=$ROOT/kokkos-src
KINST=$ROOT/kokkos-install

# 1. Kokkos depuis les sources (cache : on saute si deja installe).
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

# 2. le projet, backend Kokkos/CUDA via nvcc_wrapper.
SRC=$SLURM_SUBMIT_DIR
BUILD=$SRC/build-gh200
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KINST/bin/nvcc_wrapper" -DKokkos_ROOT="$KINST" \
  || { echo "ADC_CFG_KO"; exit 1; }
cmake --build "$BUILD" -j 10 --target coupled_kokkos diocotron_amr_kokkos \
  || { echo "ADC_BUILD_KO"; exit 1; }

echo "===== coupled_kokkos (reference GPU uniforme, auto-valide CPU vs GPU) ====="
"$BUILD/bin/coupled_kokkos"
echo "===== diocotron_amr_kokkos (le pas AMR complet sur GPU) ====="
"$BUILD/bin/diocotron_amr_kokkos"
echo "== checksum CPU attendu (serie + Kokkos-OpenMP local) = 4394594.404318 =="
echo "== FIN =="
