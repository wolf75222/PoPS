#!/usr/bin/env bash
# PREUVE DE PARALLELISME sur GH200 : montre que le pas AMR sous Kokkos/CUDA utilise des
# MILLIERS de threads (concurrency de l'espace Cuda), pas 1, et donne le temps des 80 pas a
# plusieurs tailles nc. concurrency() = nombre de threads materiels reellement utilises.
# Soumettre depuis la racine du depot sur ROMEO (Kokkos deja compile sous /scratch_p).

#SBATCH --account=r250127
#SBATCH --job-name=adc_proof
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --error=adc_proof.%J.err
#SBATCH --output=adc_proof.%J.out

set -uo pipefail
module load cuda/12.6
KINST=/scratch_p/$USER/kokkos-install
SRC=$SLURM_SUBMIT_DIR
BUILD=$SRC/build-gh200
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KINST/bin/nvcc_wrapper" -DKokkos_ROOT="$KINST" >/dev/null 2>&1
cmake --build "$BUILD" -j 8 --target diocotron_amr_kokkos || { echo BUILD_KO; exit 1; }

echo "== threads materiels (concurrency) + temps des 80 pas, par taille =="
for NC in 64 256 512 1024; do
  "$BUILD/bin/diocotron_amr_kokkos" "$NC" 2>/dev/null | grep -E "PARALLELISME"
done
echo "== FIN =="
