#!/usr/bin/env bash
# Test MULTI-GPU du diocotron AMR distribue : MPI + Kokkos/CUDA, 4 rangs sur 4 GH200 d'un
# noeud armgpu (1 rang / GPU). Exerce le parallel_copy / all_reduce entre fabs DEVICE de rangs
# differents (MPI CUDA-aware, module openmpi/aarch64/4.1.7-cuda). C'est l'etape 1 multi-GPU du
# hero-run AMR (cf. docs/HERO_RUN_AMR.md). Gate : grossier bit-identique entre repartition des
# patchs round-robin (DIST) et tous-sur-rang-0 (REF) -> l'AMR distribue multi-GPU est invariant
# a la repartition. Pre-requis : Kokkos deja compile sous /scratch_p/$USER/kokkos-install (voir
# romeo/diocotron_amr_gpu.sbatch). Soumettre depuis la racine du depot sur ROMEO.

#SBATCH --account=r250127
#SBATCH --job-name=adc_amr_mgpu
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --error=adc_amr_mgpu.%J.err
#SBATCH --output=adc_amr_mgpu.%J.out

set -uo pipefail
module load cuda/12.6
module load openmpi/aarch64/4.1.7-cuda
echo "== mpicxx=$(which mpicxx) ; nvcc=$(which nvcc) =="

KINST=/scratch_p/$USER/kokkos-install
SRC=$SLURM_SUBMIT_DIR
BUILD=$SRC/build-gh200-mpi
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON \
  -DCMAKE_CXX_COMPILER="$KINST/bin/nvcc_wrapper" -DKokkos_ROOT="$KINST" \
  || { echo "CFG_KO"; exit 1; }
cmake --build "$BUILD" -j 8 --target diocotron_amr_mpi_kokkos || { echo "BUILD_KO"; exit 1; }

echo "== GPU du noeud =="
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true
echo "== lancement 4 rangs / 4 GPU (1 rang par GH200) =="
srun -n 4 --gpu-bind=single:1 "$BUILD/bin/diocotron_amr_mpi_kokkos"
echo "== FIN =="
