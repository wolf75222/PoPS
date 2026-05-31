#!/usr/bin/env bash
# Confirmation ROMEO du diagnostic sur-taux diocotron (workflow diocotron-overshoot-diag).
# Question : le sur-taux +7-8 % est-il PLAT en resolution (-> distorsion geometrique / d'equation,
# structurelle) ou DECROISSANT (-> erreur de troncature) ? Et est-il UNIFORME sur les modes 3/4/5
# (le mode 5 est ressorti aberrant a eff 256) ?
#
# Balayage UNIFORME (refine=0, pas d'AMR pour ne pas confondre) : nc in {256,512,1024} x l in {3,4,5}.
# Observable = phi-DFT a r=r0 (deja en place dans l'exemple, = methode du papier). VanLeer (recon=1).
#
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --time=02:30:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dio_conv
#SBATCH --array=0-8
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -e
romeo_load_x64cpu_env
spack load gcc@14.2.0   # runtime libstdc++ : meme gcc que la compilation (binaire dca_conv pre-construit)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

NCS=(256 256 256 512 512 512 1024 1024 1024)
LS=(   3   4   5   3   4   5    3    4    5)
STEPS=(500 500 500 1000 1000 1000 2000 2000 2000)
i=${SLURM_ARRAY_TASK_ID}
nc=${NCS[$i]}; l=${LS[$i]}; ns=${STEPS[$i]}

bin=$HOME/adc_cpp/dca_conv
out=/scratch_p/$USER/dio_conv/e${nc}_l${l}
mkdir -p "$out" "$HOME/adc_cpp/results_conv"
echo "[$(date +%H:%M:%S)] task $i : nc=$nc l=$l nsteps=$ns OMP=$OMP_NUM_THREADS"

"$bin" "$out" "$nc" "$ns" 0 "$l" 0 1 0

# collecte du CSV (nom stable, recuperable par rsync)
cp "$out/ring_amp.csv" "$HOME/adc_cpp/results_conv/e${nc}_l${l}.csv"
echo "[$(date +%H:%M:%S)] task $i FINI -> results_conv/e${nc}_l${l}.csv"
tail -1 "$out/ring_amp.csv"
