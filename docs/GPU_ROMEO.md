# Verification GPU de la brique generee (ROMEO, NVIDIA GH200)

Le DSL `adc.dsl` genere une brique hyperbolique C++ (`emit_cpp_brick`) deja device-ready (`ADC_HD` ->
`__host__ __device__` sous nvcc, ops device-safe, `std::sqrt` comme `adc::Euler`). Ce document trace
la verification sur GPU REEL : la brique generee `EulerGen` tourne dans un kernel CUDA et donne le
meme flux que `adc::Euler` ecrite a la main.

Non integre a la CI (pas de GPU sur les runners). Test MANUEL, reproductible sur ROMEO.

## Recette

ROMEO : login x86_64, noeuds GPU aarch64 (Grace-Hopper, `armgpu`, 4x H100/GH200), SLURM, `cuda/12.6`.
nvcc doit s'executer SUR le noeud GPU (binaire aarch64), donc on compile et on tourne dans l'allocation.

```bash
# 1. generer le harnais CUDA (depuis la racine du repo, paquet adc construit dans build-py)
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_cuda_harness.py   # -> /tmp/euler_gpu.cu

# 2. envoyer en-tetes + harnais + script
rsync -az include /tmp/euler_gpu.cu python/tests/gpu/romeo_run.sh romeo:adc_dsl_gpu/
ssh romeo 'mv ~/adc_dsl_gpu/romeo_run.sh ~/adc_dsl_gpu/run.sh'

# 3. compiler (nvcc sm_90) + executer sur un noeud H100
ssh romeo 'cd ~/adc_dsl_gpu && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=8G -c 4 -t 3 bash run.sh'
```

`romeo_run.sh` : `module load cuda/12.6` puis `nvcc -std=c++20 -arch=sm_90 -I include euler_gpu.cu -o
euler_gpu && ./euler_gpu`. Le kernel `kflux` instancie `adc_generated::EulerGen` sur le device, calcule
le flux pour quelques etats, et le main compare a `adc::Euler` calcule sur l'hote.

## Resultat (obtenu)

```
noeud=romeo-a057  arch=aarch64
BUILD_OK
device=NVIDIA GH200 120GB  maxdiff(GPU EulerGen vs hote adc::Euler)=0.000e+00
```

La brique GENEREE depuis des formules Python compile avec nvcc et s'execute sur GH200 en donnant un
resultat BIT-IDENTIQUE a la brique ecrite a la main. Le codegen est donc correct jusqu'au GPU de
production.

## Limites / suite

- Test CUDA brut (nvcc), pas un build Kokkos complet : `ADC_HD` couvre les deux (Kokkos mappe sur
  `KOKKOS_INLINE_FUNCTION`), mais le solveur Kokkos complet n'est pas reconstruit ici.
- Toujours pas le dispatch de la brique generee DANS le solveur template (cf. ARCHITECTURE_CIBLE.md
  sect. 3 : exige une interface de modele type-erased). Ici la brique est compilee statiquement dans
  un harnais, ce qui suffit a valider le codegen device.
