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

## Kokkos (dispatch parallel_for, backend CUDA)

Au-dela du CUDA brut, on verifie la brique generee a travers le VRAI dispatch parallele que le solveur
utilise (`adc/mesh/for_each.hpp` -> `Kokkos::parallel_for`). On construit Kokkos depuis les sources
(pas de module sur ROMEO) puis un harnais `Kokkos::parallel_for(KOKKOS_LAMBDA ...)` qui calcule le flux
de `EulerGen` sur le device et le compare a `adc::Euler` sur l'hote.

```bash
# 1. generer le harnais Kokkos (depuis la racine, paquet adc construit dans build-py)
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_kokkos_harness.py   # -> /tmp/kokkos_euler.cpp

# 2. envoyer en-tetes + harnais (+ CMakeLists) + script, cloner Kokkos
rsync -az include /tmp/kokkos_euler.cpp python/tests/gpu/kokkos_CMakeLists.txt \
      python/tests/gpu/romeo_kokkos_build.sh romeo:adc_dsl_kk/
ssh romeo 'cd ~/adc_dsl_kk && mkdir -p harness && mv kokkos_euler.cpp harness/ \
           && mv kokkos_CMakeLists.txt harness/CMakeLists.txt && mv romeo_kokkos_build.sh kk_build.sh \
           && git clone --depth 1 -b 4.4.01 https://github.com/kokkos/kokkos.git'

# 3. build Kokkos (CUDA + Serial, HOPPER90) + harnais + run, sur un noeud GPU
ssh romeo 'cd ~/adc_dsl_kk && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=16G -c 16 -t 25 bash kk_build.sh'
```

`kk_build.sh` configure Kokkos avec `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON
-DKokkos_ENABLE_SERIAL=ON` et `nvcc_wrapper` comme compilateur, installe, puis compile le harnais
(`find_package(Kokkos)` + `Kokkos::kokkos`).

Resultat (obtenu) :
```
KOKKOS_OK
HARNESS_OK
exec_space=Cuda  maxdiff(Kokkos EulerGen vs hote adc::Euler)=5.551e-17
```

L'espace d'execution par defaut est `Cuda` (le kernel tourne bien sur le GPU). L'ecart est d'un ULP
(5.55e-17), du a la contraction FMA de nvcc_wrapper differente de l'hote, pas a un bug : la brique
generee est correcte a travers le dispatch Kokkos sur GH200.

## Cas COMPLET (time-stepping) sur GPU via le seam Kokkos d'adc

On va au-dela d'un flux isole : un cas Euler 2D complet (80 pas, CFL, Rusanov ordre 1, periodique)
avance ENTIEREMENT sur GPU a travers `adc::for_each_cell` / `for_each_cell_reduce_max|sum`
(`adc/mesh/for_each.hpp` -> `Kokkos::parallel_for` / `parallel_reduce`). On simule la MEME chose avec
la brique generee `EulerGen` et avec `adc::Euler`, et on compare les champs finaux + la masse.

```bash
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_kokkos_sim.py   # -> /tmp/kokkos_euler_sim.cpp
rsync -az include /tmp/kokkos_euler_sim.cpp romeo:adc_dsl_kk/sim/   # + kokkos_sim_CMakeLists.txt -> sim/CMakeLists.txt
rsync -az python/tests/gpu/romeo_kokkos_sim_build.sh romeo:adc_dsl_kk/kk_sim_build.sh
ssh romeo 'cd ~/adc_dsl_kk && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=16G -c 16 -t 25 bash kk_sim_build.sh'
```

Le harnais definit `#define ADC_HAS_KOKKOS` puis inclut `adc/mesh/for_each.hpp` : les boucles de
cellules passent donc par le VRAI dispatch Kokkos du solveur (le meme site d'appel que sur CPU).

Resultat (obtenu) :
```
exec=Cuda  n=64 steps=80  mass_drel=0.000e+00  rho[min,max]=[0.8912,1.0464]  maxdiff(EulerGen vs adc::Euler, GPU)=8.882e-16
```

80 pas de temps sur GH200 : masse EXACTEMENT conservee, dynamique non triviale (la bulle de pression
fait varier la densite), et la brique generee == `adc::Euler` a la precision machine (8.9e-16 cumule
sur 80 pas). Le cas complet tourne donc sur GPU a travers la machinerie Kokkos d'adc.

## Limites / suite

- Le cas complet passe par `adc/mesh/for_each.hpp` (le seam Kokkos REEL du solveur), mais on ne
  rebatit pas ici toute la pile runtime (System / AMR / MPI) sur GPU : les boucles de cellules
  empruntent le meme dispatch que la production, ce qui suffit a valider le device.
- Dispatch type-erased a l'execution : FAIT par ailleurs (adc::IModel, cf. python/tests/test_dsl_dynamic.py).
  Les harnais GPU ci-dessus compilent en revanche la brique STATIQUEMENT (perf) ; le chemin type-erased
  (vtable) est un complement HOTE, pas pour la boucle chaude GPU.
