# Backends parallèles

`adc_cpp` cible une seule pile de calcul (mesh + transport + Poisson + AMR), mais
cette pile peut s'exécuter sur sept configurations parallèles : du séquentiel mono-thread
jusqu'au multi-GPU Grace-Hopper distribué par MPI. Le point clé de conception est qu'**aucun
opérateur ne change** d'une configuration à l'autre : tout le parallélisme est confiné à
**deux seams** (coutures de dispatch). Le backend est choisi **à la compilation**, par des
options CMake.

Cette page décrit chaque configuration : ce qu'elle est, la commande de build réelle, comment
la lancer, et son **statut de validation honnête** (testé en CI vs validé manuellement sur
ROMEO). Pour la matrice de couverture test par test, voir
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (source de vérité unique) ; pour le portage
GPU phase par phase, voir [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

## Le modèle : deux seams, MPI + Kokkos

Il n'y a pas « trois couches » empilées. L'architecture est **MPI + Kokkos** :

- **MPI** distribue les sous-domaines entre rangs (un GPU par rang en mode GPU). Tout passe
  par `my_rank()` / `n_ranks()` / `all_reduce_*` de
  [`include/adc/parallel/comm.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/parallel/comm.hpp). Sans
  `ADC_HAS_MPI`, ces fonctions renvoient rang 0 / 1 rang : le code est **série par
  construction**.
- **Kokkos** parallélise le calcul **local** et abstrait le matériel via son `ExecutionSpace` :
  backend `Cuda` pour GPU NVIDIA, `Serial`/`OpenMP` pour CPU. Tout passe par `for_each_cell`
  (et `for_each_cell_reduce_*`) de
  [`include/adc/mesh/for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp), qui bascule
  CPU ↔ GPU à la compilation **sans toucher les sites d'appel**.

On n'écrit **aucun kernel CUDA à la main** : le même `.cpp` cible CPU et GPU selon le backend
Kokkos actif à la compilation. `nvcc_wrapper` n'est que le compilateur exigé par le backend
Cuda de Kokkos.

> **Le module Python `adc` est série par défaut.** L'extension `_adc` (pybind11) n'est
> construite en CI qu'en mode Serial (pas de `-DADC_USE_KOKKOS=ON`, pas de MPI). Aucun test
> Python n'exerce les chemins Kokkos, OpenMP, Cuda ou MPI. Le multi-thread, le GPU et le
> distribué se pilotent depuis la facade C++ (`System` / `AmrSystem`), pas depuis Python.

## Les options CMake réelles

Vérifiées dans [`CMakeLists.txt`](../../../CMakeLists.txt) :

| Option CMake | Effet | Défaut |
|--------------|-------|--------|
| `ADC_USE_KOKKOS` | Backend Kokkos (CPU Serial/OpenMP + GPU Cuda/HIP). Recommandé. | `OFF` |
| `ADC_USE_MPI` | Seam comm distribué (`comm.hpp` -> collectives MPI). | `OFF` |
| `ADC_USE_OPENMP` | **Déprécié** : backend OpenMP autonome (CPU seul). Préférer Kokkos. | `OFF` |
| `ADC_BUILD_PYTHON` | Module Python `adc` (pybind11) — série uniquement. | `OFF` |

Le sous-backend Kokkos (Serial / OpenMP / Cuda) n'est **pas** une option `adc_cpp` : il est
choisi au moment où l'on **installe Kokkos** (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`,
`Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_HOPPER90`), puis pointé par `-DKokkos_ROOT=...`. C'est ce
qui distingue les configurations 2/3/6 ci-dessous.

Notes :

- `ADC_USE_OPENMP` et `ADC_USE_KOKKOS` sont **mutuellement exclusifs** (erreur fatale CMake si
  les deux sont `ON`).
- Sans Kokkos, la norme C++ est 23 ; **avec Kokkos** elle retombe à **C++20** (CUDA 12.x ne
  propose pas `-std=c++23`).

---

## 1. Serial

**Ce que c'est.** Le build de référence : ni Kokkos, ni MPI. `for_each_cell` est une simple
double boucle séquentielle, `comm.hpp` répond rang 0 / 1 rang. C'est l'**oracle** : tous les
autres backends sont validés bit-à-bit (ou à l'arrondi près) contre lui.

**Build.**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Run.**

```bash
ctest --test-dir build --output-on-failure
```

**Validation : CI (gate obligatoire de chaque PR).** C'est le job `build-and-test`
(`ubuntu-latest / Release`, g++), déclenché sur **tout** `pull_request`. Couvre les 109 cibles
ctest hors-MPI **plus** les 60 tests Python (construits via `-DADC_BUILD_PYTHON=ON`, module
`_adc` série). Statut `ci-fast` dans la matrice.

---

## 2. Kokkos Serial

**Ce que c'est.** Backend Kokkos avec l'espace d'exécution `Serial` (CPU mono-thread, mais via
`Kokkos::parallel_for`). Mono-thread comme la config 1, mais il exerce le **chemin Kokkos** :
allocateur unifié (`kokkos_malloc<SharedSpace>`), `MDRangePolicy`, cycle de vie Kokkos. C'est
ce job qui avait rattrapé une régression d'init (allocation d'un `Fab` avant l'init paresseuse
de Kokkos) que le build série ne voyait pas.

**Build.** Il faut un Kokkos installé avec `Kokkos_ENABLE_SERIAL=ON`, puis :

```bash
cmake -S . -B build-kokkos -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-kokkos -j
```

**Run.**

```bash
ctest --test-dir build-kokkos --output-on-failure
```

**Validation : CI (job `ci-full`).** Job `kokkos` (`ubuntu-latest / Kokkos (Serial)`),
Kokkos 4.4.01 Serial. Ne tourne **pas** sur les PR ordinaires : seulement en mode plein (push
`master`, cron nocturne, `workflow_dispatch`, ou PR portant le label `ci-full`). Statut
`ci-full`, 91/91 ctest hors-MPI.

---

## 3. Kokkos OpenMP

**Ce que c'est.** Même backend Kokkos, espace d'exécution `OpenMP` : parallélisme multi-thread
sur CPU. `for_each_cell` devient un `parallel_for` multi-thread (avec un seuil de bascule série
pour les petites grilles du V-cycle, cf. `ADC_FOREACH_SERIAL_THRESHOLD`).

**Build.** Kokkos installé avec `Kokkos_ENABLE_OPENMP=ON`. Le `-DADC_USE_KOKKOS=ON` côté
`adc_cpp` est identique à la config 2 ; c'est l'install Kokkos qui change :

```bash
cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-kokkos-omp -j
```

**Run.** Borner le nombre de threads sur les petites machines :

```bash
OMP_NUM_THREADS=4 OMP_PROC_BIND=false \
  ctest --test-dir build-kokkos-omp --output-on-failure
```

**Validation : CI (job `ci-full`).** Job `kokkos-openmp`
(`ubuntu-latest / Kokkos (OpenMP)`), Kokkos 4.4.01 OpenMP, `OMP_NUM_THREADS=2`. Mode plein
uniquement (comme la config 2). Statut `ci-full`, 91/91 ctest, 0 échec.

> **Note d'honnêteté FP.** Sous Kokkos, la réduction **somme** réassocie l'addition flottante
> (déterministe par tuile, mais pas bit-identique à la boucle série) ; la réduction **max** est
> exacte partout. Le backend **OpenMP autonome** (`ADC_USE_OPENMP`, déprécié) et le série
> restent, eux, exacts (boucle séquentielle, pas de `reduction(+:)`). Détail dans l'en-tête de
> [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp).

---

## 4. MPI CPU

**Ce que c'est.** Build distribué sans Kokkos : `comm.hpp` passe par `MPI_Comm_rank/size` +
collectives sur `MPI_COMM_WORLD`. Le domaine est découpé en boîtes réparties sur les rangs ;
les halos s'échangent par `fill_boundary` cross-rang, le reflux/masse par `all_reduce_*`. CPU
uniquement.

**Build.**

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DADC_USE_MPI=ON
cmake --build build-mpi -j
```

**Run.** Les cibles MPI rejouent chacune np=1/2/4 sous `mpirun`. Pour `-np 4` sur une petite
machine, autoriser l'oversubscribe :

```bash
OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi --output-on-failure
```

**Validation : CI (job `ci-full`).** Job `mpi` (`ubuntu-latest / MPI`, OpenMPI). Mode plein
uniquement. Vérifie l'**invariance au nombre de rangs** : les observables (parité, AMR,
Krylov, masse) sont bit-identiques à np=1/2/4. Statut `ci-full` sur les ~21 entrées du bloc
`ADC_HAS_MPI` ; les tests non-MPI tournent à np=1 dans ce build (liés MPI, mono-process).

---

## 5. MPI + Kokkos OpenMP

**Ce que c'est.** Hybride distribué CPU : MPI entre les nœuds/rangs, Kokkos OpenMP pour le
multi-thread intra-rang. C'est le mode CPU « plein » (tous les cœurs de tous les rangs).

**Build.** Les deux options à la fois, sur un Kokkos OpenMP :

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-mpi-omp -j
```

**Run.**

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi-omp --output-on-failure
```

**Validation : ROMEO-manuel (nœud `x64cpu`).** Cette combinaison n'est **pas** dans la CI (la
CI ne joint jamais MPI et Kokkos dans le même build). Validée à la main sur le nœud `x64cpu` de
ROMEO : **52/57 runs rank-invariants** (bit-identiques np=1/2/4, dmax=0 sur les observables
parité/AMR/Krylov). Réserve honnête : 3 tests distribués-MG lourds (`mpi_cutcell_multibox`,
`mpi_amr_distributed_coarse`, `condensed_schur_source_stepper`) sont **trop lents** à np>1
(dépassent 600 s) — pathologie de **performance** (petites tuiles + halos MPI, ~5-7x de
ralentissement), pas un deadlock ni un bug de correction. Tous passent à np=1.

---

## 6. Kokkos CUDA (ROMEO / GH200 uniquement)

**Ce que c'est.** Backend Kokkos avec l'espace d'exécution `Cuda` : `for_each_cell` devient un
`Kokkos::parallel_for` qui s'exécute sur le GPU. Le même code que les configs 2/3 ; seul le
backend Kokkos change. Les `Fab` vivent en mémoire unifiée (`Kokkos::SharedSpace`), donc
device-accessibles par construction ; `for_each_cell` est **async** sous Cuda, d'où un
`device_fence()` (via `sync_host()`) avant toute lecture hôte.

**Pas de build local.** Il n'y a **pas de `nvcc` sur les postes de dev** ; `nvcc` ne tourne que
sur le nœud GPU (aarch64) de ROMEO, pas sur le login (x86). **La CI ne construit JAMAIS avec
CUDA** : toutes les cellules « Kokkos Cuda » de la matrice sont soit ROMEO-manuel, soit `?`.

**Build (sur ROMEO, nœud `armgpu`).** Kokkos installé avec
`Kokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON`, compilateur `nvcc_wrapper` :

```bash
module load cuda/12.6
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-cuda -j
```

**Run (SLURM, un GPU).**

```bash
srun --account=<compte> -p instant --constraint=armgpu --gres=gpu:1 \
  ./build-cuda/bin/<harness>
```

**Validation : ROMEO-manuel (jamais en CI).** Les harnesses GPU vivent dans
[`python/tests/gpu/*.cpp`](https://github.com/wolf75222/adc_cpp/tree/master/python/tests/gpu) (hors du graphe ctest, lancés par
sbatch/`srun`). Chacun compile la **même** logique en `exec=Cuda` ET en oracle `exec=Serial`,
puis compare cellule par cellule (`dmax = max|cuda - serial|`). Résultats réels sur GH200
(Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`) :

- Solveur **mono-grille complet** (transport + BCs + couplages + Poisson + pas de temps,
  orchestré par le `System`) : **bit-identique CPU** (phases 1-5, 7).
- Briques elliptiques post-#48 (T_e via `load_aux<5>`, EPM écranté/Helmholtz, EPM anisotrope,
  B_z par niveau AMR) : **`dmax = 0`** vs Serial, mêmes cycles MG.

**Réserves honnêtes.** Le chemin `System::add_compiled_model` (modèle DSL natif zéro-copie)
butait sur une limite `nvcc` (lambdas étendues `__host__ __device__` cross-TU) : contournée par
des **foncteurs nommés** (le chemin device réel `assemble_rhs` / `advance_amr`), mais
`test_compiled_model_parity` lui-même n'est pas encore porté device. Le capstone AMR
multi-blocs (7 tests) reste `?` sur Cuda (foncteurs nommés, en principe nvcc-compatibles, mais
sans harness ROMEO dédié).

---

## 7. MPI + Kokkos CUDA (ROMEO multi-GPU uniquement)

**Ce que c'est.** Le mode production cible : MPI distribue les sous-domaines (un GPU par rang),
Kokkos Cuda calcule sur chaque GPU, OpenMPI **CUDA-aware** échange les halos device-to-device
(UCX). C'est la config 6 + la config 4 dans un seul run.

**Pas de build local** (mêmes contraintes que la config 6 : `nvcc` ROMEO-only, jamais en CI).

**Build (sur ROMEO).** Les deux options, OpenMPI CUDA-aware :

```bash
module load cuda/12.6
cmake -S . -B build-mpicuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpicuda -j
```

**Run (SLURM, plusieurs GPU, un par rang).**

```bash
srun -n 4 --gpus-per-task=1 --constraint=armgpu ./build-mpicuda/bin/<harness>
```

**Validation : ROMEO-manuel multi-GPU (jamais en CI).** Sur un nœud à 4× GH200 (OpenMPI 4.1.7
CUDA-aware), validé np=1/2/4 (np=1 = oracle mono-GPU). Acquis :

- **10 tests** de la pile elliptique / Schur(stepper) / Poisson / system-solve / AMR
  **rank-invariants** : `dmax` cross-np = 0 (krylov_solver, mpi_poisson,
  mpi_system_solve_fields, mpi_amr_compiled_parity, mpi_amr_distributed_coarse,
  mpi_coupled_source, mpi_mbox_parity, mpi_cutcell_multibox, condensed_schur_source_stepper,
  test_schur_condensation côté invariance).
- **Validation INTÉGRÉE AmrSystem + MPI + GPU** dans **un seul run** (phase 10) : densité
  grossière bit-identique à np=1/2/4 (`dmax = 0`), masse conservée à 0.

**Réserves honnêtes.** (a) Le run intégré **ne scale pas** : le grossier est répliqué par
défaut (calcul redondant) ; le mode grossier-réparti (`distribute_coarse`) est correct et
bit-identique mais ~3.7-5x **plus lent** (le trafic de halos cross-rang domine le compute
économisé) — résultat négatif chiffré, documenté. (b) `test_schur_condensation` **échoue** côté
backend Cuda dès np=1 (défaut d'assemblage device, indépendant du nombre de rangs) ; il passe
en Serial / Kokkos Serial. (c) Sur grossier réparti, les sommes globales diffèrent à l'arrondi
entre np (ordre de réduction FMA, ~9e-13) ; le max reste bit-identique.

---

## Matrice récapitulative

| # | Backend | Options CMake (en plus de `-DCMAKE_BUILD_TYPE=Release`) | Build local ? | CI ? | Validé où |
|---|---------|---------------------------------------------------------|---------------|------|-----------|
| 1 | Serial | *(aucune)* | Oui | **Oui** (`ci-fast`, gate PR) | CI ubuntu (oracle de référence) |
| 2 | Kokkos Serial | `-DADC_USE_KOKKOS=ON` + Kokkos Serial | Oui | **Oui** (`ci-full`) | CI job `kokkos`, 91/91 ctest |
| 3 | Kokkos OpenMP | `-DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Oui | **Oui** (`ci-full`) | CI job `kokkos-openmp`, 91/91 ctest |
| 4 | MPI CPU | `-DADC_USE_MPI=ON` | Oui | **Oui** (`ci-full`) | CI job `mpi`, rank-invariant np=1/2/4 |
| 5 | MPI + Kokkos OpenMP | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Oui | **Non** (jamais MPI+Kokkos en CI) | ROMEO `x64cpu` manuel (52/57 rank-invariants) |
| 6 | Kokkos CUDA | `-DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | **Non** (pas de `nvcc` local) | **Non** (jamais CUDA en CI) | ROMEO GH200 manuel (`python/tests/gpu/`, sbatch) |
| 7 | MPI + Kokkos CUDA | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | **Non** | **Non** | ROMEO multi-GPU manuel (`srun --gpus-per-task=1`) |

**Lecture rapide :**

- Les configs **1-4** sont couvertes par la CI GitHub (1 sur chaque PR ; 2-4 en mode plein
  `ci-full` : push `master`, cron, dispatch, ou label `ci-full`).
- Les configs **5-7** ne sont **jamais** en CI : la CI ne joint jamais MPI et Kokkos, et ne
  construit jamais CUDA. Elles sont validées **manuellement sur ROMEO**, par comparaison
  bit-à-bit à l'oracle Serial (`dmax`).
- Les configs **6-7** n'ont **pas de build local** : `nvcc` ne tourne que sur le nœud GPU
  aarch64 de ROMEO.

Pour la couverture détaillée (chaque test × chaque colonne backend, avec le statut
`ci-fast` / `ci-full` / `ROMEO` / `self-skip` / `?`), la source de vérité reste
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md). Pour les phases du portage GPU et les
résultats de validation détaillés, voir [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).
