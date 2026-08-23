# Performance : advection sinusoïdale publique

Cette campagne mesure le cycle de vie PoPS **public en Python**. Le cas
[`advection_sine.py`](advection_sine.py) se lit de haut en bas : réglage des
threads, domaine périodique, modèle, MUSCL–Van Leer + flux scalaire upwind,
SSPRK2/`FixedDt`, initialisation analytique projetée, puis
`validate → resolve → compile → bind → run`.

Il n’existe aucun workload C++ autonome dans ce dossier. Le C++ reste la bonne
frontière pour corriger l’infrastructure générique (Kokkos, réduction,
communicateur), mais le cas scientifique et la mesure restent le même chemin
public que celui emprunté par un utilisateur PoPS.

## Mesure exacte

Une répétition construit une simulation fraîche hors chronométrage. Une barrière
MPI éventuelle et `RuntimeInstance.synchronize()` précèdent le départ;
`pops.run(..., console=False)` est l’unique région chronométrée; la
synchronisation Kokkos suit immédiatement l’arrêt. La métrique est donc
`public_lifecycle_wall_seconds`, jamais un kernel-only C++.

L’oracle est hors chronométrage : état global initial/final, erreur de moyenne
de cellule contre la sine analytique, intégrale native contre intégrale hôte,
finitude et boîtes locales appartenant au rang. Chaque rang écrit uniquement
`rank-xxxxx.json`; le collecteur refuse une campagne incomplète, un pavage
global incomplet/chevauchant, une concurrence/backend/MPI divergents, une UUID
CUDA absente ou dupliquée, et tout échec numérique.

Chaque résultat lie aussi l'identité de l'artifact compilé (token, clé ABI et
clé de cache) aux SHA-256 et tailles de tous les binaires `Program` utilisés
par les répétitions chronométrées. Le collecteur exige cette même autorité sur
tous les rangs d'un point.

## Fichiers

- `advection_sine.py` : cas scientifique public et linéaire ;
- `support.py` : parsing et publication JSON atomique sans écrasement ;
- `campaigns/` : les sept campagnes ROMEO, schéma v2 ;
- `run_campaign.py` : appelle `sys.executable` et le cas Python ;
- `collect_results.py` : agrège le maximum de rang à chaque répétition, puis
  médiane/MAD et débit ;
- `plot_scaling.py` : lit uniquement un `report/summary.json` couvert par le
  `COMPLETE.json` authentifié ;
- `slurm/` : construit le module Python PoPS Release dans le scratch et lance
  les campagnes complètes, une à la fois.

Les données de campagnes antérieures R1–R8 sont historiques et exclues : elles
ne satisfont pas ce contrat Python v3.

## Publication des figures

Après la collecte validée d'une campagne, le traceur lit seulement son summary
JSON et publie un répertoire neuf :

```bash
python3 benchmarks/performance/advection_sine/plot_scaling.py \
  benchmarks/performance/advection_sine/results/<campagne>/summary.json \
  --output benchmarks/performance/advection_sine/figures/<publication>
```

Les PNG et SVG sont rendus dans un staging frère, accompagnés de
`plot_manifest.json` et d'un `ANALYSIS.md` généré depuis les données scellées,
puis le répertoire complet est publié en une opération.
Ce manifeste relie la publication au hash du summary, aux options et à l'octet
du traceur, et contient les hashes de tous les médias. Une cible déjà existante
est refusée, même si elle apparaît au dernier instant de publication : l'appel
natif no-replace de macOS/Linux ne peut écraser ni figure ni publication
antérieure.

## Profilage local macOS

[`profiling/`](profiling/README.md) documente une acquisition séparée avec
`/usr/bin/sample` et `xctrace Time Profiler` sur le point OpenMP `t8` canonique.
Elle requiert cinq processus complets par outil, un handoff READY/GO après le
warmup, et des reçus/hashes. Ces temps ne vont jamais dans le scaling; voir
[`MACOS_PROFILE_ANALYSIS.md`](MACOS_PROFILE_ANALYSIS.md).

## Validation de configuration

```bash
bash benchmarks/performance/advection_sine/validate_all.sh
python3 -m py_compile benchmarks/performance/advection_sine/*.py
python3 -m py_compile benchmarks/performance/advection_sine/profiling/*.py
python3 -m unittest discover -s benchmarks/performance/advection_sine/tests -v
```

Ces vérifications n’exécutent aucun calcul scientifique. Le `--dry-run` de
`run_campaign.py` imprime la commande Python complète sans exécuter le cas.

## ROMEO

Les wrappers exportent deux sources content-addressed : PoPS et le commit Git
Kokkos propre observé à la soumission. Les archives et reçus Kokkos sont
scellés par SHA-256 dans le scratch, revérifiés puis extraits dans le répertoire
privé du job. Kokkos y est reconstruit avec code indépendant de la position
pour la route exacte (Serial, OpenMP ou CUDA Hopper90), avant de construire
`_pops` à la racine PoPS avec `POPS_BUILD_PYTHON=ON` et `Release`. Le job
utilise ensuite uniquement les installations et le paquet `build/python` de
ce scratch authentifié. Il impose
`POPS_USE_HDF5=OFF` pour toutes les routes, y compris MPI : ce benchmark ne
fait aucun I/O HDF5 et une route MPI ne doit pas introduire cette dépendance.

Avant tout lancement, `raw/build.receipt.json` vérifie l’import réel de
`pops._pops` sélectionné depuis le build-tree et lie ses octets à la source, à
la dimension, au backend MPI/Kokkos, au compilateur, au commit/archive Kokkos,
aux options effectives de son cache CMake, à `libkokkoscore`, à l'installation
Kokkos réellement résolue par PoPS, et à CUDA lorsque pertinent. Cette vérification se fait
dans une étape `srun` dédiée (avec GPU pour la route CUDA), sans intégration
temporelle. Le reçu d'allocation `observed-allocation/python-dependencies.json`
conserve en plus l'interpréteur effectif et les versions/chemins NumPy et
pybind11 vérifiés avant le build. Le collecteur recoupe ensuite
ce reçu avec `source.manifest.json`, `campaign.normalized.json`, `launch.json`,
les ressources de chaque point et le pas de temps dyadique. Ce sont des gardes
de format et de provenance : ils ne constituent pas des résultats tant qu’une
campagne complète n’a pas été validée.

Pour les routes MPI, le reçu distingue le lanceur effectif SLURM `srun` du
lanceur OpenMPI `mpirun`; les deux versions et binaires sont authentifiés.
Chaque job publie aussi stdout/stderr Slurm, `scontrol show job -dd`, la
liste/nature des nœuds, `lscpu`, l'affinité CPU/GPU et un
`observed-allocation/environment-provenance.json` limité aux variables Slurm,
OpenMP et CUDA utiles à la reproductibilité. Il n’archive jamais
l’environnement complet. Les soumissions utilisent
`sbatch --export=<allowlist-explicite>` (sans `ALL`, `NONE` ni `NIL`) avec les
seuls outils de build, dépendances Python/Kokkos et chemins de travail requis :
secrets, jetons, proxies et `PYTHONPATH` du shell de
soumission ne traversent donc pas vers l’allocation. Une valeur non sûre dans
cette allowlist est refusée avant `sbatch`. Sur GPU, le reçu conserve aussi
`nvidia-smi` complet + UUID/topologie/fréquences. Ces reçus sont inventoriés par
`COMPLETE.json`.

Prérequis : Python 3.10+ de l'architecture du nœud, CMake et Ninja Spack des
wrappers, checkout Git Kokkos propre et compilateur compatible; sur GPU CUDA
12.6, wrapper NVCC Kokkos et OpenMPI 4.1.7 `+cuda` pour MPI. Les wrappers démarrent avec
`PATH=/usr/bin:/bin`, puis sourcent directement et vérifient le setup Spack
ROMEO de leur plateforme; ils ne dépendent ni de `romeo_load_*_env`, ni de
`.bashrc`. La route GPU initialise également le module système depuis
`/etc/profile.d/modules.sh` avant `cuda/12.6`. Les versions observées
compilateur/Kokkos/Python/pybind11/CUDA/OpenMPI sont dans le reçu de build;
armgpu refuse toute compilation sur le login x86_64.

Les checkouts Kokkos par défaut sont
`$HOME/adc_cpu_mpiomp/kokkos` pour Serial/OpenMP et
`$HOME/adc_gpu_p1/kokkos` pour CUDA. Ils peuvent être remplacés à la
soumission par `POPS_KOKKOS_SERIAL_SOURCE` et
`POPS_KOKKOS_CUDA_SOURCE`. Dans tous les cas, le chemin doit être la racine
d'un dépôt Git sans modification suivie; le wrapper archive le commit au lieu
de transmettre ce chemin mutable au job.

Le Python de job est contrôlé **dans l'allocation** avant CMake : il doit être
un chemin absolu exécutable, Python 3.10+, et importer `numpy` et `pybind11`
2.13+. Les defaults ROMEO relevés le 2026-08-22 sont déjà épinglés :
`/rkvb73h` (`py-numpy@1.26.4`) + `/j4cl5xe`
(`py-pybind11@2.13.5`) sur x64cpu, et `/qwvf7fx`
(`py-numpy@2.1.2`) + `/oshtzly` (`py-pybind11@2.13.5`) sur armgpu. Pour une
qualification ultérieure, remplacer ces spécifications par les hashes exacts
observés sur l'architecture concernée :

```bash
# x64cpu
export POPS_PYTHON_NUMPY_X64_SPEC='/rkvb73h'
export POPS_PYBIND11_X64_SPEC='/j4cl5xe'
# armgpu (aarch64) -- à définir sur ROMEO depuis une allocation armgpu
export POPS_PYTHON_NUMPY_ARMGPU_SPEC='/qwvf7fx'
export POPS_PYBIND11_ARMGPU_SPEC='/oshtzly'
```

Un site peut remplacer cette activation par un fragment shell absolu et
lisible (`POPS_PYTHON_DEPENDENCY_ACTIVATION_X64` ou
`POPS_PYTHON_DEPENDENCY_ACTIVATION_ARMGPU`). Si `python -m pybind11
--cmakedir` n'est pas disponible, fournir le répertoire CMake absolu validé
dans `POPS_PYBIND11_DIR_X64` ou `POPS_PYBIND11_DIR_ARMGPU`. Le wrapper passe
ce répertoire avec `-Dpybind11_DIR` et configure
`FETCHCONTENT_FULLY_DISCONNECTED=ON` : aucune dépendance Python/native ne peut
être téléchargée silencieusement sur un nœud de calcul. Chaque invocation qui
préfixe `PYTHONPATH` par le build et la source conserve ensuite le chemin de
dépendances activé, pour que NumPy ne disparaisse pas au lancement `srun`.

```bash
export POPS_SLURM_ACCOUNT='compte-ROMEO'
# ROMEO x64cpu, relevé avec l'environnement Spack x64cpu le 2026-08-22.
export POPS_PERF_LOGIN_PYTHON='/apps/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/aocc-5.0.0/python-3.10.14-p3ga5vdvdxumbqoieap3irxh7pf2hsph/bin/python'
export POPS_PERF_JOB_PYTHON="${POPS_PERF_LOGIN_PYTHON}"
export POPS_PYTHON_NUMPY_X64_SPEC='/rkvb73h'
export POPS_PYBIND11_X64_SPEC='/j4cl5xe'
export POPS_PERF_RESULTS_ROOT="/scratch_p/${USER}/pops-sine-results"
benchmarks/performance/advection_sine/slurm/submit_x64cpu.sh \
  benchmarks/performance/advection_sine/campaigns/strong_mpi_openmp.json
```

Pour armgpu, garder ce Python x64cpu pour `POPS_PERF_LOGIN_PYTHON` et définir
`POPS_PERF_JOB_PYTHON` sur le Python aarch64 relevé :
`/apps/2025/spack_install/linux-rhel9-neoverse_v2/linux-rhel9-neoverse_v2/gcc-11.4.1/python-3.11.9-oxq4fb72flcinkm57pazy3ti7tpfp7rf/bin/python`.
Ce binaire ne peut pas être exécuté depuis le login x86_64; le wrapper le
vérifie après l'initialisation explicite de Spack, des modules et de CUDA dans
l'allocation armgpu.

Soumettre exactement une campagne complète, vérifier `COMPLETE.json` et les
reçus, puis seulement lancer la suivante. `COMPLETE.json` scelle les inventaires
SHA-256 de `raw/` et `report/` (summary et CSV compris); il est revérifié après
publication atomique. Après cette vérification uniquement, le répertoire de
travail exact du job est supprimé pour libérer le quota; en cas d’échec il reste
en place pour diagnostic. Les résultats publiables ne sont pas préremplis dans
ce dépôt.

Si une campagne complète dépasse `instant`, passer sa réservation à `short`
avec un walltime cohérent est permis; réduire une résolution, les 32 pas, le
warmup, les cinq répétitions ou un point de scaling ne l'est pas. Les sept JSON
sont une liste canonique scellée et toute dérive est refusée. Toute collecte,
tout scellement `COMPLETE.json` et toute publication exigent un commit propre et
immuable : un worktree Git sale est refusé avant l'export, et les anciens
manifests ou reçus `source_dirty=true` sont refusés par les consommateurs.
Utiliser pour cela `POPS_PERF_WALLTIME_PARTITION=short` et, si nécessaire,
`POPS_PERF_WALLTIME=HH:MM:SS` : l'override est transmis à `sbatch`, jamais au
JSON versionné, et la réservation observée est conservée par `scontrol`.

## Limites

Le workload de performance est uniforme et synchrone. Il mesure la route Python
sur Serial/OpenMP/CUDA et MPI, mais ne qualifie pas AMR, regridding, refluxing,
patch mobile ou subcycling; ces propriétés relèvent de
[`../../verification/advection/sine_wave/`](../../verification/advection/sine_wave/README.md).
