# Réception native de l'échec après préparation Halo - 2026-10-02

La préparation Halo suivie d'un échec sur le rang ciblé, le rollback et la
reprise sont reçus dans le périmètre du [reçu ROOT pinné](sdkeef92c_halo_stage_failure_reception.json).
Le vrai paquet installé utilise le build `eef92c681e4d`, ABI8, et le DSO
`8acd46dd8148`; les tests finaux sont à `5582d98ad49d`. Les 1 137 fichiers
Python/headers installés sont authentifiés et identiques avant/après chaque
exécution. Les changements intermédiaires portent sur les tests et les lecteurs,
pas sur la production installée. L'ancien SDK `3d8481`/DSO `7fd8` est conservé.

| Exécution réelle | Résultat | Mesure du harness |
| --- | --- | --- |
| Build du dépôt, Dim2 CPU Kokkos/OpenMP, MPICH/HDF5 parallèle | PASS; wheel conservée et DSO installé identique | Commande, log, CMake/Ninja et inventaires dans le reçu de build |
| Halo temporel 5/2, relais désarmé, Serial | 1 PASS | 43,56 s |
| Échec Halo @1, Serial | 1 FAIL conservé | 47,09 s |
| Échec Halo @2, Serial | 1 PASS | 46,91 s |
| Échec Halo @2, MPI2/OMP1 | 1 cas collectif PASS sur chacun des deux rangs | 51,36 s |
| Cohorte Source/host intégrée | 38 PASS | 2,73 s |

Ces temps incluent le harness et les compilations des composants nécessaires.
Ils ne mesurent pas une accélération à calcul comparable. Les deux résultats
MPI désignent un seul cas collectif. Les tests importent le paquet de
`pops-api040-halo10/site-packages`, avec `env -u PYTHONPATH`, contrôle du chemin,
du DSO, de `doctor`, de sa signature SDK et de chaque source livrée.

Le relais privé `accepted-halo-test-failure@1` est un effet de test versionné,
armé collectivement sur un owner accepté au repos. Il conserve les contrats
existants CP12, accepted9 et execution3. Les requêtes invalides, le double
armement et les requêtes MPI divergentes sont refusés avant mutation. La vraie
préparation du bloc/niveau, les copies, la fence Kokkos et leur vote réussissent
d'abord. Tous les rangs consomment ensuite la requête; seul le rang sélectionné
lève l'erreur dans une frontière votée. La publication finale du candidat n'a
pas commencé. Les bindings temporaires ont, eux, été installés pendant la
préparation. Le reçu enregistre cette causalité et les erreurs réelles.

La physique reçue demeure `dc/dt=0`, `dm/dt=m`, flux nul, Forward Euler, domaine
périodique 8×8 à deux niveaux, ratio temporel exact 5/2 avec ExplicitRemainder.
Le rollback compare les cellules valides, le stockage complet avec ghosts,
carriers, auxiliaires, Field manifests, historiques, diagnostics, géométrie et
horloge. Les checkpoints authentiques comparent tous leurs membres par
dtype/shape/octets, sauf les deux sceaux de cycle de vie explicitement déclarés.
Le test garde ses équations, guards et tolérances.

Le premier essai @1 supposait à tort que les identifiants de tentative seraient
identiques à ceux d'un contrôle sans rejet. L'analyse indépendante du payload
`POPSAND9` a trouvé 113 champs typés d'identifiant, tous `3` contre `2` après
rejet/reprise; les autres octets Program sont identiques. Le hash d'autorité
source dérivé et les deux sceaux diffèrent aussi. L'allocateur de tentatives
reste monotone et ces différences sont conservées intégralement.

Le profil de preuve @2 utilise trois vrais runtimes: reprise après rejet,
contrôle continu et contrôle soumis au même rejet puis repris. La reprise et
le contrôle aligné ont des checkpoints complets identiques hors les deux
sceaux. Le contrôle continu donne les mêmes images physiques; ses quatre
membres divergents sont inventoriés séparément. Les huit checkpoints, les
sept captures de phase et les deux erreurs/votes sont persistés. Le lecteur
indépendant examine les fichiers sauvegardés hors Native; ROOT l'a rejoué pour
Serial et MPI2, sans importer `pops`. Ses rapports conservent
`root_approval=false`; les sceaux ROOT séparés acceptent seulement cette tranche
d'exécution et de reprise.

Reproduire sur un checkout aux mêmes sources de production, avec un SDK
construit dans un préfixe distinct. Les tableaux de commandes exacts, variables
d'exécution, wheel et identités sont pinnés dans le JSON. Pour les tests reçus:

```bash
task_prefix=/Users/romaindespoulain/miniforge3/envs/pops-api040-halo10
task_output=/chemin/vers/une/nouvelle/reception
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  CONDA_PREFIX="$task_prefix" Kokkos_ROOT="$task_prefix" \
  POPS_KOKKOS_ROOT="$task_prefix" CMAKE_PREFIX_PATH="$task_prefix" \
  POPS_INCLUDE="$task_prefix/lib/python3.12/site-packages/pops/include" \
  POPS_NATIVE_DIM=2 OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  PATH="$task_prefix/bin:$PATH" "$task_prefix/bin/python" \
  docs/development/api_040/run_installed_checks.py --output "$task_output/serial" \
  --test tests/python/integration/amr/test_public_accepted_halo_stage_failure.py::test_public_accepted_halo_rank_local_failure_restores_before_publication
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  CONDA_PREFIX="$task_prefix" Kokkos_ROOT="$task_prefix" \
  POPS_KOKKOS_ROOT="$task_prefix" CMAKE_PREFIX_PATH="$task_prefix" \
  POPS_INCLUDE="$task_prefix/lib/python3.12/site-packages/pops/include" \
  POPS_NATIVE_DIM=2 OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  PATH="$task_prefix/bin:$PATH" "$task_prefix/bin/python" \
  docs/development/api_040/run_installed_mpi_checks.py --output "$task_output/mpi2" \
  --ranks 2 --dimension 2 --threads 1 --timeout 900 \
  --test tests/python/integration/amr/test_public_accepted_halo_stage_failure.py::test_public_accepted_halo_rank_local_failure_restores_before_publication
```

Le driver Serial produit son identité avant exécution; la réception conservée
ajoute une invocation `--identity-only` sous `after/`. Le driver MPI produit
lui-même ses identités avant/après et contrôle la parité des nœuds sur les rangs.
Ne pas écraser les sorties ni les préfixes déjà pinnés. Le build emploie
`scripts/build_python.sh` via `build_repository_sdk_reception.py`, après le setup
unique du worktree; sa commande exacte est conservée dans `build.command`.

Ce reçu n'établit pas une panne du transport MPI: la communication s'est
achevée avant l'erreur injectée. Le modèle n'a pas de Field/GhostBC peuplé; leur
bootstrap et leur rollback restent à recevoir. La formule indépendante du
stockage ghost complet du cas multiplicatif, le calcul mathématique du hash
d'autorité, le graphe cryptographique C++→DSO, le regrid général, Dim1/Dim3, GPU,
ROMEO scientifique et GitHub CI restent hors réception. Les autres cas reçus
sur SDK `3d8481` gardent leur artefact et ne sont pas transférés à ce DSO. Les
94 obligations demeurent ouvertes dans leur périmètre exact; prochaine tranche:
Field frais puis Ghost au vrai point initial, avec contrat distinct versionné.
