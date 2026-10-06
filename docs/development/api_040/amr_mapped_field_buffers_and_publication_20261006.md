# AMR consumed Field: solve-owned buffers and publication barriers

Les reçus du 6 octobre 2026 pour `b1203435aa5fc5017233fa22a04e78373e3f80b5` comprennent un build local, 82 PASS Source/codegen et trois PASS natifs locaux distincts. Le correctif C++ suivant `7edddce` conserve les causes d'échec collectif ; son Native reconstruit et ses exécutions MPI1/MPI2 sont reçus avec contre-revue indépendante. La campagne ROMEO **735312**, sur le Source immuable b120, termine `COMPLETED 0:0` et reçoit son build ainsi que ses deux cas AMR après recomputation indépendante. Chaque résultat garde sa révision, son artefact et sa portée.

## Corrections et routes préservées

Le correctif de production `0d15499906093ee01217667dfdfa09de86a262c4` répond au défaut révélé par le premier pas M19 sur ROMEO Source92/Native035 : le prototype du buffer de mapping appartenait au solve du Field consommé, alors que l'allocation passait par le scratch d'un bloc State. Le garde C++ d'ownership a correctement refusé ce buffer. Le candidat de mapping et son statut sont désormais alloués par le scratch hiérarchique du solve exact. Les imports du State destinataire conservent leur allocation State-owned.

Les premières données locales ont ensuite révélé qu'un layout consommateur sans solve local devait publier collectivement le Field avant son RHS. Ses publications qualifiées participent maintenant aux continuations et barrières existantes. La route originale de solve possède déjà sa propre séquence validée gather/solve/publish : le sélecteur la préserve lorsqu'un solve hiérarchique propre est présent. La production de `b1203435` est identique à celle du correctif de sélection `d63256bd112110b181b1c468948273230b089c55`, sur les 1 217 chemins et les sept unités de bindings.

Le témoin AMR original avait également une collision entre le diagnostic `accepted.npz` et le checkpoint de restart. Le test de `b1203435` utilise `accepted-checkpoint.npz` pour ce dernier et conserve le diagnostic séparément. Cette correction de basename ne change ni le runtime, ni les équations, ni les seuils, ni les gardes de restart. Les cinq oracles numériques d'origine restent inchangés.

## Résultats locaux reçus

| Domaine | Résultat reçu | Portée |
|---|---|---|
| Build et frontend | Build officiel local RC0 ; 82 PASS Source, zéro failure/error/skip | 74 tests affectés et huit tests de la route Field originale |
| Original AMR/restart | Un PASS natif sur le témoin original sélectionné | Diagnostic et checkpoint distincts, réaction sauvegardée et restart exact de ce cas |
| M19 consommé | Un PASS False, puis un PASS True via le driver installé | Deux pas acceptés, regrid, tableaux/CP12 et rollback sauvegardé après vote nonfini, monde local de taille 1 |
| Référence initiale complémentaire | Diagnostic GL4 local b5 PASS sur les deux jeux initiaux réels | Réception séparée Source b5/Native a15 ; elle n'est pas réattribuée à la campagne ROMEO ou à tous les backends |
| C++ b120 | 42 cas MPI1 PASS, dont les 27 cas ciblés ; 14 cas AMR uniques PASS sur chacun des deux rangs MPI2 | Compilation et exécution C++ locales séparées ; ne remplacent pas les programmes PDE générés |
| Vlasov–Poisson b120 | Un PASS natif ; référence FV/SSPRK2 recomputée, erreur maximale `1.11e-16` | Cas 4×8, deux pas, monde singleton ; convergence et M19 complet non reçus |

Les résultats natifs b120 du tableau utilisent le vrai wheel local Apple LLVM, Dim2/MPI, ABI11, HeaderSignature `c190…` et Native SHA `a15fe6c2d6180d0ce905d37a4dbc4ffa400978138b355caa9065094e0fcc3b7d`. Le rang singleton est enregistré. `OMP_NUM_THREADS=8` et `POPS_THREADS=8` sont des paramètres enregistrés ; la concurrence locale effective n'est pas mesurée rétroactivement.

[Validation ROOT et commandes](/Users/romaindespoulain/dev/tmp/root-amr-original-route-preservation-validation-20261006/source-command.json), [réception indépendante actuelle](/Users/romaindespoulain/dev/tmp/sol61-b120-local-native-reception-independent-20261006/report.json), pins `b0d0650f03ec4a0b181a82f9744fe69d47b8179b78c3eb4b954acbcf04d8f8d2`, lient le Source b120, le package installé, les trois reçus natifs et les vrais CPP/DSO/NPY/CP12. La recomputation offline conserve les oracles et reçoit le regrid ainsi que le rollback des états et cursors sauvegardés. Elle ne remplace pas une étude de convergence ou d'histoire complète.

Le premier lot local du correctif 0d/b5, ses 74 PASS Source et deux cas M19, reste conservé dans les [preuves initiales](/Users/romaindespoulain/dev/tmp/sol61-amr-field-buffer-production-fix-evidence-20261006/README.md) et leur [contre-revue](/Users/romaindespoulain/dev/tmp/sol61-amr-mapped-field-publication-independent-r2-20261006/report.json). Le [diagnostic IC local distinct](/Users/romaindespoulain/dev/tmp/sol61-m19-local-initial-reference-b5b1547d-20261006/README.md) vérifie les valeurs nominales et moyennes GL4 depuis les vraies coordonnées CP12, avec les bornes déjà fixées et les bits des constantes. Son périmètre reste celui des données initiales b5 reçues.

## Reproduction locale

Pour reproduire le lot historique b120, utiliser un checkout de `b1203435aa5fc5017233fa22a04e78373e3f80b5`. Depuis la production `7edddce`, les mêmes commandes exécutent les mêmes cas sur le Native reconstruit, sans réattribuer les reçus b120/a15 à ce nouvel artefact. Activer l'environnement `pops`, définir les deux racines Kokkos et utiliser un nouveau répertoire de résultats sur un filesystem compatible avec les opérations atomiques de checkpoint. Les commandes ci-dessous n'annoncent aucune nouvelle exécution.

```bash
conda activate pops
export POPS_ENV_NAME=pops
export POPS_KOKKOS_ROOT="$CONDA_PREFIX"
export Kokkos_ROOT="$CONDA_PREFIX"
export POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1
export OMP_NUM_THREADS=8 POPS_THREADS=8 PYTHONDONTWRITEBYTECODE=1
TASK_EVIDENCE=$(mktemp -d /tmp/pops-amr-reproduction-XXXXXX)

env -u PYTHONPATH -u PYTHONOPTIMIZE bash scripts/build_python.sh \
  --dim 2 --mpi --wheel-dir "$TASK_EVIDENCE/wheels"
env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" -m pytest -q \
  tests/review/test_sol61_mapped_field_route.py \
  tests/python/unit/runtime/test_mapped_field_release_admission.py \
  tests/python/unit/fields/test_program_field_problem.py \
  tests/python/unit/fields/test_amr_original_field_codegen.py \
  --junitxml="$TASK_EVIDENCE/source.xml" --basetemp="$TASK_EVIDENCE/source-data"

env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output "$TASK_EVIDENCE/original-amr" \
  --test 'tests/python/integration/runtime/test_public_amr_original_field.py::test_public_original_amr_saved_reaction_and_exact_restart[16-order0-False]'
env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output "$TASK_EVIDENCE/amr-false" \
  --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[False]'
# Exécuter True seulement après PASS authentifié de False :
env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output "$TASK_EVIDENCE/amr-true" \
  --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[True]'
```

`scripts/setup_env.sh --cpu` a déjà été exécuté une fois dans ce worktree ; le build utilise son flux incrémental habituel. Les [commandes capturées M19 False](/Users/romaindespoulain/dev/tmp/root-amr-original-route-preservation-validation-20261006/activated-m19/amr-false-command.json) et [True](/Users/romaindespoulain/dev/tmp/root-amr-original-route-preservation-validation-20261006/activated-m19/amr-true-command.json) passent par `conda run -n pops`, avec les mêmes racines Kokkos et le véritable interpréteur installé. Les XML, identités, logs et résultats du driver déterminent la réception de chaque cas.

## ROMEO et obligations restantes

L'ancien **735043** est reçu whole-Raw NEGATIVE sur Source92/Native035 : checkpoint initial reçu, échec au premier pas sur l'ownership du scratch, True absent. Son domaine historique de préservation reste **17 CPU, 7 GPU et 14 Sources**. La [réception physique](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/closed-raw-20261006/sdk41-amr-local-735043-physical/root-reception.json), le [lecteur ROOT RC0](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/sdk41-amr-node-local-retry-r2-16951221-b27e-475d-b09b-769df032579b/actual-reader-observation-735043-physical-modes.json) et la [contre-revue négative](/Users/romaindespoulain/dev/tmp/sol61-sdk41-amr-local-negative-735043-independent-20261006/report.json) conservent ce résultat ; ils ne qualifient pas le correctif b120. La sonde 735054 a reçu séparément le contexte MPI world1/OpenMP CPU2 du Native035, sans physique.

La campagne **735312**, OP `2cd8f6fa-7883-4049-8c63-699bf0e22c1c`, utilise le namespace distinct `/gpfs/projet/r250127/api040-composition15-mpich-20261002-v15/qualification-sdk41-amr-mapped-fix-b120343-20261006`. Le [statut terminal réel](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/sdk42-amr-mapped-fix-b120343-2cd8f6fa-7883-4049-8c63-699bf0e22c1c/actual-terminal-735312.json) reçoit `COMPLETED 0:0`, durée `01:17:25`, sur c025. L'[archive entière reçue](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/closed-raw-20261006/sdk42-amr-built-735312/root-reception.json) conserve les résultats originaux. La [contre-réception indépendante](/Users/romaindespoulain/dev/tmp/sol61-sdk42-amr-built-735312-independent-20261006/report.json), pins `d222c8b3f2fa26d76dee4ba8f5b32cdff57f121b71d824578e749917039a0e0f`, et la [réception ROOT](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/root-sdk42-amr-build-reception-735312.json), SHA `6af39e06bd229a0ef8fdb26da60c75992ec7cf0e44757c93a857fa1ce30ade35`, reçoivent le vrai build sept TU, wheel/install/doctor, Native `cf3d2fd4…`, monde MPI1/OpenMP8 mesuré, puis False et True, les deux pas, regrid et rollback sauvegardé après refus nonfini. Les événements Ninja et le lien sont conservés ; les fichiers objets physiques ne sont pas dans l'archive. Les 18 CPU, sept GPU, 15 Sources et registre Conda sont identiques avant/après. La sémantique complète des payloads Field/history/Aux reste non reçue. Ce lot GNU ROMEO b120/cf3 ne qualifie pas le Native local d725, MPI2, CUDA, 3D, convergence, coûts ou CI.

Les [six cibles C++ locales b120](/Users/romaindespoulain/dev/tmp/sol61-sdk42-native-cpp-amr-local-20261006/report.json) ont terminé : 42 cas MPI1 et 14 cas AMR uniques sur chacun des deux rangs MPI2, avec MPICH/OpenMP2. Les 27 cas ciblés appartiennent aux familles `ExactAuxiliaryRegistryNd` et `EllipticNamedDeviceKernels`. Les flags réellement enregistrés incluent le `-O0` imposé par le dépôt aux quatre cibles AMR ; les deux autres familles et les bibliothèques runtime utilisent `-O3`.

La [réception Vlasov–Poisson locale b120](/Users/romaindespoulain/dev/tmp/sol61-m19-vp-native-independent-20261006/report.json) conserve le vrai paquet et trois phases sauvegardées. La référence indépendante reçoit deux pas SSPRK2, positivité et spectateur exact ; la taille 4×8 ne reçoit pas l'étude PDE complète ou sa convergence.

Le [cas M19 MPI2 b120](/Users/romaindespoulain/dev/tmp/sol61-sdk42-m19-mpi2-local-20261006/report.json) reste FAILED sur les deux rangs : les deux pas, regrid et rollback CP12/NPY sont reçus, mais le wrapper de transfert C++ efface la cause nonfinie de la faute. True n'a pas été lancé dans ce lot négatif. Le correctif `7edddce1255012700fac7d098c94a3bf18d23141` remplace les deux propagations génériques par le helper collectif existant qui conserve la cause du premier rang fautif. Il conserve les kernels, le staging, la publication, les états et les gardes. Sa [contre-revue statique](/Users/romaindespoulain/dev/tmp/sol61-sdk42-m19-collective-cause-source-review-20261006/report.json) passe.

Le [nouveau lot natif 7eddd](/Users/romaindespoulain/dev/tmp/sol61-sdk42-collective-cause-native-local-20261006/report.md) reconstruit et installe réellement Native `d725dcc30a411f7782ee290de989143f83a14bf2bca6c8b17579e545a0e6994c`, ABI11/header c190. False puis True passent en MPI1 et sur chacun des deux rangs MPI2 ; les quatre audits CP12/NPY des pas, regrid et rollback passent. La cause reçue en MPI2 est `AMR transfer physical candidate staging; rank 0: AMR physical candidate is non-finite`. Les quatre cibles C++ affectées sont reconstruites : 12 PASS en MPI1, deux skips exigeant le multirang, puis 14 cas uniques PASS sur chaque rang MPI2. Les [commandes réelles](/Users/romaindespoulain/dev/tmp/sol61-sdk42-collective-cause-native-local-20261006/commands.txt) et le paquet a15 préservé sont conservés. La [contre-réception indépendante](/Users/romaindespoulain/dev/tmp/sol61-sdk42-collective-cause-native-local-independent-20261006/report.json), pins `eb90dffc9f79527e2e2a1920b96f3a7ff40eb4f5d3cca2a0b2e7c38ea3aa8feb`, recompute les quatre audits et reçoit 3 217 fichiers réguliers et cinq liens littéraux. La [réception ROOT locale](/Users/romaindespoulain/dev/tmp/root-sdk42-collective-cause-local-reception-20261006.json), SHA `36badef6adae93922833e3e6f2eac3cf0ffb8e56d827e9a91e2abc3c72d3f0b6`, authentifie ces payloads et la contre-revue. Les preuves Native a15 précédentes gardent leur source b120 ; ROMEO735312 reste un build distinct de b120. Les paramètres OMP2 par rang sont capturés ; la concurrence effective des tests M19 n'a pas été sérialisée rétroactivement.

La réception ROMEO b120 ci-dessus et la réception locale 7eddd gardent leurs scopes distincts. Les [deux cas BGK isolés 32×32 et 64×64](m19_isolated_bgk_native_reception_20261006.md) sont également reçus sur d725. CUDA, 3D, CI de la révision finale, convergence et coûts comparables restent ouverts. Les 94 IDs de mission existants, leurs équations, critères et lacunes sont maintenus ; ces résultats bornés ne clôturent pas la mission complète.
