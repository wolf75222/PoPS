# AMR physical transfers in the actual Kokkos memory space

Le port de production `069be5055a218c809f0df35e5226ff1c40446f20` est reçu sur le paquet natif local Dim2/MPI reconstruit. Les paramètres AMR originaux False puis True passent en MPI1 et MPI2, avec contre-recomputation indépendante des cinq phases sauvegardées, des deux pas, du regrid et du rollback nonfini. Cette réception complète la [réception des buffers et publications](amr_mapped_field_buffers_and_publication_20261006.md), dont chaque résultat conserve sa propre révision.

## Mécanisme de production

Le transfert physique utilisait des pointeurs de `std::vector<double>` et un contexte Host, alors que les `Fab` CUDA possèdent une autre résidence. Dans [amr_layout_transfer.cpp](../../../src/runtime/amr/amr_layout_transfer.cpp), la résidence attendue vient maintenant de `Fab::memory_space`. Le contexte d'exécution doit lui correspondre exactement ; les contrôles de type scalaire, d'autorité MPI et d'emprunt du stream restent actifs.

`IntegralBuffers<MemorySpace>` conserve le chemin Host existant. Sur un autre backend, il possède les vues Kokkos de source et de résultat, copie la source et le résultat initial vers cette mémoire, appelle le même opérateur intégral avec le stream natif exact, puis rapporte le résultat terminé vers Host. Les propriétaires restent vivants jusqu'aux fences, y compris en cas d'exception. Le fournisseur existant prépare les poids dans cette même résidence. La préparation budgète le pic de la contribution séquentielle, ses deux buffers et les poids ; un budget inférieur est refusé avant publication.

La géométrie, les intégrales, les conventions de signe, les échanges MPI et la publication transactionnelle sont conservés. Cette correction étend l'exécution du contrat physique existant, sans nouvel opcode, recette par modèle ni changement ABI : ABI11, signature de headers c190 et MPI ABI b4a615 restent inchangés. Les coûts supplémentaires non-Host sont deux allocations et copies Host vers backend, une copie retour par contribution, plus la préparation des poids existante. Leur coût comparable n'est pas encore mesuré.

## Preuves reçues

| Élément | Résultat et portée |
| --- | --- |
| Build officiel incrémental | RC0, 23.90 s ; `setup_env.sh` n'est pas répété |
| Native installé | `ce024e70e66a9258f4e39d3405f93d4919e9189f502b0d2a1c205cabed0baacf`, vrai paquet `pops` de l'environnement Conda `pops` |
| Wheel retenu | `62be9f0487e4f7cfa3b27111267a86bb17f2b9534a836d3041d344effa0d0b72`, preuve d'installation complète et doctor PASS |
| M19 original MPI1 | False puis True PASS ; fixtures 069be et descendant documentaire a084, production identique |
| M19 original MPI2 | False puis True PASS sur chaque rang ; fixture nommé `01d2cd22e7b6a75561b04feb6f1a657496efb2d3`, production 069be identique |
| Références indépendantes | Quatre recomputations des phases CP12/NPY, clocks, regrid et rollback PASS ; oracles, équations et seuils originaux inchangés |
| Quatre cibles C++ affectées | 13 cas uniques PASS MPI1 et deux skips originaux réservés au multirang ; 15 cas uniques PASS par rang MPI2, zéro failure/error/skip ; union de 16 cas |
| Préservation | Ancien paquet d725 complet, 2 563 fichiers réguliers et 223 répertoires, conservé et comparé ; nouveaux membres inchangés entre le snapshot pris après le premier MPI1 False et la fin |

Le test supplémentaire de résidence refuse un contexte qui diffère des vrais champs et vérifie l'absence de mutation. Le cas de réplicas C++ MPI2 constate deux rangs et OpenMP concurrency1 sur chaque rang, conformément au réglage CTest original ; la concurrence des autres cas n'en est pas déduite. Une sonde native séparée constate world1/OpenMP2/Host/double ; elle ne mesure pas rétroactivement la concurrence des processus M19.

Le premier MPI2 False sur la fixture 069be reste un négatif avant bind et avant PDE : les deux rangs ne publiaient pas le même binaire de composant. Quatre compilations diagnostiques des deux vrais packages ont montré que Clang/Kokkos sérialisait le chemin temporaire de la lambda de faute. La correction 01d2 nomme uniquement son foncteur `AmrTestNonfiniteDestination` ; cible, index0, NaN, RangePolicy, stream et fences sont identiques. Quatre compilations corrigées donnent deux paires de SHA identiques. Le contrôle d'identité des composants reste strict. Les huit DSO et les sources de ce diagnostic sont retenus.

Le [lot auteur fermé](/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-20261006/report.md), pins `6dc5dc8c657d9457a59253dbd6135e2acf8f9068a0664fd77bbe78ab8895300d`, et la [contre-réception indépendante](/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-independent-20261006/report.json), pins `04181e90e47b9ef860b8248db6714c858bdefec7b1d5c2075355df3fb7e9aa0b`, sont authentifiés par la [réception ROOT](/Users/romaindespoulain/dev/tmp/root-sdk43-amr-residence-local-reception-20261007.json), SHA `8ac7359ce474e2b3b269e130cae0224df62eea8a799774651bfbdd3a25d3eeb1`. Elle reçoit 3 305 fichiers réguliers, huit liens littéraux et 211 128 472 octets, plus les cinq fichiers de contre-revue. Aucun Native n'est relancé pour cette réception.

## Reproduction

Les [commandes effectivement exécutées](/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-20261006/commands.txt) donnent les chemins, caches, compilateurs, cibles C++ et relectures exacts. Pour une nouvelle exécution dans un checkout déjà configuré de 01d2 et un environnement `pops` compatible :

```bash
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 CC=/usr/bin/clang CXX=/usr/bin/clang++ MPICH_CC=/usr/bin/clang MPICH_CXX=/usr/bin/clang++ Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_HEAVY_MODULE_TU_POOL=2 CMAKE_BUILD_PARALLEL_LEVEL=2 bash scripts/build_python.sh --dim 2 --mpi --wheel-dir /tmp/pops-amr-069be-wheels
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 POPS_THREADS=2 OMP_PROC_BIND=false POPS_NATIVE_DIM=2 Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_checks.py --output /tmp/pops-amr-069be-mpi1-false --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[False]'
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 POPS_THREADS=2 OMP_PROC_BIND=false POPS_NATIVE_DIM=2 Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_mpi_checks.py --output /tmp/pops-amr-069be-mpi2-false --ranks 2 --dimension 2 --threads 2 --timeout 3000 --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[False]'
```

True suit le PASS de False et son audit, dans chaque contexte ; utiliser des dossiers distincts. Ces commandes de reproduction n'attribuent pas à une nouvelle exécution les SHA du lot reçu. Les drivers enregistrent l'identité de l'interpréteur, du vrai paquet installé et du Native chargé.

## Réception CUDA et obligations ouvertes

La campagne ROMEO CUDA **735454**, Source exact 01d2/production069be, est réellement soumise et relâchée. Le [reçu d'actuation](/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/sdk43-cuda-actuation-e3bd35e0-9b33-4d02-9cb1-7c4c738e1eea/actuation-receipt.json) conserve le binding réel ; l'observation initiale est RUNNING sur romeo-a056. Sa préparation couvre build des sept bindings, contexte GPU, quatre cibles C++, dispatch/fautes, puis M19 False et True. Aucune exécution CUDA scientifique n'est reçue par le lot CPU. Le cas C++ de réplicas exigeant deux rangs reste une obligation GPU MPI2 distincte.

Restent ouverts : résultats CUDA effectivement sauvés et reçus, GPU MPI2, 3D, CI de la révision finale, convergence et mesures à calcul comparable, sémantique de Field/history/Aux peuplés, ainsi que les autres critères des 94 IDs. Les deux pas AMR et leur rollback ne clôturent pas le cas cinétique M19 complet.
