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

Le lot auteur fermé (`/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-20261006/report.md`), pins `6dc5dc8c657d9457a59253dbd6135e2acf8f9068a0664fd77bbe78ab8895300d`, et la contre-réception indépendante (`/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-independent-20261006/report.json`), pins `04181e90e47b9ef860b8248db6714c858bdefec7b1d5c2075355df3fb7e9aa0b`, sont authentifiés par la réception ROOT (`/Users/romaindespoulain/dev/tmp/root-sdk43-amr-residence-local-reception-20261007.json`), SHA `8ac7359ce474e2b3b269e130cae0224df62eea8a799774651bfbdd3a25d3eeb1`. Elle reçoit 3 305 fichiers réguliers, huit liens littéraux et 211 128 472 octets, plus les cinq fichiers de contre-revue. Aucun Native n'est relancé pour cette réception.

## Reproduction

Les commandes effectivement exécutées (`/Users/romaindespoulain/dev/tmp/sol61-sdk43-amr-residence-native-local-20261006/commands.txt`) donnent les chemins, caches, compilateurs, cibles C++ et relectures exacts. Pour une nouvelle exécution dans un checkout déjà configuré de 01d2 et un environnement `pops` compatible :

```bash
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 CC=/usr/bin/clang CXX=/usr/bin/clang++ MPICH_CC=/usr/bin/clang MPICH_CXX=/usr/bin/clang++ Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_HEAVY_MODULE_TU_POOL=2 CMAKE_BUILD_PARALLEL_LEVEL=2 bash scripts/build_python.sh --dim 2 --mpi --wheel-dir /tmp/pops-amr-069be-wheels
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 POPS_THREADS=2 OMP_PROC_BIND=false POPS_NATIVE_DIM=2 Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_checks.py --output /tmp/pops-amr-069be-mpi1-false --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[False]'
rtk proxy conda run --no-capture-output -n pops env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 POPS_THREADS=2 OMP_PROC_BIND=false POPS_NATIVE_DIM=2 Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_mpi_checks.py --output /tmp/pops-amr-069be-mpi2-false --ranks 2 --dimension 2 --threads 2 --timeout 3000 --test 'tests/python/integration/runtime/test_m19_amr_consumed_runtime.py::test_installed_amr_consumed_field_regrid_and_nonfinite_rollback[False]'
```

True suit le PASS de False et son audit, dans chaque contexte ; utiliser des dossiers distincts. Ces commandes de reproduction n'attribuent pas à une nouvelle exécution les SHA du lot reçu. Les drivers enregistrent l'identité de l'interpréteur, du vrai paquet installé et du Native chargé.

## Réception CUDA et obligations ouvertes

La campagne ROMEO CUDA **735454**, Source exact 01d2/production069be, est réellement soumise et relâchée. Le reçu d'actuation (`/Users/romaindespoulain/dev/tmp/root-sdk28-amr-vp-units-dim2-preparation-20261003/sdk43-cuda-actuation-e3bd35e0-9b33-4d02-9cb1-7c4c738e1eea/actuation-receipt.json`) conserve le binding réel ; l'observation initiale est RUNNING sur romeo-a056. Sa préparation couvre build des sept bindings, contexte GPU, quatre cibles C++, dispatch/fautes, puis M19 False et True. Aucune exécution CUDA scientifique n'est reçue par le lot CPU. Le cas C++ de réplicas exigeant deux rangs reste une obligation GPU MPI2 distincte.

La campagne 735454 termine ensuite `FAILED 1:0`, durée `01:04:23`. Le vrai NVCC refuse la lambda Kokkos dans la classe imbriquée privée `Impl`, `system_layout_transfer.cpp:629`, avant wheel/install/Native ou calcul PDE. Le lot négatif entier (`/Users/romaindespoulain/dev/tmp/sol61-sdk43-cuda-735454-negative-20261007/report.md`), la contre-réception indépendante (`/Users/romaindespoulain/dev/tmp/sol61-sdk43-cuda-735454-negative-independent-20261007/report.json`), pins `4655a622dc079490cb0e846d739435e07dfa622e2861bf91d83af2761cfcd000`, et la réception ROOT du négatif (`/Users/romaindespoulain/dev/tmp/root-sdk43-cuda-negative-reception-735454-20261007.json`), SHA `89ed904b09163911374ae01e0572c72b5409b09e45d2dcf594a33c73ed0f07a5`, préservent cet échec. Le TAR complet contient 103 entrées, dont le domaine raw de 96 membres (78 fichiers réguliers, 18 répertoires) ; les sept entrées de publication complémentaires sont retenues. Les cinq paires de préservation sont identiques : 19 préfixes CPU, sept GPU, 16 Sources, dépendances et driver, plus le registre/configuration Conda du Home. L'allocation réellement observée est un GH200 120 GB, compute capability 9.0, avec un build `sm_80` ; elle ne constitue pas une exécution de kernel. La fermeture archive réussit ; la compilation reste négative.

Le correctif [du transfert System](../../../src/runtime/system/system_layout_transfer.cpp), commit `a17a304`, remplace uniquement cette lambda par le foncteur nommé `CopyPhysicalTransferCarrier`, hors de `Impl`. La contre-revue Source confirme les mêmes vues, indices, composants, région et fences, sans nouveau header ou ABI. Ce correctif nécessite sa propre reconstruction et ne reçoit pas de PASS CUDA de l'ancien lot.

Restent ouverts : résultats CUDA effectivement sauvés et reçus, GPU MPI2, 3D, CI de la révision finale, convergence et mesures à calcul comparable, sémantique de Field/history/Aux peuplés, ainsi que les autres critères des 94 IDs. Les deux pas AMR et leur rollback ne clôturent pas le cas cinétique M19 complet.
## Additional CUDA735570 compilation refusal

The later immutable Source `eae21965f1f156d78328f53ca94a20924eb6b854` was actually rebuilt
in job735570, which ends `FAILED 1:0`,01:07:02 on romeo-a049. It passes the earlier named
physical-carrier copy location, then NVCC's generated host stub cannot name the private
`RegionTransport<2,Kokkos::CudaSpace>::PackKernel` and `UnpackKernel` types at
`include/pops/mesh/parallel/region_transfer.hpp:475/493`. The exact errors are in the original
build log at2340/2346, SHA `d36c9f17eeecbd8c1565971219125f2845182b0dd4017d8e6683c5175b2f5086`.
The compiled TU is `src/runtime/system/system_layout_transfer.cpp`, NVCC/O3/C++20/sm80,
with `--fmad=false`; official build RC1 is not a timeout. No wheel, full-seven-unit build,
installed Native or PDE GPU result is received.

The whole archive is received and independently counter-read:104 TAR entries
(84 files,20 directories),97 raw entries (79 files,18 directories,zero links), and
114 entries in the producer manifest. ROOT receipt
`/Users/romaindespoulain/dev/tmp/root-cuda44-negative-reception-735570-20261007.json` has SHA
`5293aeec956a09667fa349929f6f86d5a188120fec38d39753a22e6cf77dd8dc`;
closed producer pins are `17657577f9a653bd4fcba6a49f882f2b3a2b30336754e87f02dd319a96b7058d`.
Archive publication succeeds, late exit0; the original failure remains unchanged.
Five before/after joins are exact, preserving19 CPU prefixes,8 GPU prefixes,17 Sources,
declared Home registry/config,87 dependency records and the borrowed driver.
The allocation records GH200120GB/compute9.0, as an observation only.

The bounded Source review proposes exposing only the two named kernel types to the NVCC
stub, retaining private transport storage and unchanged kernel bodies, views, offsets,
validation, fences and capacity. This compiler-visibility correction is still to integrate
and rebuild. It changes shipped header bytes; the job's ABI11/c190 identity and failure
cannot qualify the separate ABI12/c04a transaction-capture Source.
