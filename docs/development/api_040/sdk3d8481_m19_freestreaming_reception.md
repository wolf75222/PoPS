# M19 : transport signé reçu, Serial et MPI2

Les quatre vrais runs 32×8/64×12 Serial/MPI2 passent et sont reçus dans le profil `signed-freestreaming-original-ssprk2-uniform@1`. [Reçu, owner ROOT et approbation hashés](sdk3d8481_m19_freestreaming_reception.json). Ce profil reçoit le transport libre ; BGK et Vlasov–Poisson complets restent à exécuter.

| Maillage x×v | Serial, driver | MPI2, driver |
|---|---:|---:|
| 32×8 | 1 PASS, 54.939 s | 1 cas collectif PASS, 59.904 s |
| 64×12 | 1 PASS, 57.125 s | 1 cas collectif PASS, 65.719 s |

La composition Python conserve `ddt(f)=-div((0,v*f))`, flux v nul, x périodique, coefficient signé Aux analytique, HLL avec paire explicite `(v,v)`, FirstOrder et les deux étapes SSPRK2 exprimées dans Program. Aucun calcul Python par cellule dans la trajectoire. Le layout natif est `(nv,nx)` ; les exports NumPy `(1,nx,nv)` et Aux `(nx,nv)` respectent le renversement des axes natifs. `dt=1/(4*nx)`, état accepté à 1/8, continuation/replay à 1/4.

Les références indépendantes vérifient le symbole upwind signé et son polynôme SSPRK2, puis l'intégrale continuum moyennée en x et v, avec son moment dérivé de sinc. Les gardes restent 3e-12 pour Fourier et les trois moments physiques, `.65*t/nx` pour l'erreur continuum et `.005` pour l'évolution non triviale ; Aux signé à 1e-14. Les cinq phases et leurs checkpoints UniformCP8 authentiques sont sauvegardées ; états, caches/histoires et payloads non réservés sont bitexact après reprise/replay.

Le paquet ABI8/Kokkos/OpenMP Dim2/MPICH construit sur `3d8481c9`, DSO `7fd8c7fe…f846a44`, est conservé. Les fixtures proviennent de `471263bb`, production identique au build. ROOT rehash les 1 136 fichiers installés et leurs sources, DSO, cinq phases, binaires générés, sidecars, C++/IR Program, manifeste modèle et JUnit par rang ; les identités et inventaires avant/après sont exacts. Le non-auteur (`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sol61-m19-full-four-independent-reception.json`) a fermé indépendamment les quatre archives. Le lecteur scientifique conserve `native_authority_or_root_approval_verified=false` : ROOT ajoute séparément l'authentification et son approbation, sans modifier le résultat du lecteur.

Deux vrais échecs de fixture restent archivés : mauvais appel du wrapper Aux, puis attribut C++ modèle inexistant. Les corrections utilisent le vrai ComponentKey, CompiledModel, CompiledProblem et ModuleManifest.to_dict ; 35 tests Source intégrés passent après contre-revue. Aucun seuil ni équation n'est changé. L'absence de C++ modèle est explicitement `null` ; le C++ Program est retenu et jamais régénéré pour produire une fausse preuve. IR de sérialisation Program 5 vérifié ; le numéro du plan natif n'a pas de témoin explicite dans ces captures et reste inconnu. Le graphe C++→DSO n'est pas reconstruit.

Reproduction : utiliser les variables SDK des [commandes N8/N16](sdk3d8481_spatial_v4_scientific_reception.md) avec une sortie nouvelle et sélectionner :

```sh
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_checks.py --output /Users/romaindespoulain/dev/tmp/api040-reproduce-m19-32x8-serial --test 'tests/python/integration/runtime/test_m19_freestreaming_runtime.py::test_native_m19_signed_freestreaming_exact_restart[32-8]'
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_checks.py --identity-only --output /Users/romaindespoulain/dev/tmp/api040-reproduce-m19-32x8-serial/after
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_mpi_checks.py --ranks 2 --dimension 2 --threads 1 --timeout 1800 --output /Users/romaindespoulain/dev/tmp/api040-reproduce-m19-64x12-mpi2 --test 'tests/python/integration/runtime/test_m19_freestreaming_runtime.py::test_native_m19_signed_freestreaming_exact_restart[64-12]'
```

Les result.json liés conservent les commandes exactes des quatre runs et les JUnit indiquent le chemin du receipt. Le lecteur `tests/review/sol61_m19_freestreaming_saved_audit.py` prend ce chemin et son SHA256, sans importer PoPS ni attribuer une autorité Native à ses seuls hashes. La préparation ROOT conservée est `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/prepare_root_m19_sdk8_reception.py` ; sa sortie scellée est préservée. Un nouveau run exige une nouvelle préparation et approbation.

Scope exact : Uniform Dim2 CPU, transport libre signé et continuation. Collisions BGK, champ Poisson, AMR cinétique, GPU, ROMEO, CI et performance comparative restent ouverts. Cette réception ne ferme pas M19 complet ni les 94 obligations.
