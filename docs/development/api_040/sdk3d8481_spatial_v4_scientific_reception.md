# Réception scientifique N8/N16, profil @4 - 2026-10-02

Les quatre exécutions réelles N8/N16 Serial/MPI2 sont reçues avec des sceaux ROOT distincts et une contre-revue indépendante des archives. La fixture et le lecteur @4 sont nouveaux ; leurs prédécesseurs @3 restent inchangés. [Reçu et hashes](sdk3d8481_spatial_v4_scientific_reception.json).

| Cas | Serial, durée du driver | MPI2, durée du driver | Norme Original-F indépendante, Serial / MPI2 |
|---|---:|---:|---:|
| N8, largeur 2 | 128.886 s | 239.651 s | 4.211236792117065e-11 / 4.211237330332404e-11 |
| N16, largeur 2 | 150.194 s | 409.423 s | 4.236253752565298e-11 / 4.2362546468295966e-11 |

Chaque ligne représente un cas par backend, exécuté identiquement sur les deux rangs MPI. Les rangs ne doublent pas le nombre de cas. Le paquet est construit depuis `3d8481c9`, ABI8, CPU Kokkos/OpenMP Dim2 et MPICH ; DSO `7fd8c7fe…f846a44`. Les tests/docs proviennent de `5af964e0` en Serial et `66a1e512` en MPI, avec production identique au build. Les 1 136 sources installées et les identités avant/après sont exactes. Ce modèle conserve son IR16, CP12 et contrat accepté par défaut8.

La physique, la méthode et les critères originaux sont préservés : deux macro-pas, diffusion couplée signée constante, quantités composites Q et contrôle de restriction à 3e-8, norme physique Original-F à 1e-10. Le résidu relatif brut est conservé ; la méthode sélectionnée teste `norm <= 1e-10*max(1,r0)`. Les masques actifs comptent 48/64 cellules N8 et 192/256 N16, avec volume physique composite 1. Le contre-modèle qui moyenne T au lieu de Q est discriminé. Les carriers complets, histoires, diagnostics et horloges restent exacts après reprise et replay.

La contre-revue des quatre archives (`/Users/romaindespoulain/dev/tmp/sol61-spatial-v4-four-independent-20261002/README.md`) rejoue les lecteurs gelés, rehash les 50/51 pins et les 1 136 fichiers installés de chaque route, puis recalcule depuis les NPZ les volumes, masques et quantités Q. Vingt adversaires SourceOnly sont refusés. Ils constituent des contrôles du lecteur ; ils ne représentent pas vingt simulations natives.

Commandes de reproduction sur le paquet conservé :

```sh
cd /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-20261001
SDK=/Users/romaindespoulain/miniforge3/envs/pops-api040-halo9
export CONDA_PREFIX="$SDK" Kokkos_ROOT="$SDK" POPS_KOKKOS_ROOT="$SDK" CMAKE_PREFIX_PATH="$SDK"
export POPS_INCLUDE="$SDK/lib/python3.12/site-packages/pops/include" POPS_NATIVE_DIM=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp
export PATH="$SDK/bin:/Users/romaindespoulain/miniforge3/bin:/usr/bin:/bin:/usr/sbin:/sbin"
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_checks.py --output /Users/romaindespoulain/dev/tmp/api040-reproduce-spatial-v4-n16-serial --test 'tests/python/integration/runtime/test_public_evolved_stage_amr_spatial_v4.py::test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4[16]'
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_checks.py --identity-only --output /Users/romaindespoulain/dev/tmp/api040-reproduce-spatial-v4-n16-serial/after
rtk proxy env -u PYTHONPATH "$SDK/bin/python" docs/development/api_040/run_installed_mpi_checks.py --ranks 2 --dimension 2 --threads 1 --timeout 1800 --output /Users/romaindespoulain/dev/tmp/api040-reproduce-spatial-v4-n16-mpi2 --test 'tests/python/integration/runtime/test_public_evolved_stage_amr_spatial_v4.py::test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4[16]'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python /Users/romaindespoulain/dev/tmp/sol61-spatial-v4-four-independent-20261002/audit.py
```

Les sorties de reproduction doivent être nouvelles ; elles n'héritent pas des sceaux présents. Pour N8, sélectionner `[8]` et une autre sortie vide. Les preuves reçues restent dans `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sdk3d8481-spatial-v4-n{8,16}-root-reception-{serial,mpi2}` ; les owner4/approval4/receive et leurs hashes figurent dans le JSON.

Portée exacte : strip périodique couvrant y, D signé constant, Cartesian sans EB. Les équations complètes M06/M13, D candidat, AMR générale, autres physiques, Halo opt-in, GPU, ROMEO et GitHub CI restent à recevoir. Les corps opaques Program/Aux sont contrôlés par hash/replay ; aucun graphe cryptographique C++→DSO n'est reconstruit. Les anciennes exécutions SDKbb FAILED et erreurs de préparation sont conservées.
