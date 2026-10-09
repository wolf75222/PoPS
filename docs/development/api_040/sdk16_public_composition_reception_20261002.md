# Composition publique : réception SDK16 et correction SDK18

La mission reste active. Les profils ci-dessous qualifient des mécanismes et des trajectoires nommés ; ils ne ferment ni les 94 obligations ni tous les backends.

## Code intégré et contrats

Les équations et les méthodes sont composées en Python public puis compilées par PoPS : base monomiale immuable avec bindings explicites, update affine, flux et chemin non conservatif normalisé, et fermeture finie à six atomes écrite par un non-auteur. Les recettes scientifiques restent dans les bibliothèques/examples et tests. Le cœur exécute les opérations mathématiques communes ; aucun dispatch par modèle, article, nom de variable ou première espèce n'a été introduit.

Le contrat `pops.flux-wave-law@1` authentifie chaque loi par sa State de sortie, son axe et sa signature Rate. Des transports indépendants peuvent avoir des arités et des lois distinctes. Un vrai groupe principal joint déclare sa borne commune. Le témoin de migration conserve exactement ses matrices, équations et λ ; le fallback implicite depuis la première State est refusé. Voir [contrat](flux_wave_authority.md).

C25 retient les vraies TU et les commandes du compilateur : `pops.model.actual-compile@1`, sidecar Model v2, `pops.codegen-source@1` et export `pops.source-evidence@2`. `model_source_policy=require` exige ces preuves pour tous les modèles. La politique reste dans le plan authentifié et est retirée des seules options Program, dans les deux routes de compilation. Voir [contrat](../../review/sol61_actual_model_source_contract_v2.md) et [correction](../../review/sol61_c25_program_option_boundary_fix.md). Ces preuves ne constituent pas une preuve sémantique cryptographique C++→DSO ni l'inventaire de toutes les dépendances transitives du compilateur.

## Profils réellement construits

| Profil | Source et paquet | Backend / portée |
| --- | --- | --- |
| SDK16 | Build `89abaf8e`, tests `9bdc6acd`, 1 188 fichiers de production byte-identiques entre ces deux gels ; 1 153 fichiers installés, 1 144 sources embarquées | ROMEO x86, GCC15.3/C++20, Dim2, Kokkos5.2.1 OpenMP/Serial, MPICH4.1.2 ; un processus et deux rangs sur un nœud selon le reçu |
| SDK17 | Build731975 `d58489fc`, 1 190 sources dans le domaine du snapshot, 1 155 fichiers installés, 1 146 sources embarquées | Construction auditée, zéro objet C++ central recompilé. Les deux témoins publics échouent avant Programme/simulation ; aucune réception scientifique |
| SDK18 | Build732009 `c6b7a600`, mêmes domaines1190/1155/1146 ; correctif C25 installé et audit indépendant rejoué par ROOT | Deux témoins CPU732018/732019 et transports MPI2 :732027 reçus par recompute ROOT ; M16 MPI2 :732028 refusé pour perte du zéro signé par le getter. Aucune preuve SDK16 héritée |

Native SDK16/17/18 : SHA256 `3f4e01cfa9d69dbd1148519a947c25895814cca4b5e010c5348ff9f7e41a3fff`, headers `ae40d2c0bd718f7bed15da474dd30106f48aa69dc7c7f56303e4c06dd4e9efe4`, ABI8. Les changements17/18 sont Python/codegen ; chaque témoin compile ensuite ses véritables modèles et Programme C++.

## États sauvegardés et contre-réceptions SDK16

| Témoin | Exécutions authentifiées | Référence indépendante / résultat borné |
| --- | --- | --- |
| M16, binding explicite21 degré5 | 731840 : faute nonfinite ; 731841 : deux rangs, trois scénarios logiques nominal/nonfinite/overflow | Orbite Fraction c=4/5,s=3/5, max erreur4.44e−16 ; refus `affine_moment_update invalid_evaluation` et rollback des carriers complets/State/horloge |
| Fermeture six atomes | 731860 flux+chemin ;731861 overflow paramètre fini ;731882 B=0 ;731883 MPI2 flux+chemin ;731884 MPI2 faute | Intégrales rectangulaires/trigonométriques, inverse6×6, Rusanov signé/orienté et FE : toutes192 cellules×3 phases, erreur mesurée0, SPD et horloges ; faute prepared Cartesian face1031 et rollback exact360 grown/192valid |
| M17 Fan–Li15 publique | 731862 ordre inversé ;731885 MPI2 canonical+reverse, deux scénarios logiques, deux PASS par rang | Wick/GL24–48 puis SSPRK2 : initial8.88e−16, deux pas4.44e−16, B=0 distinct1.1669e−5 ; h3/h4 actifs, SPD/conservation et routing de base |

Les readers purs rehashent les exportations complètes, les identités installées, les couples avant/après, les DSOs et, lorsque la fixture les fournit, les vraies TU. Les adversaires repinnés sont refusés. ROOT a rejoué les readers M16, cubature et M17 MPI2 ; les sceaux et leurs SHA sont indexés dans [JSON](sdk16_public_composition_reception_20261002.json). Les données brutes et archives complètes restent dans `/project/r250127/api040-composition15-mpich-20261002-v15` et dans `/Users/romaindespoulain/dev/tmp/sol61-romeo-api040-composition15-preparation-20261002`.

Les captures de carriers MPI reçues ici sont globales/canoniques, owner−1 ; les identités, XML et logs des deux rangs sont authentifiés. Aucune image raw propre à chaque rang n'est inventée. La fixture M17 deux pas n'a pas de full-carriers/checkpoint ni de record de TU Model observée ; le Programme retenu ne remplace pas cette preuve absente.

## Échecs conservés et non-régression

SDK17 :731992 transports et731993 seconde State compilent chacun deux vrais Model/TU/DSO, puis refusent la publication sur `_compile_problem_impl() got an unexpected keyword argument model_source_policy`. Les deux exports complets32 fichiers+1lien et les inventaires avant/après sont conservés. La correction `fe549a88`, contre-revue `cd78854c`, projette les options Program après admission C25 de chaque Model ; elle ne retire pas `require` des modèles.

La cohorte Source/host du gelSDK18 passe **325 tests, zéro échec/erreur/skip, une exclusion Native explicite**,43 fichiers dédupliqués,449.12s. Les1 240 fichiers du domaine Source/host (`python/include/src/cmake/scripts`+quatre fichiers build) sont inchangés avant/après ; ce domaine diffère du snapshot1190. Import exact du vrai frontend PoPS du checkout, aucune extension Native chargée ; le helper `_pops_time_typed_program_support` est Python et classifié par loader/origin/SHA. [Commande et pins](evidence/public_composition_source_c6b7a600/command.json), [XML](evidence/public_composition_source_c6b7a600/pytest.xml), [réception ROOT](evidence/public_composition_source_c6b7a600/root-sdk18-source-nonreg-reception.json). L'exclusion `failed_program_compile_leaves` demande une autorité Native authentique ; sa réception sur l'installation est distincte de la cohorte Source.

Ce seul test exclu a ensuite passé sur l'installation SDK18 (732025,1PASS,1.429s XML). Le compilateur est explicitement mocké : il écrit des bytes partiels puis lève. Le cache ne conserve aucun binaire final/staging ni sidecar, et retient la vraie source `.failed.cpp`. Les28 pins et le lecteur indépendant sont rejoués par ROOT. Cela reçoit le nettoyage de publication, sans qualification de compilation C++ réelle ni de science.

Les témoins CPU SDK18 sont reçus séparément par ROOT après vérification des124 pins et réexécution du lecteur pur : transports indépendants,320valeurs valides/500grown sur trois phases, erreur0 ; M16 seconde State21 après spectator2,368valides/828grown sur deux phases, erreur max4.44e−16 et spectator bit-exact avec−0. Toutes les TU, commandes, sorties et DSOs des deux modèles par cas sont conservées avant bind, ainsi que Programme/IR et horloges. Les14 mutants repinnés sont refusés. Le sceau ne reçoit ni Ghost readiness ni toute la physique M16/M17.

Le MPI2 transports732027 est reçu après recompute pur ROOT et vérification du nouveau domaine158 pins/4liens : deux vrais rangs authentifiés, un PASS par rang,141.981s XML, erreur0 aux trois phases, mêmes TU/DSO/états/horloges vérifiés. Quatre mutants repinnés rang/monde/horloge/état sont refusés. Les images de carriers restent globales canoniques ; ce reçu ne reçoit pas des images raw propres à chaque rang. [Copies compactes des cinq reçus SDK18](evidence/public_composition_native_c6b7a600/receipt-index.json).

Le vrai refus MPI2 M16 est conservé :732028,un échec par rang, mêmes Source/installation avant/après. Les carriers gardent−0 dans spectator mais le getter global publie+0 : `execute_field_gather` AMR applique SUM aux doubles. Uniform transporte déjà des bytes. La correction générique du getter AMR, avec contrat v2 et byte-count vérifié, doit être reconstruite et revue avant toute nouvelle réception MPI M16 ; aucune garde n'est relâchée.

Le gel d584 conserve317PASS/1FAIL joint/1exclusion en425.65s. Trois échecs antérieurs venaient d'un runner multiprocessing sans garde `__main__`, puis ENOSPC à la clôture ; leurs logs ne sont pas transformés en passes. Le vrai refus joint a été corrigé par la déclaration explicite de la borne commune, avec math inchangée.

## Reproduction avec le paquet installé

Sur un nœud CPU obtenu par SLURM, activer l'environnement et les mêmes CC/CXX/Kokkos/MPI que la construction ; ne pas réexécuter setup_env déjà effectué dans ce worktree. Le script officiel de construction incrémentale est `scripts/build_python.sh`. Pour refaire le profil SDK18, il faut son installation et son worktree geléc6b7a600 ; un autre SDK demande une réception distincte. Ces commandes créent leurs répertoires de sortie et appellent directement le Python installé.

```bash
POPS_REPRO_ROOT=/project/r250127/api040-composition15-mpich-20261002-v15
POPS_REPRO_PREFIX="$POPS_REPRO_ROOT/envs/pops_api040_composition15_mpich_v15"
POPS_REPRO_PY="$POPS_REPRO_PREFIX/bin/python"
POPS_REPRO_OUTPUT=$(mktemp -d /tmp/pops-sdk18-repro.XXXXXX)
export PATH="$POPS_REPRO_PREFIX/bin:$PATH"
export CC="$POPS_REPRO_PREFIX/bin/gcc" CXX="$POPS_REPRO_PREFIX/bin/g++"
export POPS_KOKKOS_ROOT="$POPS_REPRO_PREFIX" POPS_NATIVE_DIM=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 POPS_REQUIRE_NATIVE_TESTS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false
export POPS_CACHE_DIR="$POPS_REPRO_OUTPUT/cache/pops" POPS_CODEGEN_DIR="$POPS_REPRO_OUTPUT/cache/codegen"
mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR"
cd "$POPS_REPRO_ROOT/source"
test "$(git rev-parse HEAD)" = c6b7a600033cb3ba42ca6adf7ca9c743e216ab87
env -u PYTHONPATH "$POPS_REPRO_PY" docs/development/api_040/run_installed_checks.py --identity-only --output "$POPS_REPRO_OUTPUT/identity"
env -u PYTHONPATH "$POPS_REPRO_PY" docs/development/api_040/run_installed_checks.py --output "$POPS_REPRO_OUTPUT/waves" --test tests/python/integration/runtime/test_scoped_flux_wave_runtime.py::test_installed_independent_state_scoped_wave_transports
env -u PYTHONPATH "$POPS_REPRO_PY" docs/development/api_040/run_installed_checks.py --output "$POPS_REPRO_OUTPUT/state2" --test tests/python/integration/runtime/test_program_affine_moment_second_state_runtime.py::test_installed_affine_degree_five_selected_second_state
env -u PYTHONPATH "$POPS_REPRO_PY" docs/development/api_040/run_installed_mpi_checks.py --output "$POPS_REPRO_OUTPUT/waves-mpi2" --ranks 2 --dimension 2 --threads 1 --timeout 6000 --test tests/python/integration/runtime/test_scoped_flux_wave_runtime.py::test_installed_independent_state_scoped_wave_transports
env -u PYTHONPATH "$POPS_REPRO_PY" docs/development/api_040/run_installed_checks.py --output "$POPS_REPRO_OUTPUT/publication-fault" --test tests/python/unit/codegen/test_compile_cache_lock.py::test_failed_program_compile_leaves_no_partial_final_or_staging_binary
```

Ces runners enregistrent l'import site-packages et le Native réel, imposent leur identité, enlèvent le shadowing Source et refusent les skips. Une collection Source ne les remplace pas. Le [sbatch exact](evidence/public_composition_native_c6b7a600/native-run-v1.sbatch), exports, compte/contraintes effectifs, captures avant/après et reçus de soumission sont conservés avec chaque job. Une nouvelle exécution doit recevoir ses propres hashes, états et critères. Pour reconstruire dans un worktree préparé : `bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$POPS_REPRO_OUTPUT/wheels"` ; cette commande crée un nouveau profil, sans hériter des sceaux ci-dessus.

## Obligations encore ouvertes

M16 : transport HyQMOM15 magnétique complet, réalisabilité, AMR et backends supplémentaires ; l'obstruction mathématique B.1 oblique est conservée et ne dispense pas les autres travaux. Un update affine21 n'est pas une fermeture de cette physique.

M17 :15 composantes=10+5, huit pasSSPRK2 originaux, autre ordre de troncature, AMR/nonconservatif d'ordre supérieur, checkpoints/restart et Riemann/chocs ; la réception deux pas ne les ferme pas. Une fermeture six atomes et Polynomial6 ne sont pas un Fan–Li d'ordre6. Voir [inventaire et prochaine trajectoire](m16_m17_remaining_obligations_after_sdk17.md).

GPU, inter-nœuds, Dim1/3, OpenMPI, coûts à calcul comparable et GitHub CI actuel restent non reçus dans ce profil. La prochaine action est de terminer les témoins SDK18, puis l'original public huit pas, avec contrats/exports versionnés et critères scientifiques inchangés.
