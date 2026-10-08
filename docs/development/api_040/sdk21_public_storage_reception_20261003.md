# SDK21 - huit pas originaux reçus CPU/MPI2, stockage et reprise AMR bornés

Gel Native `3302c4a19d94a0b08dbba9c05c4de3ae2debb232`, code MAIN équivalent
`c1dcc9a9d684fcf947734a0f2949cd19805b214d`. [Résumé JSON](sdk21_public_storage_reception_20261003.json),
[treize reçus ROOT et scripts exacts](evidence/public_composition_native_3302c4a1/receipt-index.json).
Les équations, méthodes, critères et composants C++ sont inchangés depuis SDK20.
SciPy est déclaré dans l'extra test ; aucune dépendance SciPy du cœur n'est ajoutée.

| Profil réel | Résultat reçu | Limite |
| --- | --- | --- |
| Build732100 | Script officiel incrémental ; 1 191 sources, 1 156 installés, 1 147 shipped, 1 152 membres wheel hors RECORD vérifiés ; zéro objet C++ recompilé observé | Extension/header réauthentifiés dans ce build, même contenu que SDK20 ; compile_commands absent et script distant non inclus dans les 23 fichiers bruts, script soumis local épinglé. |
| Source SDK21 | 58 fichiers, 460 collectés, 459 sélectionnés et PASS, zéro FAIL/ERROR/SKIP ; 1 377 événements liés au JUnit ; 1 241 blobs de production exacts | Vrai `python/pops` Source, helper typé explicite, zéro extension Native importée ; un node de publication Native-only exclu. Échec initial de lancement cwd rc4/0 tests conservé séparément. |
| Fan–Li canonical CPU732104/reverse CPU732110 | Huit pas acceptés, neuf captures réelles chacun ; math et stockage recomputés, ordre inverse exact | Cas smooth original N16×16, `dt=1e-4`, SSPRK2, 15=10+5 ; chemin analytique normalisé public. Pas de checkpoint/restart ou formule Ghost reçu. |
| Fan–Li canonical MPI2 732111/reverse MPI2 732112 | Un scénario logique et deux XML PASS chacun ; neuf phases × deux shards ; ordre inverse exact contre canonical CPU/MPI2 | Shards réellement `[1,0]`, sans conclure que tout le moteur du rang1 est vide. MPI sur un seul nœud. |
| V CPU732119 | N8/width1, CP12/accepted8/IR16 ; coarse48/64 active, fine64/256 ; restart/replay durable exact, histoires physiques et diagnostics tous présents | Un seul profil CPU ; jointure all-Model C25 et carrier initial complet non reçus par ce lecteur. Les TU cache `.pops-model.cpp` existent : le gap porte sur la jointure explicite, pas une absence générale de C++. |
| V MPI2 732149 | Même profil strict, deux XML PASS, coarse48/64 active et fine64/256 ; CP12 histories/diagnostics/carriers des deux rangs et restart/replay exacts ; sept contre-tests purs PASS | TU/DSO et compagnons C25 joints pour les deux Models ; manifeste Model séparé absent, digest compiler enregistré sans rehash live ni graphe transitif. Seule metadata live rank0 persiste, carrier/diagnostics/histories durables couvrent deux rangs. Carrier initial complet absent. |
| Uniform MPI2 732137 | Deux observations initiales par rang ; carrier local/complete et getter valid byte-exacts, clocks inchangés | Observation engineering idle ; aucune évolution, reprise, rollback ou readiness Ghost déduite. |
| AMR MPI2 replicated732138/partitioned732139/empty-coarse732140 | Trois blocs, deux niveaux, −0/dyadiques/trous fine conservés par getter, clocks exacts et TU/Program joints | Injection par setter valid privé ; empty-owner concerne le coarse seulement. Ghost-depth1 est une exigence du manifeste compilé, sans déduction d'allocation AMR depuis ce manifeste. |
| Publication732113 | Node installé exclu de Source maintenant PASS ; absence de binaire final/staging après panne simulée | `_run_compile`, signature et flags mockés ; aucun échec réel de compilateur ni réception scientifique. |

Tous les profils Native utilisent réellement **Dim2 CPU Kokkos OpenMP**, MPICH4.1.2,
GCC15.3/C++20, Kokkos5.2.1, HDF5 parallèle1.14.3 et Python3.12.14. World1 désigne un
rang réel et ne qualifie pas Kokkos Serial. NumPy2.5.3/SciPy1.18.1 sont observés.
Native SHA `4efc61b3a4a01ed82e41ef1eac4baaa450914afa516e806b88e282a0ac8fc0b7` ;
header `2d3aa1140e6b24d30cdb501d70012b700b1ce70ce36871756f2e7cffda7d828d`.

Pour les quatre profils Fan–Li, l'oracle indépendant original donne SSPRK2 ≤
`4.440892098500626e-16`, écart GL24/48 `0`, DOP853 final
`4.460334324107862e-10 < 3e-8`. Les ordres inverses reviennent aux mêmes neuf états,
erreur maximale `0 < 1e-12`. B n'est pas supprimé ; gardes de densité/SPD et critères
originaux préservés. Le chemin Gauss4 par défaut est une autre réalisation non
exécutée dans ces quatre profils. Le lecteur pur n'a aucune autorité Native ; la
réception ROOT joint son résultat à l'enveloppe installée réellement exécutée.

V garde `OriginalF ≤ 1e-10`, acceptance `3e-8`, FD `1e-6`, Newton et `dt=0.01`.
Le défaut historique de TagBuffer n'est plus une correction à faire :
`8f972aa109dbba96f086e22665b830dcade1044f` sépare déjà `tag_selection_buffer` et
proper nesting. Le profil732119 confirme le raffinement partiel strict ; l'ancien
échec full-refinement et ses trois pins restent conservés. Cela ne reçoit pas toutes
les topologies ou les flux nonconstants coarse/fine. MPI2 reçoit le même profil borné,
OriginalF accepté `1.6115956861858127e-13`, continu/replay
`3.521218301156558e-13`, Q/projection/balance inchangés. Les anciens reçus CPU
et leurs gaps formulés restent immuables ; la précision sur les TU cache est distincte.

Les contrats de capture/provenance restent distincts des équations et de
l'acceptation : Fan–Li capture@3, jointure all-Model/Program@2, observation Uniform@2,
gather AMR@2. Un futur checkpoint Uniform9 full-State est en revue Source dans un
autre gel ; aucune reprise Fan–Li n'est héritée du profil V CP12.

## Reproduction

Dans un checkout exact, après `scripts/setup_env.sh` une fois, le build est
`bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$run/wheels"`.
[build21.sbatch](evidence/public_composition_native_3302c4a1/build21.sbatch) et
[native-run.sbatch](evidence/public_composition_native_3302c4a1/native-run.sbatch)
conservent les commandes réellement soumises, isolation XFS/compilateur/ENV et pins.
Le prefix ROMEO courant est mutable : ne pas attribuer son prochain paquet à SDK21.
Le build732100 conserve wheel, inventaires, doctor, packages et l'installationSDK20.
Une nouvelle reproduction requiert une installation authentifiée du gel3302.

```sh
env -u PYTHONPATH POPS_DIM=2 POPS_THREADS=1 "$prefix/bin/python" docs/development/api_040/run_installed_checks.py   --output "$new_output" --test "$node"
env -u PYTHONPATH "$prefix/bin/python" docs/development/api_040/run_installed_mpi_checks.py   --output "$new_output_mpi" --dimension 2 --ranks 2 --threads 1 --test "$node"
```

Les nodes exacts, relatifs au checkout, sont :

```text
tests/python/integration/runtime/test_fan_li15_full_eight_storage_runtime.py::test_installed_original_fan_li_eight_storage[canonical]
tests/python/integration/runtime/test_fan_li15_full_eight_storage_runtime.py::test_installed_original_fan_li_eight_storage[reverse]
tests/python/integration/runtime/test_uniform_accepted_storage_observation_v2.py::test_installed_uniform_accepted_storage_observation_v2
tests/python/integration/runtime/test_amr_gather_object_bytes_runtime.py::test_installed_amr_gather_preserves_object_bytes[replicated]
tests/python/integration/runtime/test_amr_gather_object_bytes_runtime.py::test_installed_amr_gather_preserves_object_bytes[partitioned]
tests/python/integration/runtime/test_amr_gather_object_bytes_runtime.py::test_installed_amr_gather_preserves_object_bytes[empty-owner]
tests/python/integration/runtime/test_public_evolved_stage_amr.py::test_public_evolved_stage_amr_checkpoint_and_composite_Q[1-8]
tests/python/unit/codegen/test_compile_cache_lock.py::test_failed_program_compile_leaves_no_partial_final_or_staging_binary
```
Chaque commande crée une nouvelle preuve ; elle ne remplace pas une archive reçue.
Les lecteurs/commandes externes et leurs SHA figurent dans les reçus ROOT copiés.
`replay_sdk21_engineering_mpi.py` dans la racine externe des preuves rehash/recompute
les quatre rapports engineering sans importer PoPS ; ce replay ne lance pas Native.

L'échec SDK20 node732072 (SciPy absent après neuf vraies images) reste FAILED ;
son oracle favorable n'est pas réécrit. SDK19 Source RED et SDK20 Source GREEN
restent distincts. GPU, inter-nœuds, Dim1/3, coûts comparables et CI GitHub actuelle
ne sont pas reçus. M17 Riemann/DLM, autre troncature justifiée, AMR et checkpoint
restent ouverts ; M16 garde ses contradictions scientifiques originales isolées.
La mission de 94 obligations reste active.
