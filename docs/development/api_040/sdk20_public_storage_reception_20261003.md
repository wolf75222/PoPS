# SDK20 — stockage public reçu, échec scientifique installé conservé

Le gel Native `bdfea618dd5479e30f451f16e0dea5b632b12ccf` correspond aux sources
et tests de MAIN `9dc62a332df587d9ec2175a70b922e61cda97570`. Les quatorze commits
SDK20 portent les tests, captures et lecteurs ; ils ne changent ni les composants
C++ ni les équations ou critères scientifiques. [Résumé JSON](sdk20_public_storage_reception_20261003.json),
[six reçus ROOT byte-exacts](evidence/public_composition_native_bdfea618/receipt-index.json).
Les [preuves SDK19](sdk19_public_storage_reception_20261003.md) restent historiques
et inchangées, notamment le cohort Source19 rouge et le correctif du getter AMR −0.

## Résultats et autorités

| Domaine | Résultat réel | Portée et limite |
|---|---|---|
| Build ROMEO `732069` | COMPLETED `0:0`, 1 min 41 s ; 1 191 sources Git, 1 156 fichiers installés, 1 147 sources shipped, 1 152 membres wheel hors RECORD ; zéro objet C++ recompilé observé | Entrées C++ inchangées ; Native `4efc61b3a4a01ed82e41ef1eac4baaa450914afa516e806b88e282a0ac8fc0b7`, headers `2d3aa1140e6b24d30cdb501d70012b700b1ce70ce36871756f2e7cffda7d828d`. Archive install SDK19 préservée et vérifiée. Identité/build seulement. |
| Cohort Source SDK20 | 56 fichiers ; 429 collectés, 428 sélectionnés et PASS, zéro FAIL/ERROR/SKIP ; une exclusion Native explicite | 1 241 blobs de production dans un domaine distinct du snapshot build à 1 191 ; 1 284 événements setup/call/teardown joints au JUnit. Vrai package Source `python/pops`, helper typé SourceFileLoader explicite, zéro extension Native importée ; aucune CI ou science native déduite. |
| Fan–Li15, canonical, `732072` | **FAILED** : `ModuleNotFoundError: scipy` lors de la référence DOP853 | Les huit tentatives natives sont acceptées et les neuf images initial/accepted1..8 sont réellement retenues. Ce constat ne convertit pas le node en PASS. |
| Math hors ENV20 sur ces neuf images | Recompute ROOT : SSPRK2 max `4.440892098500626e-16` ; GL24/48 gap `0` ; DOP853 final `4.460334324107862e-10`, sous le seuil original `3e-8` | Wick/Gram/quadratures et oracles indépendants inchangés, SciPy dans l’environnement Source ir17. `native_authority=false` et `native_end_to_end_qualified=false` ; ENV20 ne reçoit pas DOP853. |
| Uniform observation @2, `732075` | PASS : **deux** captures local/complete/NPY/clock réellement sauvegardées, bits identiques | Initial idle seulement, 360 mots grown et 192 valid ; pas d’évolution, rollback, formule Ghost ou science. |
| AMR gather replicated, `732076` | PASS : Q0/Q1/forcing, deux niveaux, fine partiel avec trous, dyadiques et −0 exacts ; horloges inchangées | Engineering par setter valid privé et getters publics ; aucune évolution, admissibilité physique, Ghost readiness, rollback ou MPI2 reçus. |

Les trois jobs Native SDK20 exécutent réellement **Dim2 CPU OpenMP, SDK MPI activé
MPICH, un rang**. Le mot Serial dans un ancien identifiant de profil signifie
world1 ; il ne prouve pas un backend Kokkos Serial. ABI native 8, paquet 1.1.0,
Python 3.12.14, GCC 15.3, Kokkos 5.2.1 et HDF5 parallèle 1.14.3 sont authentifiés
dans le build reçu. GPU, inter-nœuds, Dim1/Dim3 et GitHub CI actuels sont ouverts.
Les temps build/JUnit ne constituent pas une mesure à calcul scientifique comparable.

## Contrats et contre-exemples

La capture `pops.fan-li15-public-composition-native-fixture@3` conserve les neuf
phases avec N16, dt `1e-4`, huit pas SSPRK2 originaux, 15=10+5 et l’ordre monomial
authentifié. Le reader `sol61.fan-li15-eight-uniform-storage-reception@1` exige
la jointure local/complete/valid bit exacte et le full-grown fini. Le refus d’un
NaN dans le grown, même repinné de façon cohérente, reste un contre-exemple de
réception. La largeur Ghost `(1,1)` vient du vrai manifest et de son allocateur,
pas d’une déduction des axes du carrier ; la formule des Ghosts reste non reçue.

Uniform observation @2 remplace le témoin @1 à une seule image persistée par deux
exports bruts, sans modifier l’ancien reçu. Quatre mutations repinnées (clock,
grown bit, valid bit, owner) sont refusées. Les 24 pins du helper indépendant
diffèrent des 44 pins de l’enveloppe Native complète (42 fichiers bruts + 2 reçus
de transfert). Pour le node Fan–Li négatif, ces domaines valent respectivement
2, 92 et 90. Les reçus ROOT @2 corrigent le libellé de domaine ; les versions @1
externes restent conservées et référencées par leurs SHA.

Le gather AMR joint les trois vrais Model CPP/DSO et le Program CPP/IR/DSO/manifest
retenus avant bind. `sol61.gather-retained-provenance-join@2` rend les six fichiers
de provenance obligatoires et authentifie ABI/header/blocs/ordre/ghost/backend.
Les trois omissions acceptées par l’ancien reader sont conservées comme rouge ;
le correctif reçoit 18 tests auteur + 13 tests non-auteur et un rejeu des exports
réels. Le hash de compilateur est celui enregistré à la compilation ; aucune
preuve universelle du graphe transitif compilateur→DSO n’est revendiquée.
Le Field forcing a un carrier grown de largeur 3, distinct du ghost-depth 1 du
Program : ces deux autorités ne sont pas interchangeables.

## Commandes exécutées et reproduction

Le [script SLURM exact SDK20](evidence/public_composition_native_bdfea618/build20.sbatch)
archive d’abord SDK19, lit les test-extras du pyproject gelé, puis exécute la
construction incrémentale officielle sur le nœud de calcul :

```sh
bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$run/wheels"
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --identity-only --output "$run/identity"
```

Le runner Native commun est le [script exact déjà conservé SDK19](evidence/public_composition_native_1b64477d/native-run-v1.sbatch)
(SHA `ff2f056d2d50619a3c02d40d70b2a3e66cd8547e4fcacb122a42f73ca04977d6`).
Les nodes exécutés, depuis un paquet installé authentifié, sont :

```text
tests/python/integration/runtime/test_fan_li15_full_eight_storage_runtime.py::test_installed_original_fan_li_eight_storage[canonical]
tests/python/integration/runtime/test_uniform_accepted_storage_observation_v2.py::test_installed_uniform_accepted_storage_observation_v2
tests/python/integration/runtime/test_amr_gather_object_bytes_runtime.py::test_installed_amr_gather_preserves_object_bytes[replicated]
```

Les runners `docs/development/api_040/run_installed_checks.py` et
`run_installed_mpi_checks.py` enregistrent l’identité réelle et refusent le
prototype/le package Source ainsi que les skips. Leurs JUnit et les archives
complètes se trouvent aux origines des six reçus copiés. Les lecteurs non-auteurs
et les commandes de rejeu ROOT sont référencés par chemin absolu et SHA dans ces
reçus ; reproduire un reçu historique exige ses exports historiques complets.
Ne pas réattribuer SDK20 au prefix courant après reconstruction SDK21.

Le correctif du test-extra ajoute SciPy, sans changement de tolérance, d’équation,
de fixture ou de runtime. Il est intégré dans le gel **distinct** SDK21
`3302c4a19d94a0b08dbba9c05c4de3ae2debb232` (MAIN équivalent `c1dcc9a9`). La prochaine
réception obligatoire est un nouveau node installé des huit pas, puis reverse et
MPI2 avec les neuf images et les mêmes critères. M17 autre troncature/AMR/CP12/
restart/Riemann et M16 transport/realizabilité complet restent ouverts. Le gather
partitioned/empty-owner demande des exécutions MPI2 séparées ; empty-owner y
désigne un owner coarse vide, pas nécessairement un moteur entièrement vide.
La mission94 reste active.
