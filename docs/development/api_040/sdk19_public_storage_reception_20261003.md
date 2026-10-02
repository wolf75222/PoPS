# SDK19 : réception du stockage public et du getter AMR

Le freeze exécuté est `1b64477d2f1c93e856e996a4c83e506fcc91b386`. Son code et ses tests sont intégrés dans le checkout principal à `802e921c6e5bbcd0880a6a6a568d410508119b8a`. Cette réception ferme le défaut de signed-zero du getter dans le témoin M16 nommé ci-dessous et reçoit l'observation initiale Uniform dans sa portée d'ingénierie. Elle ne ferme ni M16/M17 complets ni les 94 obligations de mission.

Le [relevé machine](sdk19_public_storage_reception_20261003.json) et les [sept reçus ROOT copiés exactement](evidence/public_composition_native_1b64477d/receipt-index.json) distinguent construction, Source, science bornée et ingénierie. Les archives terminales, DSOs, inventaires complets et lecteurs indépendants restent aux origines externes indiquées dans ces reçus ; les copies compactes ne remplacent pas ces archives.

## Code intégré et contrats

Le contrat [AMR field gather v2](amr_field_gather_v2.md) transporte les représentations d'objet du champ par réduction OR des octets, avec propriétaires disjoints ou source canonique répliquée, votes de préparation/allocation et vérification de la représentation de `double` avant le payload. Les cellules absentes d'une grille fine partielle restent `+0`. Le correctif est dans le vrai runtime C++/MPI ; il ne dépend d'aucun nom de modèle. La conversion préexistante de `AmrReal` en `double` ne constitue pas une promesse de conservation de tout payload NaN.

`runtime.observe_accepted_state_storage()` expose le contrat `accepted-state-storage-observation@1` par les vrais composants System, pybind et DTO Python. L'observation est immutable, readonly et collective, après bind et hors step/restart. Elle copie les blocs réellement présents, leurs ownerships et tous leurs mots grown. Elle n'effectue ni halo, ni solve, ni matérialisation supplémentaire. L'implémentation publique reçue ici est Uniform. Le DTO vérifie son enveloppe ; construire manuellement un DTO ne certifie pas son payload. L'ajout de méthode conserve l'entier ABI8 et impose néanmoins la reconstruction native/header.

## Construction réellement exécutée

Le job ROMEO `732041` a construit 24 objets C++ et installé la wheel produite par `scripts/build_python.sh --dim 2 --mpi`. La contre-réception indépendante puis le replay ROOT authentifient séparément : 1191 fichiers Source, 1156 fichiers installés, 1147 sources shipped et 1152 entrées wheel hors RECORD. Ces quatre domaines ne sont pas interchangeables.

| Identité | Valeur reçue |
| --- | --- |
| Native SHA-256 | `4efc61b3a4a01ed82e41ef1eac4baaa450914afa516e806b88e282a0ac8fc0b7` |
| Header signature | `2d3aa1140e6b24d30cdb501d70012b700b1ce70ce36871756f2e7cffda7d828d` |
| ABI / dimension | 8 / 2 |
| Compilateur / langage | GCC 15.3 / C++20 |
| Exécution | Kokkos 5.2.1 OpenMP/Serial ; MPICH 4.1.2 |
| I/O / Python | parallel HDF5 1.14.3 / Python 3.12.14 |

Les 21 fichiers du reçu de construction sont rehashés, avec les manifests Source/install/shipped/wheel et la signature header recomputée. Le log donne les 24 objets ; `compile_commands.json` n'est pas conservé dans ce reçu. La reconstruction ne prouve donc pas à elle seule le graphe sémantique transitif ou une qualification scientifique. L'installation SDK18 et sa wheel historique sont archivées avant remplacement ; les autres environnements ROMEO existants sont préservés.

## Exécutions et lectures indépendantes

| Témoin installé | Job / backend réel | Réception exacte |
| --- | --- | --- |
| M16, seconde State21, un pas FE | `732047`, CPU OpenMP, world1 d'un SDK MPI | Fraction indépendante ; getter et spectator grown bit-exacts |
| Même scénario | `732051`, MPICH, deux rangs sur `romeo-c022` | Même oracle, launcher/rangs authentifiés et signed-zero exact |
| Uniform observation initiale @1 | `732048`, CPU OpenMP, world1 | Lecture readonly ; couverture valid et grown authentifiée |
| Même observation | `732054`, MPICH, deux rangs sur `romeo-c022` | Rang0 propriétaire des 360 mots grown ; rang1 shard réellement vide |

Les quatre jobs sont terminaux `COMPLETED`, exit `0:0`. Chaque MPI2 correspond à un scénario logique avec un PASS sur chaque rang, pas à deux scénarios physiques indépendants. L'import de `pops` et du Native provient de l'installation ROMEO authentifiée avec `env -u PYTHONPATH`, avec snapshots avant/après ; aucun module du mini-runtime du handoff n'est utilisé.

M16 utilise une seconde State de 21 composantes à base explicite permutée, degré cinq, après un spectator de deux composantes dyadiques non constantes avec `-0`. Les horloges passent de `(0,0,1)` à `(0.125,1,1)` pour time/step/epoch. Les 368 valeurs valid et 828 valeurs grown sont conservées pour les phases capturées ; les 72 mots grown du spectator restent exacts. L'écart Fraction maximal reçu est `4.44e-16`. Les carriers complets de ce témoin sont globaux canoniques avec owner `-1`, et ne sont pas des archives rank-local. Les neuf mutants CPU et cinq mutants MPI repinnés sont tous refusés, dont altération signed-zero/grown.

Le refus SDK18 `732028` est conservé : le getter MPI rendait `+0` quand le carrier contenait `-0`. Les nouvelles signatures et les nouvelles captures SDK19 ferment précisément ce défaut. Ni l'ancien reçu ni son résultat scientifique refusé ne sont réécrits.

Uniform @1 conserve une seule image brute initiale. L'égalité de deux appels successifs est une assertion JUnit ; il n'existe pas deux captures brutes persistées dans ce profil. La population a six composantes sur 8×4 cellules, soit 192 mots valid et 360 mots grown. Cinq mutants CPU et sept mutants MPI repinnés sont refusés. Ce profil n'apporte pas de NPY indépendant, de véritable TU Model/Program C25 supplémentaire, de step, de rollback, ni de formule/readiness Ghost. Les versions renforcées des fixtures sont préparées séparément et attendent un nouveau freeze natif.

## Source et non-régression

Le cohort SDK19 est rouge et reste rouge dans son reçu : 48 fichiers, 373 tests collectés, 372 sélectionnés, 371 PASS, un FAIL, zéro erreur/skip et une exclusion Native explicite. Le défaut est la composabilité du garde de pureté d'un lecteur offline : il examinait les modules `pops` déjà chargés légitimement par d'autres tests Source du même processus. Le reçu terminal, le XML et les 1116 événements setup/call/teardown sont conservés. Le correctif passe son contrôle de pureté dans un subprocess isolé sans purger les modules du parent ; ses contre-tests Source ne transforment pas rétroactivement ce cohort en PASS.

Les tests de la prochaine tranche vérifieront les neuf captures Uniform des huit pas SSPRK2 Fan–Li15 originaux, deux observations readonly persistées et les topologies du gather AMR. Les équations, N=16, dt=1e-4, huit pas et critères scientifiques originaux restent inchangés. Le lecteur mathématique indépendant et les checks de stockage restent distincts de la réception d'identité Native.

## Reproduction

Le [script SLURM réellement exécuté](evidence/public_composition_native_1b64477d/native-run-v1.sbatch) est conservé exactement, SHA-256 `ff2f056d2d50619a3c02d40d70b2a3e66cd8547e4fcacb122a42f73ca04977d6`. Il refuse un Source sale ou différent, nettoie PYTHONPATH, active le prefix MPI qualifié, fixe les compilateurs du prefix et enregistre les snapshots d'identité avant/après. `scripts/setup_env.sh` a déjà été exécuté une fois dans ce worktree ; les reconstructions sont incrémentales via le script du dépôt.

Depuis la préparation locale conservée, ce lancement crée une nouvelle exécution MPI2 du témoin M16, à condition que le Source distant et l'installation soient encore ceux de SDK19 :

```bash
cd /Users/romaindespoulain/dev/tmp/sol61-romeo-api040-composition15-preparation-20261002
pops_label="sdk19-m16-repro-$(date -u +%Y%m%dt%H%M%S)"
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python \
  submit_native.py --label "$pops_label" --world-size 2 \
  --source-freeze 1b64477d2f1c93e856e996a4c83e506fcc91b386 \
  --test tests/python/integration/runtime/test_program_affine_moment_second_state_runtime.py::test_installed_affine_degree_five_selected_second_state
```

Utiliser `--world-size 1` reproduit la portée CPU ; remplacer le node par `tests/python/integration/runtime/test_uniform_accepted_storage_observation.py::test_installed_uniform_accepted_storage_observation` reproduit l'observation initiale. Après remplacement du SDK ou changement du Source, ces commandes doivent recevoir un nouveau freeze et un nouveau reçu ; une ancienne signature ne s'hérite pas. La préparation et les archives brutes résident dans les chemins absolus des reçus ; elles ne sont pas distribuées intégralement dans ce dossier compact.

## Limites restantes

Cette réception ne qualifie pas le transport et la réalisabilité M16 complets, les huit pas originaux M17, toutes les topologies AMR, une formule ou readiness Ghost générale, checkpoint/restart, ni la matrice scientifique complète. GPU, Dim1/Dim3, MPI inter-nœuds, performances à calcul comparable et GitHub CI sur le HEAD final restent non reçus. La mission demeure active ; les équations originales et les anciens échecs sont préservés.
