# SDK22 et SDK23 : checkpoint Uniform complet et continuation

Les mécanismes intégrés conservent le vrai runtime C++/Kokkos/MPI de PoPS.
Le checkpoint Uniform 9 conserve les cellules valides et les ghosts de chaque State.
La reprise et le replay complets restent ouverts : le dernier essai installé, 732282,
échoue au premier pas du replay sur l'identité d'un run précédemment clôturé.
Les équations, constantes, méthodes temporelles et critères scientifiques sont inchangés.

Les douze reçus ROOT sont copiés, sans modification, dans
[`evidence/checkpoint_sdk22_sdk23`](evidence/checkpoint_sdk22_sdk23/index.json).
L'index donne les SHA des originaux et des copies. Les rapports indépendants, lecteurs,
XML, sources C++ générées, DSO et images sauvegardées restent dans les chemins
externes pinés par ces reçus. Une réception ROOT rejoue effectivement le lecteur
indépendant sur les bruts ; les anciens rapports ne sont pas réécrits.

## Code et contrats

Le stockage complet utilise les vrais MultiFab et BoxArray. L'itération du Layout a
été corrigée vers `size()` et `operator[]` après l'échec C++ réel 732222 ; les probes
host sur le vrai conteneur couvrent Dim1/2/3. Le build Dim2 suivant a effectivement
compilé 19 objets CXX. Cela ne qualifie pas des backends natifs Dim1/3.

Le payload Uniform 9 contient `state_carriers_checkpoint`. Les contrôles natifs de
format, taille, budget, projection des cellules valides et restauration transactionnelle
restent actifs. Le préflight Python admet et exige ce membre seulement pour la version
entière exacte 9. La version 8 est distincte : sa reprise exige la sélection publique
`state_storage="valid_only_legacy8"` et conserve la portée historique des cellules
valides. La migration v2 produit toujours 8 ; elle ne fabrique pas de ghosts.

L'ABI native reste 8, le checkpoint AMR reste 12. Le symbole public d'observation
accepted-idle sans argument est conservé ; le hook de checkpoint provisoire a son
contrat séparé. Field, histoire, caches, readiness et formule des ghosts ont leurs
autorités propres : un carrier State exact ne les qualifie pas.

Les captures utilisent les vraies méthodes `time()` et `macro_step()`. La provenance
C25 retient tous les modèles, leur C++ réellement compilé, les DSO, compagnons et
manifests, ainsi que le programme, avant bind. Les corrections de capture et de
fixture de migration n'ajoutent pas de branche physique dans le cœur.

## Exécutions et limites exactes

| Source / essai | Résultat reçu | Portée et reste |
|---|---|---|
| Source `99c80569` | 545 PASS, 1 FAIL ; interruption ENOSPC antérieure conservée | Attente obsolète du ComponentManifest corrigée séparément ; aucun succès global hérité. |
| Source `7bbeb375` | 552 PASS, 71 fichiers, 1 656 événements, 1 241 blobs de production stables | Source/host seulement ; 17 migrations NativeLoader et un test de publication avec compiler-mock explicitement exclus. |
| SDK22 build 732226, source `7bbeb375` | COMPLETED 0:0 ; 19 objets CXX observés | Build, installation et identité reçus ; pas de science déduite. |
| Native 732229, source `7bbeb375` | FAILED 1:0 avant le premier pas | Méthodes d'horloge non sérialisables dans la capture ; défaut de témoin corrigé. |
| Native 732268, source `78c75abb` | FAILED 1:0 après quatre pas et écriture du CP9 | Le préflight refusait le nouveau membre ; aucun restart/replay reçu. |
| Source `12320d97` | 24 PASS dans le lot migration/admission ; 18 PASS dans la contre-revue d'admission | 17 NativeLoader exclus du lot Source ; le lot de 552 appartient au gel antérieur `7bbeb375`. |
| SDK23 build 732279, source `12320d97` | COMPLETED 0:0 en 1 min 23 s ; zéro objet CXX supplémentaire | Seul changement installé de production : `_uniform_restart_preflight.py` ; RECORD et direct_url changent également. |
| Native 732282, source `12320d97` | FAILED 1:0 en 5 min 4 s | Huit pas continus observés ; trois refus sans mutation ; CP9 rechargé bit-identique ; premier replay refusé par le ledger des runs clôturés. |

SDK22 et SDK23 sont authentifiés par 1 191 blobs Source, 1 156 fichiers installés,
1 147 sources embarquées et 1 152 membres de roue hors RECORD. SDK23 conserve
l'installation SDK22 complète dans une archive vérifiée de 1 156 fichiers. L'ancien
environnement de 2 377 fichiers est préservé. Les empreintes SDK23 sont constatées
de nouveau depuis le paquet, les bruts et Git :

```text
Source SDK23 : 12320d97c060246c35f4a9b2a36b6f53798795ab
Native       : 504c2418b028d408652426e66619d953e11a0d48d5792414090688de19dccfbd
Headers      : bc8ceef2cd0ff42555308da2f685c5663996a189a5d8e6bb4129e57b17ccca3b
Codegen @1   : 07f1760e8a617f7a18cf81fc164db47e4704e17f6a3f0bd3b9287bae8154945a
```

Le backend exécuté par 732282 est CPU OpenMP, Dim2, world1, GCC 15.3/C++20,
Kokkos 5.2.1, Python 3.12.14. MPICH 4.1.2 est compilé et authentifié ; ce job
ne reçoit pas MPI2. Les fichiers `replay5` proviennent de la capture après l'échec
et restent diagnostiques ; ils ne constituent pas un pas accepté. Aucun oracle
mathématique indépendant nouveau n'est reçu pour ce lot négatif.

La cause lifecycle doit être corrigée dans les mécanismes communs de continuation.
Le garde des runs clôturés protège les effets et sorties déjà publiés. La contre-revue
requiert une identité de continuation déterministe, un consensus des autorités
possédées avant mutation et une publication sous la transaction de restart, avec
rollback complet. Ce document ne prétend pas que ce correctif est déjà intégré.

## Reproduction

Utiliser un checkout exact, `bash scripts/setup_env.sh --dim 2` une fois par worktree,
puis le build incrémental officiel. Le préfixe ROMEO courant est mutable ; authentifier
l'installation avant d'attribuer une nouvelle exécution à SDK23. Les scripts réellement
soumis et leurs SHA sont conservés dans :

```text
/Users/romaindespoulain/dev/tmp/sol61-romeo-api040-composition15-preparation-20261002
  build23.sbatch
  submit_build23.py
  receive_build23.py
  native-run.sbatch
  submit_native.py
  receive_native.py
  seal_sdk23_source.py
  seal_sdk23_build.py
  seal_sdk23_native_negative.py
```

Dans une allocation CPU adaptée, avec le préfixe correspondant au gel choisi :

```sh
bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$new_run/wheels"
env -u PYTHONPATH POPS_DIM=2 POPS_THREADS=1 "$prefix/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output "$new_output" \
  --test 'tests/python/integration/runtime/test_uniform_state_carrier_checkpoint_runtime.py::test_installed_uniform_full_state_checkpoint_restart[fanli15]'
```

Le lot exécuté importe le vrai paquet dans `site-packages`, avec identité Python,
Native, compilateur, headers, source et inventaires avant/après enregistrés. Le module
`pops` du prototype ne fournit aucun résultat de ces campagnes.

Les prochaines variantes sont la construction indépendante `[two-transports]`,
MPI2, les 17 migrations avec NativeLoader, histoire AB2, diagnostics et retry.
Le checkpoint en publication provisoire a un témoin préparé distinct ; V@3 doit
recevoir son point initial complet et sa jointure C25 avant bind. GPU, inter-nœuds,
Dim1/3 natifs, performances à calcul comparable et GitHub CI actuelle restent ouverts.
Les critères Fan–Li, V et les 94 obligations originales sont conservés.
