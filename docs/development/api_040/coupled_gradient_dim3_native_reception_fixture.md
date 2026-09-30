# CoupledGradient: fixture de réception native Dim3

Base de production : `b468a55f814f13bb5a6813c15d99adbb4b67326d`. Ce changement ajoute uniquement des tests Python et ce rapport. Les anciennes revues et leur checkout restent conservés. Aucun en-tête, runtime, environnement installé ou artefact natif n'a été modifié.

## Cas physique et critères

La fixture publique `tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py` importe le paquet installé, exige explicitement le module natif Dim3, puis compile et lie le vrai programme. Elle utilise un domaine Uniform périodique de 6 × 4 × 3 cellules, de longueurs 1 × 2 × 3, avec deux composantes et leurs deux permutations. Chaque composante varie dans chacune des trois directions. Les valeurs initiales sont les moyennes exactes de cellules de modes de Fourier.

Le flux utilise les matrices `D = [[0.25, 0.05], [0.05, 0.10]]` symétrique définie positive et `R = [[0, -0.3], [0.3, 0]]` antisymétrique. Le taux est la somme de deux occurrences physiques de poids 0.5 et 1.5. SSPRK2 avance d'un seul pas fixé à `1e-5`. La permutation agit sur les champs et les deux axes des matrices ; les comparaisons et états sauvegardés reviennent à l'ordre physique canonique.

Le recalcul NumPy utilise les différences périodiques centrées dans les trois directions, puis les deux étapes SSPRK2. Un contrôle distinct en espace de Fourier vérifie son polynôme d'amplification discret. Le ledger natif doit contenir exactement 3456 incidences : 72 cellules × 3 axes × 2 côtés × 2 composantes × 2 occurrences × 2 étapes. Chaque incidence doit être unique, porter le bon contexte d'étape et l'identité d'occurrence, une orientation ±1, une multiplicité 1, la vraie aire tangentielle et la durée pondérée `dt/2 × poids`. Le flux est recalculé à partir de l'état initial ou du prédicteur indépendant selon l'étape.

Les critères sont fixes : liaison de l'état initial à `2e-14`, état final à `4e-12`, flux à `3e-12`, montant intégré et bilan cellule à `4e-13`, sans tolérance relative. Le bilan global de chaque composante est nul à la même tolérance. L'énergie doit diminuer d'au moins `1e-9`, l'horloge doit valoir exactement `dt`, et un seul pas doit être accepté.

## Preuves enregistrées par une vraie exécution

Pour chaque permutation, le répertoire `coupled-gradient-dim3-order-01` ou `-10` contient :

- `initial.npz` : état réellement lié, données demandées, entrée permutée, ordre et géométrie ;
- `accepted.npz` : états natifs initial et final, incrément réel, paramètres, et tableaux explicitement nommés `oracle_predictor` et `oracle_final` ;
- `native-ledger.json` : enregistrements retournés par le binding natif, regroupés par rang ;
- `ledger-rank-NNNN.bin` : bytes exacts `POPSEX02` retournés par le natif, sans reconstruction Python ;
- `increments.npz` : montants de faces natifs et variation de l'état multipliée par le volume ;
- `receipt.json` : chemins et SHA du module natif, ABI, identité d'artefact et manifeste de plateforme, SHA de la fixture, paramètres, critères, SHA des bytes de ledger, effectifs par rang et erreurs observées.

Les états et le ledger brut sont écrits avant les assertions numériques, pour conserver une preuve exploitable en cas d'échec. Les noms `oracle_*` distinguent les tableaux calculés des états produits par le runtime. Une autre revue doit recalculer le bilan à partir de ces fichiers sauvegardés.

## Réception effectuée ici et commande native à venir

Les deux cas de source, émission C++ et calcul spectral passent : erreur maximale entre stencil et Fourier `6.661338147750939e-16`. Les quatre appels d'échanges des deux étapes et deux occurrences sont présents dans chaque source émise. Le test source exige que la sélection native reste absente avant et après l'émission. La collecte du module d'intégration installé découvre les deux permutations. Cette réception n'exécute ni module Dim3, ni JIT, ni build, ni installation.

Commandes de validation locale, depuis ce checkout :

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/python/unit/codegen/test_coupled_gradient_dim3_reception_fixture.py
rtk proxy env -u PYTHONPATH POPS_REQUIRE_NATIVE_TESTS=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest --collect-only -q tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py
```

Après construction et installation du module Dim3 par l'intégrateur, lancer la réception réelle dans un répertoire de preuves neuf :

```sh
rtk proxy env -u PYTHONPATH POPS_NATIVE_DIM=3 POPS_REQUIRE_NATIVE_TESTS=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py --basetemp=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/coupled-gradient-dim3-native --junitxml=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/coupled-gradient-dim3-native.xml
```

`POPS_REQUIRE_NATIVE_TESTS=1` active la voie stricte du dépôt : aucune réussite par skip n'est admissible. La fixture ne comporte ni skip ni repli sur Dim2 ; un module Dim3 absent ou un paquet source substitué au paquet installé provoque un échec. La réussite native Dim3 reste à établir. Aucune qualification MPI, AMR, EB ou GPU n'est revendiquée par cette fixture Uniform, ni par les vérifications de source réalisées ici.
