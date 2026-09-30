# Fixture dt_bound : initialisation authentique et frontières MPI — 30 septembre 2026

Ce suivi du gel `893379b4255711135b1d1a8ab2e0e896bbc542a1` corrige seulement
les tests indépendants et ajoute le présent rapport. Production Python 9c4209a5
inchangée ; aucune mutation MAIN, header, SDK, installation, aucun build/JIT.

## Initialisation et oracles conservés

Chaque état reçoit une déclaration publique
`InitialCondition(state=block[state], value=BindArray(),
projection=ConservativeCellAverage())`. La fixture extrait les exacts
`binding.subject` retenus par `artifact.plan.initial_condition_plan` ;
`pops.bind(initial_values=...)` reçoit uniquement ces Handles résolus et les
tableaux contigus associés. Aucun `initial_state` par nom ni write privé.

Après bind, avant le premier run, la fixture rassemble les vrais états natifs
et exige time=0, macro_step=0, les données fluid `(2,5,7)` et query-only
`(11,13,17)` exactement initialisées, avec les trois composantes permutées
si demandé. Le bloc query-only est déclaré avant fluid dans la Case et reste
absent des updates. Les deux captures, les permutations et deux mises à jour
acceptées restent conservées, avec l'oracle dense sur toutes les cellules et
composantes. Cette réception vérifiera donc également l'installation effective
des données initiales avant d'interpréter un résultat de réduction.

## Ordre MPI2 et refus

La dimension native 2 est sélectionnée explicitement au lancement du test.
Résolution et préparation locale traversent `collective_call` ou
`collective_check`. `compile_resolved_plan_once` publie une seule compilation
sur rang zéro et son cache authentifié aux autres rangs ; aucun chemin de
cache local n'est présumé identique. Aucun fichier de checkpoint/output partagé
n'est créé par cette fixture.

Bind/run traversent chaque frontière collective dans le même ordre. Les
`state_snapshots` rassemblent un bloc à la fois ; chaque assertion locale est
convergée avant l'opération native suivante. Les deux états sont vérifiés sur
rang zéro sans ignorer silencieusement un tableau global vide. Le refus passe
par `collective_attempt`, puis tous les rangs exigent un diagnostic de
contraction dot_all non finie, time=0, macro_step=0, états et cursors inchangés.
Ces helpers ne prétendent pas réparer un deadlock à l'intérieur d'une opération
native ; le progrès MPI réel reste à recevoir par l'intégrateur.

## NaN initial distinct de l'overflow dot_all

Le witness public de refus contient maintenant **uniquement une entrée finie
`1e308`** dans la troisième composante query-only. La fixture exige la finitude
du tableau avant bind, puis son installation réelle intacte avant run. Son
produit au carré est ce qui doit déclencher le nouveau refus dot_all ; un refus
de bind ferait échouer le test, sans être présenté comme réception du bound.

Le précédent NaN initial est retiré de ce witness. Les routes source inspectées
séparent son admission de sa contraction : `_canonical_initial_values`
authentifie les Handles ; `BindInputs` et `BoundSnapshot` hashent les octets du
tableau ; le loader Uniform de `bound_level_zero` appelle `set_state`, dont le
marshaling vérifie shape/décomposition/payload collectif. Ces extraits ne
constituent pas une exécution complète du bind natif et n'authentifient donc
pas ici son résultat pour un NaN. Le présent gel ne prétend ni qu'un NaN a été
installé ni qu'un refus initial serait un refus de dot_all. L'intégrateur
possède séparément la réception CPP du NaN actif ; la fixture Python cible
l'overflow depuis une initiale finie authentifiée.

## Vérifications réalisées et prochaine tranche

- **7 tests source PASS**, en 8.87 s : quatre cas d'émission/routage, désormais
  assortis de l'authentification du plan BindArray et des sujets résolus ; deux
  captures temporelles refusées ; comparaison IR/semantic/hash/C++ legacy
  contre l'archive exacte du parent.
- **3 tests native collectés**, en 0.58 s ; Ruff PASS.
- Aucune compilation/bind/run native de cette fixture exécutée ici.

La sélection source utilise `-k 'not cannot_hide'` uniquement pour ce suivi de
fixture : le contre-test de commit caché reste byte-inchangé et demeure ouvert
sur production 9c. Le gel de pureté `7c68cbaa` annoncé par l'auteur sera reçu
indépendamment ensuite ; son succès n'est pas anticipé par cette sélection.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/review/test_sol61_dt_bound_capture_independent.py -k 'not cannot_hide'
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest --collect-only -q tests/python/integration/runtime/test_dt_bound_capture_independent_runtime.py
```

L'intégrateur doit exécuter les trois tests publics après reconstruction sur
la source et le package exacts ; aucune qualification native/MPI/GPU/PDE ou
consumer scalaire dt_bound <=0/nonfinite n'est attribuée à ce gel de tests.
