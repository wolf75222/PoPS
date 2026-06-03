# Exemples

Ce dépôt (`adc_cpp`) est la **bibliothèque** : cœur générique + briques physiques + bindings
Python. Les **exemples exécutables** (pilotes C++ sous `examples/`, scripts de figures sous
`scripts/`, bancs de mesure) ne vivent plus ici : ils sont dans le dépôt applicatif
[`adc_cases`](https://github.com/wolf75222/adc_cases) (un dossier Python par cas, qui importe
le module `adc`). On trouve ici, en revanche, les **recettes Python** directement utilisables
depuis le module (cf. [quickstart](quickstart.md)) et les figures de validation conservées
sous `docs/`.

## Recettes Python (module `adc`)

Les exemples annotés de composition `System` / `AmrSystem` (diocotron, Euler-Poisson, multi-espèces,
deux-fluides AP) sont dans le [quickstart](quickstart.md). Ils tournent avec le backend (série /
OpenMP / Kokkos) compilé dans `libadc`.

## Validation applicative

Les runs de validation bout-en-bout (taux de croissance diocotron, ordre de convergence des
modèles, invariants physiques, AMR multi-niveau, runs ROMEO à grande échelle) vivent dans
`adc_cases`. Détail des algorithmes et des choix :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md),
[PERFORMANCE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PERFORMANCE.md).

## GPU (Kokkos / GH200)

Les harnais GPU sous `python/tests/gpu/` sont compilés quand `-DADC_USE_KOKKOS=ON` ; ils
héritent du backend Kokkos de la cible `adc`. Les composants (System mono-grille, ops de champ
AMR, halos MPI multi-GPU, backend AOT d'un modèle DSL) ont été validés SÉPARÉMENT et
bit-identiques au CPU sur GH200 ; la validation INTÉGRÉE AmrSystem + MPI + GPU reste à faire.
Détail :
[GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).
