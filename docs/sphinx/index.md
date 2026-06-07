# adc_cpp

Solveur C++23 pour les systèmes hyperbolique-elliptique couplés sur AMR (pile mesh écrite
*from scratch*) : seam de dispatch unique série / OpenMP / Kokkos (GPU GH200) / MPI, pile
MultiFab + BoxArray + Geometry, AMR block-structured multi-niveaux et multi-patch
(Berger-Rigoutsos, reflux coverage-aware), Poisson multigrille ET FFT spectrale, couplage
diocotron (dérive E × B), Euler-Poisson auto-gravitant et deux-fluides isotherme
asymptotic-preserving. Bindings Python via pybind11.

Cette documentation est le **guide utilisateur**. Les documents de conception détaillés
([ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md),
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md),
[BACKEND_COVERAGE](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md)…)
restent la **référence contributeur** et sont liés là où c'est utile.

```{toctree}
:maxdepth: 2
:caption: Prise en main

getting_started/index
```

```{toctree}
:maxdepth: 2
:caption: Écrire un modèle

models/index
```

```{toctree}
:maxdepth: 2
:caption: Simuler

simulation/index
amr/index
```

```{toctree}
:maxdepth: 2
:caption: Exécuter

backends/index
advanced/index
```

```{toctree}
:maxdepth: 2
:caption: Référence

reference/index
```

```{toctree}
:maxdepth: 1
:caption: Référence C++

C++ API (Doxygen) <https://wolf75222.github.io/adc_cpp/cpp/>
```

## En bref

Trois axes orthogonaux (concept `PhysicalModel`, policy `NumericalFlux`, concept
`EllipticSolver`) et un seam de parallélisme unique :

- composition générique `System` : un bloc par modèle, où un modèle est une COMPOSITION de
  briques génériques (`adc.Model(state, transport, source, elliptic)`) ; le cœur ne nomme aucun
  scénario (les noms diocotron, euler_poisson… vivent côté `adc_cases`). Poisson de système
  partagé ; côté Python via `adc.System`. **Trois façons d'écrire un modèle** : composition de
  briques natives, modèle symbolique `adc.dsl.Model`, ou composition hybride — voir
  [Modèles](models/index.md).
- flux `RusanovFlux` / `HLLCFlux` / `RoeFlux`, reconstruction MUSCL (Minmod / VanLeer) + WENO5-Z ;
- `GeometricMG` (multigrille V-cycle GS red-black) / `PoissonFFTSolver` (spectral direct) ;
- AMR : `AmrSystem` mono- ET multi-bloc, multi-patch N niveaux, reflux coverage-aware,
  `AmrCouplerMP` (regrid Berger-Rigoutsos) — voir [AMR](amr/index.md) ;
- `for_each_cell` : série / OpenMP / Kokkos ; `comm.hpp` : collectives MPI — voir
  [Backends parallèles](backends/index.md).

Nouveau venu ? Commencez par la [présentation](getting_started/presentation.md),
[installez](getting_started/installation.md), puis suivez le
[tutoriel A→Z](getting_started/tutorial.md).

## Liens

- Code source : <https://github.com/wolf75222/adc_cpp>
- Référence C++ Doxygen : [/cpp/](https://wolf75222.github.io/adc_cpp/cpp/)
- Solveurs sœurs (sur `pde_core_cpp`) : [euler_cpp](https://github.com/wolf75222/euler_cpp),
  [advection_cpp](https://github.com/wolf75222/advection_cpp)
- Scénarios applicatifs : [adc_cases](https://github.com/wolf75222/adc_cases)
