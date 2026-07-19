# Tutoriels PoPS

Ces tutoriels construisent des cas complets avec l'API publique de PoPS. Ils sont organises comme
des parcours pedagogiques, alors que [`examples/final`](../../examples/final/README.md) reste le
corpus d'acceptation complet de l'architecture.

Chaque script est volontairement proche d'un cahier de calcul : les objets sont declares
dans l'ordre ou ils deviennent utiles, le cycle
`validate -> resolve -> compile -> bind -> run` reste visible, et les calculs sur les
cellules sont executes par le backend C++/Kokkos.

## Parcours disponibles

- [Advection scalaire 2D](scalar_advection/README.md) : modele conservatif, volumes finis,
  MUSCL-Van Leer, flux upwind, SSPRK2, conditions aux limites, maillages uniformes et AMR,
  avec des fichiers autonomes OpenMP/MPI.
- [Advection, relaxation et implicite](advection_relaxation/README.md) : IMEX local quand les
  cellules sont independantes, puis operateur global matrix-free et Krylov lorsque la diffusion
  les couple.
- [Source condensee et FAC composite](condensed_fac/README.md) : parcours avance AMR avec
  condensation tensorielle et solve elliptique natif sur toute la hierarchie.

Les exemples finaux restent le niveau exhaustif :
[advection scalaire](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py) et
[advection-relaxation IMEX](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py).
