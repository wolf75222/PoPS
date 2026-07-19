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

Le prochain niveau de detail est [l'exemple final d'advection
scalaire](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py), qui etend l'AMR
minimal avec tagging par gradient, sorties scientifiques et restart exact.
