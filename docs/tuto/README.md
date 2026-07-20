# Tutoriels PoPS

Ces tutoriels partent d'un probleme physique et le construisent avec l'API publique de PoPS.
Les scripts declarent les objets dans l'ordre ou ils sont utilises, comme dans un cahier de
calcul. On y retrouve le cycle `validate -> resolve -> compile -> bind -> run`. Les calculs sur
les cellules sont executes par le backend C++/Kokkos.

Le dossier [`examples/final`](../../examples/final/README.md) contient les cas d'acceptation plus
complets de l'architecture.

## Tutoriels disponibles

- [Advection scalaire 2D](scalar_advection/README.md) : modele conservatif, volumes finis,
  MUSCL-Van Leer, flux upwind, SSPRK2, conditions aux limites, maillages uniformes et AMR,
  avec des fichiers distincts pour OpenMP et MPI.
- [Systeme d'advection lineaire](linear_advection_system/README.md) : matrices de transport pleines,
  vitesses propres par direction et flux upwind caracteristique de Roe.
- [Advection et relaxation](advection_relaxation/README.md) : splitting explicite de Lie et Strang,
  IMEX local, puis operateur global matrix-free lorsque la diffusion couple les cellules.
- [Source condensee et FAC composite](condensed_fac/README.md) : cas AMR avance avec
  condensation tensorielle et solve elliptique natif sur toute la hierarchie.
- [HyQMOM a 15 moments](hyqmom/README.md) : cas constant, ondes fluide/electrostatique/magnetique,
  diocotron, tube a choc et jets croises, avec fermeture HyQMOM polynomiale, HLL, Euler, Poisson
  FFT periodique lorsque le cas le demande, et sources de Lorentz natives.

Les cas plus longs se trouvent dans
[l'exemple d'advection scalaire](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py)
et [l'exemple advection-relaxation
IMEX](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py).
