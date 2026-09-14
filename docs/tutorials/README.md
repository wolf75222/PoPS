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
  composite sur AMR lorsque le cas le demande, et sources de Lorentz natives.

Les cas plus longs se trouvent dans
[l'exemple d'advection scalaire](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py)
et [l'exemple advection-relaxation
IMEX](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py).

## Demarrer

Depuis la racine du depot, suivre [l'installation](../../README.md), puis activer
l'environnement `pops`. Les parcours sont en 2D, sauf le script scalaire 16 qui
exige une construction `bash scripts/build_python.sh --dim 3` et un nouveau
processus Python. Revenir a `--dim 2` pour les autres scripts.

```sh
env -u PYTHONPATH python docs/tutorials/scalar_advection/01_openmp_preset_ssprk2.py
env -u PYTHONPATH python docs/tutorials/advection_relaxation/01_openmp_imex_local.py
```

Les scripts OpenMP utilisent un thread par defaut. `POPS_THREADS=4` choisit quatre
threads avant l'initialisation native ; choisir une valeur compatible avec les
coeurs disponibles. Les variantes MPI scalaires utilisent un thread par rang.
Les cas HyQMOM demandent une construction MPI, y compris pour un seul rang ; voir
[leurs commandes](hyqmom/README.md#execution). Les sorties sont generees dans le
sous-dossier `results/` de chaque parcours et ne sont pas versionnees.

Le modele, ses equations et les choix numeriques restent visibles dans chaque
script. Les presets SSPRK2 et les variantes explicites sont gardes cote a cote
pour montrer le meme calcul a deux niveaux de detail.
