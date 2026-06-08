# Validation (coeur)

Etat de la validation du coeur `adc_cpp`. La matrice de couverture par backend est dans
[BACKEND_COVERAGE.md](BACKEND_COVERAGE.md) ; le detail des portages device dans
[GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md) ; la validation applicative (modeles nommes, diocotron,
runs ROMEO) dans le depot [`adc_cases`](https://github.com/wolf75222/adc_cases).

## CI

- ctests du coeur en Release et en Kokkos (Serial).
- MPI np=1/2/4, sorties bit-identiques.
- module Python : suite supplementaire (bindings et DSL).

## AMR

- reflux multi-patch conservatif a l'arrondi machine (derive de masse ~ 1e-15).
- le Poisson est resolu au niveau grossier puis injecte vers le fin : l'AMR raffine le transport, pas
  le solve elliptique (pas de solve composite multi-niveaux, pas de Schur global sur AMR).

## GPU GH200 (hors CI)

- System production np=1 valide (#97).
- multigrille geometrique device-MPI np=1/2/4 valide (#93).
- AmrSystem + MPI + GPU valides, bit-identiques (phase 10, dmax=0, #105).
- Schur et polaire device : 7/7 device-clean en Kokkos Cuda single-GPU, et MPI+Kokkos Cuda multi-GPU
  rank-invariant (10 tests, #157), plus Kokkos OpenMP en CI (#155). Couvre condensed_schur,
  polar_transport, lorentz, full_tensor, polar_poisson, krylov, schur_condensation (tous device-clean
  GH200, compute-sanitizer 0 erreur). Les 4 echecs initiaux venaient des tests (foncteurs ou pointeurs
  hote appeles dans des kernels device, ou lecture hote d'une sortie async sans fence), corriges
  #150/#152/#158 ; la bibliotheque elliptique / Schur / polaire est device-correcte.

## FFT sous MPI

`System` en MPI np>1 refuse la FFT proprement (#106, plus de segfault). `DistributedFFTSolver` existe
et est teste a part, mais n'est pas route dans `System`.
