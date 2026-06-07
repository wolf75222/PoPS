# Bibliographie

Les references qui ont informe la conception et l'implementation d'`adc_cpp` (codes AMR /
plasma consultes, manuels, articles cles). Aucun n'a ete copie ; chacun a apporte une idee. La
liste complete, annotee, est tenue dans [BIBLIOGRAPHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BIBLIOGRAPHY.md) ; ci-dessous un
resume oriente lecteur.

## Codes AMR / plasma consultes

- **AMReX** (Zhang et al. 2019, JOSS 4(37)) -- framework AMR block-structured de reference dont
  la pile mesh d'`adc_cpp` est un mini-clone ecrit *from scratch* (MultiFab, BoxArray, Geometry,
  FillBoundary, FluxRegister, MLMG ~ GeometricMG).
- **WarpX** -- code PIC-AMR electromagnetique (sur AMReX), contexte du couplage
  hyperbolique-elliptique sur AMR pour les plasmas non neutres.
- **Athena++ / PLUTO** -- frameworks hydro/MHD ; le design a axes orthogonaux de PLUTO
  (equation x reconstruction x Riemann x integrateur) a inspire le decoupage concept-template
  d'`adc_cpp`.

## Manuels

- **Birdsall & Langdon**, *Plasma Physics via Computer Simulation*, 1985 -- derive E x B,
  instabilite diocotron.
- **Chen**, *Introduction to Plasma Physics and Controlled Fusion*, 3e ed., 2016 -- cote
  repulsif d'Euler-Poisson (plasma).
- **Binney & Tremaine**, *Galactic Dynamics*, 2e ed., 2008 -- instabilite de Jeans, cote
  attractif d'Euler-Poisson (gravite).
- **Toro**, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e ed., 2009 --
  solveurs de Riemann (Rusanov, HLL, HLLC), reconstruction MUSCL.
- **Trottenberg, Oosterlee & Schuller**, *Multigrid*, 2001 -- V-cycle, lisseur Gauss-Seidel
  rouge-noir.

## Articles cles

- **Berger & Oliger**, 1984, JCP 53 -- sous-cyclage en temps des niveaux fins.
- **Berger & Colella**, 1989, JCP 82 -- reflux (FluxRegister), conservation a l'interface
  grossier-fin.
- **Berger & Rigoutsos**, 1991, IEEE Trans. SMC 21 -- clustering par signature pour le regrid.
- **Hoffart**, 2025, **arXiv:2510.11808** -- modele deux-fluides isotherme et Euler-Poisson
  magnetise, cible de validation du schema asymptotic-preserving (scenario applicatif,
  `adc_cases/two_fluid_ap/`). Voir la limite de reproduction du modele complet dans
  [limitations](limitations.md).

## Methodologie performance

- **Bryant & O'Hallaron**, *Computer Systems: A Programmer's Perspective*, 3e ed., 2016 --
  profiler d'abord, identifier le goulot, transformer, re-mesurer (voir
  [PROFILE_RESULTS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md)).
