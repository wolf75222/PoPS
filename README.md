<div align="center">

# ADC CPP

**Solveur C++23 pour systemes hyperbolique-elliptique couples, avec AMR, MPI et GPU.**

![Tests](https://img.shields.io/badge/tests-66%20%2B%2015%20MPI-brightgreen)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Instabilite diocotron AMR 3 niveaux sur ROMEO" width="640">
</p>

<div align="center">
<sub>
Instabilite diocotron (derive E x B) sur AMR 3 niveaux emboites, produite sur ROMEO (x64cpu, 96 coeurs AMD EPYC).
Une bande de charge cree un ecoulement cisaille instable qui s'enroule en cat's eyes. Les patchs fins
(cyan = niveau 1, vert = niveau 2) suivent les zones de fort gradient par regrid dynamique (Berger-Rigoutsos).
Transport v = (E x B)/B^2, phi resolu par multigrille a chaque etage SSPRK2, sous-cyclage Berger-Oliger +
reflux conservatif aux interfaces coarse-fine (derive de masse ~ 1e-15). Moteur : <code>advance_amr</code>,
multi-patch N-niveaux distribue. Reproduction :
<code>sbatch romeo/diocotron_amr3_gif.sbatch</code> puis
<code>python3 scripts/make_diocotron_amr3_gif.py out_gif_amr3 docs/anim_romeo_diocotron_amr3.gif</code>.
</sub>
</div>

---

ADC resout, sur maillage cartesien adaptatif :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi = f(U)
```

ou la partie hyperbolique (U) et la partie elliptique (phi) sont couplees a chaque pas
via aux = (phi, grad phi). Les termes diffusifs / visqueux sont hors coeur actuel (ou
traites separement s'ils existent : le repo voisin `euler_cpp` porte un flux visqueux).

Forme complete VISEE, le terme diffusif `div H(U, grad U)` n'etant pas encore une brique
generique validee du coeur :

```
d U / d t  +  div F(U, phi)  =  div H(U, grad U)  +  S(U, phi)
D phi = f(U)
```

Cas de validation : l'instabilite **diocotron** (Hoffart et al., arXiv:2510.11808)
et le **deux-fluides isotherme** plasma.

## Solveurs

| Module | Role | Detail |
|---|---|---|
| [`model::Diocotron`](include/adc/model/diocotron.hpp) | derive E x B (vorticite reduite, scalaire) | flux advectif, `elliptic_rhs = alpha (n_e - n_i0)` |
| [`model::Euler`](include/adc/model/euler.hpp) | Euler compressible (γ = 1.4, 4 var) | validé free-stream + tourbillon isentropique (ordre 1.86) |
| [`model::EulerPoisson`](include/adc/model/euler_poisson.hpp) | Euler couple Poisson : gravite OU plasma (`InteractionKind`) | source g = -grad phi, un seul signe ; Jeans (0.1%) et Bohm-Gross + Coulomb (0.1%) valides |
| [`model::LangmuirMode`, `TwoFluidLinear`](include/adc/model/two_fluid_isothermal.hpp) | noyaux 0D AP (Ä = K A) | 2 branches plasma (Langmuir + ion-acoustique), Vieta exact |
| [`operator::{RusanovFlux,HLLFlux,HLLCFlux}`](include/adc/operator/numerical_flux.hpp) | flux numeriques (politiques `ADC_HD`) | validé Sod vs Riemann exact |
| [`operator::reconstruction`](include/adc/operator/reconstruction.hpp) | MUSCL ordre 2 (NoSlope / Minmod / VanLeer) | limiteur en parametre de template |
| [`operator::assemble_rhs` / `compute_face_fluxes`](include/adc/operator/spatial_operator.hpp) | `R = -div F + S` ; flux de FACE pour le reflux | `<Limiter, NumericalFlux>`, GPU via `for_each_cell` |
| [`integrator::{ssprk2,ssprk3}`](include/adc/integrator/ssprk.hpp) | Shu-Osher SSP-RK | TVD-stable |
| [`integrator::imex_euler_step`](include/adc/integrator/imex.hpp) | IMEX (raide implicite + non-raide explicite) | **asymptotic-preserving** |
| [`integrator::{lie_step,strang_step}`](include/adc/integrator/splitting.hpp) | splitting d'operateurs | ordre 1 / 2 |
| [`integrator::two_fluid_ap`](include/adc/integrator/two_fluid_ap.hpp) | deux-fluides 2D AP (Poisson reformule) | quasi-neutre a `dt` fixe quand `λ_D -> 0` |
| [`integrator::advance_amr`](include/adc/integrator/amr_reflux_mf.hpp) | moteur AMR unifie : multi-patch N-niveaux, reflux coverage-aware, distribue MPI | mono-box = cas degenere ; pile mono-box `amr_*_mf` en `detail::` (oracle de test) |
| [`elliptic::GeometricMG`](include/adc/elliptic/geometric_mg.hpp) | multigrille geometrique (V-cycle GS rb) | compatible AMR, on-device |
| [`elliptic::PoissonFFTSolver`](include/adc/elliptic/poisson_fft_solver.hpp) | Poisson FFT spectrale directe (`EllipticSolver`, **mono-rang**) | mono-niveau periodique `n=2^k`, ~5x ; variante distribuee = `DistributedFFTSolver` (`EllipticSolver` par bandes, enveloppe `PoissonFFT`) |
| [`coupling::Coupler`](include/adc/coupling/coupler.hpp) | couplage hyperbolique-elliptique par etage | `Coupler<Model, Elliptic = GeometricMG>` |
| [`coupling::AmrCoupler`](include/adc/coupling/amr_coupler.hpp) | couplage E x B AMR mono-box (route par `advance_amr`) | conservation a 5.55e-16 |
| [`coupling::SpectralCoupler`](include/adc/coupling/spectral_coupler.hpp) | couplage E x B distribue ; delegue le Poisson a `DistributedFFTSolver` | MPI, `MPI_Alltoall` |
| [`amr::{cluster,regrid,tag_box}`](include/adc/amr) | tagging + clustering Berger-Rigoutsos + regrid | genere les patchs multi-box |
| [`solver::{Diocotron,DiocotronAmr,EulerPoisson,TwoFluidAP}Solver`](include/adc/solver) | **facades compilees** (PIMPL, `libadc`) | API stable sans template (apps, Python) ; `DiocotronAmrSolver` expose l'AMR multi-patch |

Concepts (`PhysicalModel`, `NumericalFlux`, `EllipticSolver`, `CouplingPolicy`) et
seams (`for_each_cell`, `comm`, `allocator`) : voir
[**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md). Profil run-time :
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Backends : configures UNE fois, herites partout

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc` ; **tout ce
qui lie `adc` en herite** : la facade `libadc` (`src/`), les tests, les exemples.
Aucun drapeau rebadge par cible.

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread
cmake -B build -DADC_USE_MPI=ON      # distribue (halos + FFT par MPI)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU GH200 (ou CPU portable) ; libadc compile pour le GPU
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le seam `for_each_cell` bascule serie -> `#pragma omp` -> `Kokkos::parallel_for`
(Cuda) sans toucher les operateurs. Pas couple Euler-Poisson et deux-fluides AP
valides **bit-identiques CPU vs GH200** (cf. `scripts/romeo_*.sbatch`).

## Ecosysteme

ADC est le membre **from scratch** d'une famille de solveurs PDE C++ : la ou
`euler_cpp` / `advection_cpp` reutilisent `pde_core_cpp` (mesh, fields, AMR),
**ADC porte sa propre pile AMR** (`BoxArray` / `MultiFab` / `for_each_cell`) pour viser
le GPU et le MPI sans dependance.

| Repo | Role | Socle maillage |
|---|---|---|
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | solveurs Poisson (Thomas, SOR, CG, DST, AMR + multigrille) | propre |
| [`pde_core_cpp`](https://github.com/wolf75222/pde_core_cpp) | infra partagee (mesh, fields, AMR, clustering) | propre |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection scalaire + Burgers + Chorin NS | `pde_core_cpp` |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D + viscous NS + sources plasma + Euler-Poisson | `pde_core_cpp` |
| **`adc_cpp`** (ce depot) | hyperbolique-elliptique sur **AMR** + GPU/MPI/Kokkos | **propre (from scratch)** |

## Documentation

- Tutoriels (C++ et Python en parallele, du diocotron a l'AMR multi-patch) : [tutorials/](tutorials/README.md)
- Algorithmes (formules, pseudocode, validation par methode) : [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Architecture (couches, seams, frontiere lib/demo, etat AMR) : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Schema deux-fluides AP (modele, reformulation, transport, enveloppe mesuree) : [docs/two_fluid_ap.md](docs/two_fluid_ap.md)
- Performance (profil, scaling, FFT vs MG) : [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Choix de conception : [docs/CHOICES.md](docs/CHOICES.md) ; bibliographie : [docs/BIBLIOGRAPHY.md](docs/BIBLIOGRAPHY.md) ; roadmap : [docs/ROADMAP.md](docs/ROADMAP.md)

Generer la doc hebergeable :

```bash
doxygen docs/Doxyfile                                   # reference C++ -> docs/_build/doxygen/html
pip install -r docs/sphinx/requirements.txt
python3 -m sphinx -b html docs/sphinx docs/_build/sphinx # site Python + tutoriels
```

## Quick start

### Prerequis

- C++23 (AppleClang 16+, GCC 13+, Clang 17+) ; coeur compatible C++20, la norme retombe a C++20 sous Kokkos/nvcc (build GPU)
- CMake >= 3.20
- Eigen >= 3.4 *(cote host uniquement, analyse diocotron ; optionnel `-DADC_USE_EIGEN`)*
- MPI *(optionnel `-DADC_USE_MPI=ON`)*, Kokkos *(optionnel `-DADC_USE_KOKKOS=ON`, GPU)*, HDF5 *(optionnel `-DADC_USE_HDF5=ON`)*
- Python 3.10+ *(optionnel, bindings pybind11)*

### Build

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                 # 66 tests C++
```

Options CMake :

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | compile la suite CTest |
| `ADC_BUILD_EXAMPLES` | `ON` | compile les drivers (`diocotron`, `diocotron_amr`, ...) |
| `ADC_USE_OPENMP` | `OFF` | backend de dispatch OpenMP |
| `ADC_USE_KOKKOS` | `OFF` | backend de dispatch Kokkos (CPU/GPU portable) |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |
| `ADC_BUILD_PYTHON` | `OFF` | module pybind11 `adc` (facade `libadc`) |
| `ADC_USE_EIGEN` | `ON` | outils d'analyse host (theorie diocotron) |

### Python

`src/` compile la pile template une fois en `libadc` (API stable sans template),
bindee via pybind11 (`-DADC_BUILD_PYTHON=ON`). On expose les solveurs CONCRETS :

```python
import adc, numpy as np

# diocotron (bande de charge), pas stable choisi par la facade
cfg = adc.DiocotronConfig(); cfg.n = 192; cfg.ic = adc.DiocotronIC.Band
s = adc.DiocotronSolver(cfg)
for _ in range(200):
    s.step_cfl(0.4)
rho = s.density()        # numpy (n, n)
phi = s.potential()      # numpy (n, n)

# deux-fluides isotherme 2D, regime raide (asymptotic-preserving)
tc = adc.TwoFluidAPConfig(); tc.n = 64; tc.omega_pe = 1e3
ts = adc.TwoFluidAPSolver(tc)
ts.advance(5e-3, 200)    # dt*omega_pe = 5 : stable, quasi-neutre
print(ts.max_dev(), ts.max_charge())

# Euler-Poisson : meme code, deux physiques (le signe du couplage)
ec = adc.EulerPoissonConfig(); ec.n = 128; ec.use_fft = True
ec.interaction = adc.InteractionKind.Gravity   # attractif : effondrement de Jeans
# ec.interaction = adc.InteractionKind.Plasma  # repulsif : Langmuir + Coulomb
es = adc.EulerPoissonSolver(ec)
for _ in range(100): es.step(2e-3)
print(es.mass(), es.total_momentum(0))

# diocotron sur AMR : regrid Berger-Rigoutsos pilote depuis Python
ac = adc.DiocotronAmrConfig(); ac.n = 128; ac.regrid_every = 15
asim = adc.DiocotronAmrSolver(ac)
for _ in range(480): asim.step_cfl(0.4)
print(asim.n_patches(), asim.density().shape)   # patchs fins, niveau grossier numpy
```

## Organisation du depot

```
include/adc/   coeur generique header-only (concepts, MultiFab, for_each_cell, operateurs,
               elliptique, integrateurs, AMR, modeles). Templates -> visibles a l'instanciation.
src/           facade COMPILEE libadc : solveurs concrets PIMPL (Diocotron, EulerPoisson,
               TwoFluidAP). API stable, backend herite de la cible adc.
examples/      pilotes minces (main). diocotron/diocotron_column lient adc::solver (facade) ;
               diocotron_amr/amr3/multipatch/mpi/theory lient adc::adc (moteur : AMR, MPI, Eigen).
examples/gpu/  demos Kokkos/CUDA (GH200), heritent Kokkos de adc.
tests/         CTest (+ tests MPI via mpirun). python/ : module pybind11 + test.
scripts/       generation des GIF + jobs SLURM ROMEO (MPI, GPU).
docs/          ARCHITECTURE.md, PERFORMANCE.md, animations.
```

## Validation

- **66** tests C++ (serie = OpenMP) + **15** tests MPI bit-identiques np=1/2/4 ; bindings Python verts.
- AMR conservatif : reflux multi-patch a l'arrondi machine (`~1e-15`).
- GPU GH200 : pas couple + AMR bit-identiques au CPU (checksum exact).

Runs sur ROMEO (reproduction du taux diocotron, GPU) : [docs/ROMEO.md](docs/ROMEO.md).
