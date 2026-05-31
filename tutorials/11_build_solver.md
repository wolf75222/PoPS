# 11, Construire son solveur avec adc (AMR, MPI, GPU)

Ce tutoriel montre comment utiliser `adc_cpp` comme une bibliotheque : definir son
propre modele physique, le faire tourner sur grille uniforme, puis sur AMR, puis le
distribuer en MPI et le porter sur GPU. Tout repose sur un principe : **la physique est
locale et generique, le reste (AMR, halos, MPI, backend) est branche par-dessus sans
toucher au modele**.

![diocotron AMR sur ROMEO](../docs/anim_romeo_diocotron_amr3.gif)

## Le modele mental : cinq couches

| Couche | Tu ecris / choisis | Fichiers |
|---|---|---|
| 1. Physique | un `PhysicalModel` (flux, source, terme elliptique) | `model/`, `core/physical_model.hpp` |
| 2. Numerique | un flux (`RusanovFlux`...) + une reconstruction (`VanLeer`...) | `operator/` |
| 3. Donnees + maillage | `MultiFab`, `BoxArray`, le seam `for_each_cell` | `mesh/` |
| 4. Execution | un backend a la compilation (serie / OpenMP / MPI / Kokkos) | seams `for_each.hpp`, `comm.hpp` |
| 5. Temps + couplage | un integrateur (`ssprk2`) et un coupleur (`AmrCouplerMP`) | `integrator/`, `coupling/` |

Tu n'ecris en general que la couche 1. Les couches 2 a 5 sont des composants existants
que tu instancies par template.

## Etape 1 : definir ton modele

Un modele satisfait le concept `PhysicalModel` (`core/physical_model.hpp`) : des types
`State` / `Aux`, un nombre de variables, un flux, une vitesse d'onde max, une source, et
un second membre elliptique. Exemple minimal : advection scalaire a vitesse constante
`(a, b)` (pas de couplage elliptique).

```cpp
#include <adc/core/state.hpp>

struct Advection {
  using State = adc::StateVec<1>;   // 1 variable conservee
  using Aux   = adc::Aux;           // (phi, grad phi) ; inutilise ici
  static constexpr int n_vars = 1;
  double a = 1.0, b = 0.5;          // vitesse d'advection

  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const double v = (dir == 0) ? a : b;
    return State{v * u[0]};         // F = v u
  }
  ADC_HD double max_wave_speed(const State&, const Aux&, int dir) const {
    return (dir == 0) ? std::fabs(a) : std::fabs(b);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{0.0}; }
  ADC_HD double elliptic_rhs(const State& u) const { return u[0]; }  // inutilise
};
static_assert(adc::PhysicalModel<Advection>);
```

Regles : tout est `ADC_HD` (appelable host ET device), `State`/`Aux` sont des POD, le flux
ne capture rien d'externe. `source` et `elliptic_rhs` sont obligatoires meme si triviaux
(un seul `assemble_rhs` bit-identique, cf. ARCHITECTURE.md section 4). Pour un modele
couple (diocotron, Euler-Poisson), `flux` ou `source` lisent `aux.phi` / `aux.grad_x` et
`elliptic_rhs` renvoie le second membre de `D phi = f` (ex. `alpha*(n_e - n_i0)` pour le
diocotron, `model/diocotron.hpp`).

## Etape 2 : le faire tourner sur grille uniforme

Un champ vit dans un `MultiFab` (collection distribuee de `Fab2D`). Sur une grille on
assemble le RHS via `compute_face_fluxes` / `assemble_rhs` et on avance avec un
integrateur SSP-RK. Le couplage hyperbolique-elliptique (Poisson a chaque etage) est
encapsule par `Coupler<Model, Elliptic>` (`coupling/coupler.hpp`) :

```cpp
#include <adc/coupling/coupler.hpp>
#include <adc/elliptic/geometric_mg.hpp>

adc::Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
adc::Coupler<Diocotron, adc::GeometricMG> sim(model, geom, ba, bc);
// remplir sim.state() (densite initiale), puis :
for (int s = 0; s < nsteps; ++s) sim.advance<adc::VanLeer>(dt);
```

Le coupleur ferme la boucle `Poisson -> aux = grad phi -> advance`. Pour un modele pur
hyperbolique (Euler sans Poisson), on appelle directement `advance_ssprk2` /
`advance_ssprk3` (`integrator/ssprk.hpp`).

## Etape 3 : choisir le schema (policies template)

Le flux et la reconstruction sont des **politiques** template, pas des branches a
l'execution :

- Flux (`operator/numerical_flux.hpp`) : `RusanovFlux` (robuste, diffusif), `HLLFlux`,
  `HLLCFlux` (contact, Euler).
- Reconstruction (`operator/reconstruction.hpp`) : `NoSlope` (ordre 1), MUSCL
  (`Minmod`, `VanLeer`, `MC`), `WENO5Z` (ordre 5).
- Integrateur (`integrator/`) : `ssprk2`, `ssprk3` ; `imex` / `splitting` pour le raide.

Le `Coupler` expose le limiteur et la **policy de couplage** (frequence du Poisson) ; son
flux est Rusanov :

```cpp
sim.advance<adc::VanLeer>(dt);                            // MUSCL VanLeer, Poisson par etage
sim.advance<adc::NoSlope, adc::OncePerStepCoupling>(dt);  // ordre 1, Poisson 1x/pas
```

Pour choisir AUSSI le flux numerique, on descend d'un cran : `compute_face_fluxes<Limiter,
NumericalFlux>` (operateur spatial) et `advance_amr<Limiter, NumericalFlux>` (moteur AMR)
prennent le flux en second parametre. Changer de schema ne touche jamais le modele.

## Etape 4 : passer en AMR

Le coupleur AMR multi-patch est `AmrCouplerMP<Model, Elliptic>`
(`coupling/amr_coupler_mp.hpp`). La hierarchie est une `std::vector<AmrLevelMP>` (un
`MultiFab` par niveau, ratio 2). Le moteur unique est `advance_amr` (multi-patch
N-niveaux, sous-cyclage Berger-Oliger + reflux conservatif) ; le mono-box en est le cas
degenere bit-identique.

```cpp
#include <adc/coupling/amr_coupler_mp.hpp>

std::vector<adc::AmrLevelMP> levels(2);
levels[0] = {std::move(Uc), nullptr, dxc, dyc};        // niveau grossier
levels[1] = {std::move(Uf), nullptr, dxc/2, dyc/2};    // un patch fin
adc::AmrCouplerMP<Diocotron> sim(model, geom, ba_coarse, bc, std::move(levels));

auto crit = [&](const adc::ConstArray4& a, int i, int j) {  // critere de raffinement
  return a(i, j, 0) > seuil;
};
sim.regrid(crit);                 // clustering Berger-Rigoutsos -> patchs
for (int s = 0; s < nsteps; ++s) {
  if (s % 20 == 0) sim.regrid(crit);   // re-raffine periodiquement
  sim.step(dt);                        // sync_down + Poisson + inject + advance_amr
}
double m = sim.mass();            // diagnostic conservatif (reflux a l'arrondi)
```

Le critere de raffinement est a toi (specifique au probleme). Le coupleur s'occupe du
reste : tagging, clustering, regrid avec proper nesting, injection d'aux parent->enfant,
reflux coverage-aware. Conservation a l'arrondi (`~1e-15`). Voir `04_amr_multilevel.md`
et `05_amr_multipatch.md` pour le detail, et `examples/diocotron_amr3.cpp` pour un cas
3 niveaux complet.

## Etape 5 : choisir le backend (a la compilation)

Le seam `for_each_cell` (`mesh/for_each.hpp`) bascule serie -> OpenMP -> Kokkos sans
toucher aux operateurs. On le choisit a `cmake`, une fois, et **tout ce qui lie `adc` en
herite** :

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread
cmake -B build -DADC_USE_MPI=ON      # distribue
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU (ou CPU portable)
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le meme code (`for_each_cell(box, [=] ADC_HD(int i,int j){ ... })`) compile pour les
quatre. OpenMP est garanti **bit-identique a la serie** (pas de `reduction(+:)` qui
reordonnerait la somme). ![scaling OpenMP](../docs/fig_openmp_scaling.png)

## Etape 6 : distribuer en MPI

Le seam MPI est `parallel/comm.hpp` (degenere en rang unique sans `ADC_HAS_MPI`). La
distribution se fait par `DistributionMapping` (box -> rang) ; un `MultiFab` reparti sait
echanger ses halos par `fill_boundary` et reduire par `all_reduce_*`.

```cpp
adc::DistributionMapping dm(ba.size(), adc::n_ranks());  // round-robin
adc::MultiFab U(ba, dm, 1, 1);                           // champ reparti
adc::fill_boundary(U, dom, adc::Periodicity{true, true}); // halos (MPI)
double total = adc::all_reduce_sum(local_sum);            // reduction globale
```

Lancer : `mpirun -np 4 ./build/bin/diocotron_mpi`. Le contrat du depot est que le
resultat est **bit-identique a np=1/2/4** (verifie par les tests `test_mpi_*`). Pour
l'AMR distribue, `AmrCouplerMP` prend un parametre `replicated_coarse` (defaut `true`) :
niveau 0 replique par rang (meilleur multigrille, pas de comm) ou multi-box reparti
(scalable, a activer quand la memoire du niveau 0 devient le verrou). Voir
`08_backends.md` et `examples/diocotron_mpi.cpp`.

## Etape 7 : porter sur GPU

Sous `ADC_USE_KOKKOS` + CUDA, `for_each_cell` devient un `Kokkos::parallel_for` sur
l'espace Cuda ; le stockage `Fab2D` est en memoire unifiee (`cudaMallocManaged`). Une
seule discipline : appeler `device_fence()` avant toute **lecture hote** d'un buffer
ecrit par un kernel (les reductions `for_each_cell_reduce_*` l'absorbent). Validation
GH200 : pas couple + AMR multi-patch **bit-identiques au CPU** (checksum
`4394594.404318`), voir `romeo/HERO_RESULTS.md` et `romeo/sanitizer.sbatch`.

## Carte des exemples

| Tu veux voir... | Exemple |
|---|---|
| diocotron uniforme | `examples/diocotron.cpp` |
| AMR 2 / 3 niveaux + regrid | `examples/diocotron_amr.cpp`, `diocotron_amr3.cpp` |
| multi-patch + Berger-Rigoutsos | `examples/diocotron_multipatch.cpp` |
| MPI distribue | `examples/diocotron_mpi.cpp` |
| GPU Kokkos | `examples/gpu/diocotron_amr_kokkos.cpp` |
| Euler-Poisson (gravite / plasma) | via la facade `EulerPoissonSolver` |
| deux-fluides AP | `examples/` + `TwoFluidAPSolver` |

## Depuis Python

Pour piloter sans recompiler, la facade `libadc` est bindee (`-DADC_BUILD_PYTHON=ON`) :
solveurs concrets `DiocotronSolver` / `EulerPoissonSolver` / `TwoFluidAPSolver`, champs
rendus en numpy. Voir `03_python_api.md`. Le cur generique reste C++ template (les
modeles perso de l'etape 1 s'utilisent en C++) ; Python expose les solveurs deja
compiles.

## Pour aller plus loin

- Concepts et seams en detail : [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (dont
  l'arborescence fichier par fichier, section 13).
- Algorithmes (formules + pseudocode) : [docs/ALGORITHMS.md](../docs/ALGORITHMS.md).
- Choix de conception : [docs/CHOICES.md](../docs/CHOICES.md).
