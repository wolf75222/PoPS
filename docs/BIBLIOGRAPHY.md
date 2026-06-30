# Bibliography

The references that informed the design and implementation of `adc_cpp`: existing AMR /
plasma codes consulted, textbooks, key papers. None was copied; each contributed
an idea.

## 1. AMR / plasma codes consulted

### AMReX

Reference block-structured AMR framework (LBNL), C++17 + GPU. The `adc_cpp` stack is
a mini-clone of it written *from scratch*: the direct correspondences are `MultiFab`,
`BoxArray` / `DistributionMapping`, `Geometry`, `FillBoundary`, the FluxRegister (reflux),
the MLMG (~ `GeometricMG`). Accepted divergences: no `MFIter` (we iterate `for_each_cell`
+ local fab, GPU-ready), variable-coefficient Laplacian but staircase EB (Shortley-Weller
cut-cell for the curved boundary). The multi-patch is MPI-distributed (bit-identical np=1/2/4).
[Repo](https://github.com/AMReX-Codes/amrex), Zhang et al. 2019, *AMReX*, JOSS 4(37).

### WarpX

Electromagnetic PIC-AMR code (on AMReX) for the physics of plasmas and
accelerators. Context for the hyperbolic-elliptic coupling on AMR for non-neutral
plasmas (diocotron) and the fluid model.
[Repo](https://github.com/ECP-WarpX/WarpX).

### Athena++ / PLUTO

Astrophysical hydro/MHD frameworks. The **orthogonal-axes** design of PLUTO (equation x
reconstruction x Riemann x integrator) inspired the concept-template decomposition of `adc_cpp`
(`PhysicalModel` / `NumericalFlux` / `EllipticSolver` / coupling).
[Athena++](https://github.com/PrincetonUniversity/athena),
[PLUTO](http://plutocode.ph.unito.it).

## 2. Textbooks

- **Birdsall & Langdon**, *Plasma Physics via Computer Simulation*, 1985. E x B drift,
  plasma and cyclotron frequencies, diocotron instability.
- **Chen**, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed., 2016. Langmuir
  oscillation, Bohm-Gross dispersion `omega^2 = omega_p^2 + 3 k^2 v_th^2`, Debye
  length: repulsive side of Euler-Poisson (`InteractionKind::Plasma`).
- **Binney & Tremaine**, *Galactic Dynamics*, 2nd ed., 2008. Jeans instability, gravitational
  dispersion `omega^2 = c_s^2 k^2 - 4 pi G rho0`: attractive side of Euler-Poisson
  (`InteractionKind::Gravity`).
- **Toro**, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009.
  Riemann solvers (Rusanov, HLL, HLLC), MUSCL reconstruction, conservative form.
- **Trottenberg, Oosterlee & Schüller**, *Multigrid*, 2001. V-cycle, red-black Gauss-Seidel
  smoother, restriction / prolongation.

## 3. Key papers

- **Berger & Oliger**, 1984, *Adaptive mesh refinement for hyperbolic partial differential
  equations*, JCP 53. Time subcycling of the fine levels.
- **Berger & Colella**, 1989, *Local adaptive mesh refinement for shock hydrodynamics*,
  JCP 82. Reflux (FluxRegister) at the fine-coarse interface, conservation.
- **Berger & Rigoutsos**, 1991, *An algorithm for point clustering and grid generation*,
  IEEE Trans. SMC 21. Signature clustering for the regrid.
- **Hoffart**, 2025, arXiv:2510.11808. Isothermal two-fluid model, validation target
  of the asymptotic-preserving scheme (application scenario, `adc_cases/two_fluid_ap/`).

## 4. Performance methodology

- **Bryant & O'Hallaron**, *Computer Systems: A Programmer's Perspective*, 3rd ed., 2016.
  Profile first, identify the bottleneck, transform, re-measure.
