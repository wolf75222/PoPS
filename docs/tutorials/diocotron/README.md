# Hoffart diocotron in PoPS finite volumes

`01_mpi_kokkos_hoffart_euler.py` builds the full barotropic Euler–Poisson case.
`02_mpi_kokkos_hoffart_hyqmom15.py` builds the corresponding fifteen-moment case.
Both are deliberately linear tutorials: all authoring, compilation, execution,
diagnostics, and checkpointing occur in numbered stages at module scope.
`03_render_results.py` plots genuine native snapshots and generates a GIF.

## Reference and parameters

The reference is [Hoffart et al., arXiv:2510.11808v1](https://arxiv.org/html/2510.11808v1).
The main reproduction uses the [authors' released mode-5 benchmark](https://github.com/conservation-laws/ryujin/blob/7e8177dfe35ae5f6a1ae8477f55b1781a1c01c43/prm/benchmarks/euler_poisson_barotropic-diocotron_instability-mode_5.prm),
as selected for this task: disk radius 16, annulus radii 6 and 8, background
density `1e-6`, annulus density `0.9 + 0.1*sin(mode*theta)`,
`alpha=39.4784176e12`, `Omega=-6.28318531e12`, temperature `1e-24`, and
final physical time 10. Modes 3, 4, and 5 are supported. The initial drift is
computed from the authors' magnetic-drift initialization, then integrated over
each finite volume by the native conservative quadrature.

These constants differ from the printed paper's scaling. The paper also omits
the benchmark temperature. Results must identify which constants they use;
neither the time axis nor the density is rescaled afterward to fit a figure.

## Equations and discretization

The computational rectangle is `(r,theta) in [0,16] x [0,2*pi]`. It parametrizes
the entire physical disk, including the pole; no central hole is removed.
Velocity and moment components remain Cartesian. Stored conservative unknowns
are `q_ij = r*M_ij`. Their polar fluxes are

```
F_r     = cos(theta)*F_x(q) + sin(theta)*F_y(q)
F_theta = (-sin(theta)*F_x(q) + cos(theta)*F_y(q))/r.
```

Geometry auxiliaries are native Kokkos kernels evaluated on the actual level
geometry, including after restart or regrid. The trigonometric cell factors use
the radial-face angular average and the two-sided angular-face correction.
Rusanov's central flux then preserves a constant Cartesian state in the interior
of a uniform level. First-order radial dissipation leaves an origin truncation
error whose integrated magnitude vanishes linearly under refinement; exact
pointwise free-stream preservation at the pole is not claimed. The outer
transport condition follows the authors' open/do-nothing wall, while the pole
has zero flux and the angular direction is periodic.

Electrostatics uses the mapped gradient
`C=[[cos(theta),-sin(theta)/r],[sin(theta),cos(theta)/r]]` and tensor
`K=diag(r,1/r)`. For `s=dt/2`, `J(vx,vy)=(vy,-vx)`, and `B=I-s*Omega*J`,
the full-Gauss-restart condensed tensor is

```
A   = K + s*s*alpha*q00*C.T*inverse(B)*C
rhs = q00 - s*div(C.T*inverse(B)*q_momentum).
```

The solved potential is `psi=phi/alpha`. Composite tensor FAC couples the AMR
levels, with a zero conducting potential at the outer wall and zero conormal
flux at the pole. The source update uses the reconstructed midpoint mean.
Each solve starts from zero with the stated convergence tolerance. Its diagnostic
potential history is stored without a lagged warm-start read, allowing newly
refined cells to receive the current hierarchy solution before history rotation.
At an equal-clock regrid, authenticated scalar output histories retain existing
fine-cell values and obtain only new coverage from the matching parent sample.
The two history-contract markers embedded in every snapshot identify this
implementation; old archives without them are excluded from Fourier analysis.
Run the tutorials with the native library rebuilt from the same source revision.
For HYQMOM15, one common affine Cayley velocity map updates all fifteen moments;
the higher moments are not frozen during the Lorentz/electric update.

The authored method is source-first Lie composition with Crank–Nicolson source
and SSPRK2 finite-volume transport. Its splitting order is one. Euler uses
MUSCL with Minmod; HYQMOM15 uses first-order reconstruction and its full
directional 15-by-15 closure Jacobian. The latter spectral bound alone is not a
proof of moment realizability. Actual admissibility, time refinement, and spatial
refinement must be checked before scientific interpretation. There is no moment
floor, artificial temperature, or projection onto admissible moments.

The centered mapped gradient/divergence and the finite-volume tensor operator
do not reproduce Hoffart's exact discrete energy identity. The selected
full-Gauss restart also differs from the no-restart caption of the reference
image. These are explicit method differences, not claims of identical output.

## Execution and figures

Use a native two-dimensional PoPS build with MPI, Kokkos OpenMP, and parallel
HDF5. Set `POPS_THREADS` before process startup; launch at least two MPI ranks
and two OpenMP threads per rank for the parallel qualification. The main controls
are `POPS_NR`, `POPS_NTHETA`, `POPS_MAX_LEVELS`, `POPS_MODE`, `POPS_CFL`,
`POPS_MAX_DT`, `POPS_T_END`, and `POPS_OUTPUT_INTERVAL`. The tutorials use
synchronous AMR steps, conservative transfer, accepted-state checkpoints, and
the public PoPS compile/bind/run interfaces.
The FAC coarse correction uses prepared GMRES with a fixed diagonal preconditioner.
Every matrix application retains the complete finite-Omega tensor and the conducting-disk
boundary law. GMRES uses Euclidean Arnoldi products and an explicitly authenticated
physical infinity norm for stopping. Its reference, initial and final residual checks
use that cellwise norm with the unchanged coarse tolerance. FAC independently checks
the original tensor residual, and the outer solve keeps its original composite residual
tolerances, fine smoothing and correction damping. This
is a solver change; it does not replace the equation by its drift limit. The first real
coefficient snapshot prepares the persistent GMRES sessions before iteration. Runtime
and numerical qualification of a new solver revision are recorded separately from its
implementation; no speedup is assumed from selecting GMRES.
Coarse patches are explicitly distributed across MPI ranks. `POPS_COARSE_MAX_GRID`
bounds coarse patches; `POPS_CLUSTER_MAX_GRID` bounds clusters in parent tagging
cells before factor-two refinement. The latter is not a bound in child-cell units.

Output targets are formed from exact decimal cadences, then converted once to
native binary64. Each global output interval is divided into comparable absolute
subintervals whose represented widths do not exceed `POPS_MAX_DT`; an extra
subinterval is used when floating-point rounding requires it. This avoids the
microscopic cap-limited remainder steps produced by repeated `time + max_dt`.
The same global endpoints are regenerated after restart. AdaptiveCFL may still
select smaller physical steps. `chunks.jsonl` records the actual returned time,
step count, latest accepted dt and cost of every public invocation.

Snapshots are saved at intervals of at most `0.01` through `t=1.5`, covering all
three fixed growth-fit windows; later output uses `POPS_OUTPUT_INTERVAL` and the
exact paper times. Include this early sampling cadence when estimating storage.

`POPS_RUN_OUTPUT`, `POPS_RUN_CHECKPOINT`, `POPS_RUN_RESTART`, and
`POPS_RUN_WALLTIME_SECONDS` support bounded scheduler segments. A checkpoint is
published only after an accepted native step. The calling launcher must reserve
time for compilation, checkpoint archival, and MPI shutdown. On ROMEO, native
transactions run on node-local storage and their completed immutable bytes are
archived to GPFS with checksum verification.

The attached nine-panel reference is a schlieren plot of `|grad(rho)|`.
The renderer uses the corresponding times `0.1, 1.25, 2.5, ..., 10` and records
missing times instead of synthesizing panels. Potential Fourier amplitudes at
`r=6` use each source solve's actual midpoint timestamp. The first native solve
at `5e-11` provides the near-zero normalization; it is not labeled as an exact
time-zero field solve. See [SNAPSHOTS.md](SNAPSHOTS.md) for the data contract.

Implementation checks, native manufactured solutions, short case integration,
and paper-scale scientific qualification are separate evidence. The presence of
these scripts does not establish that a long run or a figure reproduction passed.

The current Euler implementation has passed short native MPI/AMR integration
and checkpoint continuation. The current HYQMOM15 implementation remains under
qualification: the attached notes' fifth-moment formulas correspond to Appendix
B.1 of [Bryngelson, Fox and Laurent (2026)](https://comp-physics.group/papers/bryngelson-JCP-26.pdf),
and its actual cold initial data produce complex characteristic pairs. The
native speed check therefore refuses the first step. The complete published
Cartesian corrections also fail oblique checks. An exact rational Gaussian
counterexample and the distinction from alternative fifteen-moment models are
documented in [HYQMOM15_LIMITATION.md](HYQMOM15_LIMITATION.md). Neither suppressing
imaginary parts nor increasing the imaginary tolerance is an accepted fix. This
implementation milestone does not claim an evolved HYQMOM15 result.
