# Algorithms

This guide describes the numerical components implemented in the current tree and how
they enter the public compiled Program. A standalone kernel, a supported bind route and
a scientifically qualified configuration are different claims. The source links below
identify implementations; [verification scope](development/migration_verification_scope.md)
records the acceptance boundary.

## Equations and composition

A typical finite-volume block evolves cell averages of conserved quantities:

$$\partial_t U + \nabla\cdot F(U,P) = S(U,P) - \nabla\cdot J(U,P).$$

Here $P$ is a compact set of owner-qualified provider values. Field problems can supply
these values by solving an authored residual equation. Neither the variable name nor a
particular physical application chooses the numerical algorithm. The physical model,
spatial operator and time Program have separate authorities.

[`physics`](../python/pops/physics), [`model`](../python/pops/model),
[`fields`](../python/pops/fields) and [`numerics`](../python/pops/numerics) provide the
Python authoring descriptions. [`include/pops/physics`](../include/pops/physics) contains
reusable native physical bricks.

## Finite-volume transport

For Cartesian cell $i$ with volume $V_i$, the spatial residual is

$$R_i = S_i - \frac{1}{V_i}\sum_{f\in\partial i}\sigma_{if}\,\Phi_f,$$

where $\Phi_f$ is the numerical flux integrated over a face and $\sigma_{if}$ supplies
its outward orientation. A shared face contributes with opposite signs to adjacent cells.
This is the discrete conservation mechanism, subject to physical boundary/source terms
and floating-point accumulation.

[`PreparedCartesianOperator`](../include/pops/numerics/spatial/operators/cartesian_operator.hpp)
prepares layout/metric/reconstruction data, materializes a `FaceField<Dim>`, then calls
`assemble_residual` or `assemble_residual_from_face_fluxes`. The ranked
[`spatial_operator.hpp`](../include/pops/numerics/spatial_operator.hpp) is an include barrel;
there is no production `assemble_rhs` kernel to select. AMR exchange accounting consumes
the same face field.

A residual evaluation follows this order:

1. Establish the required state/provider ghosts and physical boundaries.
2. Recover and reconstruct left/right traces at each owned face.
3. Evaluate the selected physical/numerical flux and integrate by face measure.
4. Assemble signed face differences divided by cell volume, and add declared sources.
5. Keep flux exchanges associated with the Program stage and its temporal weight.

### Numerical flux and reconstruction

Rusanov uses a two-sided wave-speed bound $a$:

$$\widehat F(U_L,U_R)=\tfrac12(F(U_L)+F(U_R))
 -\tfrac12 a(U_R-U_L).$$

HLL uses left/right signal bounds; HLLC and Roe require stronger model information.
Their descriptors are not interchangeable for an arbitrary minimal physical model.
Pointwise policies are in
[`numerical_flux.hpp`](../include/pops/numerics/fv/numerical_flux.hpp) and the
[ranked finite-volume components](../include/pops/numerics/fv).

MUSCL reconstructs limited slopes. WENO5-Z combines candidate polynomial reconstructions
with smoothness-dependent weights. The spatial order depends on smoothness, boundary
stencils, available ghost depth and the chosen variable-recovery path; the descriptor
alone does not establish global convergence order. See
[`reconstruction.hpp`](../include/pops/numerics/fv/reconstruction.hpp) and
[`spatial`](../include/pops/numerics/spatial).

### Stability bounds

Explicit steps must satisfy the actual spatial/source restrictions. For constant linear
advection on a Cartesian mesh, a useful multidimensional reference is

$$\Delta t\sum_a |v_a|/\Delta x_a \le C.$$

A scalar diffusion reference is
$\Delta t\,2\nu\sum_a\Delta x_a^{-2}\le 1$ for forward Euler.
These formulas illustrate the dependence on all axes; they do not replace the authored
operator's stability contract or prove every runtime estimator conservative. Source rates,
geometry, AMR cadence and the time method can impose additional restrictions. The runtime
aggregates declared bounds and reports the limiting reason. SSP claims require the
forward-Euler spatial step itself to satisfy the claimed stability property.

## Diffusion and fitted drift-diffusion

Diffusion is a conservative face flux, for example $J=-\nu\nabla U$. The prepared
Cartesian model path requires a finite nonnegative isotropic diffusivity and compatible
Cartesian metrics. The more general
[`prepared_diffusion.hpp`](../include/pops/numerics/diffusion/prepared_diffusion.hpp)
implements declared diffusion routes, including variable coefficients and supported tensor
or fitted constructions. The [Bernoulli function](../include/pops/numerics/diffusion/bernoulli.hpp)
supports exponentially fitted drift-diffusion fluxes with stable limiting evaluations.

Explicit diffusion contributes its own stability restriction. Implicit spatial diffusion
is an authored solve in the Program; selecting an IMEX descriptor does not automatically
make every diffusion operator implicit. Diffusive interface exchanges carry the same
accepted-stage accounting requirements as advective exchanges.

## Time integration through Program

The production temporal authority is the typed Program graph. Factories under
[`pops.lib.time`](../python/pops/lib/time) compose evaluations, linear combinations,
solves, history references and commits. Manual and factory authoring lower through the
same compiler. Low-level C++ stepper objects remain useful numerical primitives and test
references; they do not install a hidden production schedule.

### Explicit Runge-Kutta

For $\dot U=L(U)$, an explicit tableau defines

$$Y_i=U^n+\Delta t\sum_{j<i}a_{ij}L(Y_j),\qquad
U^{n+1}=U^n+\Delta t\sum_i b_iL(Y_i).$$

[`rk.py`](../python/pops/lib/time/rk.py) is the generic tableau builder, including shared
field evaluation at declared stage points. Forward Euler, SSPRK and RK4 build that graph.
SSPRK2 is

$$Y=U^n+\Delta t L(U^n),\qquad
U^{n+1}=\tfrac12 U^n+\tfrac12(Y+\Delta t L(Y)).$$

SSPRK3 uses the familiar convex stages with weights $(3/4,1/4)$ and $(1/3,2/3)$.
Stage-dependent fields must be evaluated at the authored stages to obtain the intended
coupled method. A frozen or reused field is an explicit approximation, not automatically
an equivalent higher-order solve.

### Implicit, IMEX and splitting

For $\dot U=E(U)+I(U)$, additive tableaux define separate explicit and implicit stage
contributions. [`imex.py`](../python/pops/lib/time/imex.py) builds the stage residuals and
solve actions; [`dirk.py`](../python/pops/lib/time/dirk.py) supplies diagonally implicit RK.
The authored residual and solve provider determine whether a problem is local, spatial,
linear or nonlinear. Solver tolerance, iteration limits and failure policy are part of
that contract. An implicit stage is not a blanket stability or asymptotic-preserving claim.

Lie composition applies two subflows over a step; Strang applies half a first subflow,
a full second subflow, then the remaining half first subflow. The
[`splitting factories`](../python/pops/lib/time/strang.py) author those intervals explicitly.
Second-order Strang accuracy also depends on the accuracy/consistency of the subflow solves.

[`multistep.py`](../python/pops/lib/time/multistep.py) constructs Adams-Bashforth and BDF
programs with qualified history samples. Startup, cadence and restart must provide the
required history; changing the step or sampling schedule requires compatible coefficients
and identities. [Temporal execution](design/temporal-execution-contract.md) defines the
history and restart rules.

## Field and nonlinear solves

Field problems own residuals, spaces, boundary conditions, normalization and solve actions.
They are not selected by a reserved potential name. Prepared providers validate the exact
problem/layout combination before execution.

| Component | Implemented role and conditions |
| --- | --- |
| [CartesianPoissonSolver](../include/pops/numerics/elliptic/nd/cartesian_poisson.hpp) | Exact-ranked scalar Cartesian field solver with explicit boundary/nullspace policy. |
| [GeometricMG](../include/pops/numerics/elliptic/mg/geometric_mg.hpp) | Multigrid hierarchy: smoothing, residual restriction, coarse correction and prolongation. |
| [Poisson FFT](../include/pops/numerics/elliptic/poisson/poisson_fft.hpp) | Periodic constant-coefficient Cartesian Poisson route, with optional FFTW radix backend. |
| [Prepared Krylov](../include/pops/numerics/elliptic/linear/generic_krylov.hpp) | Matrix-free iterative methods on prepared vector/operator contracts. |
| [Field Newton-Krylov](../include/pops/numerics/elliptic/interface/field_newton_krylov.hpp) | Nonlinear field residual/linearization with controlled inner solves. |
| [Local nonlinear solves](../include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp) | Prepared cell-local residuals and failure reporting. |

For periodic discrete Poisson, the Cartesian Laplacian symbol is

$$\lambda(k)=-4\sum_a \sin^2(\pi k_a/N_a)/\Delta x_a^2.$$

The inverse divides resolved modes by this symbol. The current kernel treats
$|\lambda|<10^{-14}$ as the zero mode, so large-domain scaling requires specific
qualification. The zero mode requires an explicit compatibility and gauge policy. Inverting a continuous $-|k|^2$ symbol would solve a different discrete
problem. FFT constraints do not extend automatically to variable coefficients or arbitrary
physical boundaries.

Multigrid and Krylov convergence must be assessed with the problem's declared norm,
normalization and stopping criteria. Positive-definiteness and nullspace assumptions must
match the chosen method; an iteration count or successful allocation is not proof of a
correct solve.

Two CompositeFAC implementations remain live: the
[partitioned AMR implementation](../include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp)
and the [structured specialization](../include/pops/numerics/elliptic/mg/composite_fac_poisson.hpp).
They serve different prepared routes. Similar names do not make them redundant.
[The condensed-FAC tutorial](tutorials/condensed_fac/README.md) demonstrates an authored
implicit composition, with its solver and coupling choices visible.

## AMR conservation, transfer and regrid

The hierarchy supplies exact level ratios, valid regions, coverage and ownership. Fine
steps accumulate oriented interface fluxes with their temporal weights. At synchronization,
reflux corrects the discrepancy between coarse and fine integrated fluxes; fine-to-coarse
restriction updates covered coarse values. Interface preparation distinguishes coarse/fine
boundaries from fine/fine joins to avoid double counting.

The live ledgers and prepared coarse/fine operations are in
[`numerics/time/amr`](../include/pops/numerics/time/amr) and
[`amr`](../include/pops/amr). Reflux is an explicit part of the installed Program's accepted
exchange protocol, not a model-specific postprocessing hook.

Tagging and clustering propose patches subject to hierarchy constraints. Regrid prepares
new storage and transfers state, fields and required histories before publishing the new
hierarchy. A failed transfer or solve must not partially replace the accepted state.
Anisotropic ratios, MPI ownership and restart identities must be qualified together on the
selected route; a serial refinement example does not establish distributed correctness.

## Geometry and parallel execution

Cartesian data and stencils are ranked by compile-time dimension. Embedded-boundary kernels
introduce cut geometry, face measures and active-cell constraints. Small cut cells can
strengthen stability restrictions. Standalone polar ring algorithms use their own metric
terms and capability limits; their presence does not imply a polar `System` runtime.

[`mesh`](../include/pops/mesh) owns box/layout and field storage;
[`parallel`](../include/pops/parallel) owns communication seams. Kokkos supplies local
execution. MPI halo exchanges, reductions and collective publication require matching rank
participation, including failure paths. Actual device/backend support is established by
compiled artifacts and corresponding tests, not by template syntax alone.

## Choosing and checking a route

Start with a [tutorial](tutorials/README.md) matching the problem class, inspect its authored
flux/reconstruction/field/Program choices, then run the matching contract tests in
[tests/test_manifest.toml](../tests/test_manifest.toml). Complete acceptance examples live
under [examples](../examples/README.md). Performance claims require the
[benchmark protocol](../benchmarks/README.md) and comparable measurements, not just fewer
source lines or operation-count assertions.
