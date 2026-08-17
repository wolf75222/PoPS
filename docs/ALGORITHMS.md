# Algorithms

Catalog of the generic numerical methods of the `PoPS` core. For each one: the intuition, the
formula and its discretization, a pseudocode, the C++ file that implements it, and the constraints. The
core is model-agnostic; it names no scenario (diocotron, Euler-Poisson, two-fluid). These scenarios are
compositions of generic bricks and live on the application side
([`adc_cases`](https://github.com/wolf75222/adc_cases)); so do their end-to-end validations.

Each section follows the same plan: intuition (what it is for), formula and discretization (where it
comes from), pseudocode (the algorithm), code (the file and the functions), constraints and remarks (the
stability, the limits, the ctest test that covers it). All the file paths and test names cited exist in
this repository.

Architecture (layers, dispatch seam, library/application boundary):
[ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

- [Model equations](#model-equations)
- [1. Finite volumes: first-order Godunov](#1-finite-volumes-first-order-godunov)
- [2. Numerical fluxes: Rusanov, HLL, HLLC, Roe](#2-numerical-fluxes-rusanov-hll-hllc-roe)
- [3. MUSCL reconstruction (order 2) and WENO5-Z (order 5)](#3-muscl-reconstruction-order-2-and-weno5-z-order-5)
- [4. Time-integration bricks: SSPRK and object integrators](#4-time-integration-bricks-ssprk-and-object-integrators)
- [5. Stiff sources: asymptotic-preserving IMEX and partial IMEX](#5-stiff-sources-asymptotic-preserving-imex-and-partial-imex)
- [6. Operator splitting: Lie and Strang](#6-operator-splitting-lie-and-strang)
- [7. Test-only multirate reference formulas](#7-test-only-multirate-reference-formulas)
- [8. Parabolic term: diffusion as face flux](#8-parabolic-term-diffusion-as-face-flux)
- [9. Elliptic: geometric multigrid](#9-elliptic-geometric-multigrid)
- [10. Elliptic: exact-ranked Cartesian discrete Poisson FFT](#10-elliptic-exact-ranked-cartesian-discrete-poisson-fft)
- [11. Exact-ranked scalar GeometricMG](#11-exact-ranked-scalar-geometricmg)
- [12. Full-tensor elliptic: matrix-free Krylov (BiCGStab)](#12-full-tensor-elliptic-matrix-free-krylov-bicgstab)
- [13. Condensed implicit Program authoring](#13-condensed-implicit-program-authoring)
- [14. Embedded boundary: Shortley-Weller cut-cell](#14-embedded-boundary-shortley-weller-cut-cell)
- [15. Disc domain: mask, masked transport, cut-cell transport](#15-disc-domain-mask-masked-transport-cut-cell-transport)
- [16. Polar geometry: transport and Poisson on a ring (r, theta)](#16-polar-geometry-transport-and-poisson-on-a-ring-r-theta)
- [17. AMR: Program-owned subcycling + conservative reflux](#17-amr-program-owned-subcycling--conservative-reflux)
- [18. Multi-patch AMR: coverage-aware reflux, MPI-distributed](#18-multi-patch-amr-coverage-aware-reflux-mpi-distributed)
- [19. Berger-Rigoutsos clustering and regrid](#19-berger-rigoutsos-clustering-and-regrid)
- [20. Distributed mesh: global BoxArray, halos, load balancing](#20-distributed-mesh-global-boxarray-halos-load-balancing)
- [21. Extensible aux channel](#21-extensible-aux-channel)
- [22. Runtime composition and multi-species system](#22-runtime-composition-and-multi-species-system)
- [23. Symbolic DSL and authenticated native components](#23-symbolic-dsl-and-authenticated-native-components)
- [24. The dispatch seam (Kokkos: Serial / OpenMP / Cuda / MPI)](#24-the-dispatch-seam-kokkos-serial--openmp--cuda--mpi)
- [25. Capabilities to qualify (present but limited, or off master)](#25-capabilities-to-qualify-present-but-limited-or-off-master)
- [Which scheme or solver when](#which-scheme-or-solver-when)
- [References](#references)

---

## Model equations

The core solves, on an adaptive Cartesian mesh (and, optionally, on a polar ring or an immersed disc
subdomain), the generic form

$$\partial_t U + \mathrm{div} F(U, P) = S(U, P) \qquad \text{(hyperbolique, par bloc)}$$

$$\mathrm{div}(\varepsilon\,\nabla \phi) - \kappa\,\phi = f(U) \qquad \text{(elliptique, partage)}$$

The hyperbolic part `U` and field operators couple through an owner-qualified provider pack `P`.
Every consumer declares exact `ComponentKey` dependencies and receives a compact local slot plan;
the core reserves no physical `phi`, gradient, magnetic, or temperature indices. A model is a
composition `CompositeModel<Transport, Source, Elliptic>`; the coupling enters through the flux or
source provider plan under the same spatial operator. The reconstruction, flux and
source bricks are reused across geometries (Cartesian, polar, cut-cell); only the metrics and the
divergences change. The spatial discretization is finite volume (section 1); the time advance is a
method of lines (section 4); the elliptic part is solved at each step (sections 9 to 13) and re-read by
accepted provider generation.


---

## 1. Finite volumes: first-order Godunov

**Intuition.** The profile in each cell is replaced by its average. At each interface,
two averages meet: a local Riemann problem. The numerical flux solves it
(approximately) and the conservative update transports matter from one cell to the next.

**Formula / discretization.** Integration of the conservation law $\partial_t U + \mathrm{div}\,F(U,\mathrm{aux}) = S(U,\mathrm{aux})$
over the cell $(i,j)$ and the step $\Delta t$:

$$U_{ij}^{n+1} = U_{ij}^n - \frac{\Delta t}{\Delta x}\big(\hat F_{i+1/2,j} - \hat F_{i-1/2,j}\big)
                             - \frac{\Delta t}{\Delta y}\big(\hat G_{i,j+1/2} - \hat G_{i,j-1/2}\big)
                             + \Delta t\, S_{ij}$$

The conservative form (face-flux difference) guarantees exact discrete conservation: what
cell $i$ loses at its right face, cell $i+1$ gains at its left face. This is the
property on which all of AMR depends (reflux corrects exactly these face fluxes). The core does not
do the update itself: it assembles the residual of the method of lines

$$R_{\boldsymbol i} = -\,\mathrm{div}\,\hat F + S
 = S_{\boldsymbol i} - \sum_{a=0}^{D-1}
   \frac{\hat F^{(a)}_{\boldsymbol i+\frac12\boldsymbol e_a}
        -\hat F^{(a)}_{\boldsymbol i-\frac12\boldsymbol e_a}}{\Delta x_a},$$

and the time integrator applies $U^{n+1} = U^n + \Delta t\, R$ (Euler) or an SSPRK combination.
The face flux $\hat F_{i+1/2}$ is evaluated from the reconstructed states on either side:
$\hat F^{(a)} = \texttt{nflux}(\texttt{recon}_L, P_L, \texttt{recon}_R, P_R, a)$.
Cell-centered provider values are loaded from the consumer's compact plan on each side; they are not
assigned a core physical meaning or a fixed global slot. An optional parabolic term
($+\nu\,\mathrm{Lap}\,U$, guarded by the `DiffusiveModel` concept) is added with the exact-ranked
centered stencil in
`assemble_rhs`, or as a Fickian face flux $-\nu\,\nabla U$ in `compute_face_fluxes` (cf. section 8).

```
function assemble_rhs(model, U, providers, geom, R, recon_prim):
    for box li in U:                                          # seam for_each_cell : Kokkos (Serial / OpenMP / Cuda)
        for each valid Index<Dim> index:
            P = bind_provider_values(consumer_plan, providers, li, index)
            for axis in [0, Dim):
                left_state, right_state = reconstruct(U, index, axis, recon_prim)
                left  = Trace(left_state,  provider_values_at(index - e_axis))
                right = Trace(right_state, P)
                face  = FaceContext(axis=axis, oriented_measure=metric.face(index, axis))
                accumulate_divergence(nflux(physical_flux, left, right, face), face)
            S = model.source(load_state(U, index), P)
            for c in 0 .. n_vars-1:
                R(index,c) = S[c] - accumulated_divergence[c]
            if DiffusiveModel(model):                          # if constexpr -> zero codegen sinon
                R(index,c) += nu * sum_axis(centered_second_difference(U, axis))
```

**Code.** [`include/pops/numerics/spatial_operator.hpp`](../include/pops/numerics/spatial_operator.hpp):
`assemble_rhs<Limiter, NumericalFlux>` computes directly $R = -\mathrm{div}\,\hat F + S$, going through
the named device functor `detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>` (functor
rather than extended lambda: reliable device emission under nvcc from a generated native TU,
installed through the private block-loader seam). `compute_face_fluxes<Limiter, NumericalFlux>` writes the face fluxes `Fx, Fy`
(via `detail::FaceFluxXKernel` / `FaceFluxYKernel`) before the divergence: this is what the AMR reflux
needs. Same `reconstruct` and same numerical flux as `assemble_rhs`, so
$R = S - (\texttt{Fx}(i{+}1)-\texttt{Fx}(i))/\Delta x - (\texttt{Fy}(j{+}1)-\texttt{Fy}(j))/\Delta y$
gives back exactly the residual. The loop goes through the `for_each_cell` seam. An opt-in variant
`assemble_rhs_masked` (functor `AssembleRhsMaskedKernel`) restricts transport to an active
subdomain: zero normal flux on the faces touching a masked cell (conservative FV wall), zero residual
on the inactive cells. The global CFL step is read by `max_wave_speed_mf` (reduction over the local
boxes then MPI `all_reduce_max`, otherwise each rank would pick a different `dt` and diverge).

`PhysicalFlux`, `NumericalFlux` and the spatial operator are distinct contracts. The numerical
policy receives only two typed traces and a `FaceContext`; it cannot inspect a runtime model, mesh,
global auxiliary slot, or provider outside its resolved pack. It returns `FluxDensity` and a
`StabilityBound` with explicit units/convention. `apply_face_measure` is the unique density-to-
`IntegratedFaceFlux` operation, so Cartesian area, polar radius and embedded-boundary aperture are
applied exactly once by the spatial layer. A fallible evaluation maps explicitly to retry, reject or
abort transaction actions.

Primitive face reconstruction is a fallible numerical operation, not an unchecked model callback.
Every conservative-to-primitive stencil sample is evaluated through
`PreparedVariableRecovery` and returned as a `ReconstructedFaceState` carrying both the candidate
and its `RecoveryReport`. Cartesian, cached-HLL, masked, polar and embedded-boundary kernels consume
that report before calling the numerical flux. A refused candidate therefore writes only finite
transactional scratch, joins the same device/MPI failure reduction as a fallible flux, and cannot be
published. The type-erased report preserves the selected and last-attempted method kinds in addition
to their chain indices; diagnostics can therefore name the actual closed-form, nonlinear, bracketed,
repair, or custom route without reconstructing policy from an erased plan. The pointwise route is
fixed-size, `POPS_HD`, allocation-free and callback-free.

**Constraints / remarks.** CFL condition: $\Delta t \le C\,\dfrac{\min_d\Delta x_d}{\max|\lambda|}$,
where $\lambda$ is the local wave speed and $C \le 1$ at order 1; `max_wave_speed_mf` provides
$\max|\lambda|$. A model without transport ($\max|\lambda| = 0$) does not constrain the step
(`max_wave_speed_mf` returns 0). The operator writes only `R` (it touches neither `U` nor `aux`, no
ghost fill). Validation: `test_spatial_discretisation` (the reconstruction x flux pair is a
named type assembled by `assemble_rhs`) and `test_prepared_numerics_gate` (exact-ranked stability
and admissibility contracts). The Cartesian/polar invariant is checked bit-for-bit (the polar operator does not touch
this path). The end-to-end validations (diocotron, Euler-Poisson) live on the `adc_cases` side.

## 2. Numerical fluxes: Rusanov, HLL, HLLC, Roe

**Intuition.** Four levels of fidelity for the face Riemann problem, ordered by the number of waves
they resolve. Rusanov lays down a single diffusion bump (the most robust, the most diffusive); HLL
estimates two signal speeds and keeps a single star region; HLLC adds the contact wave (the
passive density discontinuity); Roe linearizes the system by the Roe average and solves
the linearized Riemann problem exactly. Each flux is a stateless policy (`POPS_HD` functor,
device-callable, no virtual) with contract
$\texttt{operator()}(m, U_L, A_L, U_R, A_R, \texttt{dir}) \to \texttt{Model::State}$, chosen by
template alongside the reconstruction limiter.

**Formula / discretization.** Rusanov (local Lax-Friedrichs), component by component (scalar
upwind, without coupling):

$$\hat F_{i+1/2} = \tfrac12\big(F(U_L)+F(U_R)\big) - \tfrac12\,\alpha\,(U_R - U_L),
\qquad \alpha = \max\big(s_L(U_L), s_R(U_R)\big),$$

where $s_{L,R}$ (`max_wave_speed`) of each state; Rusanov requires only this member, so it
applies to any base `PhysicalModel`. HLL uses the Davis estimates
$s_L = \min(s_L^{gauche}, s_L^{droit})$, $s_R = \max(s_R^{gauche}, s_R^{droit})$ via `hll_speeds`, and
falls back to the upwind flux in the supersonic regime:

$$\hat F^{HLL} = \begin{cases}
F(U_L) & s_L \ge 0 \\[2pt]
\dfrac{s_R F(U_L) - s_L F(U_R) + s_L s_R (U_R - U_L)}{s_R - s_L} & s_L < 0 < s_R \\[6pt]
F(U_R) & s_R \le 0
\end{cases}$$

HLLC restores the contact wave $S_*$ in the middle (Toro speed eq. 10.37) and reconstructs the
star states $U_L^*, U_R^*$:

$$S_* = \frac{p_R - p_L + \rho_L u_{nL}(s_L - u_{nL}) - \rho_R u_{nR}(s_R - u_{nR})}
             {\rho_L(s_L - u_{nL}) - \rho_R(s_R - u_{nR})},
\qquad \hat F^{HLLC} = F_K + s_K\,(U_K^* - U_K),\ K \in \{L, R\},$$

with $u_n$ the normal velocity and the factor $\rho_K (s_K - u_{nK}) / (s_K - S_*)$ for the star states.
Roe linearizes the system by the $\sqrt{\rho}$-weighted average:

$$\hat F^{Roe} = \tfrac12\big(F_L + F_R\big) - \tfrac12 \sum_k |\tilde\lambda_k|\,\alpha_k\,r_k,$$

waves $\{u_n - c,\ u_n,\ u_n,\ u_n + c\}$ with celerity $c$ deduced from the Roe enthalpy $H$, and
Harten's entropy fix on the acoustic waves to avoid the non-entropic shock (sonic glitch). The
entropy policy and its width belong to the Roe physical provider; the generic numerical flux never
assumes an ideal-gas layout.

```
function HLLC(provider, left, right, face):
    sL, sR = provider.wave_bounds(left, right, face)
    FL, FR = provider.flux(left, face), provider.flux(right, face)
    if sL >= 0: return FL                               # supersonique a droite -> flux amont
    if sR <= 0: return FR
    sStar = provider.contact_speed(left, right, sL, sR, face)
    side = left if sStar >= 0 else right
    speed = sL if sStar >= 0 else sR
    UStar = provider.star_state(side, speed, sStar, face)
    return provider.flux(side, face) + speed*(UStar - side.state)
```

**Code.** Stateless policies in
[`include/pops/numerics/fv/numerical_flux.hpp`](../include/pops/numerics/fv/numerical_flux.hpp): `RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux` (all `POPS_HD`). `RusanovFlux` loops component by component with
`m.max_wave_speed`; `HLLFlux`/`HLLCFlux` share the free function `hll_speeds` (Davis estimates,
requires `m.wave_speeds`). `HLLCFlux` requires `HasHLLCStructure` (`pressure`, `contact_speed`,
`hllc_star_state`) and `RoeFlux` requires `HasRoeDissipation` (`roe_dissipation`). Euler conforms
through those same capabilities. A missing capability is rejected during route resolution; there
is no component-count inference and no implicit HLL/Rusanov substitution. The
four built-ins return the common device-copyable `FluxEvaluation`. Built-in rejection reasons use
the typed `RiemannFailureCause` vocabulary before device/MPI reduction. In particular, Roe rejects
a non-finite dissipation or final candidate flux, while HLLC attributes non-finite physical flux,
pressure, contact speed, star state, and final candidate flux separately. Neither policy publishes a
successful NaN result; the runtime rolls the owning step transaction back without selecting another
solver. Every production result also carries typed requested, used and last-attempted solver
identities plus the attempt count. The typed public
`riemann.Recovery(primary=riemann.Roe(), fallbacks=(riemann.HLL(), riemann.Rusanov()))`
descriptor lowers exactly to
`PreparedRiemannRecoveryPolicy<RoeFlux, HLLFlux, RusanovFlux, RejectRiemannRecovery>` on Cartesian
Uniform and AMR routes. Other orders, duplicate candidates, candidate options, external descriptors,
and untyped values are refused during authoring; annular polar geometry is explicitly unavailable.
Only `kReject` advances to the next candidate, and the first recovery cause remains observable when
a fallback succeeds. The policy is an empty, trivially-copyable template value instantiated directly
in the face kernel: no per-face allocation, string dispatch, callback, exception or host round trip
is introduced.
The compatibility function `rusanov_flux` (in `spatial_operator.hpp`) delegates to
`RusanovFlux{}` for serial references. The flux is passed by template:
`compute_face_fluxes<Limiter, NumericalFlux, Model>` and
`assemble_rhs<Limiter, NumericalFlux, Model>` are templated on the flux policy, chosen independently
of the limiter. The `SourceFreeModel` adapter (explicit IMEX half-step) forwards
`pressure`, `wave_speeds`, and the optional HLLC/Roe structural hooks only when the wrapped model
exposes them (`requires` clauses), so the explicit half-step keeps the selected Riemann provider.
A moment hierarchy (no fluid roles, no primitive `p`) can also
drive a dense Roe-type dissipation via the DSL emitter `m.roe_from_jacobian()` (section 23): `|A|` is applied by
`pops::roe_abs_apply`
([`include/pops/numerics/linalg/dense_eig.hpp`](../include/pops/numerics/linalg/dense_eig.hpp)) behind a real-spectrum
gate. A real singular Jacobian uses the native zero-mode projector. A complex or non-converged
spectrum is rejected; the provider never substitutes another Riemann solver. Passing the typed
`entropy_fix=riemann.Harten(delta)` policy applies the Harten spectral function directly to the
dense Jacobian; `riemann.NoEntropyFix()` (the default for `roe_from_jacobian`) selects the matrix
absolute value. Role-generated Roe keeps its historical `riemann.Harten(0.1)` default and also
accepts `riemann.NoEntropyFix()` explicitly. Bare entropy scalars are rejected during authoring.
The detached artifact records `fluid_roles_v1`, `direct_action_v1`, or `flux_jacobian_v1` together
with the exact canonical entropy option. Runtime availability and inspection consume that evidence;
they never reconstruct a provider from `has_roe=True`. This provider
evaluates the flux Jacobian at the arithmetic midpoint $(U_L+U_R)/2$. It is therefore a Roe-type
linearization for a general nonlinear flux, not a claim that the resulting matrix satisfies the exact
Roe secant identity $F_R-F_L=A(U_R-U_L)$.

The related `m.wave_speeds_from_jacobian()` provider obtains HLL bounds from the extrema of each
authored dense Jacobian block. It accepts a state larger than 16 components only when `blocks=` gives
a certified block-triangular partition whose individual matrices contain at most 16 components; the
unpartitioned Roe-type provider applies to the full state and therefore accepts at most 16 components.
Both limits are checked during Python authoring, before code generation. For either provider, a complex
or non-converged dense spectrum is an invalid numerical result: the native step rejects it before
publishing state and never replaces it with a Gershgorin bound or a Rusanov flux. The default
`im_tol=None` accepts only the native roundoff floor, $64\,\epsilon$ relative to the spectral scale;
this prevents a repeated real root from being rejected solely by floating-point QR noise.
`im_tol=0` requests an exact-zero predicate. A positive explicit value relaxes the classification
relative to the spectral scale and is an opt-in numerical tolerance. When a complete partition is
structurally proven block triangular, the Roe provider certifies those diagonal blocks and still
applies its matrix function to the full Jacobian.

**Constraints / remarks.** `RusanovFlux` is the only flux compatible with the minimal `PhysicalModel`
(it reads only `max_wave_speed`): it is the robust default for scalar transport, at the cost of an
increased diffusion ($\alpha$ upper bound). `HLLFlux` still smooths the contact discontinuity (a
single star region). `HLLCFlux`/`RoeFlux` never inspect a concrete state layout: all physical
structure comes from `HasHLLCStructure` / `HasRoeDissipation`. A model without the required
capability cannot resolve that provider.
HLLC on a vacuum state (zero density) divides by zero in the star factor and
needs an upstream safeguard. Roe uses `std::sqrt` for the $\sqrt{\rho}$ average (device-clean
under Kokkos/nvcc); its key property $F_R - F_L = \tilde A\,(U_R - U_L)$ gives the exact upwind flux in
the supersonic regime, and the Harten fix avoids the non-physical expansion at the sonic point.
Validation: `test_roe_flux` (consistency $\hat F(U,U) = F(U)$, exact resolution of a linearized Riemann,
`eigenvalues()` of Euler). The `aux` coupling enters through the flux (`F` reads `aux`) or through the
source, under the same spatial operator; the same flux policies serve in Cartesian, polar and cut-cell EB.


---

## 3. MUSCL reconstruction (order 2) and WENO5-Z (order 5)

**Intuition.** First-order Godunov (section 1) replaces the profile of each cell by its average, which
is very diffusive. MUSCL reconstructs a linear profile per cell from a limited slope, then
evaluates the numerical flux on the values reconstructed at the faces; the limiter clips the slope near
extrema to stay TVD (no spurious oscillation). WENO5-Z reaches order 5 in smooth regions via a
nonlinear average of three order-3 reconstructions, without an explicit limiter, by discarding the stencil
that crosses a sharp front (the ring edge).

**Formula / discretization.** A reconstruction policy is pointwise: it takes the two non-centered finite
differences around cell $i$,

$$a = U_i - U_{i-1} \quad (\text{difference arriere}),\qquad b = U_{i+1} - U_i \quad (\text{difference avant}),$$

and returns a limited slope $\sigma_i = \mathrm{lim}(a,b)$. The three MUSCL limiters are:

$$\mathrm{minmod}(a,b) = \begin{cases} \mathrm{sgn}(a)\,\min(|a|,|b|) & ab>0\\ 0 & ab\le 0\end{cases},
\qquad
\mathrm{vanleer}(a,b) = \begin{cases} \dfrac{2ab}{a+b} & ab>0\\ 0 & ab\le 0\end{cases},$$

and $\mathrm{NoSlope}(a,b)=0$ (order 1, piecewise constant). van Leer is the harmonic mean of the
two differences: less dissipative at smooth extrema than minmod (which falls back to local order 1 on a
peak). The reconstructed states at the faces of interface $i+1/2$ are then

$$U_L = U_i + \tfrac12\,\sigma_i,\qquad U_R = U_{i+1} - \tfrac12\,\sigma_{i+1},$$

passed to the numerical flux $\hat F(U_L,U_R)$. MUSCL requires 2 ghosts (slope at $i\pm 1$).

WENO5-Z reconstructs the value at the face between $v_0$ and $v_{+1}$ from the 5-point stencil
$(v_{-2},v_{-1},v_0,v_{+1},v_{+2})$. Three order-3 reconstructions:

$$q_0 = \tfrac{2v_{-2}-7v_{-1}+11v_0}{6},\quad
  q_1 = \tfrac{-v_{-1}+5v_0+2v_{+1}}{6},\quad
  q_2 = \tfrac{2v_0+5v_{+1}-v_{+2}}{6},$$

the Jiang-Shu smoothness indicators:

$$\beta_0 = \tfrac{13}{12}(v_{-2}-2v_{-1}+v_0)^2 + \tfrac14(v_{-2}-4v_{-1}+3v_0)^2,$$
$$\beta_1 = \tfrac{13}{12}(v_{-1}-2v_0+v_{+1})^2 + \tfrac14(v_{-1}-v_{+1})^2,$$
$$\beta_2 = \tfrac{13}{12}(v_0-2v_{+1}+v_{+2})^2 + \tfrac14(3v_0-4v_{+1}+v_{+2})^2,$$

and the WENO-Z weights (Borges 2008), with $\tau_5 = |\beta_0-\beta_2|$ and optimal linear weights
$d_0=\tfrac{1}{10}, d_1=\tfrac{6}{10}, d_2=\tfrac{3}{10}$:

$$\alpha_k = d_k\Big(1 + \big(\tfrac{\tau_5}{\varepsilon+\beta_k}\big)^2\Big),
\qquad
\omega_k = \frac{\alpha_k}{\alpha_0+\alpha_1+\alpha_2},
\qquad
v_{i+1/2} = \sum_{k=0}^{2}\omega_k\, q_k .$$

In a smooth region $\tau_5 \to 0$ and $\omega_k \to d_k$: we recover order 5. The measure $\tau_5$ based on
$|\beta_0-\beta_2|$ makes WENO-Z less dissipative than classical Jiang-Shu, hence better at preserving the
growth rate of a smooth mode. 5-point stencil -> 3 ghosts. For the $-x$ face, we call the same
function with the reversed stencil $(v_{+2},v_{+1},v_0,v_{-1},v_{-2})$.

```
function reconstruct_muscl(U, i, lim):        # face i+1/2, pente limitee
    a   <- U[i]   - U[i-1]                     # difference arriere
    b   <- U[i+1] - U[i]                        # difference avant
    sig <- lim(a, b)                            # minmod / vanleer / NoSlope (=0)
    a2  <- U[i+1] - U[i]
    b2  <- U[i+2] - U[i+1]
    sig_r <- lim(a2, b2)
    U_L <- U[i]   + 0.5 * sig                   # etat gauche reconstruit
    U_R <- U[i+1] - 0.5 * sig_r                 # etat droit reconstruit
    return (U_L, U_R)

function minmod(a, b):
    if a*b <= 0: return 0
    fa <- |a| ; fb <- |b|                       # valeur absolue sans std::abs
    return a if fa < fb else b

function vanleer(a, b):
    ab <- a*b
    if ab <= 0: return 0
    return 2*ab / (a + b)                        # moyenne harmonique

function weno5z(vm2, vm1, v0, vp1, vp2):        # face entre v0 et vp1
    eps <- 1e-40
    q0  <- ( 2*vm2 - 7*vm1 + 11*v0) / 6          # 3 recon. d'ordre 3
    q1  <- (  -vm1 + 5*v0  +  2*vp1) / 6
    q2  <- ( 2*v0  + 5*vp1 -    vp2) / 6
    b0  <- 13/12*(vm2-2*vm1+v0)^2 + 1/4*(vm2-4*vm1+3*v0)^2   # indicateurs beta
    b1  <- 13/12*(vm1-2*v0+vp1)^2 + 1/4*(vm1-vp1)^2
    b2  <- 13/12*(v0-2*vp1+vp2)^2 + 1/4*(3*v0-4*vp1+vp2)^2
    tau5 <- |b0 - b2|                            # ternaire device-safe, pas std::abs
    a0  <- (1/10)*(1 + (tau5/(eps+b0))^2)        # poids WENO-Z non normalises
    a1  <- (6/10)*(1 + (tau5/(eps+b1))^2)
    a2  <- (3/10)*(1 + (tau5/(eps+b2))^2)
    inv <- 1 / (a0 + a1 + a2)
    return (a0*q0 + a1*q1 + a2*q2) * inv

# Cote operateur spatial (spatial_operator::reconstruct) :
#   n_ghost == 1 (NoSlope)  -> Godunov ordre 1, pas de lecture a +/-2
#   n_ghost == 2 (MUSCL)    -> reconstruct_muscl avec le limiteur
#   n_ghost >= 3 (Weno5)    -> weno5z(stencil direct) pour face +dir,
#                              weno5z(stencil renverse) pour face -dir
```

**Code.** Pointwise `Limiter` policies in
[`include/pops/numerics/fv/reconstruction.hpp`](../include/pops/numerics/fv/reconstruction.hpp): `NoSlope`
(`n_ghost = 1`, piecewise-constant face value), `Minmod`, `VanLeer`, `MC` and `Superbee`
(`n_ghost = 2`, `limited_slope(a,b)` returns the limited slope with device-safe scalar arithmetic), `Weno5`
(`n_ghost = 3`, a tag whose `operator()` is a no-op that just satisfies the `Limiter` concept). The
order-5 reconstruction lives in the free function `weno5z(vm2, vm1, v0, vp1, vp2)` of the same header:
it returns the value at the face between `v0` and `vp1`, and for the opposite face one passes it the
reversed stencil. All are `POPS_HD` (device-callable, static polymorphism: the limiter is a template
parameter of `assemble_rhs` / `compute_face_fluxes`, inlined on device). The mesh stencil access and the
routing by `n_ghost` are in `reconstruct` of `numerics/spatial_operator.hpp`; the policy itself
loops over no grid. The reconstruction can act on the conserved or primitive variables
(`rho, u, p`) depending on the block. Production kernels use the typed
`reconstruct_recovered`/`reconstruct_pp_recovered` entry points and consume their `RecoveryReport`
before any face flux; the value-only wrappers remain low-level compatibility helpers.

**Constraints / remarks.** The reconstruction does not change the hyperbolic stability condition: the
step stays bounded by the CFL of section 1, `dt <= C dx / max|lambda|`. Limits and pitfalls:
- `Minmod` is strictly TVD but falls back to local order 1 at extrema (it erases smooth peaks);
  for the Diocotron growth modes one prefers `VanLeer`, less dissipative at extrema.
- `MC` uses $\operatorname{minmod}((a+b)/2,2a,2b)$ and is a less diffusive TVD compromise;
  `Superbee` uses $\operatorname{maxmod}(\operatorname{minmod}(2a,b),
  \operatorname{minmod}(a,2b))$ and is the most compressive builtin MUSCL limiter. Their
  implementations avoid overflowing intermediate doubled slopes for finite inputs.
- `weno5z` is smooth (no branch on the sign: the $\beta_k$ and $\tau_5$ are squares so
  always $\ge 0$, and only $|\beta_0-\beta_2|$ goes through a ternary), which makes it fully
  device-callable; the floor `eps = 1e-40` avoids division by zero on a constant stencil.
- Reconstructing the conserved variable rather than the primitive changes the behavior at strong shocks
  (the reconstructed states can leave the admissible domain on the conserved side).
- The ghost cost drives the halo width to exchange: 1 (NoSlope), 2 (MUSCL), 3 (WENO5).

For a Python-authored model, finite primitive recovery can be strengthened with explicit physical
constraints after declaring the primitive layout:

```python
# after model.primitive_state(rho, u, v, p, conservative=(...))
model.recovery_admissibility(rho=rho > 0, p=p > 0)
```

Each keyword identifies the primitive component reported on failure; its value is a symbolic Boolean
expression over the primitive state.  Code generation emits a device-callable
`recovery_admissible(Prim, failing_component)` method.  `CompositeModel` forwards that optional
contract and `prepare_model_variable_recovery` installs it in the same ordered recovery plan as the
conversion method.  A finite candidate that violates a predicate is therefore not published: the
chain proceeds to its next declared method, or finishes with `inadmissible_candidate` when no method
remains.  Models that declare no policy retain the finite-only path and emit no extra method.

**Validation.** `test_weno_convergence` (the face reconstruction of a smooth function reaches order 5),
`test_primitive_recon` (conserved <-> primitive conversions and their use in the reconstruction),
`test_spatial_discretisation` (the reconstruction x numerical flux pair is a named type, exercised end
to end), and `test_weno_convergence` (MC/Superbee reference formulas, symmetry, homogeneity, TVD
bounds and finite extreme inputs in addition to WENO convergence), and
`test_variable_recovery_chain` (a model-declared physical predicate blocks publication
and preserves the typed failing component).


---

## 4. Time-integration bricks: SSPRK and object integrators

**Intuition.** Strong-Stability-Preserving Runge-Kutta: each stage is a convex combination
of explicit Eulers, so any stability property (TVD, positivity, bounds) held by a forward
Euler step under CFL is held by the whole scheme, at order 2 or 3. The `take_step`
objects below are stateless numerical bricks, not temporal drivers. In production,
the normalized `ProgramGraph` explicitly owns the stage sequence and
`System`/`AmrSystem` execute only that installed graph. A low-level object integrator can
therefore participate in production only through an explicitly authored typed Program
primitive and its registered lowering; no facade infers or falls back to an integrator
from a descriptor.

**Formula / discretization.** Method of lines: the space gives $\dot U = L(U)$ with
$L(U) = -\mathrm{div} F(U) + S(U)$, evaluated by $\texttt{rhs}(U, R) \Rightarrow R = L(U)$.
Forward Euler: $U^{n+1} = U^n + \Delta t\, L(U^n)$.

SSPRK2 (Shu-Osher, 2 stages, order 2, equivalent to Heun):

$$U^{(1)} = U^n + \Delta t\, L(U^n), \qquad U^{n+1} = \tfrac12 U^n + \tfrac12\big(U^{(1)} + \Delta t\, L(U^{(1)})\big).$$

SSPRK3 (Shu-Osher, 3 stages, order 3):

$$U^{(1)} = U^n + \Delta t\, L(U^n),$$
$$U^{(2)} = \tfrac34 U^n + \tfrac14\big(U^{(1)} + \Delta t\, L(U^{(1)})\big),$$
$$U^{n+1} = \tfrac13 U^n + \tfrac23\big(U^{(2)} + \Delta t\, L(U^{(2)})\big).$$

Both have SSP coefficient $C = 1$: the SSP condition is exactly the forward Euler CFL condition.
In $\texttt{MultiFab}$ operations the code uses only $\texttt{saxpy}(Y, a, X): Y \mathrel{+}= a\,X$ and
$\texttt{lincomb}(Y, a, X_1, b, X_2): Y \leftarrow a\,X_1 + b\,X_2$. The convex stage of SSPRK2 then
writes as an Euler update on the copy $U^{(1)}$ followed by
$\texttt{lincomb}(U, \tfrac12, U, \tfrac12, U^{(1)})$, algebraically identical to the convex form above.

```
function take_step_SSPRK2(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)   # scratch, aucun etat porte
    rhs(U, R)                                          # R = L(U^n)
    U1 = copy(U)
    saxpy(U1, dt, R)                                   # U1 = U^n + dt L(U^n)  (= U^(1))
    rhs(U1, R)                                         # R = L(U^(1))
    saxpy(U1, dt, R)                                   # U1 = U^(1) + dt L(U^(1))
    lincomb(U, 1/2, U, 1/2, U1)                        # U^{n+1} = 1/2 U^n + 1/2 U1

function take_step_SSPRK3(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)
    rhs(U, R);  U1 = copy(U);  saxpy(U1, dt, R)        # U^(1) = U^n + dt L(U^n)
    rhs(U1, R); U2 = copy(U1); saxpy(U2, dt, R)
    lincomb(U2, 3/4, U, 1/4, U2)                       # U^(2) = 3/4 U^n + 1/4 (U^(1)+dt L)
    rhs(U2, R); U3 = copy(U2); saxpy(U3, dt, R)
    lincomb(U, 1/3, U, 2/3, U3)                       # U^{n+1} = 1/3 U^n + 2/3 (U^(2)+dt L)

# tag -> metadonnees numeriques consommees par un lowering Program explicite
struct TimePolicy<Method, Treatment in {Explicit,Implicit,IMEX,Prescribed}, Substeps>=1, Stride>=1>
TimePolicyTraits<P>: extrait (Method, treatment, substeps, stride), defaut Explicit/1/1 sur un tag nu
ExplicitTime<M=SSPRK2,...> / ImplicitTime<...> / IMEXTime<...> / PrescribedTime  # alias de TimePolicy

# brique bas niveau : tout objet qui satisfait le concept
concept TimeStepper<I> = I.take_step(rhs, U, dt) compile
```

**Code.** Two low-level representations coexist, separating the mathematical scheme from
metadata used by an explicit Program lowering.
The tags [`include/pops/numerics/time/integrators/time_integrator.hpp`](../include/pops/numerics/time/integrators/time_integrator.hpp)
(`SSPRK2`, `SSPRK3`, `UserTimeIntegrator`) describe numerical metadata via a
`TimePolicy<Method, TimeTreatment, Substeps, Stride>`; `TimePolicyTraits` reads these fields (and accepts
a bare tag, then treated as `Explicit` with a single step). The aliases `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime` set the `TimeTreatment`; none of these descriptors schedules
an advance, an implicit solve, or a hold by itself. The object integrators
[`include/pops/numerics/time/integrators/time_steppers.hpp`](../include/pops/numerics/time/integrators/time_steppers.hpp)
(`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`) carry the method: each exposes
`take_step(rhs, U, dt)` and allocates its scratch (`R`, stages `U1`/`U2`/`U3`) only from the layout of
`U`, with no persistent state. The integrator sees only `rhs(U_stage, R)` (the method-of-lines arrow)
and the `saxpy`/`lincomb` operations of [`include/pops/mesh/storage/mf_arith.hpp`](../include/pops/mesh/storage/mf_arith.hpp):
it is agnostic of the model and of the discretization. The `TimeStepper` concept formalizes the contract, so
the retained low-level bricks and tests can accept a custom object without making it a
second production authoring route.

**Constraints / remarks.** SSP with coefficient 1: strong stability is guaranteed only as long as
$\Delta t$ respects the forward Euler CFL ($C = 1$, no margin gained on the step compared to the
order-1 scheme). In a coupled system, the order of the elliptic solve caps the global order: a Poisson
solved once per step limits the field to order 1, whatever SSPRK is chosen on the hyperbolic side
(see splitting, section 6). `Substeps` and `Stride` remain metadata used when
computing declared stability bounds. They do not create a scheduler: production subcycling,
holds, or catch-up must appear as explicit Program composition (section 7).
The `SSPRK2Step`/`SSPRK3Step` objects reproduce bit-for-bit the
old static-driver inline copies (deduplication). The installed `test_program_runtime` and native
loader suites validate the typed Program cadence and integrator routes directly.


---

## 5. Stiff sources: asymptotic-preserving IMEX and partial IMEX

**Intuition.** A stiff source (fast relaxation, Lorentz force, Debye screening `lambda_D -> 0`)
forces the explicit scheme to a `dt` of the same order as the stiffness, hence impractical. IMEX treats
transport explicitly and the stiff source implicitly (stable at fixed `dt`). The
asymptotic-preserving (AP) property guarantees that, when the small parameter `eps` (= `lambda_D^2`,
`1/omega_c`, ...) tends to 0, the scheme stays consistent and stable at fixed `dt` and captures the
limit dynamics (equilibrium, quasi-neutrality) without resolving the stiff scale.

**Formula / discretization.** On `dU/dt = T(U) + S(U)` where `S` carries the stiff part, an IMEX
Euler step (forward-backward) treats `T` explicitly and `S` implicitly:

$$U^{n+1} = U^n + \Delta t\,T(U^n) + \Delta t\,S(U^{n+1}).$$

We decompose it into two in-place operators. First the explicit transport produces the known member
$\tilde U = U^n + \Delta t\,T(U^n)$, then the implicit step solves $W = \tilde U + \Delta t\,S(W)$.
When `S` is a linear relaxation `S(U) = -(1/eps)(U - U_eq)`, the solve is analytic and
unconditionally stable; the limit `eps -> 0` gives `W -> U_eq` (equilibrium manifold `S(U)=0`)
without the constraint `dt < eps`. The scalar implicit step (per cell) is solved by the Newton residual

$$F(W) = W - \tilde U - \Delta t\,S(W) = 0,\qquad J = I - \Delta t\,\frac{\partial S}{\partial W},$$

exact in one iteration if `S` is linear in `U`, quadratic otherwise. One prepared provider accepts
a finite-difference, analytic, or automatic-differentiation Jacobian provider; the implicit-source
adapter selects the model's analytic `source_jacobian` when available and finite differences otherwise.

**Partial IMEX.** When only a subset of the variables is stiff, we integrate implicitly only
those components. The solve becomes a forward-backward Euler per component: the explicit components
advance by forward Euler at the input state, `W_e = U^n_e + dt S_e(U^n)`. The prepared residual keeps
the complete `N x N` state space: explicit rows are identity constraints fixing those known values,
while implicit rows carry the backward-Euler residual. The partitioning comes either from the model
(trait `is_implicit(c)`), or from a mask carried by the block (priority over the model default), which
allows reusing the same model with different treatments depending on the block. This
partition is read only when an explicitly authored typed implicit Program primitive is
executed. A trait, mask, `IMEXTime`, or `time="imex"` descriptor alone never schedules
Newton or an IMEX advance.

```
function imex_euler_step(U, dt, Texpl, Simpl):
    Texpl(U, dt)            # explicite en place : U <- U^n + dt*T(U^n) (membre connu)
    Simpl(U, dt)            # implicite en place : resout U <- W tel que W = U + dt*S(W)

# preparation locale, N = Model::n_vars
function prepare_implicit_source_problem(model, Un, aux, dt, controls, mask):
    S_in <- model.source(Un, aux)
    for c in 0..N-1:
        explicit_target[c] <- Un[c] + dt*S_in[c]
    residual(W)[c] <-
        W[c] - Un[c] - dt*model.source(W, aux)[c]  si c est implicite
        W[c] - explicit_target[c]                  sinon
    jacobian <- model.source_jacobian si disponible, differences finies sinon
    return PreparedLocalNonlinearProblem(
        residual, jacobian, domaine_admissible, controls, echelles)

function solve_prepared_local_nonlinear(problem, guess):
    # tableaux fixes N sur la pile, aucun callback/type-erasure/allocation dans la boucle cellule
    # residu mis a l'echelle, budget iterations/evaluations, pivot partiel, safeguard explicite
    return LocalNonlinearCellResult(candidate, status, iterations, evaluations, diagnostics)

# stepper de bloc : candidat transactionnel + publication seulement apres succes collectif
function backward_euler_source(model, aux, U, dt, controls, mask):
    candidate <- MultiFab(layout(U))
    statistics <- MultiFab(layout(U))
    for chaque fab local de U:
        for_each_cell(box, PreparedImplicitSourceKernel:
            Un <- load_state(U, i, j);  a <- load_aux(aux, i, j)
            problem <- prepare_implicit_source_problem(model, Un, a, dt, controls, mask)
            result <- solve_prepared_local_nonlinear(problem, Un)
            candidate(i,j,:) <- result.candidate
            statistics(i,j,:) <- result.status et diagnostics)
    report <- reduction MPI collective des statistics
    outcome <- SolveOutcome(report, candidate)
    # le consumer choisit collectivement accept/reject/fail
    # seul accept publie candidate dans U et les diagnostics persistants
    return outcome
```

**Code.** [`include/pops/numerics/time/schemes/imex.hpp`](../include/pops/numerics/time/schemes/imex.hpp):
`imex_euler_step(U, dt, Texpl, Simpl)` chains the in-place explicit transport then the in-place implicit
source solve (two callables `TransportStep` / `ImplicitSourceSolve`). The provider in
[`include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp`](../include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp)
owns the only cell-local nonlinear algorithm: immutable concrete functors, scaled stopping controls,
finite-difference/analytic/AD Jacobian providers, partial-pivot factorization without inverse,
safeguards, budgets and explicit statuses. The adapter in
[`include/pops/numerics/time/integrators/implicit_stepper.hpp`](../include/pops/numerics/time/integrators/implicit_stepper.hpp)
forms the forward-backward Euler residual, runs the named device kernel into a candidate field,
constructs the common collective `SolveReport`, and publishes only a globally solved candidate. The
implicit/explicit partitioning
goes through the `PartiallyImplicitModel` concept (trait `M::is_implicit(c)`), `model_is_implicit<Model>`
(default: everything implicit when the trait is absent), the POD carrier `ImplicitMask<N>` (`active`, `flag[N]`,
carried by the block, passed by value on the device) and `is_implicit_component<Model, N>` (an active mask
with priority over the model default). `ImplicitSourceStepper` models the concept
`ImplicitBlockStepper` for isolated numerical tests. Production `System` and `AmrSystem` do not
schedule that helper: their normalized `ProgramGraph` must place a typed implicit primitive
explicitly, and an unavailable lowering fails closed.

**Constraints / remarks.** The implicit step is unconditionally stable for a linear relaxation
(where a plain Picard fixed point would diverge as soon as `dt * stiffness > 1`, precisely the
stiff regime); an affine source is solved in one Newton iteration up to rounding. Local quadratic
convergence requires the usual smoothness assumptions and an exact analytic/AD Jacobian; finite
differences and safeguards retain the same outcome contract without promising that rate.
The default finite-difference relative step is `1e-7`. `LocalNewton` defaults to an absolute
threshold of `1e-12` and a budget of 20 iterations; the native implicit-source adapter maps its
central runtime policy (`abs_tol=1e-12`, `rel_tol=1e-10`, 25 iterations) into the same prepared
controls. Singular pivots, exhausted budgets, NaN/Inf, inadmissible candidates, safeguard failures
and unsupported Jacobian capabilities remain distinct outcomes. Collective priority is independent
of status numbering, so a fatal cell or MPI-rank failure cannot be hidden by a recoverable rejection.
Once that priority is known, the first failing location is selected by exact staged integer
collectives (`min(j)`, then `min(i)`, then `min(component)` at that cell). Coordinates are never
packed into a floating-point mantissa, so negative and large global `Box2D` indices keep the same
diagnostic and MPI ordering on double- and single-precision builds. These extra collectives execute
only on the failure path.
There is no warning-only or unchecked publication policy.
Limits: `imex_euler_step` is first order in time (forward-backward Euler); the AP covers the relaxation
limit, not the condensation of the potential-velocity-Lorentz couplings at high `omega_c`, which is the
domain of Schur condensation (section 13). Inactive mask and a model without the `is_implicit` trait:
when the low-level implicit solver is explicitly invoked, everything is implicit
(full backward-Euler), preserving its historical numerical behavior. This default is a
component-selection rule inside that solver, not temporal authorization. An IMEX Program
must place both the transport and the typed implicit primitive explicitly; the spatial
runtime does not infer that split. Validation:
`test_imex_ap` (AP property on a stiff linear relaxation source),
`test_ap_limit` (quantified AP limit, stiffness sweep over 8 decades at fixed `dt`),
`test_implicit_source_nd` (native implicit kernel),
`test_amr_synthetic_program_loader_transaction` (synthetic source-built loader ABI/hash/budget and
transaction artifact), and `test_amr_multiblock_implicit_transaction` (atomic two-block implicit
solve/retry/commit and no hidden integrator) are C++ kernel/loader/transaction proofs. They do not
qualify Program tableau or temporal composition. The Python-authored, compiled refined-AMR route is
proved by
`test_amr_newton_full.py::test_nonlinear_local_imex_executes_on_the_refined_amr_program` and
`test_amr_newton_full.py::test_nonlinear_local_imex_failrun_rolls_back_refined_amr_attempt`
(temporal composition and rollback). `test_newton_robustness` plus
`test_mpi_field_plan_consensus` prove exact first-failure selection across
large signed indices, including fatal-over-recoverable precedence between ranks).


---

## 6. Operator splitting: Lie and Strang

**Intuition.** When the RHS is a sum of operators with different behavior (transport + stiff
source + cyclotron rotation), we apply them in sequence rather than simultaneously: each
sub-operator keeps its own integrator, hence its own stiffness, without contaminating the other. Lie
(Godunov, order 1) chains the flows; Strang (order 2) symmetrizes the sequence around the central flow
to cancel the dominant error term.

**Formula / discretization.** We decompose

$$\frac{\mathrm{d}U}{\mathrm{d}t} = T(U) + S(U)$$

denoting $\Phi^T_{\tau}$ and $\Phi^S_{\tau}$ the exact flows (or approximate to the desired order) of
$\dot U = T(U)$ and $\dot U = S(U)$ over an interval $\tau$. Lie splitting applies one then
the other on the full step:

$$U^{n+1} = \Phi^S_{\Delta t}\big(\Phi^T_{\Delta t}(U^n)\big)$$

Strang splitting brackets the transport flow with two source half-steps:

$$U^{n+1} = \Phi^S_{\Delta t/2}\Big(\Phi^T_{\Delta t}\big(\Phi^S_{\Delta t/2}(U^n)\big)\Big)$$

The order reads off the Baker-Campbell-Hausdorff formula. The Lie composite flow equals
$\exp(\Delta t\,T)\exp(\Delta t\,S) = \exp\big(\Delta t (T+S) + \tfrac{\Delta t^2}{2}[T,S] + \dots\big)$:
the per-step error is $O(\Delta t^2)$, carried by the commutator $[T,S] = TS - ST$, hence global order 1.
The Strang symmetrization cancels the $\Delta t^2$ term: the per-step error drops to $O(\Delta t^3)$,
that is global order 2. Strang is order 2 as soon as each sub-integrator is itself; if $T$ and $S$
commute ($[T,S]=0$) the splitting is exact to all orders. The extra cost of Strang over Lie is a single
extra source half-step per macro-step (two $S(\Delta t/2)$ instead of one $S(\Delta t)$).

```
function lie_step(U, dt, T, S):
    # T, S : callables (MultiFab&, Real) -> void, avancent leur sous-systeme en place
    T(U, dt)                 # transport sur le pas plein
    S(U, dt)                 # source sur le pas plein
    # U contient maintenant U^{n+1}, ordre 1

function strang_step(U, dt, T, S):
    S(U, 0.5 * dt)           # demi-pas source
    T(U, dt)                 # pas plein transport (flot central)
    S(U, 0.5 * dt)           # demi-pas source symetrique
    # U contient maintenant U^{n+1}, ordre 2 si S et T sont chacun >= ordre 2
```

**Code.** The two generic bricks are in
[`include/pops/numerics/time/schemes/splitting.hpp`](../include/pops/numerics/time/schemes/splitting.hpp):
`lie_step(MultiFab& U, Real dt, TransportStep T, SourceStep S)` and
`strang_step(...)`. Both are templated on `TransportStep` / `SourceStep`: $T$ and $S$ are
callables `(MultiFab&, Real) -> void` that advance their subsystem in place, so the integrator is
agnostic of the physical content (in-house counterpart of `StrangSplitting` / `FractionalTime2OSplitting` of
muffin). The public production path calls `pops.lib.time.Lie(block[U], first=T, second=S)` or
`pops.lib.time.Strang(block[U], first=T, second=S)`; each factory returns an ordinary
`pops.Program`. No native stepper selects a named split scheme: every subflow, field evaluation and
endpoint stays visible in the Program IR.

**Constraints / remarks.** Strang gives order 2 only if each sub-step is itself at least
order 2: an order-1 $S$ or $T$ caps the splitting at order 1, whatever the symmetrization. In a
hyperbolic-elliptic couple, consistency requires re-solving the elliptic between the source half-steps:
otherwise the second half-step $S(\Delta t/2)$ reads a stale $\phi$ (the field of the previous half-step) and
order 2 falls (see the solve call count in the validation). The step $\Delta t$ stays subject to
the CFL of the transport $T$; the splitting does not relax this constraint, it only decouples the
stiffnesses so a stiff source can be treated implicitly (IMEX / Schur) without imposing its own
tiny $\Delta t$ on the transport. Validation: `test_splitting` measures the order of the C++
composition bricks on a non-commuting linear system, while the Program macro tests verify the authored
Lie/Strang endpoints and explicit field-solve placement.


---

## 7. Test-only multirate reference formulas

**Scope.** This section records historical formulas retained only by
`tests/cpp/support/reference_time_scheduler.hpp`. They are not installed PoPS code and cannot become
a production time engine. `System` and `AmrSystem` execute only their installed `ProgramGraph`;
production subcycling, holds, catch-up, and adaptive-step placement must be authored as typed Program
composition.

The hardware-dependent cutover proof is deliberately not a routine conformance test. The ADC-700
campaign under [`benchmarks/adc700/`](../benchmarks/adc700/) builds one AMR workload against the
pinned pre-cutover native revision and the Program-only candidate, executes both in paired ABBA
order on a real device, and emits a machine-readable candidate/pre-cutover throughput ratio. It
rejects CPU runs, incomplete device inventories, incomparable parameters, failed numerical
signatures, and a median ratio below `0.98`; no report from the campaign means no hardware
performance proof.

**Intuition.** Not all species of a coupled system require the same time step.
A stiff species (electrons) splits a macro-step into several substeps ($\text{substeps}$); a
slow species (under-resolved gas) is advanced only once every $M$ macro-steps (cadence, $\text{stride}$),
and then catches up $M$ steps in a single advance. In the retained low-level algorithm,
the two mechanisms are orthogonal and read from block metadata.

**Formula / discretization.** Each block can carry a
`TimePolicy<Method, Treatment, substeps, stride>` from which the low-level utility
extracts three integers (and the treatment, to skip prescribed blocks).
Let $\text{dt}$ be the macro-step, $n = \text{substeps}_b$, $m = \text{stride}_b$ for block $b$.

Cadence: block $b$ is held (hold) as long as macro-step $k$ verifies $(k+1) \bmod m \neq 0$, then
it catches up at the end of the window with an effective step

$$\Delta t^{\text{eff}}_b = m \, \text{dt}.$$

Over $M$ macro-steps, the block advances $M/m$ times by a step $m\,\text{dt}$: its total time stays $M\,\text{dt}$,
but it is solved only $M/m$ times (the coupling is loose for this block). Subcycling splits this
effective step into $n$ equal substeps

$$h = \frac{\Delta t^{\text{eff}}_b}{n} = \frac{m \, \text{dt}}{n}.$$

With $m = 1$ and $n = 1$ we recover identically the advance of a step $\text{dt}$ at each macro-step.

The formula $m\,\text{dt}$ is only the constant-step special case. For the whole-Program cadence of
the production facades, an adaptive window with accepted steps
$\text{dt}_0,\ldots,\text{dt}_{m-1}$ advances by

$$\Delta t^{\text{eff}} = \sum_{i=0}^{m-1}\text{dt}_i.$$

The runtime stores the window's exact accepted start, accumulated duration, and held-step count.
Those values are transactional and part of strict restart state; it never reconstructs the interval
as $m$ times the most recent step.

The macro-step can be chosen by the CFL via `step_cfl`. The stability condition bears on the
real substep $m\,\text{dt}/n \le \text{cfl}\, h_{\text{cell}} / w_b$, which gives per block

$$\text{dt}_b = \frac{\text{cfl} \; h_{\text{cell}} \; \text{substeps}_b}{\text{stride}_b \; w_b},
\qquad \text{dt} = \min_{b \,\text{evolutif}} \text{dt}_b,$$

where $h_{\text{cell}} = \min(dx, dy)$ in Cartesian, $\min(dr, r_{\min}\, d\theta)$ in polar (the
physical azimuthal step is minimal at the inner radius), and $w_b$ is the max wave speed of the block.

The test-only `ReferenceSystemDriver::step_adaptive` oracle fixes the macro-step on the fastest block,
$\Delta t = \text{cfl}\, h_{\text{cell}} / w_{\max}$, and assigns each block the runtime stride

$$m_b = \max\!\left(1,\left\lfloor\frac{w_{\max}}{w_b}\right\rfloor\right).$$

Block $b$ advances once every $m_b$ macro-steps by the effective step
$\Delta t^{\text{eff}}_b = m_b\,\Delta t$; aux is frozen on the macro-step (once-per-step coupling).
This formula is lowered by `hold_catchup_program` / `step_adaptive_program`: the Program cadence
owns the window `lcm(m_b)`, faster blocks subcycle `lcm/m_b` times, and a slow block catches up
once with `m_b dt`. `System.install_step_adaptive` and `AmrSystem.install_step_adaptive` install
that Program; `step_adaptive(cfl)` then advances the oracle macro-step
`dt = cfl h_cell / w_max`. `System` and `AmrSystem` never fall back to the test-only
`advance_subcycled` scheduler.
`Explicit(stride=M)` and `IMEX(stride=M)` now schedule: `add_equation` / `Case.block(time=...)`
record the descriptor and `install_equation_cadence` / `pops.resolve` emit the same
hold-then-catch-up Program. A user does not hand-author `Program.subcycle`.

```
function advance_subcycled(system, dt, macro_step, advance_block):
    for each block in system:                    # for_each_block, ordre stable
        if time_treatment(block) == Prescribed:
            continue                             # pilote par l'utilisateur, hors scheduler
        m = stride(block)
        if macro_step mod m != 0:
            continue                             # bloc lent : tenu ce macro-pas
        n = substeps(block)
        h = dt * m / n                           # pas effectif (catch-up) decoupe en n sous-pas
        for s in 0 .. n-1:
            advance_block(block, h, s, n)        # callable utilisateur, 1 sous-pas

# surcharge historique : macro_step = 0 -> stride toujours satisfait, tous les blocs avancent
function advance_subcycled(system, dt, advance_block):
    advance_subcycled(system, dt, 0, advance_block)

function step_cfl(cfl):                           # choix du macro-pas par CFL
    require installed whole-system Program
    evaluate_field_operator()                     # champs qualifies a l'instant courant
    h_cell = polar ? min(dr, r_min*dtheta) : min(dx, dy)
    dt = +inf
    for each block b, evolutif:
        w   = max(max_wave_speed(b.U), 1e-30)     # all_reduce_max sous MPI
        dt  = min(dt, cfl * h_cell * substeps_b / (stride_b * w))
    if dt not finite: dt = cfl * h_cell / 1e-30   # tous geles : pas degenere
    execute_installed_program(dt)                  # cadence et stages lus du ProgramGraph
    t += dt; macro_step += 1
    return dt

```

**Code.** The skeleton is
[`reference_time_scheduler.hpp`](../tests/cpp/support/reference_time_scheduler.hpp), function
`pops::test_support::advance_subcycled` (two overloads: with and without `macro_step`). It is
deliberately outside the installed headers and only the test oracle composes it into a complete
temporal driver. It reads `reference_block_substeps_v`, `reference_block_stride_v` and
`reference_block_time_treatment_v`, test-only aliases of `TimePolicyTraits` defined in
[`numerics/time/time_integrator.hpp`](../include/pops/numerics/time/integrators/time_integrator.hpp)
(`TimePolicy<Method, Treatment, substeps, stride>`, aliases `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime`). A `TimeTreatment::Prescribed` block is skipped (the guard
`!= Prescribed`). The production step choice lives in the exact-ranked
[`System<Dim>` runtime](../src/runtime/system/system.cpp): `step_cfl` computes the bound, then
dispatches the installed normalized graph. Its cadence service mechanically interprets the
graph-authored whole-Program cadence
$(k+1)\bmod m = 0$; it does not read a block `TimePolicy` or
construct an alternate schedule. During a due window it publishes the exact starting physical time
and public macro-step to every internal Program substep. The window is partitioned by explicit
endpoints (the final endpoint is exact), then committed atomically. `substeps=n` deliberately invokes
the complete authored closure $n$ times. Consequently, a typed schedule such as
`Every(AcceptedStep(clock), 1)` is evaluated once in each invocation at the same accepted-step
coordinate; the cadence wrapper performs no implicit node deduplication. A schedule that needs
hold/cache behavior must author that policy explicitly. The speed $w_b$ comes from `max_wave_speed_mf`
([`numerics/spatial_operator.hpp`](../include/pops/numerics/spatial_operator.hpp)), collective
`all_reduce_max` under MPI so that all ranks pick the same $\text{dt}$.

**Constraints / remarks.** `step_cfl` is substeps-aware since #121: the formula
$\text{dt} = \text{cfl}\,h\,\text{substeps}/(\text{stride}\,w)$ gives, for $\text{substeps}_b > 1$, a
step $\text{substeps}_b$ times larger than the old formula $\text{dt} = \text{cfl}\,h/(\text{stride}\,w)$.
Bit-identical parity with the history therefore holds only for $\text{substeps} = 1$ (at any
stride); to reproduce the historical test formula, supply the explicit historical $\text{dt}$ to
the oracle rather than its CFL helper. Under MPI, the absence of `all_reduce_max` would desynchronize the
ranks (each would see the max of its own boxes only) and would make the simulation diverge. The
stride semantics is hold-then-catch-up: the slow block is loosely coupled, which is an assumed choice
(the gas is not resolved at every step). `test_system_abstraction` proves the isolated metadata
formula; `test_program_runtime` proves the installed hold/catch-up cadence. The architecture gate proves that
no installed header and no `System` or `AmrSystem` launch can select it.

## 8. Parabolic term: diffusion as face flux

**Intuition.** A parabolic term $+\nu\,\Delta U$ (diffusion, isotropic scalar viscosity) is the
divergence of a Fickian flux $F_{\text{diff}} = -\nu\,\nabla U$. Writing it as a face flux rather
than a direct Laplacian makes it AMR-compatible: reflux sees it and corrects it at the
fine-coarse interface exactly like a hyperbolic flux, so diffusion stays conservative at level
junctions.

**Formula / discretization.** The continuous Fickian flux $F_{\text{diff}} = -\nu\,\nabla U$ adds to the
hyperbolic numerical flux before the divergence. As a face flux (centered gradient at the face, cell
values, not the level $h$):

$$F^{x}_{i+1/2,j} = -\nu\,\frac{U_{i+1,j} - U_{i,j}}{dx}, \qquad
  F^{y}_{i,j+1/2} = -\nu\,\frac{U_{i,j+1} - U_{i,j}}{dy}.$$

The divergence $-\big(F^{x}_{i+1/2} - F^{x}_{i-1/2}\big)/dx - \big(F^{y}_{j+1/2} - F^{y}_{j-1/2}\big)/dy$
gives back exactly the 5-point Laplacian:

$$+\nu\,\Delta_h U_{i,j} = \nu\left(
  \frac{U_{i+1,j} - 2U_{i,j} + U_{i-1,j}}{dx^2}
+ \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j-1}}{dy^2}\right),$$

added component by component to the residual $R = -\mathrm{div}\hat F + S$. The core `assemble_rhs`
writes this 5-point stencil directly (the AMR-less path); `compute_face_fluxes` produces the
face-flux form (the AMR reflux path). The two give a residual bit-identical to the machine.

```
# coeur : assemble_rhs, terme additif au residu (5 points), garde par DiffusiveModel
function diffusive_residual_term(model, u, i, j, dx, dy):
    if not DiffusiveModel(model):                 # if constexpr : zero codegen sinon
        return                                     # chemin hyperbolique strictement intouche
    nu   = model.diffusivity()
    idx2 = 1/(dx*dx);  idy2 = 1/(dy*dy)
    for c in 0 .. n_vars-1:
        lap = (u(i+1,j,c) - 2*u(i,j,c) + u(i-1,j,c)) * idx2
            + (u(i,j+1,c) - 2*u(i,j,c) + u(i,j-1,c)) * idy2
        r(i,j,c) += nu * lap                       # +nu Lap(U)

# AMR : compute_face_fluxes, flux de face Fickien ajoute au flux hyperbolique
function face_flux_x(model, u, aux, i, j, dx):     # face entre (i-1,j) et (i,j)
    L = reconstruct(model, u, i-1, j, dir=0, +1)   # etats reconstruits
    R = reconstruct(model, u, i,   j, dir=0, -1)
    F = numerical_flux(model, L, aux(i-1,j), R, aux(i,j), dir=0)   # hyperbolique
    if DiffusiveModel(model):
        nu = model.diffusivity()
        for c in 0 .. n_vars-1:
            F[c] += -nu * (u(i,j,c) - u(i-1,j,c)) / dx     # flux Fickien centre au face
    fx(i,j,:) = F                                  # le reflux AMR voit ce flux -> conservatif
```

**Code.** The contract is the `DiffusiveModel` concept in
[`numerics/spatial_operator.hpp`](../include/pops/numerics/spatial_operator.hpp): a model satisfies it
if and only if `m.diffusivity()` returns a `Real` ($\nu \ge 0$). The 5-point term is
added in `detail::AssembleRhsKernel::operator()` (called by `assemble_rhs`) under
`if constexpr (DiffusiveModel<Model>)`. The face-flux form lives in
`detail::FaceFluxXKernel` / `detail::FaceFluxYKernel` (called by `compute_face_fluxes`), same guard.
Both kernels are device-clean named functors (`POPS_HD`).

**Constraints / remarks.** Central invariant: a model that does not expose `diffusivity()` does not change
by a bit, the `if constexpr` being false there is no additional codegen (the hyperbolic path
strictly unchanged). The `dx`, `dy` arguments of `compute_face_fluxes` default to 0 and are
read only by the diffusive branch, so a non-diffusive model is never affected. The explicit step
on a parabolic term imposes the diffusive stability constraint
$\nu\,\Delta t \le \tfrac{1}{2}\,(dx^{-2} + dy^{-2})^{-1}$ (more restrictive in $h^2$ than the
hyperbolic CFL in $h$), not handled by `step_cfl` which only weighs the wave speed: at
diffusion-dominated, fix $\text{dt}$ explicitly. Known limit: `SourceFreeModel` (explicit half-step
IMEX) does not expose `diffusivity()`, so a diffusive IMEX block would lose its Fickian flux in the
explicit half-step (a separate refinement); and the masked path `assemble_rhs_masked` does not mask the
Laplacian. Tests: `test_diffusion` (the core $+\nu\,\Delta U$ via the divergence of the Fickian flux),
and `test_amr_program_diffusion` (a genuinely refined hierarchy smooths while its composite integral
remains conservative through the Program-owned flux ledger and reflux described in section 17).


---

## 9. Elliptic: geometric multigrid

**Intuition.** The Gauss-Seidel smoother quickly kills the high frequencies of the error but crawls on
the low ones. Multigrid restricts the low-frequency error onto coarser grids (where
it becomes high frequency again), smooths it, and prolongs it. Cost $O(N)$ per V-cycle, number of
cycles nearly independent of the mesh.

**Formula / discretization.** 5-point operator on $\mathrm{lap}(\phi) = f$ (isotropic case
$\epsilon = 1$, $\kappa = 0$):

$$(\mathrm{lap}\,\phi)_{ij} = \frac{\phi_{i+1,j} - 2\phi_{ij} + \phi_{i-1,j}}{\Delta x^2}
                            + \frac{\phi_{i,j+1} - 2\phi_{ij} + \phi_{i,j-1}}{\Delta y^2}$$

Red-black Gauss-Seidel smoother: one color $c \in \{0,1\}$ per sweep, the cell
$(i,j)$ with $(i+j) \bmod 2 = c$ is updated from its neighbors (of the other color, hence already
frozen on this sweep):

$$\phi_{ij} \leftarrow \frac{\mathrm{off}_{ij} - f_{ij}}{\mathrm{diag}},
\quad \mathrm{off}_{ij} = \frac{\phi_{i\pm1,j}}{\Delta x^2} + \frac{\phi_{i,j\pm1}}{\Delta y^2},
\quad \mathrm{diag} = \frac{2}{\Delta x^2} + \frac{2}{\Delta y^2}$$

V-cycle: $\nu_1$ pre-smoothing sweeps, residual $r = f - \mathrm{lap}\,\phi$, restriction of
$r$ by $2\times2$ average (`average_down`) onto the twice-coarser grid, recursive
resolution of the correction equation $\mathrm{lap}(e) = r$ with homogeneous conditions, prolongation
of $e$ (`interpolate`) added to $\phi$, $\nu_2$ post-smoothing sweeps. At the coarsest
level, `nbottom` sweeps stand in for an exact resolution (bottom solve). Defaults:
$\nu_1 = \nu_2 = 2$, `nbottom = 50`, `min_coarse = 2`.

```
function vcycle_rec(level l, bc):
    L = lev_[l]
    gs_smooth(L.phi, L.rhs, nu1, bc)              # pre-lissage : nu1 balayages rouge-noir
    if l est le plus grossier:
        gs_smooth(L.phi, L.rhs, nbottom, bc)      # bottom solve (longue serie de balayages)
        if masque: zero_conductor(L.phi)          # refige phi=0 dans le conducteur
        return
    poisson_residual(L.phi, L.rhs, -> L.res, bc)  # r = f - lap(phi) (porte aussi termes croises)
    average_down(L.res, C.rhs, ratio=2)           # restriction du residu (moyenne 2x2)
    C.phi = 0                                       # correction a CL homogenes
    vcycle_rec(l+1, homogeneous(bc))              # recursion grossiere
    corr = interpolate(C.phi, ratio=2)            # prolongation de la correction
    L.phi += corr                                  # saxpy
    if masque: zero_conductor(L.phi)
    gs_smooth(L.phi, L.rhs, nu2, bc)              # post-lissage

function solve(rel_tol, max_cycles):
    r0 = current_residual()                        # norm_inf(f - lap(phi)), all_reduce_max
    if r0 <= 0: return 0
    for c in 1..max_cycles:
        vcycle()                                   # warm-start : phi conserve entre appels
        if current_residual() <= rel_tol * r0: return c
    return max_cycles
```

The hierarchy is built by coarsening the domain by 2 down to `min_coarse`, but we stop if
a box does not coarsen cleanly: the test `b.coarsen(2).refine(2) == b` characterizes the
boxes that are aligned and of even size. On a multi-box domain (`max_grid_size < n`), the boxes
shrink by 2 at each level and would end at $1\times1$; `coarsen(ba,2)` would then make
several distinct fine boxes fall onto the same coarse cell (degenerate BoxArray), and
`average_down` would read out of bounds of a 1-cell fab. In serial the heap is stable, under MPI
it is shuffled and the read becomes erratic (occasional discrepancy up to blow-up). The break keeps the
current level as the coarsest grid; mono-box and non-degenerate multi-box never cross
this test, hierarchy and result strictly unchanged.

The red-black sweep makes each color data-independent (parallelizable). Between colors
and before the residual, `device_fence()` + `fill_ghosts` synchronize the device and fill the
halos; `current_residual` reduces the infinity norm by `all_reduce_max` (required for a
coarse multi-box distributed grid, otherwise the stopping criterion triggers at different iterations per
rank and desynchronizes the MPI fluxes). The `replicated` mode replicates each level on all
ranks (per-fab V-cycle without communication), which is what the AMR coupler expects (level 0 replicated).

**Code.** [`numerics/elliptic/geometric_mg.hpp`](../include/pops/numerics/elliptic/mg/geometric_mg.hpp):
`GeometricMG<Dim>` models the `EllipticSolver` concept (`rhs()`, `phi()`, `solve()`, `residual()`).
`v_cycle_` recurses through the immutable hierarchy and `solve()` returns a consumed `SolveReport`.
The $2\,Dim+1$ constant-scalar operator and damped-Jacobi smoother are shared bricks of
[`numerics/elliptic/poisson_operator.hpp`](../include/pops/numerics/elliptic/poisson/poisson_operator.hpp)
(`poisson_residual_valid`, `damped_jacobi_update_valid`). Restriction and prolongation reuse the
exact-ranked AMR transfer operators in
[`mesh/refinement.hpp`](../include/pops/mesh/layout/refinement.hpp).

**Constraints / remarks.** Fully on-device (the V-cycle goes through `for_each_cell`) and without a
CFL constraint (stationary solve). The concrete capability is the symmetric positive-definite
constant-scalar Cartesian operator, optionally with one non-negative constant reaction. Variable
diagonal coefficients, cross tensors and embedded boundaries are not silently approximated by this
engine; preparation must select another qualified provider.
**Validation.** `test_geometric_mg_nd` proves the hierarchy, capability refusal and exact-rank
execution in 1D/2D/3D. `test_poisson_convergence` proves quantitative order two for the native
specialization.

## 10. Elliptic: exact-ranked Cartesian discrete Poisson FFT

**Intuition.** On a periodic constant-coefficient domain, the discrete Laplacian is diagonal
in Fourier: one direct transform, one mode-by-mode division, one inverse transform solve
Poisson exactly (to machine residual), without iteration. Much cheaper than multigrid when
the elliptic dominates the run.

**Formula / discretization.** In native rank $d\in\{1,2,3\}$, the solver inverts the same Cartesian
second-order Laplacian as `GeometricMG`. Its eigenvalue for mode $\mathbf{k}$ is

$$\lambda(\mathbf{k}) = \sum_{a=0}^{d-1}
  \frac{2\cos(2\pi k_a/N_a)-2}{\Delta x_a^2}$$

and not $-(k_x^2 + k_y^2)$ (the exact symbol of the discrete stencil, not of the continuous
Laplacian). The resolution is $\hat\phi(k) = \hat f(k) / \lambda(k)$, with the mode $k = 0$ fixed to 0
(gauge: $\phi$ of zero mean; the right-hand side must therefore be of zero mean, otherwise $\phi$ drifts).

```
function solve():                                  # PoissonFFTSolver<Dim>, un slab/rang
    verifier la compatibilite du rhs avec le noyau constant
    rho = aplatir -rhs sur le slab local du dernier axe
    fft_.solve(rho -> phi_trial)                    # transforms Kokkos/MPI, symbole discret ND
    remplir les halos periodiques de phi_trial
    r = rhs - (-lap_h phi_trial)                    # meme operateur que celui inverse
    publier phi_trial seulement si ||r|| est dans l'enveloppe d'arrondi authentifiee
```

**Code.** [`numerics/elliptic/poisson_fft_solver.hpp`](../include/pops/numerics/elliptic/poisson/poisson_fft_solver.hpp):
`PoissonFFTSolver<Dim>` is the single exact provider for serial and MPI execution in native rank
1, 2 or 3: serial is simply the one-rank instance of the same ordered last-axis slab layout. It
models the exact-ranked `EllipticSolver`
concept and owns its immutable build request, explicit constant-nullspace workspace, trial field and
transactional publication. The residual reuses the canonical operator `poisson_residual` of
[`poisson_operator.hpp`](../include/pops/numerics/elliptic/poisson/poisson_operator.hpp); the
same wrapper fills inter-slab periodic halos before measurement and reduces by `all_reduce_max`.
The low-level transform core lives in `poisson_fft.hpp`.

**Constraints / remarks.** The FFT requires periodic BCs and a constant coefficient: neither
$\epsilon(x)$, nor an embedded mask, nor cross terms. The mode $k = 0$ must be fixed (right-hand side
of zero mean), otherwise the solve returns `kIncompatibleRhs` and leaves the published solution
unchanged. It requires exactly one canonical ordered slab of the final Cartesian axis per
communicator rank and that final extent be divisible by `n_ranks()`. Power-of-two axes use the
radix-2 Kokkos/MPI path; any other positive extent uses an explicit, diagnosed direct-DFT path that
inverts the same authenticated discrete operator rather than substituting another elliptic solver.
The raw continuous Fourier symbol remains an internal transform capability only: without a matching
apply/residual operator it cannot be advertised as an `EllipticSolver` route.
**Validation.** `test_poisson_fft` proves the concrete capability and transactional residual gate;
under MPI `test_mpi_fft_distributed` proves ordered slabs. `test_elliptic_operator` applies the same canonical
operator `poisson_residual` to the MG and FFT solutions: residuals at roundoff (`~1e-14`) and solutions
identical to `~1e-16`, so both provably invert the same discrete Laplacian.

## 11. Exact-ranked scalar GeometricMG

**Supported operator.** `GeometricMG<Dim>` owns one compile-time-ranked Cartesian operator,

$$A\phi = -\Delta_h\phi + \kappa\phi, \qquad \kappa \ge 0,$$

with a constant unit diffusion coefficient and one constant scalar reaction supplied in the immutable
`GeometricMultigridOptions`. The stencil visits the two neighbours of every axis, so the same
algorithm is instantiated in dimensions one, two and three. Setting `reaction = 0` gives Poisson;
a positive value gives the supported screened/Helmholtz form.

**Capability boundary.** The concrete solver reports
`scalar_constant_coefficient=true` and `scalar_reaction=true`. Variable diagonal coefficients,
cross tensors and embedded boundaries are explicitly `false`; there is no `SpatialProvider2D`,
`set_epsilon`, anisotropic setter or hidden 2D fallback. A requested operator outside this family
must select a separately prepared provider (for example the hierarchy tensor/FAC route) or fail
before solver construction.

**Validation.** `test_geometric_mg_nd` exercises the one algorithm in exact dimensions 1/2/3 and
checks the fail-closed capability matrix. `test_poisson_convergence` proves second-order convergence
of the native specialization. `tests/gpu/romeo/gpu_epm_validate.cpp` repeats a manufactured
constant-reaction solve on the selected Kokkos device and records the dimension and refinement
ratios; it does not advertise the retired variable/tensor/EB families.


---

## 12. Generic prepared matrix-free Krylov

**Intuition.** Krylov is an algorithm over a linear operator, not a special case of `GeometricMG`.
The final native layer therefore accepts any matrix-free callback over a `MultiFab`, a typed
mathematical-property certificate, an optional prepared preconditioner, and persistent workspace.
CG, BiCGStab, restarted GMRES, and Richardson consume the same protocol. A full-tensor elliptic
operator is one provider of that protocol; it is not baked into the solver type.

**Prepared contract.** `PreparedAffineLinearProblem` freezes and authenticates the evaluation
snapshot, topology, halo footprint, coefficients, boundary data, and optional nullspace policy before
iteration. It evaluates the exact affine constants

$$c_A=A_{\mathrm{raw}}(0), \qquad c_M=M^{-1}_{\mathrm{raw}}(0),$$

then exposes only the linear direction maps
$A_{\mathrm{lin}}(v)=A_{\mathrm{raw}}(v)-c_A$ and
$M^{-1}_{\mathrm{lin}}(v)=M^{-1}_{\mathrm{raw}}(v)-c_M$ to Krylov recurrences. The scientific
residual remains $r(u)=b-A_{\mathrm{raw}}(u)$. Thus nonzero Dirichlet or Robin data is retained in
the physical equation without injecting a constant into each search direction. The relative
reference is $\|b-A_{\mathrm{raw}}(0)\|_2$, while an authored absolute floor remains in physical
units. A warm start is tested against the true residual, not against an Arnoldi or preconditioned
estimate. A separate internal recurrence scale $\|b-A_{\mathrm{raw}}(x_0)\|_2$ exists only for a
non-converged warm start, to keep that finite residual representable. The physical reference is
deliberately excluded from this scale: a huge component already satisfied by $x_0$ must not erase a
tiny remaining residual. This internal scale never changes the physical stopping criterion or report.

```
snapshot <- authenticate(operator, topology, resources, evaluation point)
problem.prepare(snapshot)                 # freeze resources; compute c_A and c_M
workspace.bind(problem)                   # persistent fields/scalars, no hot-loop allocation

r <- b - A_raw(x)                         # physical residual for the authored warm start
reference <- ||b - c_A||_2
stop <- max(rel_tol * reference, abs_tol)
if ||r|| <= stop: publish solved(x, iters=0)

repeat according to the selected method:
    z  <- M_raw(r) - c_M                   # identity when no preconditioner is installed
    Az <- A_raw(z) - c_A                   # linear direction product
    update the CG/BiCGStab/GMRES/Richardson recurrence
    confirm convergence with ||b - A_raw(x)||_2

on breakdown, invalid evaluation, or iteration limit:
    return an explicit failed SolveReport  # no implicit best-effort solved value
```

`KrylovWorkspace` derives its exact field/scalar allocation from `KrylovFootprint` and the selected
method. GMRES restart storage is explicit and persistent. MPI products and norms are collective on
every rank, including ranks with no local box. Extension callbacks are re-probed for snapshot
mutation; generated callbacks carry an authenticated purity contract. No Python recurrence or
per-cell calculation participates in the solve.

For the full-tensor provider,
$L_{\mathrm{int}}(\phi)=\mathrm{div}(A\nabla\phi)-\kappa\phi$ uses the shared
`apply_laplacian` kernels. A fixed number of cold-start `GeometricMG` V-cycles on the diagonal block
is a prepared preconditioner; cross terms remain in the full matrix-free operator. Strong
non-symmetry may reduce preconditioner quality, but it does not change the solver contract.

**Code.** [`generic_krylov.hpp`](../include/pops/numerics/elliptic/linear/generic_krylov.hpp),
[`prepared_affine_problem.hpp`](../include/pops/numerics/elliptic/linear/prepared_affine_problem.hpp),
and [`krylov_workspace.hpp`](../include/pops/numerics/elliptic/linear/krylov_workspace.hpp).
Validation is centralized in `test_generic_krylov`: all four methods, true-residual reporting,
affine Dirichlet/Robin operators and preconditioners, nullspaces, snapshot invalidation, persistent
allocation, full symmetric/non-symmetric tensors, the MG-stall contrast, and MPI variants at
np=1/2/4.

## 13. Condensed implicit Program authoring

**Intuition.** A stiff source that couples potential, velocity and Lorentz force (diocotron at high $\omega_c$) cannot be treated component by component: the cyclotron rotation couples the two velocity components, and the potential reacts to the charge displacement. We theta-discretize the implicit source, eliminate the velocity algebraically via the closed inverse $B^{-1}$ of the 2x2 rotation, which leaves only an elliptic on the potential $\phi^{n+\theta}$ alone (Schur complement), then reconstruct the velocity.

**Formula / discretization.** The Lorentz eliminator encodes the rotation-dilation

$$B = \begin{pmatrix} 1 & -w \\ w & 1 \end{pmatrix}, \qquad B^{-1} = \frac{1}{\det B}\begin{pmatrix} 1 & w \\ -w & 1\end{pmatrix}, \qquad w = \theta\, dt\, B_z, \quad \det B = 1 + w^2 > 0,$$

closed and always invertible (no call to `std::`, four additions/multiplications, device-safe). With $c = \theta^2 dt^2 \alpha$ (Hoffart et al., arXiv:2510.11808), the condensed operator writes

$$L_{\mathrm{schur}}(\phi) = -\Delta\phi - c\mathrm{div}(\rho\, B^{-1}\nabla\phi) = -\mathrm{div}\!\big((I + c\,\rho\, B^{-1})\,\nabla\phi\big),$$

which identifies the full tensor $A = I + c\,\rho\, B^{-1}$, that is, per cell,

$$\varepsilon_x = 1 + c\rho\,B^{-1}_{11},\quad \varepsilon_y = 1 + c\rho\,B^{-1}_{22},\quad a_{xy} = c\rho\,B^{-1}_{12},\quad a_{yx} = c\rho\,B^{-1}_{21}.$$

The mass term $\kappa$ stays null (the condensation does not produce a Helmholtz). At $B_z = 0$: $w = 0$, $B^{-1} = I$, so $a_{xy} = a_{yx} = 0$ and $\varepsilon_x = \varepsilon_y = 1 + c\rho$; if in addition $c = 0$, $A = I$ and $L_{\mathrm{schur}}$ degenerates exactly into the canonical Laplacian. The condensed right-hand side is

$$\mathrm{rhs} = -\Delta\phi^n - \theta\, dt\, \alpha \mathrm{div}(\rho\, B^{-1} v^n), \qquad v^n = (m_x, m_y)/\rho,$$

where $-\Delta\phi^n$ is the canonical 5-point Laplacian negated and the divergence of the explicit flux $F = \rho B^{-1} v^n = B^{-1}(m_x, m_y)$ (applied to the momentum, which avoids the division by $\rho$) is centered order 2:

$$\mathrm{div} F(i,j) = \frac{F_x(i{+}1,j) - F_x(i{-}1,j)}{2\,dx} + \frac{F_y(i,j{+}1) - F_y(i,j{-}1)}{2\,dy}.$$

The condensed operator is in general full-tensor and is supplied to the prepared Krylov protocol of section 12. The authored provider applies $L_{\mathrm{schur}}=-\mathrm{div}(A\nabla\phi)$ directly, so the generic solver does not infer or flip a sign. After resolution, the velocity is reconstructed by $v^{n+\theta} = B^{-1}(v^n - \theta\, dt\,\nabla\phi^{n+\theta})$ (centered gradient, consistent with the RHS divergence), then extrapolated from the theta-stage to the full step by $U^{n+1} = U^n + \tfrac{1}{\theta}(U^{n+\theta} - U^n)$. The energy, if the Energy role is present, is updated only by the kinetic energy increment $E^{n+1} = E^n + \tfrac{1}{2}\rho^n(|v^{n+1}|^2 - |v^n|^2)$, the Lorentz rotation doing no work and $\rho$ being frozen.

```python
from pops.linalg import LinearProblem
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.time import FailRun

T = pops.Program("condensed_source")
q = T.state(block[state])
scope = Hierarchy()
coefficients = T.condensed_coeffs(
    "condensed_coefficients",
    state=q.n,
    linear_operator=implicit_rotation,
    subset=(mx_component, my_component),
    c=theta * theta * alpha * T.dt * T.dt,
    th_dt=theta * T.dt,
    c_rho=rho_component,
)
phi_n = T.history("phi", lag=1, ncomp=1, block=block)
rhs_storage = T.scalar_field("condensed_rhs")
condensed_rhs = T.condensed_rhs(
    rhs_storage,
    phi_n,
    q.n,
    linear_operator=implicit_rotation,
    subset=(mx_component, my_component),
    th_dt=theta * T.dt,
    g=theta * alpha * T.dt,
)
operator = T.matrix_free_operator(
    "condensed_tensor",
    scope=scope,
    # stencil_depth is inferred as 1 from apply_laplacian_coeff; an explicit
    # non-negative depth is reserved for a custom provider with a deeper stencil.
)

def apply_condensed_tensor(builder, _out, value):
    laplacian = builder.scalar_field("condensed_laplacian")
    return -1 * builder.apply_laplacian_coeff(laplacian, value, coefficients)

T.set_apply(operator, apply_condensed_tensor)
phi_theta = T.solve(
    LinearProblem(
        operator=operator,
        rhs=condensed_rhs,
        initial_guess=phi_n,
        scope=scope,
        nullspace=None,
    ),
    solver=CompositeTensorFAC(max_iter=400, rel_tol=1e-10, abs_tol=0.0),
    name="condensed_stage",
).consume(action=FailRun())
# Reconstruction, theta extrapolation, energy update and commit remain explicit Program IR.
```

The method is authored as ordinary Program IR: tensor-coefficient assembly, metric-aware right-hand
side, one typed scalar solve, reconstruction, optional theta extrapolation and optional energy
update. `LinearProblem` carries the algebra and hierarchy scope; `CompositeTensorFAC` owns the
complete flat/refined solver identity, tolerance and iteration budget. On a flat hierarchy the
generated C++ freezes the authored apply in a `PreparedAffineLinearProblem`, binds its persistent
`KrylovWorkspace`, then executes it through `ctx.solve_prepared_linear`. Preparation authenticates the
exact evaluation snapshot and separates `A(0)` from `A_lin`; the flat branch therefore supports affine
boundary/source terms without applying them to search directions. A prepared non-identity
preconditioner performs the same construction independently (`d = M_raw(0)`,
`M_lin(v) = M_raw(v) - d`), so a Dirichlet or Robin value cannot be reinjected at every Krylov
iteration. The problem and preconditioner prototypes must have the exact same components, boxes,
distribution and halo footprint; inputs may not alias mutable outputs, and workspace slots are a
private, fixed-shape native resource rather than an extension mutation seam. CG and BiCGStab replace
their complete recurrence after a recursive convergence candidate fails true-residual confirmation.
BiCGStab otherwise keeps `r = s - omega*t` and evaluates the true residual only for convergence
confirmation, failure reporting and final reporting, avoiding a third matvec on every full iteration.
GMRES batches all Arnoldi projections of a column into one vector collective, evaluates the projected
norm exactly, and applies a selective batched CGS2 pass under the DGKS norm-loss criterion. Its normal
column uses two collectives instead of one collective per existing basis vector. GMRES restart is
dynamically sized on the exact native interval $1 \le m \le \mathrm{INT\_MAX}-1$, including the
Newton-Krylov route; the upper bound is the C++/MPI count-capacity boundary, not an arbitrary
algorithmic fixed-array ceiling. The
inert scratch plan reports each allocation owner once: a matrix-free operator owns its
apply/frozen/Jv fields, a solve owns its solution, prepared problem/preconditioner and workspace,
and a condensed-coefficient node owns its four live coefficient fields. This makes shared-operator
reuse inspectable without double-counting. It also reports exact Hessenberg/rotation scalar and
batched-collective payload counts. Conditional boundary-JVP fields are a min/max range and a
geometric-MG hierarchy remains explicitly topology-dependent; neither is presented as an exact
whole-process allocation count.

The AMR install owns one complete persistent Program-resource bundle per level and
rebuilds those bundles when either the checkpointed topology epoch or the process-local
materialization generation changes. The latter is deliberately not restored, so restart rebuilds and
rejected-attempt rollbacks invalidate concrete storage even when epoch and level count are unchanged;
a level-local Krylov problem is therefore never reused against another layout.
Frozen tensor coefficients copy valid and ghost regions because face and cross stencils consume
inter-box neighbour values. On a refined hierarchy,
every level first assembles and gathers its coefficients, right-hand side and initial guess; exactly
one `ctx.solve_composite_tensor_fac` then solves the complete hierarchy; only a successful complete
solution is published to every level. The accepted synchronization subsequently applies reflux and
then average-down. The authored operator supplies $B^{-1}$; no physics-specific time preset,
source-stage stepper or System setter exists.

The prepared footprint carries the exact integer stencil depth, not a 0/1 approximation. Every typed
operation carries an immutable `StencilAccess` capability and apply regions compose those capabilities
by maximum depth, without an opcode-name table. `matrix_free_operator(stencil_depth=n)` may declare a
larger provider halo and is rejected if `n` is smaller than the composed requirement. The legacy
callback-based `preconditioners.GeometricMG()` Program route has been removed. Exact-ranked
`GeometricMG<Dim>` remains available as an elliptic field solver; a future Program preconditioner must
receive the same authenticated ranked geometry, boundary and distribution contract before it can be
published. `Identity()` remains the built-in Program preconditioner and external prepared providers
remain available through their authenticated component contract.

**Code.** The generic linear-solve protocol is in
[`python/pops/codegen/program_emit_solve.py`](../python/pops/codegen/program_emit_solve.py), and the
runtime providers in
[`include/pops/runtime/program/program_context.hpp`](../include/pops/runtime/program/program_context.hpp),
[`include/pops/runtime/program/amr_program_context.hpp`](../include/pops/runtime/program/amr_program_context.hpp),
and [`include/pops/runtime/amr/amr_tensor_elliptic.hpp`](../include/pops/runtime/amr/amr_tensor_elliptic.hpp).

**Constraints / remarks.** Stability: the theta-scheme is unconditionally stable for $\theta \geq 1/2$ ($\theta = 1$ pure implicit, the extrapolation is the identity; $\theta = 1/2$ Crank-Nicolson, extrapolation factor 2). The centered order-2 discretization fixes the Cartesian spatial order; polar emission uses the corresponding $1/r$ metric factors. The condensed sub-flow freezes $\rho$ and is composed explicitly with transport through `pops.lib.time.Strang` or `Lie`. Safeguard: $c = 0$ and $B_z = 0$ give $A = I$. The solver tolerance and iteration budget are authored controls, never hidden defaults in a native stage.

Validation belongs to the generic Program solve contract, matrix-free provider tests and the polar
tensor MPI suite; there is no named condensed time-preset test surface.

The former acceptance target "generated route throughput >= 98% of the native condensed stepper"
is not a reproducible final-architecture comparison: that stepper and its descriptors are retired,
so retaining it as an oracle would preserve a second production implementation, while compiling the
generic route twice measures the same code against itself. A fixed wall-clock ratio would also be a
flaky CI assertion across toolchains and hosts. The final deterministic performance invariants are
therefore structural and executable: condensed cell kernels are emitted inline with no physics-name
runtime dispatch; a refined stage gathers all levels and performs one hierarchy solve, not one solve
per level; the FAC object and level storage are reused while the authenticated hierarchy tiling is
unchanged and rebuilt on regrid; and the native tensor/FAC
and polar multi-box/MPI suites exercise the actual kernels. Comparative throughput belongs in an
external same-machine benchmark against a pinned released binary, never in the conformance gate.


---

## 14. Embedded boundary: exact-rank cut geometry

The production authority is one `template<int Dim>` level-set pipeline for `Dim=1,2,3`.  It samples
the signed analytic geometry on the grown Cartesian layout, materializes an active mask and a
clamped retained-volume field, and decorates the same prepared metric/operator used without EB.
There is no disc-specific native domain object, no x/y storage convention, and no separate EB
runtime.  Geometry descriptors such as a sphere or plane only emit the signed level-set expression.

The Shortley-Weller equations below are retained as the mathematical crossing reference.  The old
fixed four-face/5-point implementation and its `DiscDomain` facade have been retired; production
code stores `lower[Dim]`, `upper[Dim]`, and one diagonal and obtains them with a single axis loop.

**Intuition.** A wall not aligned on the grid (circular conductor, diocotron ring edge)
is not a staircase: the Shortley-Weller cut-cell corrects the 5-point stencil where the
disc level set cuts a face, so that the Dirichlet condition is imposed at the real position
of the interface and not at the nearest cell face. We recover order 2 where the
0/1 staircase falls to order 1.

**Formula / discretization (two-axis illustration).** A boundary may be illustrated by the disc level set
$ls(x,y) = \mathrm{hypot}(x - c_x, y - c_y) - R$, negative inside. For an active cell
$ls(x_c, y_c) < 0$, each cardinal face is cut at a linear fraction: if the neighbor is
interior ($l_n < 0$) the face is full and the half-distance equals $h$; if the level set changes
sign ($l_n \ge 0$) the linear crossing between $l_c < 0$ and $l_n \ge 0$ gives

$$\theta = \frac{l_c}{l_c - l_n}, \qquad a = \theta\, h, \qquad \theta \in [10^{-3}, 1]$$

(the lower clamp $10^{-3}$ is the anti-division guard that prevents $w \to \infty$ when the face
grazes the boundary). With the four half-distances $a_{xm}, a_{xp}, a_{ym}, a_{yp}$, and setting
$s_x = a_{xm}+a_{xp}$, $s_y = a_{ym}+a_{yp}$, the Laplacian $-\Delta\phi$ becomes a 5-point stencil with
unequal steps (Shortley-Weller) of weights

$$w_{xm} = \frac{2}{a_{xm}\, s_x},\quad w_{xp} = \frac{2}{a_{xp}\, s_x},\quad
w_{ym} = \frac{2}{a_{ym}\, s_y},\quad w_{yp} = \frac{2}{a_{yp}\, s_y},$$

$$w_{\mathrm{diag}} = \frac{2}{a_{xm}\, a_{xp}} + \frac{2}{a_{ym}\, a_{yp}},$$

and the residual on an active cell is
$L\phi = w_{xm}\phi_{i-1} + w_{xp}\phi_{i+1} + w_{ym}\phi_{i,j-1} + w_{yp}\phi_{i,j+1} - w_{\mathrm{diag}}\phi_{i,j}$
(the boundary Dirichlet value is injected via the ghost placed at $a$, not $h$). Far from the boundary all
the half-distances equal $h$: the weights give back exactly the uniform 5-point stencil
$1/h^2,\dots,-4/h^2$. For a conductor cell ($ls \ge 0$, mask at 0) the cell is skipped
(unused coefficient).

```
function shortley_weller_coefs(level L, geometry g, level_set ls):
    # one-shot au setup, par niveau MG, sur l'hote (puis lu par le V-cycle on-device)
    for each active cell (i, j) of L:            # m(i,j) != 0 ; conducteur saute
        lc  = ls(g.x_cell(i), g.y_cell(j))       # < 0 par construction (cellule active)
        axm = cut_distance(lc, ls(x - dx, y), dx)  # voisin interieur -> dx ; sinon theta*dx
        axp = cut_distance(lc, ls(x + dx, y), dx)
        aym = cut_distance(lc, ls(x, y - dy), dy)
        ayp = cut_distance(lc, ls(x, y + dy), dy)
        sx = axm + axp ; sy = aym + ayp
        c(i,j,0) = 2 / (axm * sx)                # w_xm  sur p(i-1, j)
        c(i,j,1) = 2 / (axp * sx)                # w_xp  sur p(i+1, j)
        c(i,j,2) = 2 / (aym * sy)                # w_ym  sur p(i, j-1)
        c(i,j,3) = 2 / (ayp * sy)                # w_yp  sur p(i, j+1)
        c(i,j,4) = 2/(axm*axp) + 2/(aym*ayp)     # w_diag (coefficient central)

function cut_distance(lc, ln, h):
    if ln < 0:        return h                   # face pleine (voisin interieur)
    th = lc / (lc - ln)                          # crossing lineaire
    return clamp(th, 1e-3, 1) * h                # garde anti-division par 0
```

**Code.** Ranked crossings and Shortley-Weller coefficients live in
[`include/pops/numerics/spatial/embedded_boundary/cut_geometry.hpp`](../include/pops/numerics/spatial/embedded_boundary/cut_geometry.hpp).
Collective sampling and the `kappa` clamp are prepared in
[`src/runtime/prepared_embedded_boundary.cpp`](../src/runtime/prepared_embedded_boundary.cpp); the
transport composition is
[`include/pops/numerics/spatial/embedded_boundary/operator.hpp`](../include/pops/numerics/spatial/embedded_boundary/operator.hpp).
All fields are `MultiFab<Dim>`/`FieldView<Dim>` and execute through Kokkos.  The prepared hierarchy
owns the same geometry image at every AMR level, including MPI halo and physical-boundary closure.

**Constraints / remarks.** The clamp $\theta \ge 10^{-3}$ bounds $w_{\mathrm{diag}}$ (without it a
grazing face would make the weight diverge and would break the diagonal dominance of the smoother). Compatible with
the anisotropic operator (the cut-cell weights compose with the $\varepsilon_x, \varepsilon_y$ coefficients). Validation: `test_cut_cell` (cut-cell vs staircase on a manufactured solution, order gain
), `test_cut_cell_anisotropic` (cut-cell + anisotropic operator), `test_cut_cell_anisotropic_multibox`
(multi-box single-rank), `test_mpi_cutcell_multibox` (multi-box distributed np=1/2/4; non-regression lock
of the `average_down` out-of-bounds bug on a degenerate MG hierarchy). For the elliptic on an
immersed disc, `test_poisson_disc` exercises the solver (convergence + improvement at resolution).

## 15. Generic level-set mask and EB transport

The native runtime does not own a disc mode.  Python lowers any typed `LevelSet` (disc, sphere,
plane, CSG, or user expression) to the same exact-rank analytic bytecode; System and AmrSystem then
prepare `staircase` or `cutcell` geometry transactionally.  The following two-axis equations are an
illustration of the ranked face/volume balance, not a second 2D implementation.

**Intuition.** A disc transport subdomain (diocotron ring) imposes a circular boundary
not aligned on the Cartesian grid. Three modes, from the simplest to the most precise:
`none` (mask materialized but ignored, bit-identical to the full Cartesian), `staircase` (jagged
boundary, 0/1 face gate, order 1 at the boundary), `cutcell` / embedded-boundary (continuous apertures
`alpha_f` + volume fraction `kappa`, smooth boundary, order 2 inside the disc). The cut-cell
generalizes the 0/1 gate to an aperture $\alpha_f \in [0,1]$ and divides the residual by the true immersed
volume $\kappa$, which de-jaggs the boundary and restores the order 2 that the staircase mask does not give.

**Formula / discretization.** Conservative embedded-boundary form for cell $(i,j)$ of volume
$\kappa\, dx\, dy$ (advection $\partial_t U = -\mathrm{div}\,F + S$):

$$\kappa\, dx\, dy\; \partial_t U = -\big[\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i\big] dy
 - \big[\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j\big] dx - \alpha_w |w| F_w + \kappa\, dx\, dy\, S,$$

that is, after division by $\kappa\, dx\, dy$ (with $\kappa$ clamped, cf. below):

$$R = S - \frac{1}{\kappa}\left[\frac{\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i}{dx}
 + \frac{\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j}{dy}\right]
 - \frac{1}{\kappa}\,\frac{\alpha_w |w|}{dx\, dy}\, F_w.$$

The immersed wall flux is a no-penetration $F_w = 0$ (FV counterpart of the conductor wall: the term is
identically null, written explicitly as a hook for a future non-zero flux). The apertures and $\kappa$
come from the same `cut_fraction` as the elliptic cut-cell:
$\alpha_f = a_f / h$, and $\kappa = \tfrac{1}{2}(\alpha_{xm}+\alpha_{xp})\cdot\tfrac{1}{2}(\alpha_{ym}+\alpha_{yp})$.
A face between two active cells has the same aperture on both sides
($\alpha_{xp}(i) = \alpha_{xm}(i{+}1)$, a function of the level set alone), so the fluxes telescope and the
mass $\sum_{ij} n_{ij}\, \kappa_{ij}\, dx\, dy$ is conserved to the machine; a face touching an
inactive cell is closed ($\alpha_f = 0$) and $F_w = 0$, so no mass crosses the boundary.

```
function assemble_rhs_eb(model, U, aux, ls, geom, R, kappa_min):
    # passe 1 : flux de face ponderes par l'ouverture alpha_f (MultiFab Fx, Fy temporaires)
    for each x-face i between cells (i-1,j) and (i,j):
        lL = ls(x_cell(i-1), y_cell(j)) ; lR = ls(x_cell(i), y_cell(j))
        alpha = face_aperture(lL, lR)            # voisin inactif -> 0 ; sinon cut_distance/dx
        if alpha < 1e-6:  Fx(i,j,:) = 0          # face fermee = paroi immergee, flux normal nul
        else:
            L  = reconstruct(U, i-1, j, dir=0, +1)   # reconstruction reutilisee du cartesien
            Rr = reconstruct(U, i,   j, dir=0, -1)
            Fx(i,j,:) = alpha * numerical_flux(L, Rr, dir=0)   # on stocke alpha * F
    ... idem pour les y-faces -> Fy ...

    # passe 2 : divergence EB / kappa_eff + source
    for each cell (i, j):
        if ls(x_cell(i), y_cell(j)) >= 0:        # hors disque : residu nul, cellule non avancee
            R(i,j,:) = 0 ; continue
        kappa     = cut_fraction(ls, x_cell(i), y_cell(j), dx, dy).kappa
        kappa_eff = max(kappa, kappa_min)        # clamp petite cellule (defaut 1e-2)
        S = model.source(U(i,j), aux(i,j))
        for each component c:
            div_x = (Fx(i+1,j,c) - Fx(i,j,c)) / dx     # Fx contient deja alpha*F
            div_y = (Fy(i,j+1,c) - Fy(i,j,c)) / dy
            # accumulation terme A terme : avec kappa_eff=1 et alpha=1, bit-identique au cartesien
            R(i,j,c) = S[c] - div_x/kappa_eff - div_y/kappa_eff - 0   # F_wall = 0

function face_aperture(lc, ln):
    if ln >= 0:  return 0                         # voisin inactif : face fermee (no-penetration)
    return cut_distance(lc, ln, h) / h            # voisin actif : ouverture lineaire (== mur elliptique)
```

**Code.** `set_analytic_level_set` installs the ranked expression and `set_geometry_mode` selects the
prepared policy. `embedded_boundary_mask` is the generic sidecar (all active when no geometry is
installed).  `PreparedEmbeddedBoundaryOperator<Dim>` composes
`PreparedMaskedCartesianOperator<Dim>` with `PreparedEmbeddedBoundaryMetric<Dim>`; reconstruction,
Riemann fluxes, provider values, MPI transport and Kokkos execution are therefore the same
authorities as the Cartesian route.  The operator currently advertises its centre-sampled binary
face closure explicitly; retained-volume scaling is prepared and clamped.  It does not claim the
retired continuous-aperture 2D path as a production capability.

**Constraints / remarks.** Small-cell problem: the factor $1/\kappa$ amplifies the residual
when $\kappa \to 0$ on the cut layer, which would make a fixed explicit step explode. Two stacked
guards: (i) the floor $\theta \ge 10^{-3}$ of `cut_distance` inherited from the elliptic wall (bound
$\kappa \gtrsim 2.5\times 10^{-7}$, insufficient alone); (ii) the clamp on the retained volume
$\kappa_{\mathrm{eff}} = \max(\kappa, \kappa_{\min})$, $\kappa_{\min} = 10^{-2}$ by default, which bounds
the amplification to $1/\kappa_{\min} = 100$ (implicit volume merging, calibrated for a stable fixed step
whatever the degree of cut). The clamp acts only on the denominator (volume), not on the fluxes:
the global mass stays exact, at the cost of a slight local non-conservation on the most
cut cells. The path is opt-in: an all-active mask and unit retained volume are bit-identical to the
Cartesian operator. Validation is carried by `test_embedded_boundary_generic` and
`test_prepared_embedded_boundary_nd`, which instantiate the same algorithms in 1D, 2D and 3D,
permute every active axis, check conservation/finite publication and rollback, and exercise prepared
LevelSet halos. Python tests use the corresponding generic embedded-boundary names.

## 16. Polar geometry: transport and Poisson on a ring (r, theta)

**Intuition.** For a diocotron ring, the Cartesian grid pays a structural over-rate at the edges
of the ring (the "Cartesian ring edges" lock). An annular polar grid
$r \in [r_{\min}, r_{\max}] \times \theta \in [0, 2\pi)$ aligns the geometry on the problem: theta is
periodic, r is physical (walls), and the ring excludes $r = 0$ ($r_{\min} > 0$) so no
coordinate singularity. Polar transport and Poisson reuse the same reconstruction, flux and source
bricks as the Cartesian; only the metrics change.

**Formula / discretization (transport, conservative FV).** The polar divergence
$\mathrm{div}\,F = \tfrac{1}{r}\partial_r(r F_r) + \tfrac{1}{r}\partial_\theta(F_\theta)$ is discretized
by storing the radial flux weighted by the face radius, $r_{i\pm1/2} F_r$, and the direct azimuthal flux,
then differencing:

$$R = S + S_g - \frac{1}{r_i}\frac{r_{i+1/2} F_r^{i+1} - r_{i-1/2} F_r^{i}}{dr}
 - \frac{1}{r_i}\frac{F_\theta^{j+1} - F_\theta^{j}}{d\theta}.$$

$S_g$ is the geometric curvature source ($-\rho v_\theta^2/r$ etc.), not captured by the conservative
divergence in a rotating local basis; it is carried per cell (null for a scalar ExB brick
-> bit-identical to the historical polar ExB transport). The weight $r_{i+1/2}$ of an interior face is
shared by the two neighboring cells, so the radial term telescopes; the azimuthal term telescopes
exactly (periodic). When the immutable `PreparedBoundaryPlan` assigns `NoFlux` to the two radial
faces, their evaluated numerical flux is forced to zero -> mass
$\sum n_{ij}\, r_i\, dr\, d\theta$ conserved to the machine whatever $v_r$.

**Formula / discretization (Poisson, FFT-in-theta + tridiag-in-r).** We solve
$\tfrac{1}{r}\partial_r(r\,\partial_r\phi) + \tfrac{1}{r^2}\partial_\theta^2\phi = f$ directly
(no multigrid, which stagnates on the $1/r^2$ operator). Theta being periodic with constant
coefficient, an FFT in theta diagonalizes $\partial_\theta^2$ exactly: the DFT mode $m$ has the signed
wavenumber $k(m) = m$ if $m \le n_\theta/2$, otherwise $m - n_\theta$, and the spectral eigenvalue
$-k(m)^2$ (and not the 2-point stencil $(2\cos - 2)/d\theta^2$, which is only an
$O(d\theta^2)$ approximation). The azimuthal term per cell becomes $(-k(m)^2/r_i^2)\,\hat\phi(i,m)$, diagonal in
$m$. The radial term is FV order 2,

$$\frac{1}{r_i}\left[\frac{r_{i+1/2}(\phi_{i+1}-\phi_i)}{dr} - \frac{r_{i-1/2}(\phi_i-\phi_{i-1})}{dr}\right]\frac{1}{dr},$$

so, per mode $m$, a tridiagonal system in $r$ solved by Thomas with

$$a_i = \frac{r_{i-1/2}}{r_i\, dr^2},\qquad c_i = \frac{r_{i+1/2}}{r_i\, dr^2},\qquad
b_i = -(a_i + c_i) - \frac{k(m)^2}{r_i^2}.$$

```
function polar_poisson_solve(geom, bc, rhs f, out phi):
    nr, nth = geom.nr, geom.ntheta
    # 1) FFT en theta, ligne radiale par ligne radiale : f(i, .) -> fhat(i, m)
    for i in 0..nr-1:  fhat[i] = fft( f(i, .) )
    # 2) coefficients radiaux independants du mode (geometrie pure)
    for i in 0..nr-1:
        a[i] = r_face(i)   / (r_cell(i) * dr^2)        # sous-diag
        c[i] = r_face(i+1) / (r_cell(i) * dr^2)        # sur-diag
        d_rad[i] = -(a[i] + c[i]) ;  inv_r2[i] = 1 / r_cell(i)^2
    # 3) une tridiagonale (Thomas) par mode azimutal m
    for m in 0..nth-1:
        k = (m <= nth/2) ? m : m - nth                 # nombre d'onde signe (repliement DFT)
        for i: b[i] = d_rad[i] - k*k * inv_r2[i]        # diag = radiale + azimutale spectrale
        rhs_m = fhat[., m]
        apply_radial_bc(b, rhs_m, m)                   # Dirichlet: b-=a/c, rhs-=2 a v (m=0) ; Neumann: b+=a/c
        pin0 = (deux bords Neumann) and (m == 0)        # operateur radial singulier -> jauge phi_hat(0,0)=0
        phat[., m] = thomas(a, b, c, rhs_m, pin0)
    # 4) FFT inverse en theta : phat(i, m) -> phi(i, theta) (partie reelle)
    for i in 0..nr-1:  phi(i, .) = real( ifft( phat[i] ) )
```

Boundary conditions in $r$ (via `PhysicalBoundaryConditions<2>`): Dirichlet (value $v$ at the face, reflection ghost
$\phi_{-1} = 2v - \phi_0$ -> $b_0 \mathrel{-}= a_0$, and $2 a_0 v$ to the right-hand side of the mode
$m=0$ alone) or homogeneous Neumann (Foextrap, $\phi_{-1} = \phi_0$ -> $b_0 \mathrel{+}= a_0$). Mode $m=0$
+ two Neumann boundaries: the radial operator has the constant in its kernel (singular tridiagonal); we fix
the gauge by pinning $\hat\phi(0,0) = 0$ (row 0 replaced by the identity in Thomas).

**Code.** [`include/pops/numerics/elliptic/polar/polar_geometry.hpp`](../include/pops/numerics/elliptic/polar/polar_geometry.hpp)`::PolarGeometry<2>`
and the following polar solvers remain standalone algorithm components. `pops.mesh.PolarMesh`
normalizes annular geometry for inspection/output, but the exact-ranked `System<Dim>` accepts only
Cartesian providers and refuses the annulus before artifact creation. The old dimension-erased
transport builder and its callback boundary plan have been removed; no public runtime route claims
polar transport until a metric-aware `Dim`-ranked provider owns geometry, boundaries and storage.
Poisson:
[`include/pops/numerics/elliptic/polar/polar_poisson_solver.hpp`](../include/pops/numerics/elliptic/polar/polar_poisson_solver.hpp)`::PolarPoissonSolver<2>`
(FFT-in-theta `fft1d` reused from `poisson_fft.hpp` + complex Thomas solve in r; models the
concept `PolarEllipticSolver` `rhs()/phi()/solve()/residual()/geom()`). Publishing metric-derived
auxiliary fields belongs to that future ranked provider rather than to a hidden runtime callback.

**Polar tensor operator + generated condensed Program.** When the coupled implicit source goes polar (diocotron at
high $\omega_c$), the Schur condenses a full tensor operator
$A = I + c\,\rho\, B^{-1}$ with cross terms $a_{rt}, a_{tr}$ and a theta-dependent coefficient: the
FFT-in-theta of `PolarPoissonSolver` no longer applies (it requires a constant theta coefficient
without cross coupling).
[`include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp`](../include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp)`::PolarTensorKrylovSolver<2>`
then solves by matrix-free BiCGStab (handles the non-symmetric of the cross term), preconditioned
`Jacobi` or `RadialLine` (radial Thomas per theta line, default). No MG V-cycle (stagnation on
$1/r^2$). Singular operator (pure radial Neumann + periodic theta): gauge fixed by projection onto
the subspace of zero FV mean (`project_mean`, the iterative counterpart of the mode-0 pinning). The
9-point stencil reads the diagonal corners filled by the exact `HaloSchedule<2>` followed by
`PreparedPhysicalBoundary<2>` (without which the cross term would be wrong at the
box boundary). This specialized backend is not selected by the final prepared
`Program.solve(LinearProblem(...), solver=...)` route; a future typed polar metric/operator provider
must connect it explicitly rather than reviving the removed callback-based `solve_linear_matfree`
dispatch. Multi-rank MPI /
multi-box is supported by azimuthal splitting under `RadialLine` (the Thomas sweep in r must stay local
to a box, safeguard `check_radial_columns`) and free 2D tiling under `Jacobi`.

**Constraints / remarks.** `PolarPoissonSolver<2>`: single-rank scope, single box covering the ring
(the FFT-in-theta + tridiag-in-r requires the complete theta line AND the radial column on one rank; the
distributed would impose a parallel transpose, out of Phase 2a scope) -> hard safeguard (active in Release)
if the communicator has more than one rank or the exact `BoxArray<2>` does not contain one
full-annulus patch. Theta spectral: exact for a band-limited datum
(diocotron = few azimuthal modes), `dtheta` does not enter the eigenvalue. The tridiag is
diagonally dominant (azimuthal term $\le 0$, folded BC) -> Thomas stable without pivoting. Host
reads and publication use explicit `Fab<2>::HostMirror` copies, including non-host memory spaces.
`PolarTensorKrylovSolver<2>`:
RadialLine $\sim$ moderately growing iteration count (isotropic $\times 2$ per grid doubling,
tensor $\times 2.4$); Jacobi grows in $1/h^2$ (sanity check / fallback). The cross term and the azimuthal
coupling are not in the preconditioner (an honest limit, later refinement possible).
Validation: `test_polar_transport_mms` / `test_polar_mms_vr` (polar transport MMS order 2),
`test_polar_fluid_transport`, `test_polar_lorentz_source`, `test_polar_poisson_mms`
(PolarPoissonSolver, radial order 2), `test_polar_tensor_elliptic_mms` (polar tensor operator),
`test_time_divergence` (generic matrix-free `div(grad)` Program solve) and
`test_mpi_polar_schur` (polar tensor solve multi-rank).
`test_polar_system_step` keeps the standalone coupled field-solve + local-basis aux + SSPRK3
transport + wall oracle without restoring the retired polar `System` engine. Python resolution and
direct-runtime refusal are covered by `test_layout_plan` and `test_polar_system`.


---

## 17. AMR: prepared subcycling and conservative-reflux primitives

**Intuition.** Refine only where needed. A fine level (step $\Delta x_c / r$) covers a
sub-region; to respect its CFL it does $r$ substeps of $\Delta t / r$ while the coarse does
a single step of $\Delta t$. At the fine-coarse interface, the two levels compute different fluxes,
which breaks discrete conservation. Reflux corrects the bordering coarse cell by the difference
(time-integrated fine flux minus coarse flux). The normalized `ProgramGraph` is the intended sole
authority for parent/child clocks, stage points and catch-up order; `AmrRuntime` supplies hierarchy
state, spatial evaluations, transfers and reflux primitives, but never chooses a time integrator.
The prepared primitives below are implemented and qualified independently. The production Program
driver still refuses a hierarchy with more than one level, so this section is not evidence of an
accepted end-to-end multi-level trajectory.

**Formula / discretization.** Let a fine-coarse face in $x$ between the coarse cell $(I, J)$ and the
fine patch. During the coarse step we have already advanced the coarse with its own face flux $F_c$ (over
$\Delta t$). We accumulate in parallel the fine flux of the same face over the $r$ substeps. The
correction replaces the coarse contribution by the fine contribution:

$$U_c(I,J) \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_{s=1}^{r} \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

with $\Delta t_f = \Delta t / r$ and $\bar F_f^{(s)}$ the fine flux averaged over the fine faces covering the
coarse face. The transactional Program flux ledger stores both sides already integrated with the
exact stage and time-step coefficients from the graph. For an exact-ranked spatial ratio
$\mathbf r=(r_0,\ldots,r_{D-1})$, the parent footprint is computed axis by axis; an axis with
$r_a=1$ is an identity axis. Temporal interpolation is qualified separately by its parent/child
clock relation,
$U^\star = (1-\alpha)\,U_c^{\mathrm{old}} + \alpha\,U_c^{\mathrm{new}}$.
Integral or explicitly remainder-bearing relations are graph data rather than a hidden property of
the spatial hierarchy.

The following pseudocode is the target orchestration contract. It is deliberately not presented as
the current implementation for a hierarchy with more than one level:

```
function execute_amr_program(program_graph, hierarchy, macro_dt):
    begin_atomic_attempt(hierarchy, program_graph.clocks)
    for clock_event in program_graph.recursive_parent_child_order(macro_dt):
        fill_ghosts_and_interpolate_parent_state(clock_event.point)
        rate, edge_flux = evaluate_spatial_operator(clock_event.candidate)
        candidate = combine_exactly_as_authored(program_graph, rate)
        flux_ledger.accumulate(edge_flux, program_graph.exact_coefficient(rate))

        if clock_event.child_catches_parent:
            route_reflux_integrated(flux_ledger, hierarchy)
            average_down(child_state, parent_state)

    evaluate_collective_guards()
    commit_every_level_and_clock_or_rollback_everything()
```

**Code.** `Program` is the authoring surface and its immutable `ProgramGraph` is the compiler/runtime
temporal contract. [`AmrProgramContext`](../include/pops/runtime/program/amr_program_context.hpp)
executes its single-level route transactionally over
[`AmrRuntime`](../include/pops/runtime/amr/amr_runtime.hpp) and fails closed before callbacks for a
multi-level route. The latter exposes spatial hierarchy state and prepared transfer/reflux services
only. The umbrella
[`amr_reflux_mf.hpp`](../include/pops/numerics/time/amr/reflux/amr_reflux_mf.hpp) aggregates those
spatial helpers; it is not an alternate stepper. Their named types live in
[`amr_patch_range.hpp`](../include/pops/numerics/time/amr/levels/amr_patch_range.hpp):
`PatchRange<Dim>` (the exact-ranked parent footprint of a fine patch). Reflux is represented by
the transactional `FaceFluxLedger<Dim>` and metric reconciliation; prepared AMR ghost fill owns
the corresponding coarse/fine coverage and interpolation. `TransferProvider<Dim>` prepares the
inter-level kernels: conservative average-down, MC-limited linear prolongation, explicit constant
injection, and a distinct fifth-order coarse/fine reconstruction that integrates a quartic fitted to
five parent-cell averages. During prepared block materialization, a WENO5 requirement selects this
distinct fifth-order transfer and cannot silently fall back to the linear kernel. That statement is
about the prepared transfer authority; it does not claim that the still-missing multi-level Program
orchestrator has accepted a WENO5 trajectory. Every kernel uses the same exact-rank ratio and treats
an axis of ratio one as an identity axis. Legacy `mesh/refinement.hpp` injection is not selected for
that prepared high-order transfer.

**Constraints / remarks.** Every level transition carries an explicit, possibly anisotropic,
`RefinementRatio<Dim>` and validates alignment before transfer. The order of operations is critical:
coarse/fine fluxes must be captured at the graph-authored stage points; at each catch-up, reflux
precedes `average_down`; a rejected attempt publishes neither state nor flux. Validation:
`test_refinement` (conservative average_down + interpolate), `test_amr_hierarchy` (coarse + nested
fine + ghost interpolation), `test_nd_amr_consumers` (exact-ranked parent footprints and interfaces),
`test_nd_transfer`, `test_amr_transfer_properties` and `test_prepared_amr_ghost_fill`
(quantitative 1D/2D/3D transfer, parent-average conservation and coverage-aware order-two/order-five
coarse/fine interpolation),
`test_nd_flux_ledger` and `test_program_reflux_ledger` (exact Program coefficients and transactional
ledger), `test_amr_program_diffusion` (diffusive flux crossing a coarse/fine interface), and
`test_amr_diagnostics` (mass and drift velocity via the seam reducer). `test_amr_history_ring`
currently pins the honest three-level refusal and zero-mutation guarantee; it is not a positive
multi-level advance proof.

## 18. Multi-patch AMR: coverage-aware reflux, MPI-distributed

**Intuition.** A fine level is not a single box but a set of patches. Two subtleties
follow: (a) at the joint between two neighboring patches (fine-fine interface) one must not reflux, because the
two sides are fine and the balance is already conservative; (b) the correction must go into the right
parent box when the coarse is itself multi-box or distributed across several MPI ranks.

**Formula / discretization.** The multi-patch reflux is the same operator as section 17, but filtered
by the prepared fine-coverage relation. For a bordering coarse cell $\mathbf{i}$ adjacent to the
face of a patch $g$ on axis $a$, the correction is poured only if $\mathbf{i}$ is not shadowed by
any fine patch:

$$U_c(\mathbf{i}) \mathrel{-}= \mathbb{1}\big[\lnot\,\mathrm{covered}(\mathbf{i})\big]\cdot\frac{\bar F_{f,a} - F_{c,a}\,\Delta t}{\Delta x_a}$$

where $\mathrm{covered}(\mathbf{i})$ tests membership in the coarse footprint `PatchRange<Dim>` of any fine
patch. The mask is built on the global BoxArray (all patches, known to all ranks), so independent
of the MPI distribution. The flux register has a global indexing: each rank fills its local
contributions (zero elsewhere), then $\mathrm{buf} \leftarrow \sum_{\text{rangs}} \mathrm{buf}$ by
`all_reduce_sum_inplace`; in serial the all_reduce is the identity, so bit-for-bit identical to the mono-rank.

```
function reflux_multipatch<Dim>(coarse_level, fine_boxarray_global, registers, distribution):
    # coverage on the global box array -> correct under every MPI distribution
    coverage = prepared_cf_schedule(coarse_region, fine_boxarray_global)
        for g in fine_boxarray_global:
            coverage.register(PatchRange<Dim>(g).parent_footprint())

    ledger = TransactionalFaceFluxLedger<Dim>()         # fragments identifies globally
    for patch g OWNED-LOCALLY by this rank:
        for axis a in 0..Dim-1, side in {lower, upper}, tangential index, component k:
            neighbour = coarse_cell_adjacent_to(g, a, side, tangential index)
            if not coverage.covers(neighbour):
                ledger.accumulate(face_fragment(g, a, side, k), local_flux_contribution(g, a, side, k))

    reconcile_metric_reflux_collectively(ledger)       # contribution exacte de tous les rangs

    if coarse REPLICATED (default):
        apply correction locally (chaque rang a la copie complete)
    else: # coarse DE-replique (multi-box reparti)
        parallel_copy correction into the owning coarse box
        average_down zone couverte via mf_average_down_mb / parallel_copy
```

**Code.** The exact-ranked coverage and parent footprints are prepared with the hierarchy, and
[`metric_reflux.hpp`](../include/pops/amr/reflux/metric_reflux.hpp) reconciles a
`TransactionalFaceFluxLedger<Dim>` by authenticated, collectively complete face fragments. The MPI routing of the distributed
coarse goes through `parallel_copy` in
[`mesh/refinement.hpp`](../include/pops/mesh/layout/refinement.hpp) (general redistribution between two MultiFab
on the same domain with different decompositions: local copies via `BoxHash::query`, then
`MPI_Isend`/`MPI_Irecv` jobs enumerated deterministically, tag 1). The replicated coarse fills its
periodic ghosts by `fill_periodic_local` (a purely local self-fold, without an MPI plan). The
prepared hierarchy manifest carries the parent ownership policy into `AmrRuntime`'s spatial
transfer/reflux services. Without that explicit policy, a de-replicated coarse would revert to
replicated routing (`mf_find_box` instead of `parallel_copy`).

**Constraints / remarks.** Without prepared coverage, the fine-fine joint would be refluxed twice, hence
non-conservation: the coverage relation is the central invariant of the correction. Each rank emits only its
owned face fragments; collective ledger reconciliation rejects incomplete or duplicate coverage. Bit-for-bit
reproducibility requires a deterministic enumeration order of the `parallel_copy` jobs (spatial hash on the
source, sorted candidates).
That job schedule is MEMOIZED per layout pair (dst BoxArray/DistributionMapping, src BoxArray/DistributionMapping)
by `CopyScheduleCache` in [`mesh/copy_schedule.hpp`](../include/pops/mesh/layout/copy_schedule.hpp): the cache
lives on the dst `MultiFab` (dropped when regrid move-assigns a fresh dst) but each entry is keyed on a src-layout
fingerprint, so only the live-data copy/pack/MPI/unpack reruns while the enumeration runs once. The schedule
replays in the same order as the inline loops, so the memoized path is bit-identical to a per-call rebuild
(the same design as the intra-level halo `HaloScheduleCache`, section on `fill_boundary`). The replicated-parent
coarse-fine ghost fill likewise replaces its per-cell `mf_find_box` scan with a precomputed dense cell -> local-box
lookup (`MfBoxLookup`): the parent valid boxes are disjoint, so the first-hit the linear scan returned is the only
hit, and the lookup returns the identical box index. The regrid profiler exposes `tag_density`,
`box_hash_rebuilds` and `copy_cache_hits`/`copy_cache_misses` counters for this machinery (ADC-607).
Validation: `test_amr_spatial_parity` (the spatial core of the AMR path is identical to that of `System`:
same primitive reconstruction, same HLLC/Roe flux), `test_mpi_mbox_parity` (residual invariant to the box
splitting AND to the number of ranks np = 1/2/4, dmax = 0), `test_mpi_amr_distributed_coarse` (distributed coarse
identical to the replicated coarse bit-for-bit, np = 1/2/4).

## 19. Berger-Rigoutsos clustering and regrid

**Intuition.** Given the tagged cells (strong gradient, or any physical predicate), find a
small number of rectangular boxes that cover them without too much waste. The algorithm cuts
recursively a region where the signature (histogram of tags projected onto an axis) presents a hole
(empty column), otherwise an inflection (extremum of the change of Laplacian of the signature), otherwise at the middle.

**Formula / discretization.** For a region $R$, we define the efficiency $\eta = N_{\mathrm{tag}}(R) / |R|$
(fraction of tagged cells). We accept $R$ as a box if $\eta \ge \eta_{\min}$
(`min_efficiency`, default $0.7$) or if $R$ is not splittable. Otherwise we cut. The signature on axis
$a$ is the projection $s_a[k] = \sum_{\text{ligne/col } k} \mathrm{tag}$. A hole is an interior index
$k \in [\mathrm{mb}, \mathrm{len}-\mathrm{mb}]$ with $s_a[k] = 0$, the closest to the center. Failing that, we
take the inflection: discrete Laplacian $D[k] = s[k+1] - 2 s[k] + s[k-1]$, and we cut at the index that
maximizes $|D[k] - D[k-1]|$. Failing that again, we cut at the middle of the largest splittable dimension.
The splittability criterion is $n_a \ge 2\,\mathrm{mb}$ with $\mathrm{mb} = \max(1, b)$ with $b$ = `min_box_size`.
After acceptance, each raw box is chopped into sub-boxes of side $\le$ `max_box_size`. At the regrid,
the coarse boxes are refined by `refine(ref_ratio)` in the index space of the fine level.

```
function berger_rigoutsos(tags, params):
    raw = []
    cluster_rec(tags, tags.box, params, raw)
    result = []
    for b in raw:
        result += chop(b, params.max_box_size)         # BoxArray::from_domain
    return result

function cluster_rec(tags, region, p, out):
    region = bounding_box_of_tags(region)               # trim
    if region empty: return
    eff = count_tags(region) / num_cells(region)
    mb  = max(1, p.min_box_size)
    sx  = region.nx >= 2*mb ;  sy = region.ny >= 2*mb
    if eff >= p.min_efficiency or (not sx and not sy):
        out.push(region) ; return                       # accepte
    Sx = signature(region, axis=0) ; Sy = signature(region, axis=1)
    hx = sx ? best_hole(Sx, mb) : -1                    # trou interieur le plus central
    hy = sy ? best_hole(Sy, mb) : -1
    choose (axis, kcut):
        both holes  -> couper l'axe le plus long
        one hole    -> cet axe
        no hole     -> best_inflection (max |Laplacien'|), score le plus fort
        no infl.    -> milieu de la plus grande dim splittable
    (left, right) = split(region, axis, kcut)
    cluster_rec(tags, left, p, out)
    cluster_rec(tags, right, p, out)

function regrid_level(hierarchy, coarse_lev, crit, params):
    tags  = tag_cells(data(coarse_lev), domain, crit)   # predicat (Array4, i, j) -> bool
    grown = grow_tags(tags, n_buffer, domain)           # dilatation (nesting + buffer)
    if grown.count() == 0:
        clear_above(coarse_lev) ; return                # plus rien a raffiner
    # MPI : OU global des tags avant clustering, sinon BoxArray fin diverge par rang
    all_reduce_or(grown)                                 # (grossier reparti)
    cboxes = berger_rigoutsos(grown, params.cluster)
    fboxes = [ b.refine(ref_ratio) for b in cboxes ]
    newfine = MultiFab(fboxes, DistributionMapping, ncomp, n_grow)
    interpolate(data(coarse_lev), newfine, ref_ratio)   # injection grossier -> fin
    if niveau fin existant: parallel_copy(newfine, data(coarse_lev+1))  # preserver l'ancien fin
    install_level(coarse_lev+1, fba, newfine)
```

**Code.** The ranked clustering contract is
[`ClusterProvider<Dim>`](../include/pops/amr/tagging/clustering_provider.hpp), with the deterministic
Berger--Rigoutsos implementation in
[`berger_rigoutsos.hpp`](../include/pops/amr/tagging/berger_rigoutsos.hpp). Its result authenticates
the source `LevelLayout<Dim>`, options, canonical tag shards, and output boxes. The transaction
[`prepare_regrid`](../include/pops/amr/regridding/regrid.hpp) refines every axis through one
`RefinementRatio<Dim>`, prepares ownership with the bound `PreparedLoadBalanceAuthority<Dim>`, and
retains that complete ownership proof beside the child layout. Publication occurs only through
`AmrRuntime<Dim>::publish_regrid`; an authenticated empty cluster result removes the child and all
finer levels. Under MPI, tags must be canonicalized collectively before clustering so every rank
prepares the same exact regrid contract.

**Constraints / remarks.** The clustering is pure, sequential, without physics nor MPI: it consumes a
`TagBox` already gathered. The proper nesting (each fine patch strictly interior to the parent
coverage) relies on the dilation `grow_tags` (radius `n_buffer`) and must be guaranteed after the clustering,
otherwise the inter-level ghost-fill reads outside the parent coverage. The tagging predicate is agnostic
of the physics; for a gradient criterion the caller fills the ghosts beforehand. The signature pushes the
cuts toward the real geometric holes: a full block gives 1 box, two blocks separated by an empty
band give 2 boxes. Validation: `test_cluster` (full block -> 1 box, two separated blocks -> 2 boxes,
large block chopped by `max_box_size`), `test_regrid` (a fine level is created around the tagged region,
fine data interpolated from the coarse).

**Export of the patch geometry.** The fine `BoxArray` resulting from the clustering is exposed to Python by
`AmrSystem.patch_boxes()` (a list of `(level, ilo, jlo, ihi, jhi)`, inclusive corners in the index space
of the level, ratio 2) and the facade `AmrSystem.patch_rectangles()` (conversion into physical rectangles
`(x0, y0, w, h)` on `[0, L]^2`). Same source as `n_patches()` (the same global `box_array()`, so
rank-independent and MPI-safe); it is a read between the steps, with no cost on the hot path. Wired
on both spatial stores (mono-block `AmrCouplerMP` and multi-block `AmrRuntime`). Allows tracing the real
patches (for example a GIF of the refinement) without rebuilding a proxy. Validation:
`test_amr_patch_boxes` (cardinality equal to `n_patches`, corners consistent in index and in physics, mono and
multi-block).


---

## 20. Distributed mesh: global BoxArray, halos, load balancing

**Intuition.** Distributed AMR requires that all ranks know all the boxes (global
BoxArray) to compute the multi-patch coverage and enumerate the halo exchanges in a
deterministic way, but that each rank allocates only its local fabs (via the `DistributionMapping`).
The halo exchange fills the parallel ghosts; the load balancing distributes the boxes across the ranks.

**Formula / discretization.** The tiling `from_domain(domain, m)` splits each axis `[lo, hi]`
(length `len = hi - lo + 1`) into `n = ceil(len / m)` segments distributed as evenly as possible:
the first `len mod n` segments have `floor(len/n) + 1` cells, the others `floor(len/n)` (no
greedy tail). The global box index is its position in the `y` outer / `x` inner order,
identical on all ranks.

The load balancing minimizes the imbalance `max_r charge(r) / moyenne(charge)`, the load of a box
being its number of cells (cost proxy). Two strategies: Z-order (Morton curve, contiguous
segments of target load `total / nranks` -> spatial locality) and knapsack/lpt (the heaviest box
to the least loaded rank -> minimal maximal imbalance, without locality). The Morton key interleaves
the bits of `(x, y)` brought back to the origin of the bounding box:

$$\mathrm{morton}(x, y) = \sum_{b\ge 0} \big(x_b\,4^b + y_b\,2\cdot 4^b\big),\qquad
  \mathrm{cible}_r = \frac{\text{total cellules}}{\text{nranks}}$$

The halo exchange fills the ghost `D(i,j)` of a fab from the valid cell shifted
`S(i - s_x, j - s_y)` of the neighbor, where `(s_x, s_y)` ranges over
`{0, +-L_x} x {0, +-L_y}` (the periodic shifts are active only if the direction is periodic).

```
function from_domain(domain, m):                      # BoxArray::from_domain
    sx = split_range(domain.lo[0], domain.hi[0], m)   # segments en x
    sy = split_range(domain.lo[1], domain.hi[1], m)   # segments en y
    boxes = []
    for (ylo, yhi) in sy:        # y externe, x interne -> ordre deterministe (= indice global)
        for (xlo, xhi) in sx:
            boxes.push( Box2D{{xlo, ylo}, {xhi, yhi}} )
    return BoxArray(boxes)

function split_range(lo, hi, m):
    len = hi - lo + 1 ; n = ceil(len / m)
    base = len div n ; rem = len mod n ; cur = lo
    for k in 0..n-1:
        l = base + (1 if k < rem else 0)              # rem premiers segments +1
        emit (cur, cur + l - 1) ; cur += l

function make_sfc_distribution(ba, nranks):           # load_balance.hpp (Z-order)
    order = argsort boxes by morton_key(lo - bounding_box.lo)
    target = ba.num_cells() / nranks ; acc = 0 ; r = 0 ; rank[*] = 0
    for k, b in enumerate(order):
        rank[b] = r ; acc += ba[b].num_cells()
        if r < nranks-1 and acc >= target*(r+1) and boxes_left >= ranks_left:
            r += 1                                    # garantit >=1 box/rang
    return DistributionMapping(rank)

function fill_boundary_begin(mf, domain, per):        # halos, 2 phases
    shifts = product({0,+-Lx if per.x}, {0,+-Ly if per.y})
    hash = BoxHash(mf.box_array())                    # accelere la recherche de voisins
    for li in local fabs of mf:                       # --- copies locales ---
        gbox = grow(fab(li).box, ng)
        for (sx, sy) in shifts:
            for gB in hash.query( gbox shifted by (-sx,-sy) ):
                if gB local: copy_shifted(fab(li), fab(gB), region, sx, sy)
    if n_ranks() <= 1: return
    for gF in 0..ba.size()-1:                          # --- enum globale deterministe ---
        for (sx, sy) in shifts:
            for gB in hash.query(...):
                if owner(gF)==me xor owner(gB)==me:    # une extremite locale
                    classer en send[owner(gF)] ou recv[owner(gB)]
    pack(send buffers via for_each PackKernel) ; device_fence()
    post MPI_Isend / MPI_Irecv (tag 0, pointeurs unifies -> GPUDirect si CUDA-aware)

function fill_boundary_end(mf, h):
    MPI_Waitall(h.reqs)
    unpack(recv buffers via for_each UnpackKernel) -> ghosts
```

**Code.** [`mesh/box_array.hpp`](../include/pops/mesh/layout/box_array.hpp)
(`BoxArray<Dim>::from_domain`, the vector order is the box identity);
[`mesh/distribution.hpp`](../include/pops/mesh/layout/distribution.hpp)
(`Distribution<Dim>`, exact ownership over a `RankSpace<Dim>`, with explicit partitioned or
replicated mode);
[`mesh/multifab.hpp`](../include/pops/mesh/storage/multifab.hpp) (`MultiFab<Dim>` allocates only its
local ranked Fabs and preserves the global-to-local box identity);
[`mesh/fill_boundary.hpp`](../include/pops/mesh/boundary/fill_boundary.hpp) (`fill_boundary_begin` /
`fill_boundary_end` non-blocking + `fill_boundary` blocking, `HaloExchange` owns the buffers and
`MPI_Request`, kernels `CopyShiftedKernel` / `PackKernel` / `UnpackKernel` device-clean);
[`parallel/load_balance.hpp`](../include/pops/parallel/load_balance.hpp) (`morton_key`,
`make_sfc_distribution`, `make_knapsack_distribution`, `load_imbalance`);
[`parallel/comm.hpp`](../include/pops/parallel/comm.hpp) wraps the collectives
(`all_reduce_sum`, `all_reduce_sum_inplace` for the reflux, `all_reduce_or_inplace` for the union of the
regrid tags) and degenerates cleanly in serial.

**Constraints / remarks.** The metadata (BoxArray + DistributionMapping) are replicated on
all ranks: this is what makes the enumeration of the halo jobs deterministic, so the buffers
`sbuf[A->B]` and `rbuf[B<-A]` align without negotiating sizes. If MPI is not initialized,
`my_rank()` returns 0 and `n_ranks()` returns 1 (a serial test linked against the MPI lib does not break), hence
`comm_init()` required at the beginning of `main()` for a real distributed run. The out-of-domain ghosts
without periodicity are not touched by `fill_boundary` (they are the physical BCs,
`physical_bc.hpp`). The buffers live in unified memory and are passed as-is to MPI (host
bounce avoided if the MPI stack is CUDA-aware); a `device_fence()` separates the pack from the `Isend`.
**Validation.** `test_box_array`, `test_multifab`, `test_load_balance`; under MPI:
`test_mpi_fillboundary` (halo exchange), `test_mpi_poisson` (distributed Poisson),
`test_mpi_fft_distributed` (FFT by bands), `test_mpi_redistribute`, `test_mpi_array_reduce`,
`test_mpi_coupler_inject` (np=4, results bit-identical to np=1/2/4).

## 21. Exact auxiliary provider graph

**Intuition.** Fields consumed by a flux, source, boundary, or field solve are identified by
`ComponentKey(owner_qid, space_kind, space_name, component)`. The runtime assigns storage addresses
only after all packages have registered their outputs and consumer requirements. Two components with
the same short name but different owners or spaces therefore never alias.

**Formula / discretization.** For a consumer requiring $N$ scalar providers, resolution produces a
dense local plan

$$s \in [0,N) \longmapsto (g_s,c_s),$$

where $g_s$ is an exact storage group and $c_s$ a component in that group. The device kernel receives
`ProviderValues<N>` through `ProviderStorageView<Dim,N>`; $N=0$ is a real empty pack and allocates no
dummy component. Gradient components are declared in native-axis order, while magnetic or thermal
quantities are ordinary physics-owned keys rather than reserved runtime slots.

**Code.** [`exact_aux_registry.hpp`](../include/pops/runtime/system/exact_aux_registry.hpp) owns the
provider DAG, contracts, generations and accepted/candidate publication.
[`provider_storage_binding.hpp`](../include/pops/runtime/system/provider_storage_binding.hpp) binds
each consumer slot to its resolved group address, and
[`flux_interfaces.hpp`](../include/pops/numerics/fv/flux_interfaces.hpp) carries compact provider
values into device-callable physics. Generated packages register routes before the global registry
is sealed; native blocks are installed only after that seal.

**Constraints / remarks.** Registration rejects duplicate producers, missing dependencies, cycles,
contract or rank mismatches, and stale generations before mutation. Inputs and derived outputs are
staged into candidates, halos and physical boundaries are filled through prepared exact-ranked
transports, and publication is collective and transactional. A local provider failure on one MPI
rank rejects the same candidate on every rank. **Validation.** `test_exact_aux_registry_nd` covers
zero/N providers, multiple storage groups, permutations, transactions and 1D/2D/3D boundaries;
`test_mpi_auxiliary_ghost_fill` covers remote peers and empty ranks; `test_aux_single_source` and the
Python provider-pack suites cover physics/code-generation binding without physical slot names.

## 22. Runtime composition and multi-species system

**Intuition.** Python composes what (one block per species: composite model + spatial scheme + temporal
policy), C++ computes per cell. N species interact in the elliptic right-hand side
(`f = sum_s q_s n_s`) and in the inter-species source, never in the flux: a block's flux only
sees its own state and the exact provider values declared by that consumer.

**Formula / discretization.** At each authored field stage, the Program solves the shared Poisson
whose right-hand side is the co-located sum of the elliptic bricks of all selected blocks and
publishes its owner-qualified potential/gradient outputs. The same Program explicitly orders each transport, source,
coupling and commit:

$$f_{ij} = \sum_{b} \mathrm{elliptic\_rhs}_b(U^b_{ij}),\qquad
  \frac{dU^b}{dt} = -\,\mathrm{div}\,F_b(U^b, \mathrm{aux}) + S_b(U^b, \mathrm{aux})$$

A model is assembled by `dispatch_model(spec, visitor)`: it builds the transport brick
(`exb` / `compressible` / `isothermal`), the source brick (`none` / `potential` / `gravity` /
`magnetic` / `potential_magnetic`, the fluid sources requiring `NV >= 3`), the elliptic brick
(`charge` / `background` / `gravity`), combines them into `CompositeModel<TR, Src, Ell>` and calls
`visitor(model)`. The core names no scenario; a scenario is this composition, named on the
`adc_cases` side.

The `exb` route is `CartesianExBDrift`: a scalar Cartesian transport for native dimensions 1, 2
and 3. It consumes one explicit potential-gradient provider per native axis plus the three
Cartesian magnetic providers, and the single Levi-Civita kernel evaluates
\(v_i = \epsilon_{ijk} B_j\,\partial_k\phi/|B|^2\). The exact-ranked gradient is embedded in
Cartesian 3-space: 3-D consumes every component, 2-D can explicitly provide its normal field,
and 1-D obtains the mathematical longitudinal projection without a separate branch.

```
function dispatch_model(spec, visitor):              # model_factory.hpp
    dispatch_transport(spec, TR ->                    # exb | compressible | isothermal
      dispatch_source<TR::n_vars>(spec, Src ->        # none | potential | gravity | (potential_)magnetic
        dispatch_elliptic(spec, Ell ->                # charge | background | gravity
          visitor( CompositeModel<TR, Src, Ell>{TR, Src, Ell} ))))
    # combinaison invalide (source fluide sur transport scalaire) -> throw

Program.step_strategy(controller):                   # authoring, before compile
    declare the typed dt/CFL/error controller
    declare provisional stores and acceptance guards

pops.run(bound_instance, t_end=..., max_steps=...): # sole execution transition
    authenticate the instance produced by pops.bind
    execute the compiled Program transaction
    publish states, fields and consumers only after accepted attempts
    return immutable RunReport(accepted/rejected attempts, final clock, stop reason, identities)
    # max_steps exhaustion or any terminal failure raises; no successful report is fabricated
```

**Code.** [`runtime/system.hpp`](../include/pops/runtime/system.hpp) and
[`runtime/amr_system.hpp`](../include/pops/runtime/amr_system.hpp) are private native executors
materialized from one resolved `Case`; neither is an authoring API. `RuntimeInstance` installs
qualified blocks, field plans, the temporal graph, layout authorities and consumer graph from the
compiled artifact in one authenticated transaction. The single-level executor shares Poisson across
the selected blocks; the adaptive executor runs the same graph over a shared hierarchy (same BoxArray,
exact `Distribution<Dim>` and geometry per level), performs coarse co-located
Poisson assembly, and conservatively transfers/refluxes every declared state. Inter-species sources
are typed component-interface implementations in the graph; the bytecode and native registration
calls remain internal lowering details. On the coupling side,
`coupling/system/amr_system_coupler.hpp` authenticates only shared exact-ranked AMR topology; it
owns neither physics assembly nor a time scheme.
[`runtime/model_factory.hpp`](../include/pops/runtime/builders/factory/model_factory.hpp):
`dispatch_model` / `dispatch_transport` / `dispatch_source` / `dispatch_elliptic` assemble a
`CompositeModel` from a `ModelSpec` (the core names no scenario).

**Constraints / remarks.** The blocks share an aux and a Poisson; the coupling between species
goes through the elliptic right-hand side (sum) and the coupled sources, not through the flux. An
inter-species coupling is a TYPED operator (`CouplingOperator`,
[`include/pops/coupling/source/coupling_operator.hpp`](../include/pops/coupling/source/coupling_operator.hpp)):
it carries a DECLARED conservation contract (conserved versus created roles) validated at registration,
so the Program applies a term-set whose invariants are machine-checked rather than trusted;
ionization is declared NON-conservative in density (it net-sources an electron/ion pair) while collision
conserves momentum and thermal exchange conserves energy. The named couplings are presets lowering to
this one representation, inspectable read-only through `coupled_operators()`. The production facade
applies `substeps` and `stride` only to the whole installed Program. Per-block runtime-adaptive
cadence is unsupported until its explicit `ProgramGraph` lowering exists; only the test oracle
retains its historical formula.
In multi-block AMR, `regrid_every > 0` is supported (the union-tag regrid rebuilds the hierarchy from all blocks' tags; `regrid_every == 0` keeps it frozen)
and `set_conservative_state` accepts a complete block-qualified conservative state for every native
or deferred compiled (`.so`) block. Without an explicit IMEX mask
(`implicit_vars` / `implicit_roles` empty), an explicitly authored typed implicit
Program primitive uses the model's component-selection default. The empty mask, an
`IMEXTime` policy, or `time="imex"` never creates or schedules that primitive.
**Validation.** `test_system_abstraction` and `test_coupled_source` cover the exact-ranked static
contracts; `test_coupling_operator_contract`, `test_program_runtime`,
`test_generated_amr_system_block`, and `test_variable_role` cover the installed multi-block runtime.

## 23. Symbolic DSL and authenticated native components

**Intuition.** The Python DSL describes physics and Programs as immutable symbolic data. Resolution
selects small component interfaces and an exact target; compilation specializes that resolved plan
into a native artifact. There is one production extension contract: the artifact must authenticate
its manifest, binary, target and generated interface tables before any runtime state is mutated.
Host-callback and flat-array model ABIs are not alternate execution modes.

**Formula / discretization.** Code generation still emits the same local formulas and CSE as the
builtin path. Authentication does not change the numerical operator: a selected numerical flux is
called on two qualified `FaceTrace` values, the spatial operator applies face measures and divergence,
and a Program orders the resulting rates. What crosses the extension boundary is a versioned POD
request/table protocol, not a C++ object hierarchy or an algorithm name.

```
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=target.layout)
compiled = pops.compile(resolved)             # exact platform + component identities
instance = pops.bind(compiled, params=...)    # atomic installation into RuntimeInstance
report = pops.run(instance, t_end=..., max_steps=...)

function load_component(binary, expected, execution):
    verify binary identity and complete ComponentManifest identity
    api = resolve("pops_component_interface_v1")
    verify catalog digest, interface ids, versions and complete table sizes
    prepared = api.prepare(parameters, exact_target, execution)
    require prepared.status == CONTINUE
    return owned handle + prepared resources         # binary remains loaded while callable
```

Each hot-path interface is resolved and prepared once. Calls then use its typed bulk request and an
explicit execution context (dimension, scalar and precision, memory space, backend/device, stream and
communicator identities). An unsupported target or a `retry` / `reject` / `failed` outcome propagates
as a declared transaction action; it is never converted to a neutral numerical value.

**Code.** The public package lifecycle is documented in
[`design/external-component-packages.md`](design/external-component-packages.md). The generated C/POD
tables live in
[`runtime/generated_component_abi.hpp`](../include/pops/runtime/config/generated_component_abi.hpp),
and [`runtime/component_loader.hpp`](../include/pops/runtime/dynamic/component_loader.hpp) authenticates
the expected component/catalog identities, exact interfaces and execution context before preparing
resources. Interface consumers are independent; adding a writer, boundary, tagging, transfer, reflux,
field-solve or numerical-flux implementation does not add a central scientific switch.

A generated whole-block artifact still reaches `System` or `AmrSystem` through the private binding
seam named `_install_native_block`. It is owned by `pops.bind` and the resulting `RuntimeInstance`, not
by public authoring. [`runtime/native_loader.hpp`](../include/pops/runtime/builders/compiled/native_loader.hpp)
checks the ABI key, route manifest, parameter count and required installer symbol before generated
code calls [`runtime/dsl_block.hpp`](../include/pops/runtime/builders/compiled/dsl_block.hpp) on the real
grid context. This is an internal specialization of the same resolved plan, not a second user-facing
registration API.

External numerical-flux packages execute through the exact-ranked fields and geometry carried by
[`runtime/external_riemann_brick.hpp`](../include/pops/runtime/program/external_riemann_brick.hpp).
There is no local square-grid or flat-array adapter: the resolved native rank, patch layout and
distribution remain authoritative across the component boundary.

**Constraints / remarks.** A shared-object path alone proves nothing. Installation refuses a missing
or unexpected symbol, digest, catalog version, interface, table prefix, platform identity or execution
context before publishing the component. Prepared resources retain their owning handle and are tied
to one exact execution identity. The runtime freezes registrations after bind, and output/effect
components publish only for an accepted transaction.

**Validation.** `test_amr_native_loader` covers authenticated interface discovery, preparation,
table truncation and forged/missing declarations. `test_external_component_package` covers package,
binary and symbol authentication; `test_external_interface_backend` locks generated interface-table
selection. `test_multi_layout_runtime` and `test_shared_interface_runtime` exercise installed
components through the runtime layouts rather than an alternate engine. The existing
device/MPI suites validate the named-functor kernels used by generated native blocks.

## 24. The dispatch seam (Kokkos: Serial / OpenMP / Cuda / MPI)

**Intuition.** Not a numerical algorithm but the switch point that makes them all portable.
`for_each_cell(box, f)` dispatches the loop over the cells of a `Box2D` to Kokkos, the only on-node
backend; the execution space (Serial sequential, OpenMP multi-thread, Cuda/HIP GPU) is chosen AT
THE INSTALLATION OF KOKKOS, not by an pops flag. The operators (assemble_rhs, V-cycle, couplers)
never see the execution space and no CUDA kernel is hand-written. Detail in
[ARCHITECTURE.md](ARCHITECTURE.md) section 4 (execution layer).

**Formula / discretization.** The functor `f(i, j)` is taken by value and captures only
`Array4` handles (POD), never the `Fab` nor anything virtual: exactly the constraint of a device
kernel. It always becomes `Kokkos::parallel_for(MDRangePolicy<Rank<2>, IndexType<int>>)` (signed
indices for the ghost boxes with negative bounds), instantiated for the Kokkos execution space
chosen at install (Serial, OpenMP or Cuda/HIP); no `#pragma omp` nor hand-written double loop
is a production path. Bit-identity: `for_each_cell` has no
inter-iteration dependency (each `f(i,j)` writes the single cell `(i,j)` and reads cells it does
not write in the same call: red-black GS smoother, residual/restriction/prolongation write
a distinct destination), so the result is independent of the order. The reductions carry a
FP choice: the `Kokkos::Sum` sum re-associates the addition per tile (non-associative in IEEE754), so
`sum` is deterministic/idempotent (same data, same Kokkos space -> same bits) but is NOT
bit-identical to a hand-written lexicographic sum, and this holds for ALL spaces (Serial,
OpenMP, Cuda) since there is only a single Kokkos path. The max is exact everywhere (associative/commutative,
without roundoff). A threshold `POPS_FOREACH_SERIAL_THRESHOLD` (default 4096 cells) switches to a small
sequential host loop (an optimization INTERNAL to the Kokkos path, not a separate backend) for the
small boxes (coarse V-cycle levels ~2x2..32x32) where the fork/join would crush the computation, but
only if the default Kokkos execution space is the host space (`if constexpr`: on device,
parallel_for whatever the size, otherwise a data race).

```
function for_each_cell(box b, f):                    # for_each.hpp  (#error sans POPS_HAS_KOKKOS)
    if DefaultExecSpace == DefaultHostExecSpace:     # if constexpr (espace Kokkos hote)
        if (b.nx * b.ny) < foreach_serial_threshold():
            for j in b: for i in b: f(i, j)          # petite boucle hote, INTERNE au chemin Kokkos
            return
    Kokkos::parallel_for( MDRangePolicy<Rank<2>, IndexType<int>>(lo, hi+1), f )  # Serial/OpenMP/Cuda

function for_each_cell_reduce_sum(b, f):             # Kokkos::Sum deterministe par tuile
    Kokkos::parallel_reduce(..., acc += f(i,j), Sum<Real>)   # reassocie : non bit-id a une somme lexicographique

function sync_host():  device_fence()                # avant un acces hote (memoire unifiee)
function sync_device(): pass                          # no-op sous SharedSpace (scaffolding)
```

**Code.** [`mesh/for_each.hpp`](../include/pops/mesh/execution/for_each.hpp): `for_each_cell` (`Kokkos::parallel_for`
on the execution space chosen at install, `#error` without `POPS_HAS_KOKKOS`, guard `if constexpr` device,
threshold `foreach_serial_threshold` for the internal small host loop),
`for_each_cell_reduce_sum` / `_max` (reducers `Kokkos::Sum` / `Max` deterministic), the variants
with a reducer functor `reduce_sum_cell` / `reduce_max_cell` (passed directly to `parallel_reduce`
without a wrapper lambda, a device-clean cross-TU path for a Model-template kernel), and the
coherence seam `sync_host()` (= targeted `device_fence()`) / `sync_device()` (no-op under unified memory).
The fabs and the reduction `sum(MultiFab)` (all-reduce on all ranks) live in
[`mesh/multifab.hpp`](../include/pops/mesh/storage/multifab.hpp). The MPI collectives are wrapped in
[`parallel/comm.hpp`](../include/pops/parallel/comm.hpp) (`all_reduce_sum`, `all_reduce_max`,
`all_reduce_sum_inplace`, `all_reduce_or_inplace`, `barrier`, `comm_init` / `comm_finalize`), which
degenerate into the serial identity.

**Constraints / remarks.** The CPU -> GPU switch changes no call site: one changes the
Kokkos execution space at install, the physics stays unchanged. The functor must be device-callable
under Kokkos (annotated `POPS_HD`, POD captured by value); capturing an object with a vtable or a `Fab`
breaks the device. The switch to the small host loop of the threshold is safe only under a host Kokkos space
(the `if constexpr` evaporates on device, zero overhead, the GPU path strictly unchanged). GPU
discipline: `device_fence()` (via `sync_host`) between a device kernel and a host loop on the same unified
memory, otherwise a host-write / kernel race (cf. CHOICES.md). **Validation.** The seam is exercised
transversally by the whole suite; specifically the MPI tests of section 20
(`test_mpi_fillboundary`, `test_mpi_poisson`, `test_mpi_array_reduce`, np=1/2/4 bit-identical) and the
GH200 device validations (GPU_RUNTIME_PORT.md) which confirm that the Kokkos Serial, OpenMP and
Cuda spaces give the same results (up to the FP choice of the Kokkos sum, documented).


---

## 25. Capabilities to qualify (present but limited, or off master)

What exists with a restricted scope, or what is written/designed without being on `master` as of the date
of this page. The goal is not to present a partial capability as complete.

- GaussPolicy restart/evolve. An experimental policy (re-imposing Gauss at each step, or keeping the
  `phi` evolved by Schur) on a branch (PR #237); the associated experiment is discarded. Not on
  `master`.
- Condensed implicit Program on AMR. A hierarchy-scoped `LinearProblem` gathers coefficients, RHS and
  the authored initial guess on every level, then an executable Krylov solver with an explicit
  `CompositeTensorFAC` provider performs one composite solve before reconstruction on every level.
  The current `AmrTensorElliptic` provider accepts nested ratio-2 hierarchies with a replicated
  mono-box coarse level on one MPI rank. Its FAC backend supports equal tensor diagonals plus cross
  terms; unequal `eps_x`/`eps_y`, multilevel MPI and multi-block Program scope return an explicit
  capability failure. The Program solve/provider protocol itself is not tied to these limitations.
- FFT layout/capability. `PoissonFFTSolver<Dim>` supports serial and MPI ordered slabs for Cartesian
  native ranks 1, 2 and 3. Non-canonical decompositions fail closed; radix-2 extents use the fast
  path and other extents use the diagnosed direct-DFT path. `CartesianCG` remains the uniform
  non-periodic or variable-topology alternative.
- Polar Poisson. `PolarPoissonSolver` (FFT in theta, Thomas in r, section 16) is mono-rank and
  mono-box. The polar tensor/Krylov path (polar Schur) lifts this limit on its perimeter.
- Cut-cell and Hoffart fidelity. The cut-cell (sections 14, 15) is a numerical capability of the core; it
  is not presented as a proven correction of the growth rates of the Hoffart benchmark.
- Energy under condensed implicit integration. The authored method in section 13 adjusts kinetic energy if an `Energy` role is
  declared; the isothermal case does not use the energy equation.

---

## Which scheme or solver when

| Probleme | Choix | Pourquoi |
|---|---|---|
| general hyperbolic transport | Godunov finite volumes + Rusanov flux | robust, works for any equation (section 1, 2) |
| compressible Euler with shocks | primitive reconstruction + HLLC or Roe | resolves the contact, less diffusive than Rusanov (section 2, 3) |
| smooth zones, high precision | WENO5-Z + SSPRK3 | order 5, low dissipation (section 3, 4) |
| stiff source (Lorentz, relaxation) | local IMEX, or global Schur condensation | implicit, no exploding time step (section 5, 13) |
| periodic Poisson, $n = 2^k$ | `poisson_fft_solver` | direct, $O(N \log N)$ (section 10) |
| uniform constant-coefficient Cartesian Poisson | `CartesianCG` | exact-ranked 1D/2D/3D CG; static periodic/Dirichlet/Neumann BC (section 12) |
| AMR constant-scalar Poisson / reaction | `GeometricMG` + FAC | genuine exact-ranked multigrid hierarchy, not an alias for uniform CG (section 9, 11) |
| full-tensor Cartesian operator | prepared GMRES or BiCGStab with an explicit provider | generic, no matrix assembly (section 12) |
| full-tensor polar operator | dedicated metric-aware polar Krylov solver | polar measure and radial line preconditioner (section 16) |
| localized feature (front, ring) | structured `pops.layouts.AMR` descriptor | adaptive refinement, conservative reflux (section 17 to 19) |
| inter-species sources | `CoupledSource` bytecode as a typed `CouplingOperator` | declared conservation contract, validated at registration (section 22) |
| non-rectangular domain | EB cut-cell (disc) or polar ring | curved boundary without staircase (section 14 to 16) |

## References

- Finite volumes and Riemann fluxes: LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge, 2002. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009
  (HLLC, Roe).
- WENO reconstruction: Jiang & Shu, *Efficient Implementation of Weighted ENO Schemes*, JCP 126
  (1996). WENO-Z: Borges et al., JCP 227 (2008).
- SSP integration: Gottlieb, Shu, Tadmor, *Strong Stability-Preserving Time Discretization Methods*,
  SIAM Review 43 (2001).
- IMEX: Ascher, Ruuth, Spiteri, *Implicit-explicit Runge-Kutta methods*, Appl. Numer. Math. 25 (1997).
- Multigrid: Briggs, Henson, McCormick, *A Multigrid Tutorial*, SIAM, 2000.
- Krylov: Saad, *Iterative Methods for Sparse Linear Systems*, SIAM, 2003 (BiCGStab, section 7).
- AMR: Berger & Oliger, *Adaptive mesh refinement for hyperbolic partial differential equations*, JCP
  53 (1984). Berger & Colella, *Local adaptive mesh refinement for shock hydrodynamics*, JCP 82 (1989).
  Berger & Rigoutsos, *An algorithm for point clustering and grid generation*, IEEE Trans. SMC 21 (1991).
- Cut-cell: Shortley & Weller, *The numerical solution of Laplace's equation*, J. Appl. Phys. 9 (1938).
- Condensed Schur (magnetized Euler-Poisson): implicit source coupling potential / velocity update.
