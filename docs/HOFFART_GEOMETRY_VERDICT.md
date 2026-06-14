# Geometry experiment verdict (June 2026) -- the cut-cell does NOT recover the rate

Discriminating experiment on ROMEO GH200 (job 647507), COMPLETE system-schur model,
n=256, t_end=2.0, paper windows, l=3,4,5, three transport geometries:
square (square box, default), staircase (disk mask 0/1 at R=16), cutcell
(embedded-boundary aperture+kappa at R=16).

## Raw result (rate gamma_numeric, raw, paper windows)

| mode | square | staircase | cutcell | paper | error |
|------|--------|-----------|---------|--------|--------|
| 3 | 0.037182 | 0.037182 | 0.037182 | 0.772 | -95.2 % |
| 4 | 0.048897 | 0.048897 | 0.048897 | 0.911 | -94.6 % |
| 5 | 0.121080 | 0.121080 | 0.121080 | 0.683 | -82.3 % |

The three geometries give the SAME rate (differences ~1e-11 = machine rounding;
the mask/EB is indeed ACTIVE but physically without effect). DIRECT CONCLUSION:
the domain mask/cut-cell at the outer boundary R=16 DOES NOT CHANGE the diocotron rate.

## Why the cut-cell-at-R was the wrong tool

The diocotron instability lives on the ring r0=6 / r1=8, DEEPLY inside the
domain R=16. The disk mask acts at the outer boundary R: it only touches the
corners (radius > 16), which carry rho_min and are dynamically inert. The Poisson
wall (Dirichlet on the circle R=16) already imposes the disk for phi. So
confining transport to ||x|| < 16 changes nothing in the ring dynamics.

## The point that REORIENTS the diagnostic: RESOLUTION-INDEPENDENT deficit

The complete model deficit is ~ -95 % at n=256 AND at n=384 (nearly identical). A
ring-edge DIFFUSION (smoothing of the sharp ring by the cartesian grid) would
DECREASE with resolution (smaller cell => less smoothed ring). The deficit does
NOT move with n. So:

- this is NOT (only) a ring-edge diffusion resolvable by larger n (the "high-res
  cartesian n=768/1024" route will probably not suffice);
- this is NOT the outer-boundary geometry (cut-cell without effect);
- so it is very probably STRUCTURAL: normalization / time scale / coupling of the
  complete system-schur path. The raw rate ~0.037 is a PLATEAU.

## Contrast with the REDUCED model (which does recover)

The REDUCED model (scalar ExB drift):
- on POLAR grid: l=4 EXACT (0.913 vs 0.911), l=3/5 close (diag_polar_omega);
- on CARTESIAN grid: -5 to -27 % at n=192 minmod, and IMPROVES with order/resolution
  (WENO5 sweep) -> classical diffusion behavior, resolution-DEPENDENT.

The complete model (resolution-independent at -95 %) behaves DIFFERENTLY from the reduced.
The raw factor between reduced-polar (0.155) and complete-cartesian (0.037) is ~4x,
close to the ring-diffusion factor invoked in M1, BUT the resolution independence
of the complete contradicts a purely diffusive explanation.

## Honest verdict

- The cut-cell (#218/#222/#224) is a real tested capability (MMS order ~2, conserved
  mass) but does NOT fix the diocotron rate: wrong tool for this blocker.
- The native ABI bug (#225) is fixed (native GH200 runs possible); the case is
  executable natively (DISC #14). These engineering gains remain valid.
- The complete model deficit is RESOLUTION-INDEPENDENT and (boundary-)GEOMETRY-
  INDEPENDENT => suspect = STRUCTURAL (normalization / scale / coupling of the
  complete system-schur), NOT mesh diffusion nor the outer boundary.
- NO reproduction of the complete model claimed.

## Recommended next step (diagnostic, not large GPU)

Isolate the structural factor: compare, on the SAME minimal setup, the raw rate of the
COMPLETE system-schur path vs the reduced ExB path, to localize where the
plateau ~0.037 comes from (2pi normalization / time scale of the complete? Schur coupling
strength? initial drift velocity?). This is a normalization/structure study,
not a resolution increase nor a new geometry.

## UPDATE: ROUTE 1 attempt (complete model on polar grid) -- WELL-BALANCING wall

The polar path (ring r0/r1 resolved by a grid axis) was assembled (PR adc_cases
#18: polar isothermal fluid #209 + Lorentz + polar Schur #215 + polar Poisson;
observable phi on r=r0). It ASSEMBLES and STARTS but diverges before the fit window.
Three-level characterization:
1. NaN at t~0.02; smaller dt only DELAYS (t=0.02 -> 0.101 at dt=1e-4) -> NOT the CFL.
2. Rotating equilibrium IC derived (radial balance: centrifugal rho v_theta^2/r - d_r p
   - rho d_r phi + rho B_z v_theta = 0; ExB-continued root; signs verified vs the engine,
   PR adc_cases #20). Correct in the CONTINUOUS.
3. BUT the continuous equilibrium is NOT discretely stationary: a delta=0 run (without
   perturbation, nr=256) grows ALL azimuthal modes from 0 to ~1e9 in 200 steps.
   The discrete operators (centrifugal source polar_geom_source vs flux divergence;
   and/or the Schur stage) do not preserve the continuous balance.

VERDICT: the complete polar fluid at the stiff parameters requires a WELL-BALANCED SCHEME
(which discretely preserves the source-equilibrium rotating equilibrium) -- a real CFD work item,
not a knob nor a continuous IC. The REDUCED scalar ExB model avoids it (no moment
equation) and that is why IT recovers l=4 exact in polar: the proof that the ring
resolution is the key exists, but the COMPLETE fluid does not run stable without well-balancing.
Work item in progress: polar-wellbalanced workflow (discrete residual diagnostic + well-balanced
fix + delta=0 stationarity test). NO reproduction of the complete model
claimed (neither cartesian -82/-95%, nor polar blocked).

## Engineering gains from the campaign (independent of the scientific verdict)
- Polar Schur MULTI-RANK MPI (#227 merged, plus mono-rank; parity np=1/2/4 ~1e-13);
  MULTI-BOX extension (#229, Kokkos fix in progress).
- cut-cell EB (#218/#222/#224), generic Strang (#217), native ABI fix GH200 (#225,
  + non-regression CI test), hoffart case executable natively (DISC #14).

## UPDATE: ROUTE 1 (complete polar model) -- option (c) frozen-equilibrium, 4 GH200 campaigns

Following the well-balancing blocker identified above, option (c) frozen-equilibrium was
delivered and validated on ROMEO GH200 (mono-rank, Kokkos). Principle: we precompute the FROZEN
equilibrium residual R_eq = step(U_eq) - U_eq ONCE on the axisymmetric ring (zero
perturbation), then advance the CORRECTED map U <- step(U) - R_eq. By construction
(step - R_eq)(U_eq) = U_eq: the axisymmetric equilibrium becomes an exact discrete fixed point.

### What is ESTABLISHED (robust)

1. AXISYMMETRIC WELL-BALANCING RESOLVED. Campaign 1 (delta=0, frozen, nr=ntheta=256, dt=1e-3,
   >=200 steps): max||U^n - U_eq||_inf = 4.150e-20, far below the floor
   C*eps*||U_eq||_inf = 2.320e-13 (||U_eq||_inf = 1.045). U_eq is thus an exact discrete fixed
   point at machine precision. Corollary: step() on GPU is DETERMINISTIC (otherwise R_eq would
   not cancel at 4e-20). NB: this check is partly tautological (R_eq := step(U_eq) - U_eq)
   and does NOT prove the stability of the linearized: it only establishes that the axisymmetric
   O(1) balance is cancelled (||R_eq||_inf = 83.6 at n=256) and that step() is reproducible.

2. THE PERTURBED PATH DIVERGES ANYWAY, before the measurement window. The blow-up (NaN) occurs
   at t ~ 0.01, i.e. ~100x faster than the O(1) window of the diocotron mode (gamma_paper ~
   0.772/0.911/0.683 for l=3/4/5 in omega_d=1 units). The frozen-eq corrects exactly the
   axisymmetric balance but does NOT prevent the divergence of the non-axisymmetric perturbation.

3. THE DIVERGENCE IS INDEPENDENT OF THE INITIAL AMPLITUDE delta. Campaign 2 (l=3, frozen,
   dt=1e-3, t_end=2, nr=256): death time ~ 0.01-0.02 for delta from 1e-1 to 1e-5 (5 orders of
   magnitude). This EXCLUDES a power law driven by the amplitude (a residual forcing O(delta)
   without an unstable operator would give t_death ~ 1/delta). CAVEAT: the measurement (death at ~10-20 steps
   on a sampling grid t_end=2) is too coarse to distinguish strict
   independence from a logarithmic shift t_death ~ -(1/sigma) ln(delta). Correct statement:
   COMPATIBLE with a linear instability (independent OR log-delta), EXCLUDES a power law.
   This pillar does NOT discriminate by itself "numerical" from "physical" (the linear phase of a
   physical mode is also delta-independent): it establishes the linearity, nothing more.

4. REFINING dt DOES NOT FIX (at fixed operator). Campaign 3 (l=3, delta=0.1, frozen, nr=256,
   t_end=0.05): the NUMBER of steps before death grows ~linearly in 1/dt (~9 steps at dt=1e-3;
   ~15462 steps at dt=1e-6, i.e. ~10x per decade), which EXCLUDES a temporal instability at
   fixed step-number. IMPORTANT CAVEAT: the PHYSICAL death time is NOT flat; it
   still grows by +29% over the last decade (0.01308 at dt=1e-5 -> 0.015462 at dt=1e-6). A
   geometric extrapolation of the increments suggests a finite limit t_inf ~ 0.018, but it
   rests on a SINGLE increment ratio (3 points -> 2 gaps) and the dt=1e-3 point is even
   non-monotone. "dt-converged" is thus a misnomer: what is demonstrated is
   "refining dt at identical operator DELAYS death but does not avoid it". Point that saves the
   practical scope: even the most generous limit t_inf ~ 0.018 represents only ~0.01-0.02
   e-folding of the diocotron mode (1/gamma ~ 1.1-1.5), i.e. 50-70x too short to measure the rate.
   No dt refinement opens an exploitable window.

5. THE DIVERGENCE WORSENS WITH RESOLUTION. Campaign 4 (l=3, delta=0.1, frozen, dt=1e-4,
   t_end=0.05): n=128 -> death t=0.035, ||R_eq||=40.6; n=256 -> 0.0083, ||R_eq||=83.6;
   n=512 -> 0.0017, ||R_eq||=165.1. log-log slope d(log t_death)/d(log N) ~ -2.2 (death ~ 1/N^2),
   ||R_eq|| grows ~linearly in N. The perturbation is injected at a FIXED PHYSICAL wavenumber
   (l=3, l/r0 ~ 0.5, N-independent): a PHYSICAL mode of l=3 with finite gamma would converge
   with h (rate that stabilizes as N grows). Here the effective rate GROWS without saturation:
   it is the OPPOSITE signature of a convergent physical mode, and it is the most probative pillar.
   ||R_eq|| growing with N corroborates (unbounded truncation error) without being a standalone
   proof. NB: ||R_eq|| at n=256 equals 83.6 both at dt=1e-3 (campaign 1) and at dt=1e-4
   (campaign 4): the dominant term of the residual is O(1) (Lorentz source omega_c=1e12 condensed
   by Schur), NOT O(dt) -- which reinforces point 4 and points to the stiff source.

### Diagnostic (the blocker: semi-discrete spatial instability, but root cause not closed)

The divergence of the complete polar perturbed path is attributed to an INSTABILITY OF THE
SEMI-DISCRETE SPATIAL OPERATOR (WENO5-Z + Rusanov + polar 1/r geometric source + Schur coupling) at the
paper stiffness. Pillars 3 and 5 (linearity + ~1/N^2 worsening at fixed physical wavenumber)
are the most solid; pillars 1, 2 and 4 are corroborating but not discriminating taken
in isolation. The frozen-eq corrects exactly the axisymmetric O(1) balance but can do nothing on
the operator acting on the non-axisymmetric perturbation: this is consistent with the persistence
of the divergence.

CAVEAT on "ill-posed": we AVOID the strong term ill-posed in the Hadamard sense. No direct
measurement (jacobian spectrum, smooth data) has established it, and a credible competing
mechanism reproduces ALL the observations without invoking an intrinsic ill-posedness: a
CONSISTENT but NON POSITIVITY-PRESERVING scheme failing on the near-vacuum data. The initial
density has a contrast 1e6 (ring rho_max=1 vs halo rho_min=1e-6, model.py:38-39);
RusanovFlux is component-by-component without floor (numerical_flux.hpp:52-67); the geometric
source and to_primitive divide by rho (hyperbolic.hpp:228-238, 143-148). WENO5
reconstructing a 1e6 jump over ~1 cell can produce a NEGATIVE reconstructed rho (or p=cs2 rho)
-> 1/rho and pressure with inverted sign -> local anti-diffusion -> finite-time singularity.
This mechanism also explains pillar 5 (steeper gradient at high N -> larger overshoot
-> t_death ~ 1/N^2) and the fact that the REDUCED model survives (rho there is a passive scalar:
neither 1/rho, nor pressure, nor moment, so no overshoot toward rho/p < 0 possible). Retained statement:
semi-discrete spatial operator UNSTABLE at the paper stiffness on the non-axisymmetric path,
probably via a NON POSITIVE RECONSTRUCTION at the stiff ring edge (and/or the stiff Lorentz
source condensed). This remains a defect of the SPATIAL SCHEME (spirit of the conclusion intact),
but we do not claim the generic ill-posedness.

CAVEAT on "NOT IMEX": the dt sweep keeps the operator AND the Schur condensation
identical (stepper theta=0.5, Crank-Nicolson, NOT L-stable). It excludes "smaller dt at
identical operator", NOT a DIFFERENT temporal treatment of the stiff source. An L-stable scheme
(theta=1 / BDF / stiff-stable IMEX) or an EXACT/exponential integration of the Lorentz
rotation would not fix a failing spatial scheme but COULD MASK the blow-up by
damping the mesh-mode (false stabilization). We thus do NOT assert "IMEX changes
nothing"; we assert that the correct cure is SPATIAL.

### Actionable fix

The blocker is in the semi-discrete SPATIAL SCHEME, not in a knob, not in dt, not (in the sense
of a real correction) in an IMEX. Targeted redesign, in order of probability suggested by the
competing mechanism:
1. POSITIVITY-PRESERVING RECONSTRUCTION / RIEMANN at the ring edge (positivity limiter
   on reconstructed rho and p, vacuum-compatible Riemann state, density floor). It is the most
   direct lead given the 1e6 contrast and the current absence of a safeguard.
2. WELL-BALANCED treatment + STABLE dissipation/upwinding of the geometric source 1/r and of the
   stiff Lorentz source.
An L-stable/exponential temporal scheme for the stiff source is useful BUT must be verified
as a real cure (and not a masking of the mesh-mode) by measuring whether the slow diocotron
rate O(1) actually emerges.

### Why the REDUCED ExB model escapes it

The reduced model (scalar ExB drift, l=4 EXACT 0.913 vs 0.911 paper in polar) has NEITHER
moment equation, NOR stiff Lorentz source, NOR 1/rho, NOR pressure: rho there is a passive
scalar transported. No reconstruction overshoot toward negative rho/p is possible, and there
is no stiff algebraic source to condense. This is exactly what makes the spatial operator
of the complete path fragile and that of the reduced robust.

### Residual uncertainties (honest)

- DECISIVE discriminating test not done: spectrum of the semi-discrete jacobian D[step](U_eq) for
  N=128/256/512. If max Re(lambda) ~ +C/h^2 with sawtooth eigenvector at k_Nyquist ->
  numerical instability confirmed; if Re(lambda) saturates with smooth eigenvector at fixed k ->
  physical mode. The -2.2 slope predicts the first case but does not measure it.
- Direct probe not done: min(rho), min(reconstructed p) and LOCATION of the first NaN (cell
  (i,j)) just before death. At the edge r=r1=8 -> vacuum/positivity mechanism; spread out -> ill-posedness
  more generic.
- Regularization tests not done: (a) hyperdiffusion eps*h^p or artificial viscosity;
  (b) positivity limiter + floor; (c) reduced contrast (rho_min=0.1) or smooth ring (tanh)
  instead of top-hat; (d) bounded smooth data without near-vacuum. If one makes the slow rate
  O(1) emerge, the announced fix is valid.
- dt-sweep not extended (1e-7, 1e-8) nor delta-sweep redone at fine dt: the plateau t_inf ~ 0.018
  is not established (1 single increment ratio). This does NOT change the practical scope (window
  still 50-70x too short).
- Component-by-component ablation (WENO5 alone -> +source 1/r -> +Schur -> +stiff Lorentz) not
  done: it would identify WHICH term makes the operator unstable and validate the prescription.

### Status

NO reproduction of the complete model claimed (neither cartesian -82/-95%, nor polar that diverges).
The AXISYMMETRIC well-balancing is RESOLVED (exact discrete fixed point, 4e-20). The remaining
blocker is the INSTABILITY of the semi-discrete spatial operator on the non-axisymmetric perturbed path at
the paper stiffness, whose most probable root cause is a non positive reconstruction at the
stiff ring edge (and/or the stiff Lorentz source). The reduced ExB model recovers the rate in
polar and remains the credible reproduction route; the complete fluid requires a spatial redesign
before any rate fit.
