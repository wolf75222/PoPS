# Hoffart benchmark with the separately named Fan–Li15 system

[`04_mpi_kokkos_hoffart_fan_li15.py`](04_mpi_kokkos_hoffart_fan_li15.py) authors a full-temperature, two-velocity, degree-four Fan–Li system for the selected Hoffart disk benchmark. It is a separate model from exact HYQMOM15. Neither its Hermite closure nor its hyperbolicity domain resolves the obstruction documented in [`HYQMOM15_LIMITATION.md`](HYQMOM15_LIMITATION.md).

**Qualification status:** the complete linear tutorial and native transport route are implemented. The official Dim2 MPI build has passed selected native MPI1/MPI2 tests, and an actual mode-5 MPI2/OpenMP2 run on a 16×64 two-level hierarchy reached `t=0.002`. All saved raw-moment states passed strict finite/H1 checks. A genuine fresh-world restart reproduced its numerical fields and histories exactly but exposed missing native AMR attempt-counter restoration. Full restart qualification therefore failed and requires a repaired native build and a new actual checkpoint chain. Neither this short run nor structural tests establish sustained admissibility or paper-scale reproduction.

## Physical model and constants

The benchmark uses the authors' released constants, including the temperature supplied in their benchmark files:

| Quantity | Value |
|---|---:|
| Grounded disk radius | `16` |
| Ring radii | `6`, `8` |
| Background / mean ring density | `1e-6` / `0.9` |
| Density perturbation | `0.1*sin(mode*theta)` |
| Modes | `3`, `4`, `5` |
| `alpha` | `39.4784176e12` |
| `Omega` | `-6.28318531e12` |
| Initial Gaussian covariance | `1e-24 I` |
| Final benchmark time | `10` |

The choice of the released benchmark, rather than the differing printed scaling, was explicit. In particular, `Omega` must not be replaced by the nearby ideal value `-2*pi*1e12`: that would alter the gyro phase by about 2.82 radians over a `0.001` interval. The model solves with the full finite `Omega`; a guiding-center equation is not substituted.

There are fifteen raw Cartesian velocity moments `M_pq`, `p+q<=4`, in the PoPS order

`M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04`.

PoPS stores `q_pq=r*M_pq` on the rectangular computational domain `(r,theta) in (0,16) x (0,2*pi)`. A positive common factor `r` changes density but leaves normalized velocity and covariance unchanged. `M10` and `M01` remain Cartesian momentum components. The origin is not removed; the radial lower transport face has zero flux, angle is periodic, the outer transport wall is open, and the potential is grounded at the outer wall.

The finite-volume method explicitly declares `zero_measure_faces=(frame.boundaries.x_min,)`. For bounded physical raw moments, the disk pole has zero physical face measure, so both conservative flux and nonconservative path contributions vanish there. The native method must apply this geometric contract before attempting to recover an inadmissible `q=0` trace. This is independent of `NoFlux`: a generic zero conservative-flux boundary does not authorize discarding its nonconservative product.

For `rho=M00`, define `u=M10/rho`, `v=M01/rho`, and

\[
\Theta=\begin{pmatrix}
M20/\rho-u^2&M11/\rho-uv\\
M11/\rho-uv&M02/\rho-v^2
\end{pmatrix}.
\]

The generalized-Hermite expansion uses this full tensor. Its degree-three and degree-four coefficients are retained. The fifth moments in the conservative Grad flux follow the generalized-Hermite truncation, and the Fan–Li regularization supplies a separate nonconservative term. A Gaussian closure, an isotropic temperature constraint, and a flux-only Grad system would each define a different model. The constitutive reference is [Fan and Li, arXiv:1401.4639, equations (4.16), (5.9), and Theorem 4.13](https://arxiv.org/abs/1401.4639).

## Both transport terms and one mapped direction

The authored equation is

\[
q_t+\partial_r F_{g_r}(q)+\partial_\theta F_{g_\theta}(q)
 +B_{g_r}(q)\partial_rq+B_{g_\theta}(q)\partial_\theta q=S(q).
\]

`fan_li15_expressions(U)` supplies the Grad flux and the matrices `B`. Both are used in `model.rate(..., equation=ddt(U) == -div(physical_flux) - nonconservative)`. The physical product is not a state-only source. Its only nonzero rows are the degree-four indices `(4,8,11,13,14)`. Their complement `(0,1,2,3,5,6,7,9,10,12)` gives the ten conservative transport rows, and the declaration names every one explicitly. External electric/magnetic forces and open-wall fluxes can still change their domain integrals.

One `covectors` mapping is passed to the physical flux, the physical product, and `FanLi15RawMomentPath`. Its radial components are the cell angular averages of `(cos(theta),sin(theta))`; its angular components are `(-sin(theta),cos(theta))/(r*cos(delta_theta/2))`. These are the same trigonometric metric corrections used by the preceding disk cases. They preserve the stated interior uniform-level constant-flow identity when used consistently; they are not an assertion of an exact pole, outer-wall, or coarse/fine identity.

The change from `M` to `q=r*M` introduces no omitted radial term in `B`. Indeed, `B(q)q=0`: an increment proportional to all raw moments leaves normalized `u`, `v`, and `Theta` fixed, and those are the only differentials in this regularization. Therefore `r*B(M)*d_r(M)=B(q)*d_r(q)`.

For a fixed mapped direction `g`, the characteristic matrix is `A_g=DF_g+B_g`, with radius

\[
|g\cdot u|+\sqrt{5+\sqrt{10}}\sqrt{g^T\Theta g}.
\]

The tutorial does not ask PoPS for eigenvalues of the Grad flux Jacobian and does not supply substitute wave speeds. The native Fan–Li interface must supply both dissipation and CFL from its authenticated full-system path bound.

## Declared weak path and finite volumes

`FanLi15RawMomentPath` chooses the straight path in the **complete raw state**,

\[
\Phi(s)=(1-s)q_L+s q_R,\qquad
P_g=\int_0^1 B_g(\Phi(s))(q_R-q_L)\,ds.
\]

The path contract authenticates both the declared Grad flux and the full product against the constitutive expressions. Its native analytic integral uses density-oriented polynomial/logarithm weights. It does not discard high moments or replace the path by a sampling rule with unmeasured quadrature error.

`PathConservativeFiniteVolume` selects conservative variables, first-order reconstruction, and Rusanov fluctuations. For one common face covector `g`, the split is

\[
F^*=\tfrac12(F_L+F_R-a(q_R-q_L)),\qquad
J=F_R-F_L+P_g,\quad D^-=(J-a\Delta q)/2,\quad D^+=(J+a\Delta q)/2.
\]

Thus the common conservative flux telescopes, while `-P_g/2` enters the residual on **both** sides with their own cell measures. The product is not a conservative face transfer. This follows the path-consistent fluctuation form of [Parés, 2006](https://doi.org/10.1137/050628052).

The native method's common geometry is the arithmetic average of the two trace covectors. It must use that same `g` in both physical flux evaluations, the path integral, and the speed bound. An endpoint majorant that bounds the entire admissible raw path is

\[
a=\sqrt{6+\sqrt{10}}\sqrt{\max\left(
\frac{g_x^2q_{20,L}+2g_xg_yq_{11,L}+g_y^2q_{02,L}}{q_{00,L}},
\frac{g_x^2q_{20,R}+2g_xg_yq_{11,R}+g_y^2q_{02,R}}{q_{00,R}}
\right)}.
\]

Separate endpoint geometric directions do not satisfy this proof. The bound also does not prove that the explicit update preserves the admissible set for arbitrary high moments. SSPRK2 transport is used, but spatial reconstruction remains first order; a higher-order path method would additionally require a within-cell product integral.

Each actual transport stage checks `dt * sum_faces(area * a) / volume <= CFL` on active cells, after canonical fine-subface replacement. The native controller reduces a maximum over coordinate directions, so its separate D2 proposal hook returns four times the directional bound. This gives the conservative proposal `CFL*h_min/(4*a_max)` for two face pairs. It does not multiply Rusanov dissipation, replace the stage check, or prove a bound for a changed source/predictor state. A violated stage bound refuses the attempt.

## Python physics and generic native execution

The physical Fan–Li coefficients remain in `pops.moments.fan_li`. Its multi-index
definition supplies both the symbolic nonconservative matrix and the generated
C++ polynomial products. The Python lowering plan selects the fifteen moments,
the five regularized rows, the `f5=0` closure and the spectral constant. HYQMOM15
keeps its separate constitutive definition.

The installed C++ SDK supplies `PathRusanovFlux<N>`, typed conservative and side
results, and model hooks for the directional flux, path integral and common
geometry. Cartesian and AMR consumers depend on those contracts rather than a
named physical model or fifteen-component storage. The same Cartesian path
consumer is tested on the one-dimensional two-component system
`u_t + u_x = 0`, `v_t + u v_x = 0`, whose nonconservative product is nonzero.

Numerical moment utilities remain native: interval certification of a raw 2D
covariance, normalization, complete bivariate Hermite transforms through degree
four, bounded polynomial arithmetic and analytic density/logarithm integration.
Their dimension and supported orders describe mathematical algorithms, not a
Fan–Li closure. The generated caller supplies the physical coefficients and
whole-path speed theorem. The robust midpoint, compensated sums, fixed path
orientation and strict floating-point refusals are retained. Generated Fan–Li
C++ fixtures live under tests; no Fan–Li C++ model is installed in the SDK.

## Admissibility is not full moment realizability

The Fan–Li hyperbolicity domain requires finite moments, `rho>0`, and `Theta` strictly positive definite. Equivalently, in exact arithmetic,

\[
H_1=\begin{pmatrix}
\rho&M10&M01\\ M10&M20&M11\\M01&M11&M02
\end{pmatrix}\succ0.
\]

The tutorial declares zero-threshold recovery checks for density, both covariance diagonal entries, and the covariance determinant. These are additional floating checks; they are **not** a certificate for the exactly represented raw matrix. There is no tolerance, density floor, covariance floor, clipping, or replacement distribution in these predicates.

An executed development guard witness found 93 false accepts, including six exact-zero determinants, among 3,234 exactly retained binary64 near-singular inputs when direct floating covariance tests were compared with exact rational signs. The repaired native helper uses conservative outward bounds and a distinct `IndeterminateCovariance` refusal when strict positivity cannot be established. It certifies the represented raw state's covariance before the ordinary floating recovery checks; its proof assumes strict IEEE arithmetic and gradual underflow. The retained host replay has zero false accepts and 1,052 conservative refusals of exact-SPD adversarial inputs. All 3,072 selected saved native initial benchmark states were exact-SPD and accepted. These are isolated constitutive-header observations, not full Fan–Li runtime or backend qualification. Do not bypass a refusal by weakening the symbolic check or forcing the determinant positive.

The generated model and separate native path materializer apply this certificate to raw recovery and actual face inputs. Candidate publication and AMR synchronization retain their native recovery checks. Source review and isolated header witnesses do not establish those execution points in a full simulation; that remains an explicit runtime qualification requirement.

`H1>0` does not imply degree-four moment realizability. For example, `rho=1`, zero mean, `Theta=I`, and `M40=-1` meet the Fan–Li hyperbolicity condition but cannot be moments of a positive distribution. The degree-four Gram matrix over `(1,v_x,v_y,v_x^2,v_x*v_y,v_y^2)` gives a useful additional necessary diagnostic; it is not used here as an unproved sufficient certificate. No claim that the evolving truncated Hermite density is positive is made.

## Full Schur mean and exact centered gyro phase

The full-Gauss-restart source uses the original finite-`Omega` Schur system. With `s=dt/2`, `J=[[0,1],[-1,0]]`, `B=I-s*Omega*J`, physical gradient map `C`, and polar tensor `K`, the condensed tensor is

\[
A=K+s^2\alpha q_{00}C^TB^{-1}C.
\]

The charge RHS retains the Gauss projection. The field solve retains outer cap 300, relative tolerance `1e-10`, absolute tolerance `1e-12`, damping `0.5`, 64 fine sweeps, and coarse GMRES cap 512/restart 64. The opt-in polar Poisson inverse preconditions GMRES; it does not replace `A`, the original residual, or its stopping test. The initial field guess is zero.

The supplied mean endpoint remains CN. The explicitly selected `rotation="exponential"` applies

\[
v^+=\exp(dt\,\Omega J)(v^0-u^0)+u_{CN}
\]

to every raw moment, using the actual represented full interval and a compensated phase product. For a homogeneous source cell with constant `Omega`, this rotates central moments exactly up to representation/arithmetic error, even for time-dependent cell-uniform electric acceleration. It copies density and the supplied first moments exactly. It does not correct the CN mean error and is not a full exact or AP source integrator. A nonfinite full interval or phase is refused without changing the equation to Cayley.

In exact arithmetic, an orthogonal central rotation retains the local CN work identity. That identity is distinct from global field-plus-particle energy balance and from physical gyro-phase accuracy. Source-first Lie composition remains first order in splitting time, and the very large `|Omega*dt|` requires a separate stiff-limit and time-refinement assessment.

## Initialization, AMR, and genuine output

Initialization retains the authors' Gaussian covariance `1e-24 I`. The tutorial uses the grounded-disk mode Green function for the drift and computes all fifteen Gaussian raw moments before conservative spatial cell projection. A spatial average of varying Gaussians generally has nonzero third/fourth Hermite coefficients and resolvable subcell covariance. Those contributions are retained. The cold variance itself can disappear in a raw binary64 representation about an order-one mean; the resulting guard failure must be investigated, not heated away.

Synchronous dynamic AMR uses complete-hierarchy field solves, conservative parent injection, coarse/fine injection, volume averaging, distributed root patches, and the existing load balancing. Convex transfer preserves `H1>0` in exact arithmetic, but that alone does not certify the floating implementation or the finite-volume update.

The output calendar and checkpoint pipeline follow the existing HYQMOM15 module: exact decimal target construction, cap-safe subdivisions anchored at the absolute scheduled interval, accepted-step checkpoints, actual midpoint potential timestamps, and a near-zero first accepted interval for Fourier normalization. The density snapshots retain `q0_levelN` and `psi_levelN`; `moments_levelN` additionally stores all fifteen actual raw fields for covariance and high-moment diagnostics. Use the saved patch table and finest active-cell coverage when evaluating these diagnostics; array positions outside a fine patch do not represent additional physical cells. Checkpoints retain native state/history and artifact identity. Additional moment snapshots increase storage and must be included in measured campaign storage estimates.

The saved-state schedule includes the attached sequence at `0.1,1.25,2.5,...,10`, along with the finer early growth cadence. These declarations do not constitute saved states. Paper figures, schlieren, or GIFs may be rendered only after real accepted states are produced and inspected; none has been generated for this Fan–Li case.

## Integration and remaining acceptance

The solver explicitly selects **`interface_coupling="fine_flux"`**. The conservative full-tensor implementation is integrated from revision `8d351df`. The actual linear tutorial passes public authoring, validation, resolution, native compilation and the short execution described above, including the explicit pole-face contract, one Schur source prefix and both SSPRK2 hierarchy barriers. The selected interface method retains the full finite-`Omega` tensor and its original residual tolerance.

The native transport route carries the common-face covector and full-system bound, conservative flux, and two separate nonconservative side contributions. Its hierarchy barrier replaces coarse-side contributions with canonical fine subfaces before dependent stages and records only the conservative flux in the reflux basis. Refluxing `P_g` as an antisymmetric flux would change the method. Exact parent injection is required for both prolongation and coarse/fine traces; reconstructed coarse traces would need the missing within-cell path treatment. Selected MPI1/MPI2 tests cover the path route, hierarchy barriers, exact transfer and collective refusal; these local native checks are distinct from sustained case qualification.

After the shared attempt-counter repair and coherent native rebuild, new low-resolution runs must establish full checkpoint/restart parity and actual expanded-AMR behavior. Sustained admissibility, source/field timestamps, conservation of the appropriate transport rows, time/space studies, stiff-source limitations, mass and energy budgets, and measured time/storage costs determine campaign readiness. No Fan–Li deployment or high-resolution allocation has yet been qualified.
