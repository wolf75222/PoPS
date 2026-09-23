# Scientific limitation of the requested HYQMOM15 case

The exact fifteen-moment case fails its initial characteristic-speed check on
the retained Hoffart finite-volume data. An independent audit found no
implementation or moment-conversion defect in the compared PoPS and active
RIEMOM2D paths. It establishes two distinct mathematical results:

- At the unit isotropic Maxwellian, a common positive definite symmetrizer proves
  real diagonalizability in **every spatial direction**.
- At a particular strictly realizable correlated Gaussian, and at representative
  actual Gauss-averaged Hoffart initial cells, oblique characteristics are
  genuinely complex even though the Cartesian spectra are real.

The formulas in the original closure attachment correspond to **Appendix B.1**
of the published paper; **Eq. (40)** is a different closure. A subsequently
supplied, different `main.pdf` studies the Maxwellian linearization. Its result
does not contradict the correlated-state counterexample. The exact case remains
fail-closed; this note changes neither the closure nor its guards and makes no
claim that every HYQMOM method or correction is impossible.

For a two-dimensional conservation law, hyperbolicity requires real
diagonalizability of `n_x J_x + n_y J_y` for every real spatial direction `n`.
Checking the Cartesian axes alone does not establish this property.

## The supplied Maxwellian analysis

The new document is Sacha Dupuy's *Waves dispersion in 2D 15 moments magnetic
HyQMOM model* (19 March 2026), six pages, 151561 bytes, SHA256
`dae8fd182835fbdddbe2d26f122e34711e7a8beb87bc388e51076449829c3d54`.
Its page-3 order, also used by this case, is

`[M00,M10,M20,M30,M40,M01,M11,M21,M31,M02,M12,M22,M03,M13,M04]`.

Set density to one, mean to zero and covariance to the identity. Removing all
magnetic and electric-field terms from the printed matrix and stripping the
Fourier factor `i` gives `kx Jx + ky Jy`. Independent exact differentiation of
the [PoPS closure](../../../python/pops/moments/closures/hyqmom15.py) through the
complete [raw/central/standardized conversion chain](../../../python/pops/moments/model_builder.py)
agrees with **all 225 entries of each** matrix in the
[RIEMOM Maxwellian Jacobian](https://github.com/Ahcas28/RIEMOM2D/blob/0f2a1967485256d0525e5255ec8a1c4b4de5e2ea/linearized_Jacobian_fluid.m).

The literal printed PDF differs at two combined entries, using one-based indices:

| Row / column | Literal printed entry | Closure and RIEMOM entry |
|---|---|---|
| 12 / 4: M22 / M30 | `i ky` | `i kx` |
| 12 / 8: M22 / M21 | `-3 i ky` | `+3 i ky` |

These are independently detectable shared-moment inconsistencies:
`Jy[row M31] = Jx[row M22] = grad(M32)` and
`Jy[row M22] = Jx[row M13] = grad(M23)` must hold. Both identities fail in the
literal printed matrix and hold exactly in the source-derived matrices; the
corresponding M41 and M14 identities pass in both. The literal printed Jy has
the pair `+/- i sqrt(3)`, inconsistent with the document's statement that its
x and y speeds agree. The printed matrix is retained separately from its
source-consistent interpretation in the audit; these document discrepancies
are not a difference between Appendix B.1 and Eq. (40).

For the source-derived Maxwellian matrices, either Cartesian characteristic
polynomial is

\[
\lambda^3(\lambda-1)^2(\lambda+1)^2
(\lambda^2-6)(\lambda^2-3)(\lambda^4-6\lambda^2+3).
\]

This agrees with the document's stated block polynomials. Repeated eigenvalues
have complete eigenspaces. More strongly, let `T` convert raw moments to the
fixed tensor probabilists-Hermite moments
`h_pq = integral He_p(vx) He_q(vy) f dv`, through total degree four. Use weights
`1/(p!q!)`, replacing the two pure fourth-order weights by `1/6`:

\[
W_{pq}=\frac{1}{p!q!},\qquad W_{40}=W_{04}=\frac16.
\]

Exact rational arithmetic verifies `det(T)=1` and, with `H=T^T W T`,

\[
H>0,\qquad HJ_x=(HJ_x)^T,\qquad HJ_y=(HJ_y)^T.
\]

Hence `H^(1/2) (nx Jx + ny Jy) H^(-1/2)` is real symmetric for every real
direction: the isotropic Maxwellian has a complete real characteristic basis
in all directions. This is a pointwise Maxwellian proof, not merely an angular
scan and not a theorem for arbitrary realizable moments. The speeds can still
depend on angle: for unit normal `(1,1)/sqrt(2)`, the largest speed is
`2.81344556503634`, versus `sqrt(6)` on either axis.

## Exact correlated-Gaussian counterexample

Take unit density, zero mean, and a Gaussian velocity distribution with covariance

\[
\Theta=\begin{pmatrix}1&1/2\\1/2&1\end{pmatrix}.
\]

This Gaussian is not an isotropic Maxwellian. Its covariance eigenvalues are
`1/2` and `3/2`. Its full degree-four moment matrix
is positive definite: the integral of the square of any nonzero quadratic
polynomial against this positive Gaussian is strictly positive. The raw moments
`M_pq = integral(v_x^p v_y^q f dv)` in the case's component order are:

| Index | Moment | Exact value |
|---:|---|---:|
| 0 | M00 | 1 |
| 1 | M10 | 0 |
| 2 | M20 | 1 |
| 3 | M30 | 0 |
| 4 | M40 | 3 |
| 5 | M01 | 0 |
| 6 | M11 | 1/2 |
| 7 | M21 | 0 |
| 8 | M31 | 3/2 |
| 9 | M02 | 1 |
| 10 | M12 | 0 |
| 11 | M22 | 3/2 |
| 12 | M03 | 0 |
| 13 | M13 | 3/2 |
| 14 | M04 | 3 |

The following definitions specify the closure and its differentiation without
requiring a particular PoPS build. Set `rho=M00`, `u=M10/rho`, `v=M01/rho`,
`sigma_x²=M20/rho-u²`, and `sigma_y²=M02/rho-v²`. For `p+q≤4`, define

\[
S_{pq}=\frac{1}{\rho\sigma_x^p\sigma_y^q}
\sum_{i=0}^{p}\sum_{j=0}^{q}
\binom pi\binom qj(-u)^{p-i}(-v)^{q-j}M_{ij}.
\]

In particular, `S00=1`, `S10=S01=0`, and `S20=S02=1`.
Appendix B.1 closes degree five by

\[
\begin{aligned}
S_{50}&=\tfrac12 S_{30}(5S_{40}-3S_{30}^2-1),\\
S_{41}&=-\tfrac14 S_{30}(8S_{40}-9S_{30}^2-4)S_{11}
+\tfrac14(10S_{40}-15S_{30}^2-6)S_{21}+2S_{30}S_{31},\\
S_{32}&=\tfrac12(2S_{40}-3S_{30}^2)S_{12}
+\tfrac12(3S_{22}-1)S_{30}.
\end{aligned}
\]

`S05`, `S14`, and `S23` follow by exchanging the x and y indices in `S50`,
`S41`, and `S32`, respectively. Eq. (40) instead adds

\[
\Delta S_{32}=S_{03}\left(S_{31}-S_{40}S_{11}
+\tfrac32 S_{30}^2S_{11}-\tfrac32 S_{30}S_{21}\right)
\]

to the B.1 expression, and adds its x/y exchange to `S23`; the other four formulas
are unchanged. At the Gaussian above, these differences **and all their first
derivatives vanish**, so both closures give the same flux Jacobians.

Recover each fifth raw moment using

\[
M_{pq}=\rho\sum_{i=0}^{p}\sum_{j=0}^{q}
\binom pi\binom qj u^{p-i}v^{q-j}\sigma_x^i\sigma_y^j S_{ij},
\qquad p+q=5.
\]

For each transported component `(p,q)`, set `F_x,pq=M_(p+1,q)` and
`F_y,pq=M_(p,q+1)`. Differentiate these fluxes with respect to **all fifteen raw
moments**, including the dependence of the mean and standardization on those
moments. Evaluating `A=J_x-J_y` at the table above gives

\[
\det(\lambda I-A)=P(\lambda)=
\lambda^{15}-\tfrac{67}{2}\lambda^{13}
+\tfrac{5641}{16}\lambda^{11}-1422\lambda^9
+\tfrac{9747}{4}\lambda^7-1593\lambda^5
+\tfrac{4131}{16}\lambda^3.
\]

This polynomial was computed using exact rational forward differentiation and
the Faddeev–LeVerrier recurrence; the exact Cayley–Hamilton residual is zero.
The new independent source-derived audit reproduces it and its factorization:

\[
P(\lambda)=\frac{\lambda^3(\lambda^2-1)}{16}
(4\lambda^4-39\lambda^2+9)
(4\lambda^6-91\lambda^4+384\lambda^2-459).
\]

The final factor is the cubic `4z^3-91z^2+384z-459` in `z=lambda^2`.
Its discriminant is exactly `-4627764`, so it has two nonreal roots and one
real root. The latter is positive because the cubic is strictly negative for
`z<=0`. This gives **four nonreal lambda roots**. Exact real-root isolation
independently finds eleven real roots with multiplicity and four nonreal roots.
Both Cartesian-axis spectra at this correlated state remain real.

For the **unit** normal `(1,-1)/sqrt(2)`, the nonreal roots are

\[
\lambda=\pm1.12660841487331220086731
\;\pm\;0.0648866203337799047907267\,i.
\]

The imaginary magnitudes agree at 80 and 140 decimal digits. The largest relative
eigenpair residual over the fifteen eigenpairs is below `6.03e-141`, using
`||Av-lambda v||inf / ((||A||inf+|lambda|)||v||inf)`.
This counterexample excludes a common SPD symmetrizer **at this correlated
state**, while the isotropic Maxwellian above has one.

## Consequence for the Hoffart initialization

The obstruction also occurs in the actual PoPS Gauss4 finite-volume initialization,
using the authors' benchmark constants and thermal variance `1e-24`, on a `16×64`
base grid with one factor-two refined level. The exported binary64 states were
inspected without changing them. All 512 active base cells
and 2048 active refined cells had positive-definite degree-four raw moment Gram
matrices, certified by exact rational elimination; the 512 covered base cells
were classified separately and were also positive definite.

These are Gauss-averaged cell moments of spatially varying initial velocity
distributions, not pointwise isotropic Maxwellians and not the analytic
correlated Gaussian above. Two active refined-cell examples on the `32×128`
level, in zero-based `(r,theta)` indices, were independently rechecked at 100 and
160 decimal digits from the exact stored binary64 values:

| Cell | Mapped direction | Representative complex pair | Maximum relative eigenpair residual, 160 digits |
|---|---|---|---:|
| (16,27) | radial | `0.5167910924994291 ± 0.02848187156306240 i` | `3.30e-161` |
| (12,53) | angular | `0.09100196784776332 ± 0.01461281139778020 i` | `8.94e-161` |

PoPS stores `q=r*M`; the closure Jacobians on q and physical M=q/r agree exactly
by density homogeneity, also verified entry by entry for these cells. The full
source-builder square-root/standardization chain agrees with independently
derived exact rational Jacobians, with maximum relative matrix error below
`6.68e-158` at 160 digits. The fixed raw-to-centered/scaled similarity error is
below `1.21e-160`. Both Cartesian-axis calculations have zero computed imaginary
part at 100 and 160 digits. Both cells have exactly positive degree-four Gram
leading minors.

Mapped covectors were reconstructed from retained coordinates and documented
geometry, not obtained from a new native auxiliary-field observation. Dividing
each covector by its length preserves nonreal unit-normal characteristics:
the largest imaginary magnitudes become approximately `0.02848473131` and
`0.09130256433`. The archive's values and all benchmark constants are unchanged;
this is an offline recheck of historical time-zero evidence, not a new native
bind, accepted step, checkpoint or trajectory.

The native run stopped
before its first accepted step with an invalid local maximum speed; there is no
accepted HYQMOM15 trajectory or checkpoint to interpret as a successful run.
The published Cartesian corrections and even a full Gaussian higher-moment
target preserving density, mean and the entire covariance still fail oblique
checks. A smaller time step, a larger finite speed bound, or better eigenvalue
conditioning cannot turn these inviscid transport characteristics into real ones.

## Source and scope

The [published paper](https://comp-physics.group/papers/bryngelson-JCP-26.pdf),
§3.8 and Appendix B, supplies the two variants and their pointwise correction
algorithms. They must not be conflated or described as a transcription error.
The inspected reference implementation is
[HyQMOM.jl at `54cf5770d7017c9abdee620da3583912e7bafeab`](https://github.com/comp-physics/HyQMOM.jl/tree/54cf5770d7017c9abdee620da3583912e7bafeab).
Its [eigenvalue correction](https://github.com/comp-physics/HyQMOM.jl/blob/54cf5770d7017c9abdee620da3583912e7bafeab/src/numerics/eigenvalues6_hyperbolic_3D.jl)
checks Cartesian axes; this does not provide an all-normal proof for the disk.

The active [RIEMOM2D closure at
`0f2a1967485256d0525e5255ec8a1c4b4de5e2ea`](https://github.com/Ahcas28/RIEMOM2D/blob/0f2a1967485256d0525e5255ec8a1c4b4de5e2ea/closureS5.m#L14)
has exactly the six B.1 polynomials used here. An exact rational comparison also
matches its generated raw-moment Jacobian at the Gaussian counterexample above.
The [active flux path](https://github.com/Ahcas28/RIEMOM2D/blob/0f2a1967485256d0525e5255ec8a1c4b4de5e2ea/Flux_closure15_2D.m)
uses `M2CS4_15` and diagonal standardization. All six polynomials agree
coefficient by coefficient; the compared M-to-C, C-to-S and closed C5-to-M5
expressions agree in values and derivatives at the Maxwellian, correlated
Gaussian and a positive skew Gaussian mixture. No implementation or conversion
defect was found in these compared paths.

The supplied explanation that RIEMOM's `real(...)` conversion was only for
typing is accepted. No intent to remove instability is inferred from it, and
the results here do not depend on that routine. RIEMOM2D's electrostatic example
uses a warm periodic square with different constants, so its execution does not
by itself qualify the selected conducting-disk benchmark.

The revised [Riemann35.jl planar
closure](https://github.com/comp-physics/Riemann35.jl/blob/3dd0fef3d69faee4c333e07604af3d6a68fa8db0/src/moments/hyqmom_3D.jl#L15)
and its [device flux](https://github.com/comp-physics/Riemann35.jl/blob/3dd0fef3d69faee4c333e07604af3d6a68fa8db0/src/numerics/flux_closure_dev.jl#L248)
also match B.1. Both Julia projects evolve 35 moments in three velocity
dimensions; selecting one spatial layer does not produce a fifteen-moment model.
These reference comparisons preserve the distinction between B.1 and Eq. (40)
and do not remove the exact oblique counterexample.

This note records a scientific limitation, not an implementation of a correction.
No moment projection, covariance floor, heating, characteristic-guard bypass,
normal-frame closure replacement, or alternative model was applied. The exact
case remains available for reproducing the refusal. A separately named model
would require its own equations and validation: for example, the
[Fan–Li generalized-Hermite regularization](https://arxiv.org/pdf/1401.4639)
has an all-normal theorem, but changes the transport equations and cannot be
labelled the attached HYQMOM15 closure. These findings do not establish that
every future HYQMOM15 algorithm or correction is impossible.

## Reproducible evidence

The independent audit was evaluated against PoPS source
`65b0658617b44b185b4068c6d71215986607ac43` without loading PoPS or running its
native runtime. The sealed campaign archive, maintained separately from the
source checkout, is
`output/evidence/hyqmom15-maxwellian-independent-20260915-v1/`.
Its archive-relative files and SHA256 hashes identify the evidence without
depending on a particular user's filesystem layout:

| Artifact | SHA256 |
|---|---|
| `REPORT.md` | `bf28480e89d4872dba3083327f09d19755d52199d7f8371ad4a7c3bc3b33cf1f` |
| `SHA256_MANIFEST.json` | `c1ae5fb8c4d3fcb316c3441b10dd178d5c68fa6e9858453271d91790022d4e9b` |
| `MAXWELLIAN_SYMMETRIZER.json` | `d4692fb0f9696cbbdb72d6698f7ab1265afb90e5696aac3b81a93eb112f480f7` |
| `ACTUAL_CELLS.json` | `520c78bfcdadd100510137f0bd9a4717232f874ed64a590a4a84e8c86a46ccaa` |

The bundle includes the exact source snapshots and new PDF, all 450 printed
versus source entry comparisons, every source-derived Jacobian, exact
characteristic polynomials, complete high-precision spectra/residuals, and
standalone `audit.py`, `export_and_symmetrizer.py` and `actual_cells.py` scripts.
The historical native initial-state archive has SHA256
`7a5eb9040b20e777edb7e0f0a3143e2b84db7ccd0e04ab2217cf37cdc4aa4137`;
the two cell records were matched to it bit for bit before differentiation.
