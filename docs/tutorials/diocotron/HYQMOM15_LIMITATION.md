# Scientific limitation of the requested HYQMOM15 case

The exact fifteen-moment case currently fails its initial characteristic-speed
check. This is a mathematical limitation of the supplied transport closure on
these data, not merely an ill-conditioned eigensolver. The formulas in the attached
`main.pdf` correspond to **Appendix B.1** of the published paper; the paper's
**Eq. (40)** is a different closure. Both have genuinely complex characteristics
in oblique directions, including for strictly realizable Gaussian states.
The case remains fail-closed. No alternative closure is implemented by this note.

For a two-dimensional conservation law, hyperbolicity requires real
diagonalizability of `n_x J_x + n_y J_y` for every real spatial direction `n`.
Checking the Cartesian axes alone does not establish this property.

## Exact Gaussian counterexample

Take unit density, zero mean, and a Gaussian velocity distribution with covariance

\[
\Theta=\begin{pmatrix}1&1/2\\1/2&1\end{pmatrix}.
\]

Its covariance eigenvalues are `1/2` and `3/2`. Its full degree-four moment matrix
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
It can be reproduced from the formulas above with rational arithmetic or a
symbolic algebra system. For an independent exact root count:

1. Compute `gcd(P,P')=lambda²` and the degree-13 squarefree polynomial `Q=P/lambda²`.
2. Form the Sturm sequence `Q, Q', -rem(Q,Q'), ...`, without changing signs of its
   members.
3. Its signs at negative infinity are
   `-,+,-,+,-,+,-,+,-,+,+,+,-,+` (11 variations).
   At positive infinity they are
   `+,+,+,+,+,+,+,+,+,+,-,+,+,+` (2 variations).

Thus `Q` has nine distinct real roots and **four distinct nonreal roots**.
Normalizing `(1,-1)` by `sqrt(2)` rescales the eigenvalues and does not make them
real. This counterexample also excludes a common SPD symmetrizer at this state:
such a symmetrizer would make every directional Jacobian similar to a symmetric
matrix.

## Consequence for the Hoffart initialization

The obstruction also occurs in the actual PoPS Gauss4 finite-volume initialization,
using the authors' benchmark constants and thermal variance `1e-24`, on a `16×64`
base grid with one factor-two refined level. The exported binary64 states were
inspected without changing them. All 512 active base cells
and 2048 active refined cells had positive-definite degree-four raw moment Gram
matrices, certified by exact rational elimination; the 512 covered base cells
were classified separately and were also positive definite.

Two active refined-cell examples on the `32×128` level, in zero-based `(r,theta)`
indices, have the following B.1 characteristic pairs. Independent 100-digit
calculations used the exact stored binary64 moments:

| Cell | Mapped direction | Complex characteristic pair |
|---|---|---|
| (16,27) | radial | `0.516791092499429 ± 0.0284818715630624 i` |
| (12,53) | angular | `0.0910019678477633 ± 0.0146128113977802 i` |

The relative eigenpair residuals are below `3.4e-101`. The native run stopped
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

This note records a scientific limitation, not an implementation of a correction.
No moment projection, covariance floor, heating, characteristic-guard bypass,
normal-frame closure replacement, or alternative model was applied. The exact
case remains available for reproducing the refusal. A separately named model
would require its own equations and validation: for example, the
[Fan–Li generalized-Hermite regularization](https://arxiv.org/pdf/1401.4639)
has an all-normal theorem, but changes the transport equations and cannot be
labelled the attached HYQMOM15 closure. These findings do not establish that
every future HYQMOM15 algorithm or correction is impossible.
