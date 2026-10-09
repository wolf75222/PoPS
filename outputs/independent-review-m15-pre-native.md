# M15 axial B.1: independent criteria before native reception

Scope: the existing `api040_m15_hyqmom_axial_b1.py` represents one 1D
order-four axial marginal of Appendix B.1 with five moments. The original
corpus M15 asks for Fox–Laurent recurrences at orders 1–4; this bounded
variant does not implement or qualify those other orders. It does not
establish the 1D–2V fifteen-moment M16 system.

The independent oracle in `tests/python/support/api040_m15_independent_oracle.py`
uses only NumPy/Python arithmetic. It integrates the positive spatial mixture
of two normal velocity distributions over each cell analytically using sinc,
checks those means against 40-point Gauss–Legendre integration, evaluates B.1
through a centered-moment expression algebraically distinct from the PoPS IR,
forms the full 5×5 flux Jacobian by complex-step differentiation, and takes
the nonsymmetric Jacobian eigenvalues for signed HLL speeds. It advances
first-order periodic HLL finite volumes with Forward Euler at
`dt=1/(100N)`, `N=32,64,128`, to `t=.02` in `64,128,256` steps.

The acceptance thresholds are fixed here before any new M15 native run:
initial cell means absolute error ≤`2e-14`; full native state versus independent
FV oracle ≤`3e-11`; each conserved moment integral drift ≤`2e-12`; time error
≤`2e-14`; permuted component result versus canonical ≤`3e-12`; every accepted
cell has `rho>0`, `H2=rho*M2-M1²>0`, and the 3×3 Hankel Gram positive
semidefinite, with no clipping or projection. The initial maximum separation
of B.1 fifth flux from a Gaussian fifth moment must exceed `1e-2`.
Rank-boundary data are outside this positive-interior witness.

Pure preflight (no native artifact): the independent initial Gram minimum is
`0.265918`, `0.265788`, `0.265757` for N=32,64,128. B.1 versus Gaussian
fifth-flux gaps are `0.233918`, `0.234009`, `0.234150`. The final independent
FV Gram minima are `0.269820`, `0.269495`, `0.269357`; maximum moment-integral
drifts are below `8e-17`, and state changes exceed `0.013`. The independent
B.1 flux agrees with the source expression at 32 initial cells to
`5.56e-17`; this is an algebra check, not native qualification.

The native test should authenticate exact Dim1/package/header identity, save
initial and final states for both canonical and permuted component orders,
compare actual data to this oracle, and exercise native rejection of negative
density, zero variance and negative Hankel determinant with rollback. None of
those native outcomes has been observed in this preflight report.
