# Actual AMR request: first-Jacobian source/math counter-review

The typed native diagnostic received by Root identifies **N32, permutation
012**, at Newton iteration 0: 240 columns exhausted, finite pivot and residual
above the original stop. N16/012 and N16/201 passed earlier in that native loop.
This supersedes the unproved initial hypothesis that the first failing case
must be N16. No native test was run by this reviewer.

## Frozen operator rather than a substitute SPD problem

`sol61_amr_original_first_jacobian_oracle.py` derives a small dense matrix from
the actual frozen request and kernels at
`9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`. Seven exact Git sources and SHA256s
are recorded in `amr_original_first_jacobian_source_math_sol61.json`. Diffusion,
reaction, means, waves, cubic coefficient and linear controls are parsed from
the original C++ witness. No generic model name or equation is added to PoPS.

The physical request has two patches per level, ratio 2, homogeneous Neumann
faces at x=0/1 and partial refinement on x=1/4..3/4. Its active unknowns are
coarse cells outside that interval plus all fine cells inside it. The matrix
rows follow the actual source seams:

- Covered parents are averages of their two fine children before application.
- C/F ghost interpolation is quadratic, with `s=+/-1/4`; its three parent
  weights are `(-3,30,5)/32` or `(5,30,-3)/32`.
- Same-level neighbors include the patch seam; Neumann physical ghosts reflect
  the actual boundary cell.
- The scalar image uses the actual arithmetic constant coefficient stencil.
  Its coarse face is replaced using the actual matrix-entry reflux sign -1
  and 1D `fine_face_weight=2`.
- Every active component has metric dx on coarse and dx/2 on fine. Covered
  parents are not unknowns and MPI replicas are not introduced into the matrix.
- With the actual null seed, the first Jacobian is `S ⊗ D + I ⊗ R` (the cubic
  derivative is zero). The actual discrete manufactured forcing is
  `A*q_exact + 0.2*q_exact^3`, using the original means and cosine waves.

This gives 24 scalar/72 total DOFs at N16 and 48 scalar/144 total DOFs at N32.
The scalar matrix satisfies `S*1=0` and `w^T*S=0` exactly. The full matrix is
**not symmetric in its physical metric**; no pseudo-SPD operator replaces the
actual quadratic C/F operator. The weighted similarity condition number in
the exploratory N32 calculation is approximately 23146.

The N32 source forcing norm is 8.752282870145205, giving stop
`0x1.6f18ebf69915cp-14`. Root's native stop is
`0x1.6f18ebf6991e7p-14`, a relative discrepancy around 8e-14, consistent with
different floating summation/stencil evaluation. This is a check of source
derivation against the supplied native context, not bit-exact native execution
or relaxation of the solver's tolerance.

## What the same authored controls actually do

The dense linear-algebra probe uses the unchanged restart 80, budget 240 and
relative linear tolerance 1e-5. `W^(1/2)` merely changes coordinates so the
Euclidean norm is exactly the authored physical norm. Every cycle independently
checks `b-A*x` in that norm, including apparent projected convergence.
Two-pass MGS is an unconditional orthogonalization comparison, not a claim to
have received a particular GPU/MPI DGKS implementation.

| Case | MGS passes | Right preconditioner | Columns | Actual residual | Stop | Result |
|---|---:|---|---:|---:|---:|---|
| N16/012 | 1 | Identity | 47 | 1.5192283e-5 | 8.7291704e-5 | Pass |
| N16/012 | 2 | Identity | 47 | 1.5180702e-5 | 8.7291704e-5 | Pass |
| N32/012 | 1 | Identity | 240 | 2.5587208e-4 | 8.7522829e-5 | Fail |
| N32/012 | 2 | Identity | 240 | 2.5587593e-4 | 8.7522829e-5 | Fail |
| N32/012 | 2 | Spatial diagonal | 233 | 8.4205093e-5 | 8.7522829e-5 | Pass |
| N32/201 | 1 | Identity | 240 | 2.4998222e-4 | 8.7522829e-5 | Fail |
| N32/201 | 2 | Identity | 240 | 2.4997296e-4 | 8.7522829e-5 | Fail |
| N32/201 | 2 | Spatial diagonal | 233 | 8.4363297e-5 | 8.7522829e-5 | Pass |

Root's native N32/012 beta is `0x1.05538b6317a3ap-12`. The source-matrix result
has the same iteration-limit category and scale, but differs numerically.
It is **not asserted to reproduce the native beta**: the matrix represents the
exact first Jacobian, while native JVP is central finite difference and uses
Kokkos/MPI accumulation. At the zero seed and unit physical direction, the
exact cubic central-difference truncation is bounded by
`0.2*h^2/min(cell_measure) = 1.28e-9` for N32/h=1e-5; this bound does not include
floating cancellation or claim native equivalence.

These comparisons demonstrate that extra orthogonalization alone is not a
source/math closure of the N32 slow convergence. Euler's independent MGS2 plus
actual-JVP recheck change addresses the separate internal projected-acceptance
guard from c37; it must not be relabeled as receiving this N32 case.

## Generic preconditioning candidate, with obligations

The right Jacobi comparison uses only `diag(S ⊗ D)`, the actual spatial
operator diagonal, whose minimum in this request is 1331.2. It does not read
the model-specific local reaction/cubic expression and does not change A, b,
the metric, state space, permissions, source or numerical controls. It changes
unknown coordinates via an invertible prepared diagonal and measures the true
residual of the original equation.

This supports a generic spatial-provider preconditioning path for independent
native reception. It is not a PoPS implementation. A real port must retain
prepared authority and failure collectives, all components/levels, active
coverage, ownership, and original-equation recheck; it must authenticate finite
usable diagonal entries before any solve mutation. The diagonal must reflect
the actual C/F/physical stencil, or be explicitly an approximate preconditioner.
`2*D/h^2` must not be described as the exact interface diagonal. No physical
equation, tolerance or budget may be changed, and N32 must not be excluded from
production to make N16 pass. Near-singular/signed/general coefficients and
missing diagonal capability require an honest declared realization policy,
not model detection or an invented quotient/gauge.

## Executed source/math checks and remaining reception

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -B tests/review/sol61_amr_original_first_jacobian_oracle.py --output docs/development/api_040/amr_original_first_jacobian_source_math_sol61.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q -p no:cacheprovider tests/review/test_sol61_amr_original_first_jacobian.py tests/review/test_sol61_amr_gmres_projection_math.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/sol61_amr_original_first_jacobian_oracle.py tests/review/test_sol61_amr_original_first_jacobian.py
rtk git diff --check
```

**14 source/math PASS in 2.40s**, Ruff and diff-check PASS. Tests protect the
actual metric/support/conservation, all four original cases, two-pass-alone
failures and true original residual under spatial right scaling. The oracle
does no nonlinear or temporal solve, field allocation, AMR runtime, resource
lifecycle or publication. No native saved states are manufactured or consumed.
No PoPS/native import, SDK/JIT/build, MAIN/environment/donor edit occurs here.
Native serial/MPI N32 convergence, original nonlinear residual and unchanged
candidate rollback/publication remain Root's pending reception. The distinct
discard API/Outcome lifecycle change is outside this numerical oracle.
