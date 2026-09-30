# T3 nonlinear spatial field: independent oracle and reception obligations

Date: 2026-09-30. Author: independent GPT-6.1 Sol reviewer.

This preparatory receipt does **not** receive an implementation commit. The author
is working from `2599d2214bc34bbb351dec2bda63bcbab381bf89`; no T3 implementation SHA
has been frozen or authenticated. The exclusive future reception checkout is
`PoPS-sol61-spatial-nonlinear-review`, at that exact base and left unchanged.
The oracle is kept in the reviewer's existing exclusive review checkout. No
production Python, C++ header, principal checkout, or installed environment is
modified by this receipt.

## Independent discrete relation

`tests/review/test_sol61_spatial_nonlinear_math_oracle.py` imports NumPy and pytest,
but does not import PoPS. It constructs a closed periodic 5-by-4 cell grid on a
rectangle of lengths 1 and 1.7. The centered discrete Laplacian uses the two
different spacings. For 2, 3, and 5 co-localized scalar fields the original relation
is

```
R_i(u; d) = α_i u_i + β_i u_i³
            + A_i(x) u_i Σ_j C_ij u_j + Σ_j B_ij u_j²
            − Σ_j D_ij Δ_h u_j − f_i(x).
```

All coefficients, the spatial captures `A_i`, and the manufactured right-hand
sides `f_i` are fixed data. `D` is nonsymmetric, with distinct components and
nonzero cross diffusion. A prescribed nonconstant field creates `f` by evaluating
this original discrete relation. This is a manufactured **discrete** witness;
there is no claim of mesh convergence or an independently solved physical M27.

A small dense analytical Newton reference with explicit Armijo backtracking
converges from two distinct seeds. Each result is checked against both the known
field and every equation of the original residual. The seeds and captures remain
unchanged. A simultaneous permutation of unknowns, equations, coefficients,
captures and seeds preserves the residual and Jacobian, and the inverse
permutation recovers the same solution.

The explicit central finite difference at `1e-7` is compared to the analytical
Jacobian action. Additional larger steps verify the cubic central-difference
error and its quadratic step scaling; a forward difference fails that witness.
Counter-cases reject an implicit replacement of nonlinear terms by seed values,
a transposed diffusion matrix, reordered spatial captures or RHS routes, and a
tail-component defect. Exhausting the **reference** Newton updates leaves its
inputs unchanged; this does not receive production transactionality.

## Exact candidate obligations after its freeze

The proposed public route is `FieldProblem` with
`CellCenteredNonlinearCoupled(finite_difference_step=...)`, `Newton`, a distinct
`solve_spatial_field` operation, and `pops.spatial-field-residual@1`. These are
author declarations pending source authentication, not currently received facts.

1. Authenticate the exact URI, schema/version promotion, resolved body, frozen
   physical contract, derivative policy and solver options. Old linear graph
   identities and explicit refusals must retain their existing meaning. A
   benchmark name or an assumed width of two or three cannot select behavior.
2. Preserve the original closed residual body. Products of different unknowns
   must remain inside the solve, with exact handle and point routes. Fixed
   captures must remain distinct from seed guesses and evolving iterates. Any
   closure rejection must precede publication. No reaction-matrix inverse,
   constant-reaction requirement, or local condensation may silently replace
   the new nonlinear relation.
3. Receive actual generated calls into the existing native Newton/GMRES engine,
   including residual callback, central-FD JVP, tolerances, bounded iterations,
   line search and original-residual acceptance. Source emission alone cannot
   establish native iteration or convergence. A physically declared contraction
   must not be presented as a universal heterogeneous norm.
4. Receive a public compile/bind/solve witness with exact initialized states,
   captures and two seeds. Observe all real solved states, component
   permutations, unknown/equation ordering and an unrelated query/capture block;
   inspect the original residual independently from native solver receipts.
   Renaming a body or unknown cannot alter semantics.
5. Test transactional failures: malformed closure, unsupported boundary or
   method, nonfinite data at the stage where it is actually admitted, exhausted
   Newton and GMRES, and invalid original residual. Snapshot every affected
   state and accepted clock/cursor before failure; require unchanged publication
   afterward. A failure at bind cannot be reported as a Newton failure.
6. Inspect global field storage and collective ordering: compatible layouts and
   component routes, owned active cells, AMR coverage if supported, empty ranks,
   replica contribution, and rank-local callback/JVP failures converged before
   collectives. Unsupported topology must fail explicitly rather than inherit
   an unrelated legacy visitor's assumptions.

The API 0.4 reference `reference-replay/document/02_contracts.tex`, contracts
C15–C19, motivates these obligations: an unknown product is allowed; fixed data
and the initial guess have distinct roles; the residual remains closed; bounded
Newton checks the original residual; failures do not partially publish.
The 128-unknown dense profile in C17 is not treated as an authority for arbitrary
global field inversion. The dense oracle here has at most 100 scalar entries and
is only an independent mathematical reference.

## Verification and limits

Executed on 2026-09-30 with the existing environment interpreter, without
installing or rebuilding anything:

```
rtk proxy env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/review/test_sol61_spatial_nonlinear_math_oracle.py -q
```

Result: **7 passed**. Ruff and staged diff checks are checked before freeze.
No PoPS source behavior, native Newton/GMRES, real MultiFab, MPI, Kokkos, GPU,
checkpoint codec, nonlinear M27, or PDE end-to-end qualification is claimed.
Those remain separate exact-SHA receipts after the author's freeze and the
coordinator's native rebuild.
