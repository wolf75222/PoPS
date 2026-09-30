# Original FieldProblem: diffusion from frozen State captures

The physical model still declares its original equations before the Program. A
diffusion entry may now be an expression of the exact State captures supplied to
the original FieldProblem adapter. For example, `D0 * (1 + material[0])` remains
the declared physical coefficient in `-div(D grad(q)) + R(q, captures) - f`.
There is no mutable RuntimeParam, uniform replacement State, model-name dispatch,
or rewrite of the source/reaction equation.

Select the discretization explicitly:

```python
CellCenteredNonlinearCoupled(
    finite_difference_step=1e-6,
    face_policy="Arithmetic@1",
)
```

`Arithmetic@1` names `pops.field.face-mean.arithmetic@1`: each face matrix is the
arithmetic mean of the two adjacent cell matrices. The route admits finite zero,
negative, and nonsymmetric entries and makes no SPD certificate. This is an
explicit numerical realization, authenticated by the registered Case method,
the source contract, the SolveRequest, and the emitted Program identity. It is
not a change of the physical equation. It can also be selected for constant D to
compare discretizations of the same physical problem.

The legacy `face_policy=None` route retains `pops.spatial-field-residual@1` and
its old serialized and emitted bytes. In Uniform, its diagonal-only/scalar native
default is harmonic and its full-matrix default is arithmetic. The existing AMR
original-field provider already uses its general arithmetic route. The explicit
new contract selects arithmetic in both implementations, including Uniform
scalar diffusion. Captured diffusion with the legacy method is refused at
authoring. A coefficient that reads the Newton unknown is refused: this release
has no full spatial nonlinear JVP realization for D(q).

## Evaluation, authority, and publication

The coefficient node carries the same exact capture tuple as the local body;
qualified AST indices retain their physical State declarations. Its dependency
metadata contains the reads actually used by D. Validation checks input object
identity, component/dependency declarations, coefficient AST, physical problem,
point, boundary, and the registered numerical method. Replacing or reordering
captures, changing the point, or resealing a different equation/face policy is
refused before emission.

The existing pointwise coefficient kernel reads those State views at the declared
evaluation point, checks layout/distribution and finite coefficients, then fills
the real coefficient halos. Uniform reuses the existing immutable prepared input.
AMR assembles the actual coefficient hierarchy at that point, retains its strong
owners, and prepares the original full-tower provider collectively. The existing
provider freezes its coefficient snapshot and generation; Newton JVPs therefore
vary q while keeping the authored captured D fixed. Existing coverage, quadratic
coarse/fine ghosts, arithmetic face/reflux handling, execution-lane votes,
SolveOutcome authority, and atomic all-level publication are reused. No solve per
level or host PDE loop was introduced.

The original local reaction and forcing are evaluated by their existing native
body. Final publication still checks the original residual of the entire
candidate. The seven Newton controls, finite-difference step, restart and budget
are unchanged by the face policy; right-preconditioner selection remains a
separate explicit option. This source extension does not qualify convergence for
signed/degenerate D or any particular scientific campaign.

## Versioning and bounded reception

The selected face realization uses residual contract @2, SolveRequest schema 2
with `realization.coefficient_face_policy`, and conditional Program IR version 10.
Legacy original-field IR 8 and explicit right-preconditioner IR 9 remain unchanged
when no face extension is selected. Native ABI and wire ordinals are unchanged;
the public-header manifest schema remains, but the actual native header signature
changes and every installed native dimension must rebuild.

Six production files implement this extension. Source tests author real
Model/Case/Program instances, resolve and emit Uniform scalar/vector and AMR
three/five-component variants with permutations. An independent exact Fraction
face calculation checks conservative signed, zero and nonsymmetric discontinuous
matrices. Mutation tests exercise captures, dependencies, point, named policy and
resealed equation authority. The constant route's IR, emitted C++, and three
physical module hashes are compared byte-for-byte with the earlier c7 receipt.
One real Kokkos/MPI syntax translation unit instantiates explicit arithmetic
application in Dim1, Dim2 and Dim3. These are source, math and syntax checks only;
installed-native execution, MPI rollback, nonconstant coarse/fine convergence,
checkpoint/restart and saved-state scientific reception belong to the subsequent
ROOT campaign. Its previous SDK375 campaign remains a historical receipt and is
not attributed to this implementation.

Run the source suite from this private checkout without using its installed DSO:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c \
  'import sys;sys.path.insert(0,"python");import pytest;raise SystemExit(pytest.main(["tests/review/test_sol61_original_captured_diffusion.py"]))'
```

The adjacent source receipt records the actual checks and their bounded scope.
