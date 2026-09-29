# Finite linear support contract v1 — W06 / M09

This increment implements explicit finite linear maps in the common native Program
expression mechanism. It does **not** implement a globally distributed mesh solve,
a discretized gradient, or the conducting-disk Hoffart PDE. The finite source
witness is exactly the 8+4 DOF algebra specified by W06/M09.

## Public authoring and representation

`FiniteSupport(name, dofs)` declares a nonempty ordered tuple of unique DOF labels.
Its identity is the exact `(name, ordered labels)` pair. Equal dimensions do not
imply equal supports. `support.bind(field)` requires the field's complete component
labels in that order; binding an explicit tuple is an explicit packing operation.
This mathematical support does not own a mesh, communicator or runtime resource.

`FiniteLinearMap(source, target, coefficients)` captures a rectangular constant
matrix in row-major target-by-source order. All coefficients must be finite Python
integer/float constants, copied into immutable tuples. This first contract does
not yet accept runtime coefficient matrices. No width 4/8, two-block, 12-variable,
or model-name dispatch exists. Native template sizes and available stack resources
remain actual implementation limits; the caller must use genuinely small supports.

`apply(vector)` yields the target support; `solve(rhs)` accepts a square map and
yields its source support. Square does not assert invertible. Both carry exact
source/target contracts and coefficients into versioned `finite_linear_v1` DAG
nodes, projected through `finite_projection_v1`. CSE keys include all those fields.
The common region-aware CSE preserves lazy `where` branches. There is no Python
runtime callback, Python inverse, precomputed native answer, or per-cell allocation.

Native application uses fixed arrays and explicit row sums. A nonfinite product or
sum invalidates the complete joint result, even if another projection is finite.
It does not promise extended-precision cancellation recovery. Native solve calls
the existing `block_apply_inverse` and its versioned balanced relative pivot
policy (`block_inverse_scaling_v1.md`). A singular pivot or nonrepresentable solve
produces an invalid joint result; existing pointwise/nonlinear status admission
rejects it before publication. This does not claim that the *whole coupled system*
is singular. No universal Schur inversion assumption is introduced.

`vector.materialize(P, name, template=state, at=point)` explicitly identifies the
output State/DOF packing. `finite_support_v1` and `finite_template_index` are stored
in the Program node. All captured values retain exact SSA/block ownership and
field provenance. Authoring requires equal layout, centering, spatial support,
sampling, frame and clock; native emission collectively votes exact layout,
distribution and local rank **before any `fab(li)` access**. The vote uses `long`.
Ordinary `P.value` cross-block admission is unchanged. Inactive cells preserve the
explicit output template, including when the first captured input has another width.

## Exact finite witness and acceptance

`api040_m09_finite_native.py` accepts only the independent oracle's coefficients
and old captures. The fixture's existing NumPy RNG sequence is unchanged:
G:4→8, D=−Gᵀ, K=GᵀG, rho=(1,1.1,.9,1.2), s=.03, alpha=.5. A=I−sJ,
B=sG and C=s alpha D Rho. Both original equations are

```
A v + B phi − v_old = 0
C v + K phi − K phi_old = 0.
```

A singleton native Cartesian batch stores two component vectors (8 and 4 DOFs).
It is **one algebraic sample**, not eight velocity cells and four potential cells.
The monolithic route uses the existing 12-variable LocalResidual/LocalNewton
provider. The condensed route uses four unknowns, evaluates
`v=A.solve(v_old−B.apply(phi))` inside the residual, then reconstructs v through the
same native operation after `consume`. There is no second artificial solve.
Both original residual vectors are freshly evaluated after reconstruction and
both `<1e-11` guards precede every commit. Named scalar tolerances are fixed before
native measurements: LocalNewton absolute tolerance 1e-13, corpus residual,
monolithic comparison and CN energy defect all 1e-11.

The independent NumPy oracle is used only to compare saved native outputs. The
runtime test writes/reopens actual midpoint states; CN endpoints/energy are
computed from those states and the old captures. A permutation changes both DOF
orders, coefficient packing and block insertion order. A separate stress fixture
sets A=diag(0,0,0,0,1,1,1,1): the full 12×12 system is invertible, while the chosen
elimination pivot is singular. Monolithic success is judged by the unchanged
original-residual threshold; condensed refusal must preserve all states and time.
The stress system is ill-conditioned enough that direct solution-component
agreement at an absolute 1e-11 is not its acceptance criterion (observed host
roundoff 1.09e-11 at a component of magnitude 32722). No residual tolerance was changed.

## Evidence and remaining work

The initial source test failed because `FiniteLinearMap` was absent. A second
real authoring failure exposed same-block-only pointwise reconstruction; the
explicit materialization contract above closes that seam with a native layout vote.

Local result before the first installed reception: **36/36 targeted source/host tests passed**
(including the 14 new finite-map checks and existing product/capture/layout tests);
Ruff and `git diff --check` passed. Six runtime tests collected; none executed.

Source/host checks compile emitted residual/reconstruction code with actual PoPS
and Kokkos headers and invoke the existing prepared nonlinear solver. They are
not loader, MPI or GPU qualification. The six installed tests are prepared at
`tests/python/integration/runtime/test_finite_m09_runtime.py`; root owns their
actual native compilation and serial/MPI2 execution. No native campaign was run
in this worker. The global meshed/distributed mixed residual/provider problem,
redistribution between different spatial layouts and dynamic matrix coefficient
capture remain concrete implementation gaps beyond this finite-support increment.

```
python -m pytest -q tests/python/unit/numerics/test_finite_linear_maps.py \
  tests/python/unit/codegen/test_finite_m09_source.py \
  tests/python/unit/codegen/test_finite_m09_host.py
python -m pytest -q tests/python/integration/runtime/test_finite_m09_runtime.py
mpiexec -n 2 python -m pytest -q tests/python/integration/runtime/test_finite_m09_runtime.py
```

Use the authenticated rebuilt package for the last two commands, with a timeout
for MPI; source injection is not installed-runtime evidence.
