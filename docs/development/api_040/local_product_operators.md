# LocalResidual product local operators (Python IR v2)

This extension follows the corrected local State storage carrier (3009947,
934206a). It adds no C++ provider or solver: the original residual product still
uses `prepare_local_nonlinear_problem<N>` and its existing atomic scratch/status
publication. `N` is the sum of the selected State widths, with no chosen-model or
cardinality limit. Direct component-expression products retain their v1 IR.

The public body may call `P.source` and `P.apply` with authenticated local
operator handles. It may materialize local component/affine expressions before
these calls. The v2 subgraph records argument positions and exact model, block,
Space, evaluation point and field provenance. Validation requires a closed local
graph; a field solve, gradient or other nonlocal operation is not relabelled as a
local residual. Serialization, detachment and Program optimization preserve this
subgraph and its captures.

The scalar and product routes share `_emit_local_residual_nodes`. Each physical
call binds its actual argument inside the residual lambda, including a derived
candidate or a frozen State. Parameters come from that call's exact block table.
The common expression emitter retains lazy `where`, rounding barriers and finite
checks. An invalid active physical row marks the complete original residual
invalid; no reduced equation or converged surrogate replaces it.

A captured field remains the result of its exact external field solve. The body
must also capture the State occurrence used by that field and read it under that
same FieldContext. An unknown may have the same outer seed as this frozen State;
its Newton placeholder nevertheless remains distinct. Two capture names for the
same frozen occurrence share one placeholder. Rebinding a field to the candidate
is refused, including after normalization.

Provider reads on an exact frozen State are prepared through the existing solve
action and collective outcome route. Two calls may not overwrite a shared
provider key from different frozen States or evaluation points before the joint
kernel. Such simultaneous snapshots require a further native realization.
Candidate-dependent auxiliary provider evaluation is also still an **IMPL** gap:
code generation refuses it explicitly instead of computing the provider at the
seed and freezing it. Local source expressions in conservative/primitive
variables and RuntimeParam values are evaluated at every candidate. This is not
an assertion that an arbitrary field problem is local.

## Bounded evidence

Source witnesses resolve public Cases with 1+2, 2+3 and 1+2+4 components in both
block orders, normalize their Programs, and emit the Program plus every complete
native State storage loader without invented flux. A host C++ witness compiles
the real emitted seven-variable residual and real prepared Newton provider at
`-O2 -fno-fast-math`. Its physical source reads `2*z`, its linear operator reads
`z`, and all equations couple through the joint sum. An independent NumPy
equation checks two parameter sets, two seeds, permutations and a nonfinite
candidate. The original scalar, direct-product, stagnation and conditional
consumer tests provide regression coverage.

These are source/host results, not installed MultiFab, MPI, AMR or rollback
acceptance. The centralized rebuild and native reception remain required.

The final bounded suite passed **59/59** in 24.32 s with
`env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python
-m pytest -q --tb=short -o pythonpath=python` and these files under
`tests/python/unit/codegen/`:

- `test_local_product_operators.py`
- `test_local_product_operator_host.py`
- `test_local_auxiliary_outcome.py`
- `test_local_residual_product.py`
- `test_local_residual_product_host.py`
- `test_local_product_independent_review.py`
- `test_api040_t3_original_residual_stagnation.py`
- `test_conditional_consumers.py`

The v2 public Case test also crosses `detach_compiled_program` and requires an
unchanged IR hash and emitted C++, with no surviving authoring registries. This
does not assert that the separate direct-product v1 detachment defect reported
by the installed reception has been repaired by this extension.
