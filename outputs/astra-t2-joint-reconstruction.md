# T2 joint/vector reconstruction delivery

Base: MAIN `83b2b1239973f3e744f9e3975734a40a3abf65d6`, isolated branch
`codex/api040-joint-reconstruction` in `PoPS-resource-lifetime`.
No shared checkout edit, package installation, or heavy build.

## Concrete gap and correction

The initial public 2+3 witness failed before construction because
`reconstruction.User` rejected `state=`. A scalar component sampler could not
express an output in state U reading components of V. The authored protocol now
supports `User(body, state=U, sampling=(V,...), formal_order=...)`: vector row
output, explicit sampled state handles, inferred per-component reads and halo,
live runtime captures. Scalar User without state retains its existing v1 route.

Native changes are confined to the two common reconstruction headers. A joint
policy returns a fixed `std::array`; conservative and primitive samplers expose
oriented `(offset,component)` reads. The principal wrapper maps resolved rows,
calls each vector body once with that row's own RuntimeParams, and assembles the
complete trace. Primitive recovery remains lazy with one shared conversion cache
per selected offset. Existing domain/status and collective face rejection remain
the publication authority. No model dispatch, new solver or per-cell allocation.

The independent review found a real intermediate flaw: mutating sampling to a
foreign Model's homonymous state initially preserved the source digest. The final
code retains exact source handle capabilities, authenticates them before Case
qualification and carries resolved capabilities across the immutable boundary.
The source digest stays deterministic, without object addresses. Explicitly
foreign block captures are refused; ordinary declaration captures remain bound
through each block's parameter carrier.

## Evidence actually executed

- Baseline red: `test_cross_two_plus_three_body_is_a_real_vector_not_componentwise`
  failed with `User() got an unexpected keyword argument 'state'` (1 failed).
  Receipt: `outputs/t2-joint-reconstruction-red.xml`.
- Final source suite: **66 passed in 45.46 s**, including legacy scalar,
  multiblock identity, lazy/nonfinite CSE, runtime-capture routing and principal
  capture/stability tests. Receipt: `outputs/t2-joint-source.xml`.
  Command: `env -u PYTHONPATH .../pops-api040/bin/python -m pytest -q --tb=short
  -o pythonpath=python tests/python/unit/codegen/test_user_joint_reconstruction.py
  tests/python/unit/codegen/test_user_reconstruction_gap_review.py
  tests/python/unit/codegen/test_user_reconstruction_multiblock.py
  tests/python/unit/codegen/test_user_numerical_capture_routing.py
  tests/python/unit/codegen/test_principal_numerical_captures.py
  tests/python/unit/numerics/test_user_reconstruction_adversarial.py`.
- The source suite compiles and executes the actual emitted policies with C++20
  O2, an independently calculated 5-component oracle, both orientations, distinct
  row capture values/rebinds, and active/inactive invalid arithmetic. This tests
  policy evaluation, not the full native mesh runtime.
- Full `test_weno_convergence.cpp` TU: syntax-only passed against current headers
  using the existing build's compiler/dependency flags. Log:
  `outputs/t2-joint-weno-syntax.log`.
- Exact new gtest body plus its model/helper were compiled in a temporary standalone
  executable against current reconstruction/Fab headers and installed Kokkos/gtest
  libraries: **1/1 passed**. It uses actual CPU Fab buffers, proves cross-component
  conservative results, lazy Primitive recovery shared between components and
  failure under reversed orientation. Log: `outputs/t2-joint-header-host.log`.
- Independent Sol source tests: 5/5 passed against this source after the authority
  correction; three independent runtime-case authorings passed validate/resolve/
  graph/emission. Those are the independent reviewer's observations, not native runs.
- `git diff --check` passed. Ruff passes on added/new helper files and other changed
  Python files except `principal_lowering.py`'s existing unrelated loop-closure/
  missing-strict diagnostics (the changed lines add no diagnostic).

## Central reception and limits

Rebuild `test_weno_convergence`; run
`test_weno_convergence.joint_stencil_cross_components_preserve_lazy_primitive_status`
alongside existing scalar/Primitive tests. The independent public native witness is
Sol's commit `4aedc7e`, file
`tests/python/integration/runtime/test_user_joint_reconstruction_independent_runtime.py`:
true FV 5x5 oracle, permutations, +3 cross-state halo, live rebind, invalid square
root rollback and zero Python callbacks during execution. It was prepared on the
confirmed public API and remains **not executed natively** until the SDK rebuild.

Root's separate `_compile_emit.py` correction for principal StateStorage loaders
must accompany native reception: group policies must remain on the group helper,
not instantiate an isolated row FV model without physical wave speeds. This tranche
does not add a fictitious row wave speed or modify that emitter.

Contract: `docs/development/api_040/joint_reconstruction_v2.md`. No automatic TVD,
positivity, characteristic decomposition or formal-order theorem is claimed.
The shape is generic, not restricted to 2+3; the public body produces one vector
row and the principal group assembles all rows. GPU/MPI and full native Uniform/
AMR/Primitive package acceptance remain pending central execution.
