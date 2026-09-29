# Local State storage authority at resolution

The installed product/H05 reception on `4bb0639` compiled the local Programs,
then failed at bind: `inspect_compiled._ghost_depth_by_block` found no declared
ghost depth in blocks with neither a numerical method nor a spatial descriptor.
The native carrier already required one ghost cell. That requirement existed
only after lowering, so the resolved plan and bind inspection disagreed.

Resolution now publishes `StateStorage(ghost_depth=1)` for an exactly selected
local cell/multifab State with an authored physical frame, when the block has no
explicit numerical or spatial method. The minimum is shared with
`prepare_state_storage_requirements`; the frozen resolved descriptor is consumed
by compilation and the unchanged strict bind inspector.

The admission reads the canonical operator signatures of that selected State.
It excludes grid operators, diffusive laws, and balances with spatial terms.
It does not infer a dimension from the installed backend or invent a flux.
An explicit descriptor, including a depth greater than one, remains authoritative.
The private carrier uses the same admission; authoring models are not mutated.
Repeated A → B → A selection in a reusable multi-state model remains exact.

## Evidence

`test_resolved_local_storage_authority.py` initially reproduced three failures
with the installed reception's exact missing-depth error. Its seven final cases
cover product permutations, multi-state selection, a spatial neighbour,
diffusion exclusion, explicit depth four, and continued inspector refusal when
the storage authority is removed from a plan.

Two overlapping source suites passed:

- 33 tests: storage authority, local loader, source storage, joint reconstruction
  storage, and the M11/W10 constrained composition.
- 35 tests: the final seven authority cases, generic diffusion, H05, and the
  independent local product source/apply oracle.

These checks exercise public validation/resolution and real model graph lowering,
without JIT or package installation. Native serial/MPI bind and evolution of the
original installed product/H05 cases remain for central reception. No native
execution, GPU qualification, or complete M11/M22 PDE qualification is claimed.

Reproduce the focused source check in the configured environment:

```sh
env -u PYTHONPATH python -m pytest -q -o pythonpath=python \
  tests/python/unit/codegen/test_resolved_local_storage_authority.py
```
