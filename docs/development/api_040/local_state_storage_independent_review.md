# Independent review: resolved local State storage

Reviewed `0ae3db0` against its parent `4e0d363`. The change publishes a real
`StateStorage(ghost_depth=1)` on the resolved block before the unchanged strict
bind inspector reads it. The selected `StateSpace` comes from the single exact
state of each Case block; the authored physical frame and cell/multifab layout
are checked before admission. The existing explicit spatial or numerical
descriptor wins, including a deeper stencil. Grid and diffusive operators on
the selected State remain excluded.

The independent test here uses one reusable Model with A and B, a spatial flux
declared only on B, and blocks A → B → A. Public validate/resolve followed by
`ProgramModelGraph.from_resolved_blocks` gives storage only to A, both A native
carriers have ghost depth one and no flux, and the authored model hash and Case
block spatial declarations are unchanged. Together with the seven author tests,
the focused source suite passes 8/8.

One separate multi-state lowering limit remains: if B also declares `waves`,
lowering A rejects `set_eigenvalues must cover the exact set_flux axis set`.
`module_lowering.py` is unchanged by `0ae3db0`; this does not contradict the
storage selection test, which has no wave declaration. It must not be treated
as proof that every multi-state transport composition compiles. Native bind,
MPI, and execution of product/H05 remain pending central reception.

Focused source check:

```sh
env -u PYTHONPATH python -m pytest -q -o pythonpath=python \
  tests/python/unit/codegen/test_resolved_local_storage_authority.py \
  tests/python/unit/codegen/test_resolved_local_storage_independent_review.py
```
