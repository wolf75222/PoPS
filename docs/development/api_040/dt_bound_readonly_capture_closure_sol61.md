# Read-only dt-bound DAG closure and exact block routing

Base: `5d97b7a5a18532c71f030b7dc85e7e534e9ed805` (including `40e9b87`
and the inspect-only readonly-input correction). Implementation is Python/codegen
only; no C++ header, shared environment, build or JIT cache changed.

The public witnesses from independent reviewer Ptolemy, frozen at
`a7ecd3ca47b636aa058514a382786b9d75c58e93`, exposed two actual source failures:
`dot_all(u.n,u.n)` in a bound captured an already authored top-level state and
raised `KeyError(0)`; reading a separate state only inside the bound raised the
explicit missing block-index error. Fixing compiled bind inspection alone could
not close either emission seam.

`readonly_dt_bound_nodes` now builds an authenticated dependency view of the
query and its captures. It traverses transitive `inputs` in stable DAG order,
deduplicates shared values, rejects cycles, and uses the existing read-only
operation whitelist for both local nodes and captures. In particular a captured
reduction cannot conceal a physical affine update or control-flow operation.
The traversal is iterative, so a long valid scalar DAG does not inherit Python's
recursion limit. It never issues, copies, renumbers or moves an authored node.

`set_dt_bound` checks that closure before publishing its bound; existing atomic
authoring restores IDs, nodes and metadata on refusal. `_emit_dt_bound` lowers
the view, including previously missing captured state/scalar dependencies.
`_block_indices` preserves all top-level declaration indices and appends only
query state owners without an existing route. No index-zero fallback is added.
The block ABI and serialized `block_order` therefore include a query-only state,
without adding it to the publication/commit table or changing top-level nodes.

Source reception: **32 tests passed in 4.69 s**, covering actual public
validate/resolve → ProgramModelGraph → C++ emission for both independent cases
and a shared captured scalar DAG; precise block1 routing for `bound_data`; one
query evaluation of each shared reduction; unchanged authoring nodes/commits;
direct and transitively captured affine-update refusal; control-flow refusal;
atomicity, typed provenance and the existing readonly bind-input fixture. A
1500-operation captured scalar chain also closes without recursion failure.
Ruff and whitespace checks pass.

Two complete pre-fix public source baselines, with and without a local pure bound,
were calculated against the actual base checkout. Tests require exact equality
of IR hash, semantic-data SHA256 and full emitted C++ SHA256. No route or graph
extension is needed for those programs; all six original fingerprints remain
unchanged. Their values are frozen in the new test, not recomputed as an oracle.

The unchanged independent public script was separately replayed from the new
checkout: exit **0**, both `existing_state` and `readonly_block` report real
emission `received`, IR version **7**, and the actual source-package path.
Its bytes SHA256 is
`c1326ce1bac92487303f1d7bd879793def1ffe11f8c9921b14f8ea050f01e945`;
the preserved receipt is
`outputs/sol61-dot-all-a77-independent/dt-bound-receipt.json`.
The copied historical probe and output stay outside this implementation commit;
their original independently frozen source remains authoritative.

Exact test command, from this checkout:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I - <<'PY'
import pathlib, sys
root = pathlib.Path.cwd()
sys.path[:0] = [str(root / 'python'), str(root)]
import pops, pytest
assert pathlib.Path(pops.__file__).resolve() == root / 'python/pops/__init__.py'
raise SystemExit(pytest.main(['-q',
    'tests/python/unit/codegen/test_dt_bound_capture_closure.py',
    'tests/python/unit/codegen/test_readonly_control_input_review.py',
    'tests/python/unit/time/test_program_authoring_atomicity.py',
    'tests/python/unit/time/test_typed_provenance_guards.py']))
PY
```

This reception proves authoring, routing and source emission. It does not claim
execution of the returned bound, native MPI reduction/field solve, field-provider
cache behavior or restart. Existing field-route/lowerability checks remain in
force; the whitelist is not enlarged to replay arbitrary temporal graphs.
Root owns rebuilt-package/native reception; Ptolemy owns independent review.
