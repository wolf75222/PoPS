# Independent symbolic-layering review — 30 September 2026

Candidate `908508a7ce8ccc05c6241f6239818380fd0760e3`, exact parent
`bc37b0af4012a10fa3e9bd7ecaccd1f7befc897b`. Exclusive worktree
`PoPS-sol61-layering-review`, branch `codex/api040-sol61-layering-review`.
No production, native header, shared environment or principal checkout changed.

The dependency correction is received at source level, but **Program DAG identity
is not preserved in mixed scalar/finite composition**. Approval of that invariant
requires a separate fix. The independent source comparison below exercises real
Program authoring, not a mocked hash function.

## Confirmed regression

Create one two-component `FiniteLinearMap.apply` result `vec`, then author
`program.value("mapped", (u.n[0] + vec[0], u.n[1] + vec[1]), at=u.next.point)`.
On the parent, both finite projections refer to one application. On the candidate,
`Expr._wrap` independently calls `lower_finite_scalars` for each projection. Each
conversion owns a fresh memo, so the encoded Program contains two applications.

The exact authored Program hashes differ:

| Image | Parent | Candidate |
| --- | --- | --- |
| Plain finite materialization | `4e37055ac8f5584e0697bb67ad26dcd7fb313fbd7df437d81312e75dbdeae386` | same |
| Mixed scalar/finite composition | `78d710ece2e46d11e66a9d3e9fa33d044b6905c0b9781c18e8e31b4ecea6beba` | `d4ebedf7e287cfcf5017dcc7815a7d6b7dccab07fc0ce612c3fe3b837dae34cc` |

A smaller independent `encode_expressions((Const(1)+vec[0], Const(2)+vec[1]), None)`
probe produces one application/nine nodes on the parent and two applications/
thirteen nodes on the candidate. Finite-vector arithmetic also changes encoded
node sharing, even when its mathematical DAG stays identical.

`canonical_hash_data` structurally interns equal subexpressions, so its mathematical
identity stays equal in these cases. A later compiler CSE can likewise merge equal
work. Neither fact repairs the already changed Program serialization/hash. No
claim of a numerically wrong native trajectory follows from this review; the
demonstrated defect is lost joint encoding and Program identity parity during a
layering refactor. The author and integrating worker were informed before freeze.

## Received invariants and explicit differences

The new independent `tests/review/sol61_layering_908508a.py` authenticates separate
parent/candidate `git archive` Python snapshots and imports each in a fresh process.
It receives 31 checks, including:

- Seven identical structural mathematical DAGs: apply, solve, exact
  Fraction/Decimal inputs, finite-first arithmetic, finite-vector arithmetic,
  unary operations and comparisons, plus the mixed composition above.
- Identical scalar reconstruction, cross-state joint reconstruction and source
  face-policy bodies/options/source identities; identical state-handle inspection
  images; identical real Program hash/attributes for plain finite materialization.
- Ten explicitly refused conditions: foreign/permuted support, NaN coefficient,
  infinite input, width mismatch, vector return from scalar reconstruction,
  Boolean offset, missing face stability, face width mismatch and symbolic truth.
- Independent all-scope AST scans including function-local and relative imports
  in the five affected ownership modules, and byte-identical architecture gate.

The scalar/application plans use explicit `pops.finite-scalar-plan@1` and
`pops.finite-application-plan@1` tags. The IR consumer validates operation,
support order, coefficient shape/finitude and input width and refuses malformed
protocol/cycles in the replayed author tests. Existing `finite_linear_v1`,
`finite_projection_v1`, `finite_support_v1`, reconstruction and face protocol
tags remain. `StateShapeHandle` is a nominal model-owned abstract contract;
board and registry implementations retain their original public classes.
No duck shape is admitted by the received author test. Numerical policies depend
on the allowed model contract; finite algebra depends only on standard libraries.
The import graph/allowlists were not weakened.

Refusal semantics remain fail closed, but wording/classes are not literally
unchanged: infinite input now says "finite scalar literal must be finite";
symbolic finite truth raises plain `TypeError` rather than the parent's more
specific `SymbolicTruthValueError` and loses its contextual suggestion. These
differences are recorded rather than hidden by normalizing exception receipts.

## Reception and reproduction

126 selected source tests pass in 12.64 s: unchanged import/time architecture,
the author's layering tests, finite maps, temporal expressions/adversarial
expressions, scalar/joint reconstruction, source face authoring/adversarial
policies and exact literal codegen. Ruff and `git diff --check` pass.

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_layering_908508a.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/python/unit/numerics/test_symbolic_policy_layering.py \
  tests/python/unit/numerics/test_finite_linear_maps.py \
  tests/python/unit/time/test_program_expressions.py \
  tests/python/unit/time/test_program_expressions_adversarial.py \
  tests/python/unit/numerics/test_user_reconstruction_adversarial.py \
  tests/python/unit/numerics/test_user_joint_reconstruction_independent.py \
  tests/python/unit/numerics/test_user_face_authoring.py \
  tests/python/unit/numerics/test_user_face_independent_review.py \
  tests/python/unit/codegen/test_exact_scalar_literals.py \
  tests/python/architecture/test_import_graph.py \
  tests/python/architecture/test_time_package_layout.py
```

An initial attempt setting `POPS_NATIVE_DIM=2` failed loader preflight because
this isolated source tree has no native variants manifest. The successful source
run explicitly unsets that selector; it neither bypasses a failing native test
nor claims installed/native execution. No setup/build/install was needed or run.

Raw receipts and source identities are emitted under
`outputs/sol61-layering-908508a/receipt.json`. Python archive SHA-256:
parent `b2a8d77345da0cf03f0f297c3543ac8f7838c60686e101a7d5e65b0f10a53d53`,
candidate `704613c269689bc8694b3ecf2beb2941e46cfcb4ac549ae222d931efa8d26c82`.
The script prints the confirmed regression separately from compatible checks.
Native compile/bind/run, MPI, GPU, checkpoint reception and GitHub CI remain
outside this bounded source review.
