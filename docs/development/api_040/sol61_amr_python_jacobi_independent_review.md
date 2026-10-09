# Independent Python reception of the AMR right realization

Reviewed on 1 October 2026: production
`c28a8f4370bbf2e9d781ddf52ddcc59597ce2e65`, parent
`3a03a2ef49d37c6fe779db85672fd2b1a89cf86f`, and frozen author tests/docs
`44cb4a874e61c117b7e37a9dfbf10b2eb534f8e1`. The exclusive review checkout is
`PoPS-sol61-amr-python-jacobi-review`. No MAIN, shared environment, installed SDK,
C++ header, JIT or native build was changed. The independent files contain no
production correction. No new P1/P2 is demonstrated in this bounded Python review.

The prior native-core **source/host** review is a separate freeze, `7e0fb5b2`.
It requires c7 + ec25713 + 3a03a2e together: the prepared provider authenticates the
supplied lane inside the prepared lane's collective context. Neither freeze
qualifies Kokkos/MPI execution, a completed public AMR trajectory or GPU operation.

## Actual Python lowering and refusals

The new independent test constructs the public three-component permuted original
FieldProblem witness, with distinct State captures and a separate seed. Its
physical construction helper is reused, but its contract attacks and predicates
are distinct from the author's 21 tests. It runs actual validate/resolve/emit:

- Selected C++ contains exactly one
  `AmrFieldRightPreconditioner::kSpatialBasisJacobi` prepare argument. The entire
  emitted original callback/solve/consumer suffix is byte-identical to identity
  realization at the same Program node ID.
- The original equation identity and seven Newton controls agree exactly.
  Solver identity, authored IR and realized C++ distinguish the realization.
  The selected immutable prepared identity is computed independently as
  `prepared-spatial-newton-v2` over controls and the exact
  `pops.amr.original-spatial-jacobi.basis-response@1` URI.
- Actual scratch metadata adds exactly one persistent inverse field. The
  explicit cost note says 1 + stored DOFs composite operator applications.
  This is not a linear-work complexity guarantee: each basis evaluation applies
  the full original operator. There is no size cap or efficient FAC claim.
- Request schema 2, realization URI, solver identity, absent/null policy and a
  fully redigested unknown policy are checked by the real request validator.
  Every injected mutation refuses without changing the issued Program.
  Prepared-policy changes and descriptor private-storage corruption also refuse.
- Uniform resolution rejects before the emitter is called and preserves the
  authored Program hash. Other adapter/installed-field-plan refusals are received
  separately through the author source selection; no silent fallback is added.
- A routing-only nested-region probe receives recursive IR8/IR9 promotion using
  the actual serializer. It deliberately does **not** qualify a lazy original
  solve: its public builder requires top-level authoring. An actual owner-issued
  solve injected into dt_bound is rejected for its non-readonly operation before
  serialization. The negative injection is restored and the Program hash agrees.

Default options omit the policy; selected options retain it, and the prepared
identity separates it from the seven numeric controls. The request URI/schema and
IR discriminator enter the compiled identity/cache. No new choice is inferred
from a field name, shape, coefficient sign, or a physical equation. The witness
has constant cross-diffusion and two captures used by loads/local reactions. This
reception does not cover the future captured-diffusion extension.

## Fresh legacy equality with complete provenance

`sol61_amr_python_jacobi_parent_parity.py` is an explicit external review helper,
not a unit test requiring a historical Git object. Each fresh interpreter imports
the requested checkout's Python implementation and invokes the **same fixed
physical fixture and callsite**. The parent checkout's Python is byte-equivalent
to 3a/c7; its additional independent core tests do not modify that implementation.
There is no fetching or installation. The helper returns full request/manifest
objects before comparison, with no provenance normalization.

Three fresh parent/candidate profiles agree exactly: width 2 AMR / (1,0), width 3
Uniform / (2,0,1), width 5 AMR / (4,2,0,3,1), each with an explicit distinct seed.
Compared fields are authored IR, resolved IR, IR8 discriminator, complete emitted
C++ digest, three Module hashes, three full Module manifests, original equation,
solver identity and complete SolveRequest. Full manifests include their actual
provenance. This is stronger than replacing path-dependent golden plan IDs.

The companion `sol61_amr_python_jacobi_independent_receipt.json` stores compact
canonical SHA256s of every manifest/request field after strict full-object
comparison. The private full comparison remains under
`outputs/sol61-amr-python-jacobi-independent/full-parity-receipt.json`, with its
hash in the committed receipt. The independent helper hash is also recorded.
No installed/native receipt is synthesized from these source profiles.

Reproduce the external comparison from this checkout:

```sh
PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" tests/review/sol61_amr_python_jacobi_parent_parity.py ../PoPS-sol61-amr-right-preconditioner-review > /tmp/sol61-jacobi-parent.json
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" tests/review/sol61_amr_python_jacobi_parent_parity.py . > /tmp/sol61-jacobi-current.json
# Compare both profiles arrays in full, ignoring only the top-level source/package
# inventory paths. Do not normalize any profile or provenance field.
```

## Checks and limits

```sh
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q tests/review/test_sol61_amr_python_jacobi_independent.py
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q tests/review/test_sol61_amr_python_spatial_jacobi.py
rtk proxy "$PY" -m ruff check tests/review/test_sol61_amr_python_jacobi_independent.py tests/review/sol61_amr_python_jacobi_parent_parity.py
rtk git diff --check
```

Independent final selection: **11 PASS in 48.52 s**. Author source selection:
**21 PASS in 115.87 s**, run separately on the same frozen implementation. Thus
32 distinct source cases passed; no omnibus native suite is inferred. Fresh legacy
comparison: **3 exact-equal profiles**, including complete provenance. These are
source checks, not GitHub CI or native execution.

The new native lane test `52f44138` was also read independently: stage calls the
actual authority callback twice, pre-Accept validates storage/finitude on the
prepared lane before the third callback, only rank 0 supplies borrowed world,
rejection leaves the report/candidate/publication intact, and authenticated retry
calls the fourth validation before publication. This is source-order reception;
its actual serial/MPI acceptance and rollback remain root's responsibility.

Root still needs the matching rebuilt SDK/source and real serial/MPI original
residual/authority/candidate/rollback/checkpoint tests. Source parity and changed
realization do not qualify the identity N32 budget-240 profile, original physical
AMR convergence, variable captured diffusion, GPU, or a scalable diagonal provider.
