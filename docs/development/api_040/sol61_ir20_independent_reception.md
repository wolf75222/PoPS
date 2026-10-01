# Independent IR20 original-interaction source reception

2026-10-01. Reviewer checkout `PoPS-sol61-ir20-review`, branch
`codex/api040-sol61-ir20-exact-review`. Production received:
`0a5748aa6c19c9afa58ff566ee4dbe33671675a8`, then the two distinct corrections
`652e8605ff62aa46b6a1c15f7b66ceaa9827ef9e` (geometry getter) and
`5222a1b99bbdde2041b535ddc5c65a6bb38f875e` (uint64 identity).
Author tests/docs `e64a8c5d` are not independent execution evidence.
No MAIN, installed environment, native build, or native artifact was modified.

Two concrete source defects were demonstrated before correction:

* The public `Case.field → bind_program_inputs → Program.solve` route admitted
  `DirectSpatialInteraction(2**63)` and `2**64-1`, then failed in
  `_program_nonlinear_problem._request_data`, `make_identity("solve-equation", …)`:
  `canonical CBOR integer at $.payload.interactions.realization.max_workspace_bytes
  is outside signed int64`. The initial independent run had 10 passing checks
  and those two failures. `5222a1b9` retains low-range integer descriptors and
  encodes high uint64 through the existing exact integer literal protocol;
  all three boundary budgets now reach the actual emitter as exact `ULL` values.
  No budget or identity integer-domain limit was relaxed.
* `original_candidate_interaction` called the actual potentially throwing
  `prepared_amr_level_geometry` before `interaction_phase`. The extracted
  source loop, with a throwing facade substitute, refused without reaching a
  vote. `652e8605` moves that getter inside the phase. The same counter now
  reaches the vote before rethrow; the historical `0a5748aa` loop remains a
  separately pinned outside-vote counter. This is a source/host exception-order
  proof, not a demonstrated MPI deadlock.

Independent checks are in `test_sol61_ir20_authority_independent.py` and
`test_sol61_ir20_original_math_independent.py`:

* Actual public two-unknown FieldProblem authoring, ordered components and
  signed scale, IR20/request `pops.spatial-field-residual@4`, full central-FD
  declaration, row/column/scale tampering refusals and capture retiming refusal.
  Actual Case validation/resolution/emission runs for Uniform and two-level AMR,
  with permuted unknowns, cells 7×5, non-origin physical bounds, and maximum
  uint64 budget. Emission uses the live F candidate, not the IR19 accepted-source
  sealer. These are emission checks: no compilation, bind, or solve is claimed.
* Verbatim `require_evaluation`, producer-lease block, and reentry guard from
  the actual PreparedAmrFieldResidual header are compiled and executed under
  UBSan. Explicit vector-storage, authority, provider and lane substitutes
  receive exact source/tower identity, wrong same-layout tower, foreign provider,
  foreign lane, stale epoch, stale nonce, reentry, revocation before the physical
  body, exception unwinding, monotone retry nonce, and nonce overflow. Reusing
  the same buffer for different plus/minus values changes the observed producer
  result; a previous nonce is refused. No Kokkos/device storage, MPI communicator,
  full Newton/GMRES solve or all-rank consensus is executed by this host probe.
* An autonomous Fraction/NumPy oracle contracts signed nonsymmetric kernels
  against positive nonuniform measures for sizes 3/5/7 and widths 2/3/5. It
  discriminates seed freezing, omission of the nonlocal variation in JVP,
  same-address pointer caching, component/source substitution, permutations,
  rank-local omission and duplicate contribution. Its candidate-prefactor
  variant is an additional mathematical counter-model, **not** a claim that
  multiplying the authored SpatialInteraction term by an arbitrary candidate
  expression is admitted by the current public API.

Source inspection also follows both synchronized branches: frozen D copies and
synchronizes q before the interaction producer; candidate D synchronizes q,
evaluates/prepares D, applies the operator, then invokes that producer on the
same physical q. The lease is lexical and revoked before `add_local`, or on an
exception. The generated AMR producer calls the direct kernel each invocation;
its ordinal is per F, not the shared Newton index. The detached output tower is
not an accepted publication. The context rechecks attempt/point/provider/epoch/
materialization before return and verifies the Core lease again.

The per-term explicit budget does not bound total solver RAM or the sum of
several simultaneously retained interaction terms. Existing quotient ownership,
active/coverage masks and volume measure are delegated to the common direct
kernel; this reception does not requalify its actual MPI transport or device
reduction. Unknown physical-units schema gaps are not filled with invented
physics. Private leases are not checkpoint resources. No complete nonlocal PDE,
Vlasov model, GPU execution, AMR EB, or native convergence is received here.
The author's six fresh legacy byte-parities are separate evidence; this cohort
checks unchanged low-range resource descriptors and omission of IR19 snapshot
routes, and does not claim a new independent full legacy image campaign.

Reproduction (source-only Python, no setup/install):

Final cohort: **25 SOURCE_ONLY/math/host checks passed in 9.94 seconds**.
Ruff and `git diff --check` passed. The host compiler executes only the small
extracted fragments; no full native translation unit was built by this reviewer.

```sh
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_ir20_authority_independent.py \
  tests/review/test_sol61_ir20_original_math_independent.py \
  -p no:cacheprovider --tb=short
```

The historical geometry probe explicitly requires the local `0a5748aa` Git
object; it never fetches or installs anything. Native execution on the rebuilt
SDK remains ROOT's separate reception responsibility.
