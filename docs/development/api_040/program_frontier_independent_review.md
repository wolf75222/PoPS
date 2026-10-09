# Independent Program computed-frontier reception

Reviewer: GPT-6.1 Sol, independent of the production author. Date: 2026-09-30.
Review checkout: `PoPS-sol61-coverage-review`, branch `codex/api040-sol61-coverage-review`.
No production files, shared environment, native extension or primary checkout were modified.

## Source identities and result

The reviewed correction is `ccfa25af07f592638c6a4b09925f116a08543804` in
`PoPS-computed-frontier`. `freeze_frontier_source.py` obtains the inputs from
`git archive`, excludes native extensions and hashes every archived file before
the independent probe imports the package. The 750-file source projection has
SHA-256 `be7a2ea8a40769bf4a700318cceca86f40841780da16f7ec96a7ddedb7c03450`.
The checked-in receipt `tests/review/evidence/sol61_frontier_ccfa25a.json` records
the exact source, probe, three header hashes, interpreter and host compiler.

Final result: **5/5 independent source/math/host checks pass**, exit 0 in 3.75 s.
The production author's unit tests were not used as the independent oracle.

The earlier mutable source snapshot, based on `f6aad44`, was separately frozen
with tree SHA-256 `57ee7ea08320d929adedb025d019a7c5fbf19180005517c212e841e48f1fc965`.
It produced a real source defect: a public FV rate inside either branch region
escaped the top-level support gate, emitting two `neg_div_flux` calls and one
`reached_duration` despite the `no_spatial_exchanges` contract. The defect was
reported immediately to the author and integrator. The same public counterexample
is refused by the independently archived `ccfa25a` correction. Old red results
are discovery evidence, not reception of the correction.

`sol61_frontier_f6aad44.py` is recovered byte-for-byte as historical evidence.
Its old ExternalTimeGrid corruption results belong to the original `f6aad44`
source only; they do not qualify the new ComputedDt implementation.

## Independent evidence

| Contract | Evidence | Result and limit |
| --- | --- | --- |
| Relaxed RK duration is computed | Fraction oracle for rotation, initial `(1,0)`, Heun increment `(-1/2,1)` gives gamma `4/5`, candidate `(3/5,4/5)`, squared norm `1`; actual public Program emits two dot reductions and no hardcoded `0.8` | Math and real lowering pass; native field values remain integrator-owned |
| Stage authority and request remain distinct | Public source evaluations have distinct points; actual extracted C++ setter accepts `.8` while retaining requested duration and partition step `1` | Host setter passes; no relabeling of the numerical request |
| Shared Scalar representation and ownership | Dot and requested-duration values share exact type and scalar vtype; foreign Program Scalar is refused; immutable ProgramGraph retains frontier operations without changing authoring IR hash | Conversion and authoring pass; this does not claim graph-native execution |
| Interval restrictions apply inside scopes | Public FV branch in both arms is rejected by recursive gate | Corrected source gate passes; AMR target also explicitly refuses v1 without interval remapping |
| Invalid endpoint cannot publish Python temporal state | Controller seam rejects zero, negative, NaN, infinity and one-ULP overshoot with strict zero endpoint tolerance | Temporal publication remains at entry; fake native has no rollback semantics |
| History duration is actual accepted duration | Extracted production setter retimes a pending publication starting at entry to `.8`, preserving past sample duration `1` and start bits | Host C++20 passes; duplicate and rounding-stalled endpoint are refused |
| Restart and live receipt authority | Honest JSON round trip; malformed request duration, last accepted dt, missing receipt, rejection count and endpoint limit rejected; live corruption prevents the next native call | Actual Python controller/restart functions pass |
| All-rank preparation agreement | Patched collective seam disagrees independently on policy, time, step, endpoint and prior receipt | Each refusal occurs before `native.step`; no real MPI execution |

The controller source accepts retry only after typed numerical rejection,
shrinks the request by the policy and enforces its rejection budget. Native
publication resides in the existing `RuntimeInstance._accepted_step_transaction`
envelope: snapshot/begin precede advance, then consumers and commit/finalize;
exceptions restore the native/Python snapshot. The source ordering is coherent.
This review does not certify native rollback of history, queued retries, scheduler
caches or accepted exchanges. Those require the integrator's rebuilt native tests.

The fixed legacy public Program was independently constructed and serialized
against archived `f6aad44` and `ccfa25a`. Both are schema version 5 and have exactly
the same IR/serialization SHA-256
`b83cde50f4464b399e803a7395ec59073b096890142d61d88f2cad056ab43414`
and semantic projection SHA-256
`d91cbc52e33fa5b13d58f9153a5345b31584ac99b398065eda536ba2a3e61e1f`.
This is one meaningful compatibility fixture, not an exhaustive legacy inventory.

## Exact reproduction commands

Run from the review checkout. A snapshot output must be new; do not overwrite
the retained evidence trees.

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/freeze_frontier_source.py --checkout ../PoPS-computed-frontier --revision ccfa25af07f592638c6a4b09925f116a08543804 --output outputs/frontier-sol61-ccfa25a-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_program.py --source outputs/frontier-sol61-ccfa25a-source --receipt tests/review/evidence/sol61_frontier_ccfa25a.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_legacy_identity.py outputs/frontier-sol61-f6aad44-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_legacy_identity.py outputs/frontier-sol61-ccfa25a-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/freeze_frontier_source.py tests/review/sol61_frontier_program.py tests/review/sol61_frontier_legacy_identity.py
rtk git diff --check
```

No remaining source/host blocker was found on this frozen revision. Native RK
values and exact stage clock observations, transaction rollback, actual MPI
agreement/deadlock behavior, and rebuilt ABI/extension provenance remain outside
this review's qualification.
