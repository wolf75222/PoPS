# T5 independent offline scientific reception: prepared contract

Status: **pending authentic native receipts**. This commit prepares an offline
oracle and its source/protocol tests; it qualifies no PoPS execution. No saved
physical states, positive native receipts or replacement runtime data were
fabricated. No PoPS import, MAIN/environment change, native build or JIT is needed.

The exclusive checkout starts at
`8eabb8e2875158e3f4f8a0ed0255b0a63db4b6c4`. Its frozen source contract is
`test_public_integral_feedback.py`, `test_integral_candidate_capture.py` and
`integral_state_receipts.py`. The 21 tests receive protocol refusal, deterministic
CBOR vectors, exact frame metadata, frozen source/file contracts and a CLI run
whose import hook forbids PoPS. They do **not** exercise the oracle against
synthetic fields. Native paths have not yet been supplied by the integrator.

## Discrete oracle from saved endpoints

The actual public fixture is Uniform Dim2 on the unit square, `(N,1)` cells,
periodic in y, outflow in x, one density component, velocity +1, first-order
upwind numerical flux. It receives N=8 and N=16. Constants are unchanged:
`h=.01`, `gamma=.3`, `q0=.7`, absolute tolerance `3e-13`, zero relative tolerance.
The declared initial cell averages are `1+.2*(i+.5)/N`.

For **each saved pre-step state** and its saved q, independently evaluate:

```
U*_i = U_i (1 - gamma q h)
U'_i = U*_i - h N (U*_i - U*_{max(0,i-1)})
q'   = q + h U*_{N-1}
```

The source is the original reaction `-gamma*U`, scaled by the typed candidate q
captured at the pre-step point. There is one reaction update, followed by one
transport Euler update; this fixture is Lie composition, not SSPRK2. The oracle
uses explicit independent cell-face incidences and `math.fsum`; it reads actual
saved arrays and never writes predicted arrays as runtime snapshots.

The mean-density budget is
`mean(U')-mean(U) = -gamma*q*h*mean(U) + h*(U*_0-U*_{N-1})`.
q receives only the outgoing right trace. The offline checks also require exactly
`4N` unique global cell incidences, all native flux densities/measures/components,
periodic/exterior flags and incidence orientations. Their actual native ledger
sum must telescope to the same boundary budget. Across all rank images there must
be **one** selected x+ trace and **one** matching consumed key, with orientation
-1, multiplicity 1, measure 1, weight exactly h, component 0, and quadrature
`cell:N-1:0/axis:0/side:1`. The consumed transfer scale -1 therefore increases q
by the positive delivery. Runtime frame clock, tick, stage, phase, dt bits,
physical-time bits and source-evaluation identity are checked, using the declared
primary clock from the authenticated temporal checkpoint.

This is the oracle for this particular public fixture. The shape and incidence
conditions are neither a generic production restriction nor AMR/3D qualification.

## Files and external authentication

Each actual `save_public_snapshot` produces:

- `<phase>-state.npz`: `state`, `level_0`, q, time, step;
- `<phase>-receipt.json`: artifact/platform/dimension, quantity identity/value,
  accepted clock, checkpoint path/hash, rank-image hashes and decoded ledgers;
- a native checkpoint NPZ at the recorded checkpoint path.

The helper hashes the checkpoint and ledger images, but **does not hash the
saved-state NPZ**. The receiver therefore requires externally pinned SHA256 for
all three files, plus a pinned authentic runtime identity JSON. It does not trust
an artifact's own newly recomputed checksum as sufficient provenance.

The pins manifest schema is `sol61.integral-feedback.offline-pins@1`. Required
top-level fields are `schema`, `source_commit`, `abi_key`, `native_sha256`,
`identity_file`, `dimension`, `size`, `cases`. `identity_file` and every file pin
have exactly `{path,sha256}`; relative paths resolve beside the pins manifest.
The authentic identity JSON must agree on source, ABI and native hash. The source
contract SHA above describes the oracle's fixture, while `source_commit` must
name the **actual rebuilt runtime**, which can be a later integrated commit.
The identity JSON is documentary provenance, not a fresh DSO execution/hash run.

Each case specifies `kind`, `cells`, actual `artifact`, actual `platform`, actual
`quantity_identity`, and `phases`. Every phase specifies pinned `receipt`, `state`
and `checkpoint`. The complete campaign is required exactly once each:

| Case | N | Required phases |
| --- | --- | --- |
| restart | 8 | initial, accepted, continuous, restored, replayed |
| restart | 16 | initial, accepted, continuous, restored, replayed |
| retry | 8 | before, rejected, retried |

No mock/example receipt manifest is shipped. `--describe-contract` prints these
structural requirements with `scientific_status=pending_receipts`.

Before reading arrays, the reader validates file pins and ZIP member inventory
and bounds compressed/decompressed bytes. A 64 MiB reader budget is specific to
this small reception fixture; it is not a runtime rank/field limit. `allow_pickle`
is false. The reader cross-checks saved state against the actual physical
`state_fluid` checkpoint array, ABI and clocks; independently reconstructs typed
array evidence and the deterministic CBOR restart envelope digest; partitions
the actual `program_exchange_state` using exact rank offsets; independently
decodes POPSEX02 with bounded counts/strings, no trailing bytes, finite amounts,
unique occurrences and consumed-key support; and compares those bytes and their
hashes with the JSON receipt. Initial q declarations must match on every rank.

## Retry, restart and publication limits

Initial/before checkpoints require explicit schema-v2 `bound_initial` authority,
no run identity, and canonical +0 clock. After the **real rejected run**, a
schema-v1 run identity is legitimate at clock zero: `begin_run` publishes the run
manifest before the attempted step. This distinction follows the real lifecycle
and must not be replaced by `step==0 => bound_initial`.

The rejected attempt must preserve physical state, q bits, accepted clock and each
rank's entire exchange image. Accepted clock/schedule/synchronization/history/cache
cursors must also be preserved. Failed-attempt statistics and run provenance can
legitimately change. A safe retry must satisfy the fresh-step oracle and match the
independent fresh N8 accepted state/q exactly. Restore must match accepted state,
q, clock, rank-ledger bytes and accepted cursors; replay must match the continuous
run exactly and satisfy the second-step oracle using saved pre-step q.

The snapshots do not save the caught exception text. An offline result cannot
prove the rejection's `user_face_numerical_stability` diagnostic or native test
success by itself; the integrator must attach the authentic pytest/log result.
Likewise, rank images permit global counting, but this tool does not launch MPI,
prove absence of collective divergence or replace native serial/MPI reception.
No endpoint budget, equation, test constant or author fixture was changed.

## Commands

Source/protocol reception (21 PASS):

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); import pytest; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_integral_feedback_offline_contract.py"]))'
```

Describe the pending contract:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_integral_feedback_offline_oracle.py --describe-contract
```

After the integrator supplies actual receipts and pins, invoke the same standalone
script with `--pins` and the authentic manifest path, optionally `--output` for a
new independent JSON report. A scientific PASS is emitted only after all real
phases are received. Missing pins/files or any failed invariant aborts before an
output report is created. There is currently **no scientific PASS report**.
