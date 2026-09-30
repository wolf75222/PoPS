# Bound initial checkpoint envelope — 30 September 2026

Base: `5e69b3425184446ea2cd318fe2d31d038cca3035`, exclusive
`PoPS-sol61-initial-checkpoint`. This is the second production lifecycle correction
after initial controller installation in `03cf450`.

The parent's installed native reception exposed two further refusals before
evolution: Uniform/AMR capture plans dereferenced `last_run_identity=None`, then
the canonical sealer required a prior `pops.run`. Both seams now authenticate an
explicit initial bind origin instead of inventing a run or its execution controls.

Initial envelopes use schema **2**, `origin={"schema_version":1,"kind":"bound_initial"}`
and `run_identity=null`. Sealing authenticates semantic/artifact/bind identities,
the actual native clock, existing temporal controller/cursors and zero attempted
transactions. The initial clock is currently native `+0/q0`, established by the
existing constructors; this is an authenticated current bind invariant, not a
mathematical restriction on future explicitly declared nonzero initial origins.
Checkpoint is read-only. Missing external runtime controls still refuse normally.

Run-origin envelopes retain explicit emission version **1**, their exact fields
and tokens. This constant is independent of generated supported-schema metadata.
The parent owns the central supported version 2 change in `ff726dc`; that commit
is not copied here because this review branch retains its original native SDK.
Both manifest validators admit the exact two schemas, and the derived manifest
character budget covers the new origin. Payload codec versions are unchanged.

Uniform/AMR capture plans retain a real run identity when present and carry the
initial origin otherwise. An initial restart restores no run identity/lineage;
the first actual run creates its normal RunManifest with actual execution controls.
The existing authenticated restart receipt remains available. Post-run restoration
and post-run regrid lineage preserve their existing behavior. Regridding an initial
source does not manufacture a run before any execution request exists.

The public RuntimeInstance reseal obtains the accepted temporal state through
an explicit executor authority route and exposes the executor's actual run
manifest. It owns neither a duplicate controller nor an invented run. This fixes
the first native reception's ten initial-checkpoint failures at the facade layer.
The installed ABI5/SDK307 reception subsequently passes 84 temporal/checkpoint
tests and all six transport/restart/refusal fixtures. Three of four diffusive
fixtures pass; the affine two-level AMR boundary balance remains unqualified.
No wire version or legacy envelope token changes for this private ownership fix.

Composite capture validates every live temporal leaf and retains actual child
checkpoint bytes. At restart, each child is decoded/authenticated by its existing
native preflight and resolved resource budget first; only then the composite
checks initial child origin/temporal authority before publication. Envelope
inspection does not inflate nested archives ahead of child resource admission.

The installation-tail test now authenticates the actual implementation module files
under the loaded `pops` package, so source and installed reception are both valid.
It no longer requires the checkout's Python path or admits an unrelated prototype.

## Evidence

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/python/unit/runtime/test_bound_initial_checkpoint_envelope.py \
  tests/python/unit/runtime/test_initial_checkpoint_strategy.py \
  tests/python/unit/time/test_step_transaction_contract.py -q \
  --junitxml=outputs/bound-initial-source-20260930.xml
```

Result: **40 passed**, no skips, 1.12 s. Source authority is this worktree's package;
storage/ABI probes are explicit unit seams. Coverage includes real Uniform/AMR
installation tails, exact controller and initial ErrorControlled proposal queue,
actual NPZ serialization/inspection, both strict manifest validators, read-only
state, initial restore followed by the real run-manifest creation seam, legacy
seal/token equality, composite child origin, and rehashed forged origin/clock/run
refusals. Unbound/missing-controller/native-clock/temporal-clock/transaction
divergence fail before initial envelope publication. The supported metadata alias
is checked against the actual generated value; legacy emission remains 1 and
initial emission 2 even when integrated metadata becomes 2.

Ruff and `git diff --check` pass. Existing public native Integral/ND tests are
unchanged. This worker did not execute native evolution/restoration, MPI/GPU,
regrid-on-restart, SDK installation or GitHub CI. The parent's six public native
receptions and independent reviewer remain the integration acceptance evidence.
No C++ header, shared environment or central release-contract file was changed.
