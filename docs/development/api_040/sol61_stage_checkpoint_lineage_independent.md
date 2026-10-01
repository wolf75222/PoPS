# Independent real Stage checkpoint lineage review

The closed ROOT campaign is
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-corrected-serial-dim2`.
Its actual JUnit has **eight failures in eight cases**, not scientific acceptance.
JUnit SHA256: `9c33efd0dad9a4d452f4afaf1b2f8dfc6840f70b711a91398a2cb4613b21b6a1`.
Log SHA256: `67caaf8b268cfbe392f002054e677d163960606b2a1d6700babf617fb669f194`.
The failure at raw checkpoint file equality precedes `check_saved`; no positive
Stage science or complete restart qualification is inferred from this review.

The independent reader uses NumPy and stdlib only, without importing PoPS,
loading native code, compiling, or editing the donor. It reads the direct
eight `s0`–`s7` directories, avoiding pytest's `scurrent` alias. All 24 real
checkpoint manifests, array-evidence hashes, restart identities and tokens
authenticate against an independently implemented deterministic CBOR encoder.
All 24 run identities independently recompose exactly.

For every scalar case, 47 of 49 NPZ leaves are identical in dtype, shape and
raw bytes between continuous and replay; for every coupled case, 74 of 76
are identical. The two different leaves are `pops_checkpoint_manifest` and
`pops_restart_identity`. Within the manifest, only the run/restart identity
hex digests differ. Arrays, clock, semantic/artifact/bind identity and all
other manifest fields are strictly equal.

The difference is required by the actual source contract, not an inferred
cache exception. `RunManifest` schema3 includes `continuation_identity` in
its payload. `begin_run` obtains it from `_restart_lineage_identity`.
`LifecycleMixin._restore_checkpoint_run_identity` publishes the authenticated
source run both as `_last_run_identity` and `_restart_lineage_identity`.
An exact Uniform restart is supplied that source run by RuntimeInstance's
authenticated restore transaction. The first run has start0, enddt, step0
and no lineage. The continuous second run has startdt, end2dt, step1 and no
lineage. Replay has the same latter clocks/controls with the accepted first
run identity as lineage. These three independently computed RunManifest
digests match the real files for all eight cases. The checkpoint restart
identity hashes the complete base manifest including this run identity;
its corresponding difference is therefore required as well.

Six countermodels per case (**48 refusals**) mutate disk copies of the actual
replay NPZ, recompute their array evidence and restart seals, then reload
them. State `Q0`, oldest `T0` history and native diagnostic bytes refuse at
strict payload comparison; a rescaled, consistently resealed clock refuses
at `t`; the continuous run digest substituted into replay refuses at `run
lineage mismatch`; a foreign artifact digest refuses at `artifact_identity`.
No seal or content is changed in the real archive. Twenty explicit synthetic
protocol tests verify CBOR wire vectors and reject ambiguous JSON/opaque
values. They never claim to be native checkpoints or scientific states.

```sh
rtk proxy env -u PYTHONPATH \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B \
  tests/review/sol61_stage_checkpoint_lineage.py \
  /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-corrected-serial-dim2 \
  --countermodels > /tmp/sol61-stage-native-checkpoint-lineage.json
rtk proxy env -u PYTHONPATH \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
  tests/review/test_sol61_stage_checkpoint_lineage_protocol.py
```

The actual resulting JSON contains the unchanged paths and SHA256 of all 24
files and the exact refusal messages; its SHA256 is
`f4316a96faeb007c5812c2397f9ae6365bddbc61d5cfb622a5219f8fe5c6f815`.
This is a local observed-byte review, not an external ROOT owner approval,
an independently authenticated build receipt, or a CPP-to-DSO link proof.
The reader is deliberately specific to these closed, fixed-dt, exact Uniform
restart witnesses; no general lineage policy or AMR regrid authority is
inferred. Runtime validation must continue to authenticate each complete
checkpoint before mutation. A fixture replacement must validate provenance
from the actual runtimes, compare all payload leaves strictly, and compare
the remaining manifest fields exactly rather than drop both manifests.
