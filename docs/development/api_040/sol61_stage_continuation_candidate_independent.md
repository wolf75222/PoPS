# Independent candidate continuation comparator reception

Candidate `a0f92caad8da3c5c1e6bd99b58aff7cf07573e5b`, exact parent
`c3802111e9f0151c7b9ab43e2fd629c967a56a60`. This commit changes two review
test/document files only. No native execution, JIT, build, installation or
donor modification occurred.

The exact candidate `compare_checkpoint_replay` function is extracted from
its fixture AST and executed against the eight real failed-campaign trios.
Its real source `inspect_checkpoint_payload_integrity`, `RunManifest` and
identity methods remain in use. Expected authority data is constructed from
the independent NumPy/stdlib reader's own CBOR derivation, not the author's
tests, `FixedDt` helpers or seal implementation. That independent reader
recomposed all 24 run digests/restart seals exactly in review `8a7174bc`.

All eight real pairs pass the candidate's exact comparison. Forty-eight
mutations of disk copies, fully resealed by the independent implementation,
refuse: state, history, native diagnostic bytes, changed clock, wrong run
lineage and foreign artifact. Mutated copies are temporary and never replace
the donor. To reach deeper checks, the mutable test restart authority is also
updated to the new outer seal; the artifact attack additionally updates its
claimed artifact authority. Full manifest comparison still refuses it.
No expected result comes from a native callback or from author oracle code.

The live producer's source protocol is coherent: it calls actual
`authenticate_checkpoint_payload(runtime, ...)` before collecting authority;
this verifies the complete checkpoint seal, semantic/artifact/bind identities
against the real RuntimeInstance install plan, and ABI against loaded runtime.
The RuntimeInstance's `_checkpoint_identities`, `time`, `macro_step` methods
and three lifecycle identity/manifest properties genuinely exist. The helper
then requires the actual run manifest's identity to match runtime.last_run_identity.
The comparator reauthenticates every full file, typed run manifest, identities,
clock, and continuation relation. It compares every non-envelope leaf in
dtype, shape and raw bytes, and every remaining manifest field exactly.
Only individually authenticated run/restart identity fields differ; they
are not silently deleted without validation.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B \
  tests/review/sol61_stage_continuation_candidate_reception.py \
  --campaign /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-corrected-serial-dim2 \
  --independent-reader /Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-checkpoint-lineage-review/tests/review/sol61_stage_checkpoint_lineage.py
```

Result: eight pairs and 48 refusals, source runtime API checks PASS; Ruff and
diff checks PASS. No new blocker is demonstrated. The live creator itself is
**not** exercised on a genuine RuntimeInstance here: runtime/source tracing
is distinct from ROOT's pending native execution of that producer.
Expected host authority data is explicitly reconstructed, not presented as
a historical live authority sidecar. The eight historical JUnit failures
remain failures, and scientific `check_saved` was not reached in that
campaign. These tests do not replace fresh Stage science/restart acceptance,
MPI/AMR coverage, an external owner approval, or a compiler-to-DSO link proof.
