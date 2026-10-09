# Independent typed Field/Ghost capture review

Reviewed exact author freeze 83a5c265652f328c3f3d31d3a3ea5eb4dbe318e8 in the separate field-candidate-oracle-independent worktree (local copied author commit 88a8e6f9). No production, Native, ENV, Main or original evidence writes.

## Red evidence retained

Independent positive receive against the real SDK14 export failed before correction: 1 FAIL in 0.30s, string indices must be integers at reader v2 line38. Signature is a 20109-character lossy string with nested mappingproxy representations. Commit a88a63a211984cfa2c4598d1ee15110fe7637e9f retains that reproduction. ROOT chose a new authentic capture rather than decoding this representation. The old export remains failed; it is not retrospectively received.

External authority: /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk8d438518-src2b0b7524-initial-field-ghost-serial-dim2/root-export-pins.json

Actual SHA256: c5211dcba1ec28ba10958561620375628d123e518e28c30b04c23903340fa3d0. All 28 pinned files rehashed equal. The old export now receives an explicit signature ValueError from reader v3, after complete inventory/hash verification. Missing/extra inventory and wrong digests are refused before signature or math. No replacement pins or receipts are minted by these tests.

## Source correction and independent checks

Fixture uses the genuine ComponentManifest.to_data()['signature'], JSON serialization with allow_nan=False and no default coercion. Profile is versioned accepted-initial-field-ghost-public@4; reader v3 requires a typed dictionary, and refuses both Python repr and JSON strings as signature substitutes. Old reader v2 source was not modified. The actual public Source model resolves and its nested typed manifest serializes/roundtrips; this is a Source witness, not an installed Native capture.

Independent nested adversaries refuse nonfinite values, boolean keys, mappingproxy objects and tuples; JSON duplicate keys and nonfinite constants are refused. Declared flat State layout accepts only float64 length 2*N*N for N8/N16, with component-major reshape. Ranked arrays, wrong lengths/dtypes, unknown and boolean sizes are refused rather than guessed. Actual initial/accepted/reloaded flat arrays128/512 cross-check in bits against all nine recorded carrier valid patch slices. This proves the layout of those pinned data only; the old signature remains unacceptable.

Coherent command (Source-only, external evidence read-only):

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 SOL61_EXPORT_PINS=/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk8d438518-src2b0b7524-initial-field-ghost-serial-dim2/root-export-pins.json /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_field_candidate_oracle_independent.py tests/review/test_sol61_field_candidate_math_migration.py tests/review/test_sol61_initial_field_ghost_native_preparation.py tests/review/test_sol61_field_ghost_typed_provenance.py tests/review/test_sol61_typed_signature_independent.py tests/review/evidence_sol61_field_candidate_signature_independent.py -q --tb=short

Actual result: **59 PASS in 6.72s, zero skip**. This includes the previous four temporal/consumed-strip counterexamples unchanged and five explicit external-evidence checks. Evidence test filename is deliberately outside automatic test_ collection and requires an explicitly supplied authority path; it is not a portable Native fixture.

## Verdict and limits

No remaining blocker in this bounded typed capture/reader Source preparation. OriginalF1e-10, FE consumer point versus owner stamp, and retained-stage cache distinction are unchanged. Producer candidates initial/accepted alone support endpoint OriginalF; reload is observation-absent and cache bits only. Pins supply external provenance authority; reader does not issue Root approval or independently establish source/DSO identity. No positive receive of an authentic new typed Native export has occurred here. ROOT must rerun the fixture and authenticate the new export before SCI reception. No new MPI, Field corner/general halo readiness, or bootstrap failure qualification is implied.
