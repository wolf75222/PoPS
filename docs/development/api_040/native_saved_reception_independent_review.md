# Independent retained-native reception review, 2026-09-30

This review reads the genuine retained native payloads. It never imports PoPS,
calls a native backend, runs a JIT/install, imports the benchmark author's oracle,
or writes to the principal checkout or shared environment. Each corruption is
performed on a fresh private copy; the originals remain unchanged. Forged arrays
are adversarial fixtures, never substituted reception results.

## Frozen scope

- Checker SHA256: `224bbd62366a0aa9d084a8ae4027b2c5223a964e9edc972b6c8d32af00b2db4b`.
- Archive manifest SHA256: `cb06d1c9ce21dcbd8456e81bd4eaf77e974e57f46a6cfa4765c5191f7a58dfa8`.
- Archive: `docs/development/api_040/evidence/native-fc0-20260930` in the principal checkout.
- Inventory: 161 payloads, 6,079,553 bytes; 16 scientific snapshots (six M26,
  eight M27, two ND2). Snapshot source, logs, receipts and failed outer runs are
  retained together. These describe ABI4 SDK `fc0bfd9a…`, not subsequent ABI5 builds.
- Private checker source is retained as
  `evidence/native_saved_reception_checker_224bbd6.py`; observations are
  `evidence/native_saved_reception_independent_224bbd6.json`.

## Original data and mathematical criteria

All 16 genuine snapshots pass the frozen checker. Independent receipt-link
checks also pass: nine native/source identity receipts, 52 frozen source-file
digests, 14 saved-state witness digests, eleven JUnit summaries and one MPI
before/after native/ABI/source-files identity comparison. This corroborates the
retained internal links; a mutable manifest is not an external signature.

M26's dense kernel is `cos(2π(x_i-x_j))/12`. The measure appears both in applying
the kernel and in the pairings. The seven pairings, adjoint identity and quadratic
energy increment have the correct normalization. A second calculation uses the
analytic first Fourier mode of each prescribed density rather than the dense
kernel or the author's imported oracle. Its maximum potential error is
`1.1102230246251565e-16`. The maximum original pairing error is `2.22e-16`.
This is a twelve-DOF measured interaction, not an aggregation PDE.

M27 is backward Euler for `c_t=Δμ`, `μ=c-ε²Δc` with `dt=.01`, `ε=.08` and the
periodic centered finite-volume Laplacian. With positive Laplacian magnitude
`λ`, its multiplier is `1/(1+dt λ(1+ε²λ))`. The checker tests all retained `c`
and actual `μ` fields, both original equations, mean mass, discrete energy and
time. An independent dense `2N × 2N` solve of the original mixed equations,
starting from the integrated prescribed cell means, agrees with the retained
fields to `1.6286971771251046e-13`. The largest original mass-equation residual
is `7.70e-13`, below `1e-10`. This is the frozen linear pair.

ND2's physical rate is `2(D+R)Δu`, because the two occurrences have weights `.5`
and `1.5`. The SSPRK2 multiplier is `I+dt A+.5 dt² A²`. An independent spectral
calculation from the prescribed rectangular cell means agrees with the genuine
final fields to `2.220446049250313e-16`; prescribed initial error is zero.
The ledger has 1,536 unique stage/occurrence/cell/axis/side/component incidences.
Signed amounts use outward orientation, physical face area and `dt/2` times the
occurrence weight. The maximum amount-balance error is `8.33e-18`. The frozen
checker's missing initial anchor is an adversarial defect despite these honest
data being correct.

## Twelve independent corruption experiments

The harness only redigests outer `manifest.files` SHA256/byte counts when marked
resealed. Nested native/witness receipts remain untouched. It requires both a
refusal and the relevant failure category. Against the original checker it exits
1 intentionally: five expected refusals succeed and seven false positives are
recorded.

| Mechanism | Mutation | Observed result |
| --- | --- | --- |
| Unsealed integrity | Flip one NPZ byte | Refused: receipt integrity |
| Unsealed integrity | Perturb one ledger flux | Refused: receipt integrity |
| Resealed science | M26 pairing += .001 | Refused: pairing error |
| Resealed science | M27 actual μ entry += .001 | Refused: μ error |
| Resealed science | ND2 actual final entry += .001 | Refused: state error |
| Resealed science | ND2 initial/final/increments zero, all flux/amount entries zero | **False positive:** 16 snapshots pass |
| Resealed nonfinite ledger | One numerical flux is NaN | **False positive:** 16 snapshots pass |
| Resealed nonfinite ledger | One face measure is NaN | **False positive:** 16 snapshots pass |
| Metadata | Remove all `scientific_states` entries | **False positive:** passes with zero recomputations |
| Resealed metadata | ND2 native identity hash becomes 64 zeros | **False positive:** 16 snapshots pass |
| Resealed metadata | M27 saved-state witness hash becomes 64 zeros | **False positive:** 16 snapshots pass |
| Metadata | Failed JUnit count 26 becomes zero | **False positive:** 16 snapshots pass |

ND2 computes its expected trajectory from the supplied initial array, allowing
a different initial-value problem. Its scalar ledger errors use `max(previous,
abs(value-reference))`; a NaN second argument can leave the previous finite error
unchanged. Explicit finite numeric checks must precede these reductions. The
metadata defects arise because the original checker only checks outer file
digests and never validates the listed provenance/receipt/JUnit relationships.
An archive-specific recomputation should require complete, unique scientific
inventory instead of accepting a user-emptied list.

## Reproduction and limits

From the exclusive review checkout, using the actual principal archive path:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  docs/development/api_040/evidence/native_saved_reception_independent_review.py \
  --checker docs/development/api_040/evidence/native_saved_reception_checker_224bbd6.py \
  --archive /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS/docs/development/api_040/evidence/native-fc0-20260930 \
  --output /tmp/native-reception-independent-review.json
```

The report pins its checker and archive, so a corrected checker can be replayed
with the same probes using `--checker` without modifying this historical result.
No current installation, binary hash, MPI execution, checkpoint restore or native
ABI5 behavior is qualified by this offline review. Native binaries and the full
installed source package are not retained here, so those hashes cannot be
recomputed from the archive. The bounded injections do not constitute exhaustive
malicious NPZ/JSON/filesystem parser validation. The two ND2 native cases belong
to an outer eight-test run with six Integral checkpoint failures; those failures
remain explicit and are not reclassified as passing.

## Corrected checker replay

The independent harness was replayed unchanged on a private copy of the corrected
checker SHA256 `0a196a7bb2d5b5d6e44c65de08b4540e1841a06dadd7ec80b6d598a5ae676759`
and manifest SHA256 `c3d13928659fa7b094d5b4f55c0127b8b96adb0f974cd77ffd90d8c77f6adfb1`
(manifest version 2). The archive now includes two additional genuine M27 N16/N32
snapshots from the original run whose subsequent N64 case failed. All eighteen
genuine snapshots pass; that outer failed run remains failed.

All twelve corruptions are refused for the expected category, exit 0. The M26/M27
perturbations are now refused earlier by their untouched witness saved-state
digests. ND2's coherent zero trajectory is refused by its prescribed initial
anchor (`initial_error=1.0923055579518806`). NaNs are refused by strict JSON parsing;
scientific inventory, foreign native identity, stale saved-state witness and
forged JUnit counts are refused by their exact consistency checks. Independent
alternative mathematics and the original retained source/witness/JUnit/MPI links
continue to pass. Full observations are retained in
`evidence/native_saved_reception_independent_0a196a7.json`.

The final reception additionally uses `--complete-witness-reseal` for two
equation-only attacks. Each starts from another genuine private archive copy,
perturbs its M26 pairing or M27 actual μ entry, updates that snapshot's hash in
the genuine witness receipt, then updates both outer manifest records. These
reach the scientific guards and are refused by `pairing_error≈.001` and
`mu_error≈.001`, respectively. All fourteen attacks reject for the correct
category, preserving the original twelve observations. The final evidence is
`evidence/native_saved_reception_independent_0a196a7_complete.json`. No native
payload in the principal archive has been modified.
