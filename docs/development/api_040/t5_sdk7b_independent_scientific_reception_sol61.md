# Historical T5 scientific reception from authentic SDK7b saved states

The independent offline reception **passes** for all three actual T5 cases:
N8 and N16 reaction/transport/q feedback with restart, and N8 rejected attempt
with safe retry. No production code changed. The oracle imports NumPy/stdlib
only; it never imports PoPS, invokes JIT/builds, modifies MAIN/the environment,
or writes replacement positive native states.

## Authentication and quantitative result

Root supplied the owner pins at
`outputs/native-integral-feedback-sdk7b-root-owner-pins-20260930.json`, SHA256
`7a6d07c39e434b256e08792ff6c89d991c8525068eb0df0c0a37e43c12521bc2`.
The independent harness checks that exact manifest digest before reading the
externally pinned identity plus **39 phase files**. It receives source
`634cba3511fef957e65a21fcae3ab257f164b360`, native DSO SHA256
`8a217a0fff5a131729a842dd222e2e08346034eb53b4f461fc17fc8cc291b563`, and SDK
`7b503163f41cc33040c296679e3b3530a9716c91ecfe4fdce6f69dca6f07dad0`.
The identity records installed production PoPS, Dim2, CPU/OpenMP, binary64,
one thread and an MPI-enabled build. The run and pins have **one actual rank**.

The authentic input directory is
`outputs/installed-spatial-original-nonlinear-integral-feedback-dim2-sdk7b-20260930`.
Its read-only pytest log reports **8 PASS in 240.18 seconds**: five T3 cases and
three T5 cases. This report receives only the latter three. The fixture catches
and verifies the expected native rejection diagnostic; exception text itself is
not serialized in the snapshots, so the log/source result remains separate
evidence from the offline invariants.

Re-running the frozen `07d14e5` oracle independently produces a JSON result
exactly equal to root's
`outputs/native-integral-feedback-sdk7b-offline-reception-20260930.json`.
Seven saved-step oracle evaluations (including restart replay) give:

| Quantity | Maximum residual |
| --- | ---: |
| field infinity norm | 2.220446049250313e-16 |
| source plus boundary inventory | 8.304988641238964e-17 |
| native ledger telescoping balance | 1.951563910473908e-18 |

The original absolute tolerance **3e-13** is unchanged. All native incidences
are received: 32 for N8, 64 for N16. Each accepted image has exactly one selected
right exterior record and exactly one consumed key, with actual flux, measure,
weight/dt and physical endpoint support. Exact runtime clock/point/duration bits,
source evaluation, typed integral declaration and checkpoint envelope digests
are received. Reaction uses the **saved pre-step q**, including the updated q
on the second step.

| Case | q after first step | q after second/replayed step |
| --- | ---: | ---: |
| N8 restart | 0.7118500624999999 | 0.7236549032181168 |
| N16 restart | 0.7119124312499999 | 0.7237795053013649 |
| N8 safe retry | 0.7118500624999999 | — |

Rejected-attempt state/q/clock/rank-ledger bytes and accepted cursors remain
identical to the before image. Restore matches the accepted image; replay matches
the continuous image exactly. Safe retry matches fresh N8 acceptance exactly.
The complete positive metrics, file digests and negative-control evidence are
saved in `t5_sdk7b_independent_scientific_reception_sol61.json` beside this report.

## Fully resealed scientific negative controls

The separate harness clones the actual pinned files into explicit adversarial
directories. It changes only negative inputs, mirrors field/q changes into their
paired checkpoint/receipt, rewrites actual POPSEX02 bytes when needed, and
recomputes every affected typed-array digest, deterministic CBOR restart envelope,
checkpoint SHA, native image SHA and external counter-model file pin. These are
**NOT root-owner pins and NOT positive native receipts**; their manifests carry
both labels explicitly. The authentic originals are received again afterward
and remain unchanged.

All nine controls are refused for the expected invariant, rather than stale
checksums. All 40 external counter-model file pins are checked. For eight controls,
the mutated phase additionally passes the oracle's file/checkpoint/image
authentication before the scientific check fails. The transpose control reaches
the source-declared shape guard immediately after external file authentication;
its envelope is also resealed by the harness, but the oracle correctly refuses
before loading that incompatible physical checkpoint shape.

| Negative input | Demonstrated refusal |
| --- | --- |
| endpoint q used in original reaction instead of pre-step q; field delta 4.214474728136963e-5 | reaction/candidate q or FV field update |
| actual right-face flux increased by .001 | physical flux/measure/duration/orientation |
| trace scale -1 omitted; q delta -0.02370012499999996 | integral update equation |
| physical axes transposed from `(1,1,8)` to `(1,8,1)` | declared field shape |
| interior trace substituted, even with the correct right amount and matching consumed key | actual right endpoint cell |
| receipt duration changed by one ULP while weight and state remain correct | exact interval/duration authority |
| foreign clock with otherwise valid frame | declared primary-clock authority |
| restored cell changed by approximately .001, with checkpoint mirrored | exact restore state/q/clock/ledger |
| rejected-attempt cell changed by approximately .001, with checkpoint mirrored | rollback state/q/clock/ledger |

The 21 existing source/protocol tests remain a **distinct** result. Their checks
are not relabelled as scientific/native receipt tests. Negative controls operate
on copied saved data; they do not inject faults into a running native Program or
prove that a native restart would admit the forged data.

Private negative artifacts are preserved under
`work/PoPS-sol61-t5-offline-reception/outputs/sol61-t5-sdk7b-negative-countermodels-v2`:
369 files, approximately 9 MiB. Their complete resealed-pin manifests can be
inspected; the independent harness reproduces them. They are excluded from the
proposed authentic archive.

## Scope and compact archive mapping

This is historical qualification of the public **Program composition** witness:
the original source S is evaluated and multiplied by typed captured q in the
Program, followed by the actual FV operator and accepted exterior trace update.
It does not qualify the forthcoming direct physical Equation/FieldProblem
connector for q*S. It is also not MPI multi-rank, AMR/EB, GPU, Dim3 or continuum
convergence qualification. N8/N16 are two actual discrete-oracle receptions.
Later source/native revisions require their own authenticated receipt.

`t5_sdk7b_authentic_archive_mapping_sol61.json` proposes exact archive names and
retains all root-owner hashes for the **39 authentic phase files plus identity**.
Total input bytes: **904,051**. No archive has been created or qualified here.
JIT caches, SDK/build/environment trees and negative counter-models are excluded.
Root can package this mapping, preserve the original owner manifest/log alongside
it, then authenticate a relocated path manifest and receive the actual archive.
The archived checkpoint basenames and bytes must remain unchanged; the oracle
permits relocation of externally pinned paths, not scientific rebaselining.

## Exact commands

Authentic positive oracle:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_integral_feedback_offline_oracle.py --pins /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/native-integral-feedback-sdk7b-root-owner-pins-20260930.json --output outputs/sol61-t5-sdk7b-independent-positive.json
```

Nine resealed controls (the destination must be absent/empty; use a fresh name
when reproducing preserved artifacts):

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_integral_feedback_real_countermodels.py --owner-pins /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/native-integral-feedback-sdk7b-root-owner-pins-20260930.json --countermodels-dir outputs/sol61-t5-sdk7b-negative-countermodels-v2 --output outputs/sol61-t5-sdk7b-countermodel-reception-v2.json
```

Source/protocol checks remain the standalone pytest command documented in
`t5_offline_feedback_reception_contract_sol61.md` (21 PASS). No native executable
or installed PoPS module is called by these commands.
