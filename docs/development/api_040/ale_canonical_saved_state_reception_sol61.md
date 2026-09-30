# Independent reception of canonical ALE saved states

This review consumes the real historical Uniform1D serial and MPI2 outputs of
the e49 SDK. It runs stdlib/NumPy offline, without importing PoPS or executing
the native module. It changes tests/readers only. The runtime JUnit evidence and
the external owner seals remain Root's evidence; no seal is generated here for
a positive campaign.

## External authorities

Workspace: `/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops`.
Private checkout: `work/PoPS-sol61-ale-canonical-serial-offline`, parent
`004a6757b76dceba0e60ff8bf4bf2ff9eb84798c`.

| Campaign | Root inventory | Externally supplied SHA256 |
|---|---|---|
| Serial | `outputs/public-ale-canonical-serial-owner-reception-20260930/inventory/oracle-pins.json` | `9cf00652fe3de74c437ba19f91fb9687adacde99d4c57cdb0e1d9939a37e8b4b` |
| MPI2 | `outputs/public-ale-canonical-mpi2-owner-reception-20260930/inventory/oracle-pins.json` | `8afc2f2333922b846b4ed4df6c435bc504c651172ace4759e61f12f9c7686ee9` |

Each inventory contains 285 actual files: 78 state/receipt/checkpoint triples,
50 observer publications and one source/native identity receipt, across 14
cases (16/32 cells, scalar/three components/permutation, source on/off,
restart and rejected attempt). The serial seal reports 2,142,516 bytes; MPI2
reports 2,154,756 bytes. The native DSO SHA256 is
`e0a63a78e54f0a670db0f76335a3901d4603bd70ccfa9026cc35c5894aafc491`.
Serial source is `0a8148f823fb78b378e43f8b16a6c61810d67d03`; MPI2 source is
`a7834520dfabc4931b497d62662f687775cdff68`; the source identities
in the original inventories are retained verbatim. Root authenticated 1114
production files to Git and received eight native JUnit tests per campaign/rank.
Those native tests were not executed by this reviewer.

## Reader defect and strict repair

The old reader asked for `synchronization_state` and `cache_generations`.
Neither is a field of the actual `TemporalRestartState.to_data()` schema 2.
The serializer has fourteen exact keys, including `synchronization_cursors`,
`history_cursors`, and `cache_cursors`. This is a reader defect, not a failure of
the saved native calculation.

The repaired reader requires Uniform checkpoint payload 8 and the exact temporal
schema 2 key set and scalar types. It validates the actual ordinary root clock,
accepted synchronization, schedule/clock cursor boundaries, empty undeclared
synchronization/history/cache cursors, event queue and transaction counters.
It compares the **entire** canonical temporal envelope at accepted/restored,
continuous/replayed, and before/rejected boundaries, including controller
origin/count and all counters. It also compares every typed native checkpoint
array byte for byte, excluding only the two authenticated bind/run restart
envelope members whose provenance legitimately changes on restoration.
State, physical nodes, measures, generation and native exchange wire remain
exactly compared. No missing field is made optional.

The observer reader now recomputes the complete scientific-output manifest
identity, in addition to typed array hashes, physical state/geometry,
accepted clock and authenticated bind/run provenance.

## Scientific reception

The existing independent equations remain unchanged: physical endpoints follow
the declared sinusoidal coordinate law; endpoint differences define measures;
GCL uses the actual endpoint displacements; Reynolds uses the old physical
inventory, original upwind/Rusanov flux with signed relative characteristics,
and the original source evaluated once on the old state/measure. Every native
face amount is checked bit exactly against the canonical fused
`F_phys - density * sweep` expression using exact rational binary64 products and
one final binary64 rounding. The source and geometry incidences, occurrence
identities, contexts and support are checked independently.

The first repaired real serial and MPI2 receptions both pass all 14 cases.
Their observed maxima are identical:

| Quantity | Maximum absolute discrepancy |
|---|---:|
| Physical field | 8.881784197001252e-16 |
| GCL | 0 |
| Reynolds | 6.504366001965772e-17 |
| Native face amount vs independent original physics | 8.673617379884035e-19 |
| Source amount | 3.3881317890172014e-21 |
| Coordinate law | 0 |
| Global inventory | 8.881784197001252e-16 |

MPI2 is a genuine empty-rank reception: each actual image declares a
non-replicated global box `[0,N-1]`, owner 0; rank 0 owns its one local patch,
rank 1 has no local patches. Both rank images are decoded and their global
topology/ownership must agree. This does not qualify two nonempty partitions.

## Counter-model and process-isolation obligations

`sol61_moving_interval_real_countermodels.py` first authenticates Root's serial
inventory against the externally supplied SHA above. It operates only on
private byte copies of those real files, keeps the original case declarations,
and marks every altered seal **NOT root owner authority**. Native carrier and
ledger serializers must reproduce the real seed image byte for byte before
altering it. Modified state, wire, checkpoint typed evidence/restart digest,
observer arrays/snapshot evidence/manifest identity, receipt and all 285 file
hashes are rescaled consistently. Original owner files must remain unchanged.

The adversarial cases distinguish conservation from original physics: paired
face incidences with zero global imbalance but a one-bit amount change or
non-fused rounding; a different physical velocity; a doubled original source;
an altered coordinate law with balanced GCL/Reynolds; false legacy temporal
keys; foreign synchronization cursors; changed restored controller origin;
changed restored native history metadata. The three alternative equations
recompute matching states and ledger, so a mere balance-only oracle cannot be
the rejection reason.

The shared-process `pops not in sys.modules` assertion was invalid when earlier
tests had legitimately imported PoPS. Its replacement uses the real helper and
oracle in a subprocess `Python -I`, temporary cwd, absolute helper path and a
meta-path import guard which raises on any PoPS import. It does not clear the
parent process module cache or weaken the no-PoPS requirement.

Final source/protocol suite: **59 PASS**. Ruff and `git diff --check`: PASS.
Historical failed private scratch `outputs/ale-countermodels-v1` records the
first unsuitable constant step1 seed: it had no face distinguishing FMA from
non-fused rounding. The corrected negative seed uses actual accepted abc/source
data, which contains three distinguishing faces. This was a test-seed issue.

The final harness receives **9/9 refusals**, with all 285 file hashes valid in
each altered campaign and the original 285 files reverified unchanged. The two
balanced face negatives differ at face 10 by one binary64 bit:
`abebe3a875ee47bf` becomes `aaebe3a875ee47bf`; their net signed face delta is
zero. The three wrong-equation counter-models satisfy their own Reynolds
balance within 2.92e-17 and GCL exactly zero, but fail the original velocity,
source, or coordinate-law check. Controller and native-history mutations fail
the complete exact restart comparisons. Quantitative results and all refusal
messages are saved in `ale_canonical_saved_state_scientific_reception_sol61.json`.

## Observed I/O failure and bounded reading

The first readall-based counter-model process, pid 97820, was observed after
6min32s with only 0.21s CPU. A macOS sample showed
`_io_FileIO_readall_impl -> _Py_read -> read`; `lsof` showed ordinary 2–16KB real
NPZ/JSON donor files, with very slow progression. A separate read-only
`os.pread(fd,2188,0)` of one such file returned all bytes in 0.000134s with
SHA256 `c686a5d741e0f7020f94c88bd5f936b3b46a966f5a4d769ab34e623e5f36d540`.
The observations distinguish the two access patterns; they do not establish
the underlying kernel/filesystem cause. Root separately observed CMake I/O
stalls, whose build receipts remain outside this scientific reception.

The ALE reader now opens a nonblocking descriptor, requires a regular file and
the existing byte budget from `fstat` on that same descriptor, reads exactly
its declared extent with bounded `pread`, and requires stable device/inode,
mode, size, modification and change timestamps afterwards. Empty/truncated
reads fail closed. The external SHA256 is still checked on all bytes. Unit
probes cover short reads, truncation, modification during reading, FIFO,
oversize and wrong hash. No scientific tolerance, checkpoint guard or native
code changes. Final serial/MPI receptions both pass with this reader.

Private failed scratch `outputs/ale-countermodels-v2` is also retained: the
first observer reseal omitted the mandatory identity algorithm field, and the
strict identity validator rejected that harness error. The final v3 run fixes
the reseal, passes its original authentic campaign before/after, and rejects
all nine scientific negatives. The failed runs are not counted as receptions.

## Commands and limits

Run in the private checkout with the explicitly selected Python:

```sh
rtk proxy env -u PYTHONPATH PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_moving_interval_offline_contract.py tests/review/test_sol61_moving_receipt_seal.py tests/review/test_sol61_moving_temporal_v2_contract.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_moving_interval_offline_oracle.py --pins /ABS/ROOT/inventory/oracle-pins.json --output /ABS/PRIVATE/reception.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_moving_interval_real_countermodels.py --pins /ABS/SERIAL/inventory/oracle-pins.json --owner-sha256 9cf00652fe3de74c437ba19f91fb9687adacde99d4c57cdb0e1d9939a37e8b4b --countermodels-dir /ABS/PRIVATE/EMPTY-COPIES --output /ABS/PRIVATE/negatives.json
```

The source/math tests contain structural metadata primitives, not fabricated
positive physical states. The scientific positives are the externally sealed
historical saved files only. No PoPS/native import, build, JIT, installed
environment mutation or donor edit occurred in offline reception. AMR, 2D/3D,
GPU and distributed nonempty partition qualification remain outside this lot.
