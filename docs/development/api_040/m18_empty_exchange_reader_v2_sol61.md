# M18 empty exchange reader correction, version 2

Source base `33dee906850d56eec542dd0a7d0b21954aa244d3`. This three-file
oracle/tests/docs change corrects an offline-reader assumption. Native codec,
checkpoint arrays, actual receipt donors, MAIN and environment are unchanged.

`AcceptedExchangeLedger::checkpoint` in
`include/pops/runtime/program/accepted_exchange.hpp` writes eight magic bytes
then a little-endian uint64 record count. With no integrals and no forced
extension, the exact empty `POPSEX01` image is **16 bytes**. Extended `POPSEX02`
also writes integral and consumed counts: its exact empty image is **32 bytes**.
The C++ reader rejects trailing bytes. The old oracle incorrectly required all
rank spans to be 32 and admitted `POPSEX01` followed by 24 zero bytes; that
noncanonical image is now rejected.

`empty_exchange_images` admits only the exact pair:

```
POPSEX01 + uint64_le(0)
POPSEX02 + uint64_le(0) + uint64_le(0) + uint64_le(0)
```

Every rank boundary is checked independently; any record, integral quantity,
consumption, missing/trailing byte or other wire version is rejected. Offsets
must be the native capture's exact int64 type, start at 0, increase strictly,
end at the exact uint8 image extent and contain exactly ranks + 1 entries.
There is no inferred empty image and no missing-member fallback.

The reader contract is `sol61.m18-empty-exchange-reader@2`; offline contract
and reception now emit schemas `sol61.m18-offline-contract@2` and
`sol61.m18-offline-reception@2`. External owner-pins schema remains @1. Original
equations, five-node quadrature, three unknowns, immutable read-only target,
Newton 12/tolerance 2e-11, entropy/cone criteria, phases, exact restart/rollback
and safe-rebind checks are unchanged. A test authenticates their source
prefix byte-for-byte against Git 33dee. This reader supports the empty mailbox
of this M18 witness; it makes no general producer/M19/AMR qualification.

## Actual source/protocol result

From private checkout `PoPS-sol61-m18-empty-wire-reception`:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q -p no:cacheprovider tests/review/test_sol61_m18_empty_exchange_codec.py tests/review/test_sol61_m18_entropy_offline_contract.py tests/review/test_sol61_m18_owner_assemble.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_m18_empty_exchange_codec.py tests/review/sol61_m18_entropy_offline_oracle.py
rtk git diff --cached --check
```

**122 PASS in 1.58 s**, Ruff PASS. Negatives include each nonzero count position,
high-bit/max counts, invalid/truncated/extended/trailing legacy images, wrong
offset type/extent/rank assignment and one bad peer among three. Empty-image
protocol byte vectors are codec controls, not fabricated scientific states.

## Read-only observation of authentic ROOT files

External native-owner seals supplied by ROOT:

- Serial: `6f8bf5c628b80e8ab189177a1500ec5a1b0e84fbd6c214ced40fa23488d53fc1`.
- MPI2: `e8bd8f3f618a2344ebbc67ea5cd6d12177cc40223c1c4f5e3c8161261064d143`.

The standalone test-file CLI authenticates each external seal and all 31
hashed phase leaves before reading the ten real checkpoints. Serial offsets
are `[0,16]`; MPI2 offsets are `[0,16,32]`. Every actual rank image is
`504f5053455830310000000000000000`, precisely the native 16-byte empty01 image.
All ten phases in each campaign pass this byte-only check, including both
outside refusals and restart/safe-rebind snapshots. CLI elapsed times were
0.084 s Serial and 0.077 s MPI2. It explicitly emits
`scientific_reception=false` and `native_execution_here=false`: complete M18
scientific reception awaits ROOT's separately approved mathematical pins.

Exact commands, with workspace root assigned to the task-specific `m18_ws`:

```sh
m18_ws=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -B tests/review/test_sol61_m18_empty_exchange_codec.py --owner "$m18_ws/outputs/installed-m18-entropy-serial-dim2-sdk375f-owner-codec-20261001/root-native-owner-auth.json" --owner-sha256 6f8bf5c628b80e8ab189177a1500ec5a1b0e84fbd6c214ced40fa23488d53fc1 --output outputs/m18-real-serial-empty-wire-observation.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -B tests/review/test_sol61_m18_empty_exchange_codec.py --owner "$m18_ws/outputs/installed-m18-entropy-mpi2-dim2-sdk375f-owner-codec-20261001/root-native-owner-auth.json" --owner-sha256 e8bd8f3f618a2344ebbc67ea5cd6d12177cc40223c1c4f5e3c8161261064d143 --output outputs/m18-real-mpi2-empty-wire-observation.json
```

The two machine reports remain in the private checkout's `outputs/`; they
are observations, not ROOT owner seals. No PoPS import, native execution,
compiler, JIT, installation, rebuilt evidence or mutation of originals was
performed by these offline observations.
