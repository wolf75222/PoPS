# Independent C38 wire9 capacity review

Verdict: no blocking capacity defect found in the reviewed source. This is a
wire/capacity host result, not an installed regrid, MPI or GPU qualification.

Reviewed `PoPS-principal-group`, branch `codex/api040-c38-pending-qualification`,
base `90da6e8` plus the author's frozen working changes:

- `amr_program_checkpoint.hpp` SHA256
  `780287346eb2c300d8de7fd206ecafa4fb64c1a453996e94dae86fbfa8e4138b`.
- `src/runtime/amr/amr_system.cpp` SHA256
  `b0cc98a997ee580046db479541761373e282100a6e5fd8d455b89b285554275b`.

The bind capacity path requires an artifact-backed installed Program and frozen
checkpoint metadata (`checkpoint_program_state_capacity_`, around line 20974).
It copies every history name, owner, state/space/clock/interpolation identity,
depth and width from `program.checkpoint_metadata_.histories` into the capacity
shape (around line 21088). The pending-record count is bounded by all frozen
histories times all configured parent-child transitions, including inactive
levels. The existing live image is checked against this authority afterwards;
it is not used to invent the ceiling.

The new variable string is bounded by running the same
`write_pending_ring_contract` with those frozen descriptors through
`CountingWriter`. Slot and clock fields have fixed wire widths, so stub values do
not shorten their encoding. Taking the largest descriptor contract for every
possible pending entry is conservative even for histories with different names
and identities. It introduces no empirical character budget. The 20 fixed words
of each wire9 pending record include both string framing lengths (key and retained
ring contract); the new variable bytes are added separately. Checked repeated
addition/multiplication rejects overflow.

Independent executable evidence:
`evidence/c38_wire9_capacity_review.cpp` constructs three unequal history
descriptors, five levels and all twelve possible pending records. Four cases add
0, 1, 8192 and 131071 bytes to the largest interpolation identity. All pass:

- actual Writer size equals Sizer;
- serialized bytes fit the derived capacity and deserialize exactly;
- capacity increases by exactly 13 times the added bytes (one descriptor plus
  the maximum contract bound for twelve pending records);
- `pending_history_remap_count = SIZE_MAX` throws `length_error`.

The first fixture was intentionally ordinary but incorrectly ordered its pending
keys by ring then level; production validation refused it. The fixture was fixed
to sort canonical keys. That rejection was not a production defect.

Executed command from `PoPS-resource-lifetime` (no CMake build or installation):

```sh
/usr/bin/clang++ -std=c++20 -O2 -DPOPS_HAS_KOKKOS -DPOPS_HAS_MPI -DPOPS_NATIVE_DIM=2 \
 -Xpreprocessor -fopenmp -I../PoPS-principal-group/include \
 -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
 docs/development/api_040/evidence/c38_wire9_capacity_review.cpp \
 -o outputs/c38-wire9-capacity-review \
 -L/Users/romaindespoulain/miniforge3/envs/pops-api040/lib -lkokkoscore -lmpi -lomp \
 -Wl,-rpath,/Users/romaindespoulain/miniforge3/envs/pops-api040/lib
outputs/c38-wire9-capacity-review
```

Output: `wire9 capacity: four unequal long-contract cases and overflow passed`.
No producer core file was edited during this review. The actual restart/regrid
trajectory, collective rollback, resource allocation at bind and cross-rank
agreement remain the central native acceptance obligations.
