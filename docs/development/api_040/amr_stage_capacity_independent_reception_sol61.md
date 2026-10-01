# Independent AMR history capacity reception, 2026-10-01

Production received: `ea13bf6950b0e41b2aecb8724b7dc58c321b4561`, exact parent
`0abbe25395a44e570d8d5525693b8e2dcbf4d387`. The actual four-case native bind
failure and its original receipts remain ROOT evidence; this review does not
rerun or declare that native failure repaired at runtime.

The diff contains eleven additions in two Python emitters. Both missing sites
now use `global_history_storage.descriptor(program,name)`: checkpoint shape
metadata and hierarchy phase registrations. The initial registration already
used that exact immutable issuance. The scalar observation retains its own
width and space; no physical owner-State identity is substituted for it. No
IR, physics, numerical controls, capacities, or C++ implementation changed.

Independent regression: `tests/review/test_sol61_amr_history_capacity_independent.py`.
Seventeen SOURCE_ONLY tests pass. They use the earlier independent public
Case/Program witness, including a real already-captured two-component physical State and a
one-component global history, then execute the authentic shape emitter and
nested registration body. The expected string is the canonical JSON of the
actual serialized `store_history.attrs.global_field_storage` in its IR. It is
not minted from a checkpoint, a mutable registry, or a reimplemented descriptor
function. Shapes and phase registrations match that string exactly, without
changing the IR hash. Freeze/to_graph preserve the issued observation. Local
git parent0abb is explicitly required for the historical probes; no fetch or
installation occurs.

The same actual Program evaluated by the historical shape emitter still emits
`scalar-history:exact-global-history`, reproducing the source mismatch. The
historical hierarchy registration has the same wrong fallback. The repaired
sites refuse ten owner/point/bool-component/removed-qualification/registered
width mutations before returning an emitted string. A legacy physical State
history keeps both shape and hierarchy registration bytes exactly equal to the
parent. This is a targeted byte comparison, not a claim of an independently
executed full native compilation or all legacy C++ profiles.

The exact native capacity guard is unchanged:

```
if (live.histories != shape.histories || live_clocks != shape.logical_clock_identities ||
    live.temporal_partition.provider_identity != shape.temporal_provider_identity ||
    live.temporal_partition.cells.size() > shape.temporal_cell_count ||
    live_bytes.size() > accepted)
  throw ...;
```

`src/runtime/amr/amr_system.cpp`, `amr_program_checkpoint.hpp` and the native
history publication guard are compared byte for byte with parent0abb. This
does not weaken the production mismatch refusal or increase a capacity.

The compiler driver passes `source=src` to the actual Program artifact-spec
projection; its `generated_source` component is SHA256 of those CPP bytes.
The test executes that body and the real identity primitives with explicit
SOURCE_ONLY toolchain/catalog inputs, retaining real Program semantic data.
Same IR/semantic inputs produce a changed source component and artifact-spec
identity for repaired CPP. No installed native loader contract or DSO is
substituted, queried for this trace, or declared received. This proves the
cache projection rule, not a complete CPP-to-DSO compilation graph.

The offline readers `6108fbf0` and `1dab28a4` contain no `scalar-history:`
hardcode. Their accepted Program payloads remain explicitly opaque, authenticated
and replay-compared. They do not claim semantic decoding of the native history
descriptor table, and no descriptor is reconstructed from raw CP. A future
extension of that decoding must compare the actual native table against the
authentic serialized IR-issued descriptor, not silently promote raw CP JSON.
Those two readers are unchanged in this commit.

A separate pre-existing limit was observed: a storage TimeState declared but
never materialized as a `State.n` node is absent from `_block_indices`, so AMR
metadata emission refuses its owner. That minimal public reproduction was sent
to Coverage for a distinct fix. The metadata probes here use an already-captured
physical owner; they add no identity commit or artificial read to materialize
an unused owner.
The four Stage fixtures use updated Q0 as owner and are unaffected by this
limit. It is not repaired or newly introduced by ea13 and is not claimed as
received support for unused storage-only TimeStates.

Command (source checkout identity, no installation):

```
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 PYTHON -m pytest \
  tests/review/test_sol61_amr_history_capacity_independent.py \
  -q -p no:cacheprovider --tb=short
```

Result: **17 PASS, 31.18 s**. The test environment is the source package with
the preserved Python interpreter; its installed DSO is not used to qualify
the fix. The first cache trace correctly refused that incompatible loader
contract; the final trace deliberately supplies fixed local resource inputs
and preserves the actual source/identity computation. ROOT owns installation,
actual compile/bind, and the fresh Serial/MPI campaign. No Native reception,
SDK mutation, JIT, heavy C++ build, or environment edit was performed here.
