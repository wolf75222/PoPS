# Typed history sample identity and restart compatibility

This note records the current source contract for history sample provenance in
Uniform and AMR checkpoints. It is an additive explanation of the existing
temporal contract; it does not change the migration result status.

The evidence here is source-level. It makes no native build, runtime, numerical,
or performance qualification claim. The open matrix status remains in
[migration M3-M8 results](migration_m3_m8_results.md) and the acceptance
vocabulary remains in the [M0-M2 contract](migration_m0_m2_contract.md).
The broader schedule and accepted-boundary rules are in the
[temporal execution contract](../design/temporal-execution-contract.md).

## Identity states

The C++ [identity type and ring manager](../../include/pops/runtime/program/program_runtime_state.hpp#L128-L316)
define three states. A state is attached to each logical history ring slot.

| Kind | Meaning | Allowed metadata |
| --- | --- | --- |
| `UnknownLegacy` | The archive or payload has no authenticated publication identity. The C++ default and an absent Python identity member both represent this state. | `start_bits`, `interval_bits`, and `ordinal` are all zero. It is not an authenticated sample. |
| `RegisteredZeroStart` | Registration allocated a cold ring before its first accepted store. | All three fields are zero and the ring must still be uninitialized. This is allocation state, not publication evidence. |
| `Publication` | An accepted store published one physical time window. | The exact window start and interval are finite, positive where required, and `ordinal` is nonzero. The ring must be initialized. |

The Python [identity validator](../../python/pops/runtime/_history_sample_identity.py#L1-L84)
enforces the same kind and initialization relationships before restore or
capture. It does not reconstruct a publication from field values, a facade
last-step value, or an old archive's missing metadata.

## Exact window identity

A `Publication` stores:

- `start_bits`: the IEEE binary64 bit pattern of the window start;
- `interval_bits`: the IEEE binary64 bit pattern of the outgoing interval;
- `ordinal`: an unsigned 64-bit occurrence number within that exact
  start/interval window.

The [C++ allocator](../../include/pops/runtime/program/program_runtime_state.hpp#L169-L192)
uses exact bit equality for the physical window. It starts an ordinal at one
and advances it only among existing publications with the same two bit
patterns. A new start or interval begins its own ordinal sequence. Signed-zero
bits and the full u64 ordinal remain part of the identity.

The outgoing interval is checked against the actual native `Real` width.
The [Python seam](../../python/pops/runtime/_history_sample_identity.py#L19-L70)
requires `runtime_environment_report()["real_bytes"]` to be exactly four or
eight. Four-byte `Real` compares against the binary32-rounded interval;
eight-byte `Real` compares against the binary64 interval. A static precision
label or fallback cannot authorize a publication. This is a parity and
native-width contract, not a native execution result.

## Ring lifecycle

Registration creates a ring of depth `lag + 1`, marks it uninitialized with
`fill_count = 0`, and assigns `RegisteredZeroStart` to every slot. See
[Program registration](../../include/pops/runtime/program/program_context.hpp#L1194-L1269)
and [AMR registration](../../include/pops/runtime/program/amr_program_context_history_checkpoint_public.inc#L72-L135).

A first store creates one `Publication` from the exact current start and
interval. It writes the value, interval, and identity into slot zero and
cold-fills deeper slots with the same value, interval, and identity. Those
copies make the ring evaluable; they are one actual accepted sample, not
additional time samples. The C++ store path sets `store_pending`, while
`fill_count` advances only when a logical rotation consumes that pending
store. Thus a first cold store followed by one rotation has
`fill_count = 1`, even when the ring depth is greater than one. The
[published store path](../../include/pops/runtime/program/program_context.hpp#L1935-L1990)
and [AMR store path](../../include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc#L1-L100)
stage the complete ring metadata together.

A second write before rotation is one pending logical sample. If slot zero
already contains a publication, its start and interval must match exactly;
the existing slot-zero identity is retained. A changed physical window is
rejected. After rotation, the next publication in the same exact window gets
the next u64 ordinal. [Rotation](../../include/pops/runtime/program/program_runtime_state.hpp#L286-L325)
moves values, outgoing intervals, and identities together, then increments
`fill_count` once and clears `store_pending`.

## POPSHID1 persistence

Each Uniform ring uses the key
`history_sample_identity_<name>`. Each AMR level uses
`history_sample_identity_<name>_level_<n>`; the AMR enclosing checkpoint
still owns the level and descriptor authority.

The [POPSHID1 codec](../../include/pops/runtime/program/history_sample_identity_codec.hpp#L1-L77)
encodes the fixed header

`POPSHID1` + little-endian u64 UTF-8 name length + name bytes + signed
i64 level (`-1` for Uniform) + u64 depth,

followed by four little-endian u64 words per slot:
kind, start bits, interval bits, and ordinal. The exact payload size is
`32 + len(name.encode("utf-8")) + 32 * depth` bytes for each ring. The
Python capture stores it as a one-dimensional `uint8` array and restore keeps
the bytes unchanged; no float conversion is used for identity fields. The
[checkpoint resource budget](../../python/pops/runtime/_checkpoint_resource_budget.py#L201-L212)
accounts for the same per-ring size.

## Restore, rotation, rollback, and regrid

Python capture validates every ring's initialization, fill count, outgoing-dt
ledger, and identity before the first gather. Python restore validates the
whole payload first, installs all numeric anchors and provenance, and only then
allows selective replay. The [capture/restore protocol](../../python/pops/runtime/_system_io_history.py#L254-L676)
also requires an explicit identity restore seam. An absent identity member is
passed as an empty payload, which explicitly clears the live identity to
`UnknownLegacy` before any replay; it is not filled from `dt`.

The C++ AMR store stages values, intervals, identities, and flux expressions.
If a valid-copy fails, it restores every stable ring element before metadata
publication. The [AMR service transaction](../../include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc#L1-L100)
therefore retains the prior accepted ring image at that failure boundary.
Rotation performs the corresponding ring, interval, identity, and expression
moves together. A source-level rollback guarantee here describes publication
ordering; it is not a native qualification claim.

An accepted AMR checkpoint retains each history descriptor and slot's
initialized flag, fill count, outgoing interval, and four identity words. It
also retains pending history remaps with their level, topology/materialization
generations, accepted macro step, temporal ratio, source interval, target
interval, and consumed flag. See the [AMR checkpoint writer](../../include/pops/runtime/program/amr_program_checkpoint.hpp#L894-L978)
and [accepted-state validation](../../include/pops/runtime/program/amr_program_checkpoint.hpp#L664-L840).
The [AMR remap/import path](../../include/pops/runtime/program/amr_program_context_history_checkpoint_runtime.inc#L205-L330)
compares the restored history provenance with the live restored rings before
swapping accepted auxiliary state. Remap publication stages an immutable
accepted image before swapping its pending map; a failed preparation or
publication stage does not publish that staged pending map. The enclosing
restart transaction remains the rollback authority for a failed accepted-state
import.

For a regrid, a pending remap records the old and new hierarchy authority and
the exact source/target interval relation. A deferred lag read is admitted
only with a live authenticated marker, the expected current store, and the
retained interval ledger; it does not mint identity for an unknown legacy
slot. A pending remap is cleared only after its corresponding logical store
and rotation complete. See [AMR history remap services](../../include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc#L220-L396).

When a regrid uses the `ParentAlignedState` projection, the source requires
authenticated matching sample identities for each mixed-source slot: retained
child cells that overlap the new fine support and parent-projected uncovered
cells must describe the same selected sample. Two `UnknownLegacy` values do not
authorize that blend. A completely disjoint or parent-only projection can retain
unknown provenance on its resulting cells without inventing a publication
identity. The [AMR history remap preparation](../../src/runtime/amr/amr_system.cpp#L8890-L8935)
contains these checks; the selected-prior path still checks only the selected
lag through `matching_authenticated_sample`.

## Legacy readers and composite ancestors

The POPSAND7 reader accepts the older v4, v5, and v6 slot layouts. Those
layouts contain no identity words, so the reader leaves each sample at its
default `UnknownLegacy`; it never infers a publication from the outgoing
interval or field values. Current POPSAND7 slots carry kind, start bits,
interval bits, and ordinal. See the [reader](../../include/pops/runtime/program/amr_program_checkpoint.hpp#L1183-L1250).

Older archives remain readable through the empty-identity compatibility seam,
and their stored anchors can follow the existing replay protocol when the
current storage policy and accepted-state checks allow it. This compatibility
preserves old data access; it does not authenticate an old publication.

An AMR composite history ancestor has a stricter requirement. The
[condensed-prior path](../../include/pops/runtime/program/amr_program_context_spatial_operations.inc#L488-L531)
calls `matching_authenticated_sample` for the selected coarse and fine
history samples and rejects a pending remap or a different retained sample.
Therefore a legacy `UnknownLegacy` prior cannot be made current by allowing
a blocked step to proceed. When an old unknown prior would be consumed before
a first new store, select an explicit fresh-history/cold-start workflow:
register the new ring, use zero-start access only where the contract permits
it, and publish a first authenticated sample before the composite ancestor
read. This is the supported recovery boundary exposed by the source
contracts; no implicit legacy self-refresh is promised.

## Status boundary

This guide documents source contracts and compatibility decisions only.
Parity, native `Real` width, failure-before-publication behavior, and
legacy-reader handling remain inputs to the appropriate gates. No source
inspection in this document changes the [M3-M8/R1 results ledger](migration_m3_m8_results.md)
or supplies native, numerical, or performance evidence.
