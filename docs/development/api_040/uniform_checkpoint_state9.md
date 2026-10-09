# Uniform checkpoint State storage @9 (Source preparation)

Checkpoint9 adds `state_carriers_checkpoint`, a rank-one uint8 array containing
canonical complete POPSCAR1. It authenticates every State block/component/patch,
valid and grown geometry and object bits, together with the existing checkpoint
clock and lifecycle envelope. Field potentials, auxiliary state, history samples,
cache and readiness retain their existing distinct contracts. No generic ghost
formula, history-grown or whole-mission scientific qualification follows.

`runtime.restart(path)` and direct `System.restart(path)` default to
`state_storage="full"`: only version9 is accepted. The explicit public option
`state_storage="valid_only_legacy8"` accepts version8 and rejects a claimed full
carrier member. It preserves historical valid-array restoration; it does not
restore or certify source ghost bits. Other ABI, program, spatial, lifecycle,
clock, cursor and manifest checks remain mandatory. AMR and multilevel layouts
cannot use this Uniform compatibility option. It is not an identity bypass.

The public offline v2 migration continues to produce version8. Its metadata
authority may be authenticated version8 or9; a version9 authority's carrier is
removed, never copied into the migrated state. Explicit valid-only8 restart is
the consumer for this output. Strict9 rejects it. Missing v2 ghost information
cannot be inferred from a freshly bound engine or metadata authority.

Local nonmutating native validation precedes the outer collective preparation
vote. Restore requires the existing uncommitted restart transaction, prepares
all local target-storage candidates, checks saved valid-array bits against the
carrier, votes failures and exact payload agreement, then copies into existing
allocations with fences. The outer snapshot abort restores prior State/clocks
if publication fails. GPU copies are fallible, not advertised as noexcept.
Recorded owners may be routed to current owners under exact geometry. No empty
engine claim follows from an empty local shard, especially replicated storage.

The allocation-derived POPSCAR1 capacity uses actual blocks, ghosts, components
and layouts with codec framing and the one-cell fragmentation upper bound;
resource limits are installed before archive reads, not learned from payloads.
Internal checkpoint capture preserves the existing provisional accepted-effect
route at transaction depth1; ordinary public observation remains idle-only.

Release catalog drift existed at baseline3302 before this extension. Historical
and recomputed values are pinned in
`tests/review/sol61_uniform_checkpoint9_release_authority.json`. Generate with
`python3 scripts/generate_release_contract.py` and check with `--check`.
Payload9 is distinct from Native ABI8: additive methods/header signature need a
fresh SDK build, without changing public instance/config layouts or claiming ABI9.

Source probes exercise the actual restore bodies and real codec through an
explicit serial host storage adapter (Dim1/2/3, signed zero, NaN payload and valid
contradiction). They are not Kokkos/MPI/installed-Native reception. The future
installed fixture has two publicly authored physical structures, full-carrier
checkpoint/restart/replay and resealed malformed/contradictory/legacy variants.
ROOT must independently review, rebuild and execute before qualification.

The future Fan–Li witness runs eight original Uniform SSPRK2 steps (checkpoint
at4, continuous5–8, restart4, replay5–8). The independent two-transport model
uses checkpoint1 and replay2. These are prospective Native contracts, not received
science. Compatibility variants reseal an actual projected version8 archive,
require strict9 rejection, then explicitly restore scope8 before full9 restore.
Run/capture failures are persisted immediately and the existing helper rethrows
the original exception. No observer NPY replaces the full grown carrier.

Principles→decision→file→oracle→command→status: 1.1/1.3 physical equations and names
remain public in fixtures; native State codec/restore are independent of them.
1.4/1.6 actual native bytes enter `_system_io` and transaction restore. 1.7 capacity
is allocation-derived without observed allowances. 1.8 vector/scalar anisotropic
host cases plus two unrelated public structures exercise the common storage effect.
Source commands use `test_sol61_uniform_checkpoint9_preparation.py`,
`test_sol61_uniform_carrier_restore_host.py`, `test_sol61_uniform_storage_observation_host.py`
and `test_sol61_state_carriers_resource_capacity.py`; Native execution of
`test_uniform_state_carrier_checkpoint_runtime.py` remains ROOT-owned and unrun.

The catalog correction is a distinct preceding commit at unchanged payload8:
`sol61_release_catalog_baseline3302_authority.json` records historical digests,
actual catalog digest and baseline guard failure. The CP9 authority file then
records the separate payload9 change, still Native ABI8.

Additive observer ABI correction: the historical C++
`observe_accepted_state_storage() const` signature/export is preserved exactly.
The boolean capture helper is private. A separate public checkpoint-carrier route
permits the existing provisional checkpoint effect only under its uncommitted
external step transaction; it does not publish or label that image accepted.
The observer remains unconditionally accepted-idle. Old noarg method pointers
remain source-compatible and the native explicit noarg symbol is retained.
