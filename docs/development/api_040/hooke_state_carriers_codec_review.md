# Hooke independent carrier-codec review

Reviewer real GPT-6.1 Sol; author Banach. Frozen author SHA `86a71b63` on baseline `3b2b0509`. Separate review worktree; tests/docs only. Design principles 1.1–1.8 and central goal reused from the actual earlier read. No setup, PoPS build, JIT or native runtime execution.

## Decision before native compilation

No concrete codec correctness blocker found by bounded source review and executed host witnesses. Native mandatory acceptance remains open: this receipt cannot establish restart hashes, MPI collective failure safety or device rollback. Keep the exact ghost/valid-state hash oracle.

Capture stages allocations, host mirrors and fences under catch before the collective vote (amr_system.cpp:11062–11094). ExecutionLane::allgather_bytes (execution_lane.hpp:639–675) votes allocation/copy failures internally. Merge checks rank-index authority, envelopes, duplicate owners and all-rank exact replica payload consensus (state_carriers.hpp:153–185). Decoder bounds row/name/payload counts by input bytes, rejects trailing bytes and overflowing shape, and preserves 32/64-bit payloads without numerical normalization.

Restore freezes producer bytes, collectively preflights source-only geometry and identity, requires a restart transaction, checks all global target geometry, then compares every valid bit against uploaded scientific arrays before staging full ghosts (amr_system.cpp:11137–11204). Publication uses copy_full_field_in_place: existing allocations remain bound, and copy/fence exceptions enter a collective failure vote. Outer restart rollback is required; the method itself does not roll back. rollback_restart_transaction (18687–18717) retains its snapshot on restore failure. No field refresh occurs; evaluation caches are discarded noexcept. Native asymmetric allocation, copy/fence failure and outer rollback injection remain necessary.

## Material performance finding [P2]

state_carriers.hpp:208–212 performs P(P-1)/2 overlap comparisons per level on block zero. At 100,000 patches this is 4,999,950,000 candidate pairs per validation, repeated at capture canonicalization, Python source validation and restore validation. This is not negligible and carries no finite validation budget. Cross-block coverage recount at 225–227 additionally scans all rows for each block/level.

Existing generic BoxArray disjointness is also quadratic but explicitly requires BoxArrayValidationBudget (mesh/layout/box_array.hpp:18–23,214–224). A finite budget that refuses valid scientific cases would narrow the reception and is not a final scalable remedy. Generic BoxHash (mesh/index/box_hash.hpp:61) offers bounded spatial candidate generation with explicit bin budget, if coordinate range and exact overlap semantics are preserved. An exact sorted endpoint sweep is another option: sort by axis-0 lower coordinate, retire intervals whose upper coordinate is strictly below the next lower endpoint, compare remaining candidate boxes in every dimension. Report worst-case quadratic behavior for the sweep; do not claim universal linear scaling. Cache per-level counts while validating instead of rescanning all rows. These are common geometry operations, not model treatments.

| Principle | Decision / limit |
|---|---|
| 1.1–1.2 | Byte carrier persistence is a common execution effect; does not impose new scientific equations or model assembly |
| 1.3–1.5 | Opaque block identities, arbitrary component counts, explicit Real width and full patch storage; no model name dispatch |
| 1.6 | Independent host authority witnesses executed; native restart receipt still open |
| 1.7 | Quadratic unbudgeted source geometry validation is a material scalability concern; no native/performance receipt |
| 1.8 | Keep one versioned codec; use existing common geometry mechanisms, no per-model branch |

## Executed commands

```sh
/usr/bin/clang++ -std=c++20 -O0 -I include tests/review/hooke_state_carriers_host.cpp -o /tmp/pops-hooke-state-carriers-host
/tmp/pops-hooke-state-carriers-host
/usr/bin/clang++ -std=c++20 -O0 -I include tests/review/sol61_state_carriers_codec_host.cpp -o /tmp/pops-author-state-carriers-host
/tmp/pops-author-state-carriers-host
```

Independent stdout: HOST_ONLY independent: mixed ownership, missing replica, 32-bit payload, overlap, extent overflow, missing block refused; signed-zero/NaN bits preserved.

Author stdout: SOURCE_ONLY codec: ghost bits, signed zero/NaN, N components, shards and refusal cases passed.

No native package was loaded; these compile only standard-library codec header executables. Legacy v11/v12 Python source handling and scientific restart run are owned by Root and were not requalified here. Source archive hash consensus authenticates encoded bytes; exact scientific NPZ/carrier agreement is checked by runtime valid-bit comparison and still needs native execution.

## Inspected author-file SHA256

```text
fcfcb090de0071d0982cf3b76b0df9613f729e27738ad55db81943df5005e78a  include/pops/runtime/checkpoint/state_carriers.hpp
a5a3cabb0281ce72563a7b65731daf32c4e946e893c5fd7801c34edcda8ceee7  src/runtime/amr/amr_system.cpp
13784da953b9880880b378151ce7a39e743b612c275195d7f8b7ea41057719ae  include/pops/runtime/amr_system.hpp
67d3489916ac6a1e4da5b0dc819e827577a6a8128fc33e971559469ceb5cdea3  python/bindings/core/init/init_amr.cpp
48a6475ca7d3e658f0f66533e2d200b281f2a7717b16dfc3fcaf3f8f7435168d  include/pops_headers.manifest
a331460d35c2466c0dd99940e327a0647d2d5546f6a5297bec4602d92d74425a  tests/review/sol61_state_carriers_codec_host.cpp
```
