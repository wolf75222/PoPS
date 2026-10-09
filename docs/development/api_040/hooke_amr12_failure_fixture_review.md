# Hooke independent AMR12 refusal fixture review

Reviewer actual GPT-6.1 Sol, independent of fixture author Galileo. Frozen author SHA703d29f9 onb4ced740/f1edec55; docs-only review worktree. Source review and AST/in-memory compilation only. ROOT owns actual installed-native serial/MPI2 executions and receipts. No JIT/native/env mutation.

Decision: no concrete source blocker found; proceed to actual ROOT runtime qualification while preserving all strict guards and exact hashes. Snapshot operations were checked against actual native method definitions and Python collective boundaries, not only fixture intent.

Native wrong type/shape/truncation on rank0 is passed by every rank into validate_checkpoint_state_carriers; producer/type/decode failures enter collectively_rethrow_exception before exact archive consensus. MPI exception helper propagates root diagnostic to every peer, so expected boundary substring is available on all ranks. Python int16/2D uint8 injections enter prepare_checkpoint_state_carriers, whose checkpoint preflight votes errors before invoking native validation; test collective_attempt then gathers refusal records. The test helper cannot repair hangs inside native calls; the actual guarded native phases are the reason these injections are admissible. Structurally valid rank divergence tests exact native image consensus only in MPI2.

_capture_image=False active snapshots do not call accepted-only POPSCAR1/POPSDIA1 capture. They use native rank-local full-grown carrier manifest, readonly diagnostic map IEEE bits, histories, lifecycle and temporal relations. checkpoint_temporal_relations and carrier manifest have no accepted-only transaction guard. The fixture explicitly checks capture during transaction is refused and does not remove that guard.

The valid-cell injection uses real set_block_level_state on Q0 coarse storage: rank0 supplies nextafter values, peers original values; native setter writes local field and discards cached evaluation without a hidden transport collective. All ranks subsequently capture global projection. Both actual changed rank0 manifest and changed valid array bytes are required. Restore compares authentic accepted archive valid bits to live state during candidate preparation before any publication. Fixture requires mutated snapshot to remain exactly unchanged after refusal, then real rollback transaction to restore accepted POPSCAR1, diagnostics, carrier hashes, histories and clocks. Finally blocks roll back each genuinely begun transaction. This is direct codec/transaction coverage; it intentionally does not claim public history/auxiliary commit/finalize coverage or device/OOM failure injection.

The all-rank phase registry added byb4ced740 retains real native row/rank order and pins its generated file in the public AMR receipt. No result file exists merely because the fixture was prepared. Serial counts nine named refusals; MPI2 ten, not double-counted by rank.

| Principle | Bounded review decision |
|---|---|
| 1.3–1.5 | Common byte-persistence and transactional guards, existing physical fixture unchanged |
| 1.6 | Real source fixture prepared; ROOT actual native result required |
| 1.7 | No new backend or performance receipt from preparation |
| 1.8 | No model workaround or relaxed bit oracle |

Source verification: `/usr/bin/python3` ast.parse and compile(source, path, exec) without executing or importing PoPS; passed. Expected failure assertions reject missing imports/bindings and arbitrary assertions as successful native refusal.

Inspected SHA256:

```text
c5b91086f536318a239daf1c42f5bb874ddfa9994be1bf57a7b26a197d5d4236  tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py
6110d3f2b8c84d1807cd99a943aaacb652d9f939cbaec34b94eabb7e28a85a61  tests/python/integration/runtime/test_public_evolved_stage_amr.py
8cf9e516d413312cfe4cc235b91aeaf11e43d5cc308b1ffc46e0f8fcc81d8f71  python/pops/runtime/_checkpoint_state_carriers.py
d1fb358cbce7edecb1d2f171c5e2ebad48565cb2aa8cfd78cbbed7f7692576f9  include/pops/parallel/collective_exception.hpp
cb01d980db60d5287efda954db2f6c3313578b495228629978b78c8f9a19a813  src/runtime/amr/amr_system.cpp
```
