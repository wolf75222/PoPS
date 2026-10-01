# Hooke actual ExecutionLane transport review

Independent reviewer actual GPT-6.1 Sol; frozen author commit `0690571a`, parent `1b9ca827`. Separate docs-only review fork; previous receipts preserved. No environment setup, PoPS build, JIT or native run.

Correction to earlier source receipt: `execution_lane.hpp:639` allgather_bytes belongs to ObserverMpiLane (class begins at497), while ExecutionLane ends at488. I incorrectly attributed its enclosing class. Actual native build of837606ed correctly rejected checkpoint_state_carriers at11096. The earlier claim that that call had internally gated ExecutionLane allocation safety was therefore wrong. Native build log supplied by Root was read; it is evidence of failure, not success.

New source decision: no concrete blocker found on0690571a before rebuild. checkpoint_state_carriers now uses the existing actual ExecutionLane overload broadcast_bytes_inplace at execution_lane.hpp:942–944, which forwards the authenticated lane communicator to comm.hpp:874–897. There is no ObserverMpiLane or MPI_COMM_WORLD transport substitution. Canonical rank order gives each source exactly one length and payload broadcast. uint64 length is encoded into eight plain bytes; unsigned-char decoding avoids signed-char extension; uint64-to-size_t is checked. Source copy, peer resize and result-vector allocation are guarded by collective exception votes before peers enter the corresponding byte transfer. The existing broadcast primitive chunks payloads at INT_MAX. Empty bytes remain valid for transport but are rejected later by the codec. Source archive shards remain authenticated by merge rank index and replica consensus.

The code borrows the already required package lane and does not create/free a communicator or change pinned storage. Capture host mirror allocations/fence still enter the first collective staging vote. validate_checkpoint_state_carriers retains source-only geometry, Real/block envelope and exact encoded-byte identity consensus. Restore still freezes bytes, requires transaction, authenticates target geometry and checks every valid cell bit before full in-place copy; device copy/fence failure is collectively propagated for outer rollback. All three explicit native instantiations and bindings are present. This inspection does not prove MPI failure injection, numerical restart hashes, native linking or GPU behavior. Existing MPI transport wrappers govern MPI-call failures; this patch adds no alternative communicator recovery semantics.

| Principle | Decision |
|---|---|
| 1.3–1.5 | Existing common lane-scoped byte transport, no model treatment |
| 1.6 | Prior source API error exposed by actual native build; corrected source now awaits actual rebuilt native evidence |
| 1.7 | Serial broadcasts of rank shards are control traffic; no new measured MPI/performance receipt |
| 1.8 | Reuse actual existing primitive; no new facade or observer lane |

Inspected frozen files SHA256:

```text
fcdb3c26b5277aed0af1d0b2acaf2b78c4ec66d19b54dfc05ea96b7f66ae32ca  src/runtime/amr/amr_system.cpp
5fb4b63e0d3b06ce6e66b075df91c818680a8e1e8116eff894ede4356a83fd09  include/pops/parallel/execution_lane.hpp
ea33342d30b86f2f4affff5e3f2946d5194b61cb3ae66dfac5d30819c550c291  include/pops/parallel/comm.hpp
```

## Actual coherent host probe

Executed `/usr/bin/python3 tests/review/sol61_state_carriers_lane_host.py` in this frozen review worktree. The author probe extracts the exact changed CPP transport block and compiles against the real ExecutionLane and collective_exception headers with C++20 -Wall -Wextra -Werror. Stdout: `SOURCE_ONLY exact transport/real ExecutionLane API: empty, NUL, 70000 bytes passed`. This closes the earlier missing-method source API witness for the serial host; it remains no MPI or full native translation-unit proof.
