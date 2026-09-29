# C36 — native resource lifetime implementation (bounded)

Checkout: `PoPS-resource-lifetime`, branch `codex/api040-resource-lifetime`, base `022f5acbb9181eb85a6d9cd05e30a92aad1c4872`.

## Concrete defect and ownership

The former `launch_for` returned after queue submission while its functor could capture raw pointers. Neither a cache replacement nor an executor destructor protected an external captured allocation destroyed before the executor. A prepared resource cache also had no exact attempt/completion registration through accepted rollback.

The existing cache now owns a shared native State: current actual resource holders, monotonically increasing versions/attempts, and controls for actual native submissions. A lease owns the actual holder and its version; its version and attempt refer weakly to State. Controls retain executor/buffer holders; no strong ownership cycle returns to State. The Program hook captures only weak State, and remains safe across cache moves, context destruction and artifact install snapshots.

Each asynchronous `submit_for` requires the exact cache attempt, exact executor lease, and a lease from that same cache for every externally captured allocation. The executor creates a CUDA/HIP event and records it on the actual selected Kokkos stream. CPU and unsupported-event backends use an actual instance fence and claim no asynchronous completion. The cache retains the control even if callers drop their ticket; dropping a ticket cannot hide a failed/cancelled required submission.

`launch_for` is now synchronous before return, so its legacy raw captures are safe. The only found existing non-test caller, `benchmarks/adc757/heterogeneous_numerics.cpp`, now uses owned `submit_for`: original kernels, numerical work, baselines/candidates and timing order remain, each submission has its own native completion event, and independent lanes are still enqueued without intermediate fences in the concurrent route. Its synthetic rollback now rejects and drains before workspace overwrite.

## Actual transitions

- Collective acquire: failed construction preserves old buffer/version; successful replacement revokes the old version. Reacquiring a mutable resource drains submissions that still borrow its exact version.
- `begin_attempt`: rejects and drains previous work before mutable reuse, increments a counter held outside accepted snapshots, then clears old controls. The Context overload converges allocation/drain preflight across its actual ExecutionLane before Program collective kernels.
- `ProgramContext` / `AmrProgramContext`: install weak lifetime hooks; `begin_step` creates the attempt; `prepared_resource_lease` and `submit_prepared_for` expose the existing native resources/executor through the actual context.
- `ProgramRuntimeState::advance_cadence_region`: drains and verifies submissions before returning a completed region or a map barrier; rejection drains before unwinding. These are the actual shared Uniform/AMR numerical dispatch hooks.
- `prepare_accepted_restore`: revokes then drains before Uniform or AMR writes accepted carriers. A remote-rank failure therefore also revokes a locally successful task before collective rollback.
- Uniform and AMR `AcceptedSnapshot` constructors drain before their first carrier copy. Snapshot quiescence is separate from numerical success, so an old rejected attempt does not prevent the next retry snapshot.
- Invalidation, restart regrid preflight, artifact step replacement and failed-install rollback are connected. The install snapshot preserves the weak callback, but never rewinds the cache attempt counter.
- AMR generation invalidation revokes cached resources and drains before new context scratch is exposed.

## Failure behavior

Logical revocation immediately removes result consumption authority but leaves physical completion observable/drainable. Query failure is never accepted as completion: it records failure and must successfully fence. A failed record similarly fences before unwinding. A failed drain leaves all owners retained; destruction/rollback that cannot establish drainage terminates rather than releasing buffers or overwriting accepted state. Event/callback owners are destroyed before the leased executor releases externally owned native streams. No public API accepts an invented completion/success boolean.

## Tests and actual validation

No build/install or native test executable was launched by this worker (root centralizes compilation). Four **syntax-only** checks passed against the isolated source with `/usr/bin/clang++`, C++20, real installed Kokkos/OpenMP/MPI headers and main compile_commands options adapted to the isolated source/env:

- `test_prepared_resource_cache.cpp`: exit 0.
- `test_prepared_stream_executor.cpp`: exit 0.
- `test_program_context_contract.cpp`: exit 0, including actual System/AMR context templates and the new System transaction test.
- `benchmarks/adc757/heterogeneous_numerics.cpp`: exit 0.

Only the existing Googletest char8_t warning appears in test TUs. Logs are `outputs/lifetime-syntax-*.log`. `git diff --check` passes. `scripts/check_packaging_manifest.py` passes after classifying both new headers as sdk-support (167 api, 7 abi, 18 sdk-root, 162 sdk-support, 8 test-only). Syntax checks do not establish runtime behavior, linked ABI compatibility, MPI success or GPU event support.

Added test coverage (awaiting root native execution):

- Exact buffer ownership through replacement and clear; failed candidate preserves current lease on every rank.
- Actual CPU Kokkos kernel, logical cancellation, late completion acknowledgement, and retained-buffer destruction counts. The CPU kernel is physically fenced before return: this is explicitly **not** a GPU pending-event proof.
- Stale versions/attempts, invalid/foreign submissions, cross-cache buffer refusal, monotone retry, dropped cancelled handles.
- Destruction with external ticket surviving and with only the cache registry retaining the submission.
- Weak callback after move/destruction; legacy raw capture completion before caller release.
- `ProgramContextContract.PreparedTaskRollbackRevokesEveryRankAndRetryUsesNewAttempt`: actual System accepted transaction with rank-zero injected failure after actual Kokkos submission; all ranks must restore their own accepted state, revoke consumption, drain, and retry with a larger attempt id.
- `ProgramContextContract.ArtifactInstallRollbackRestoresWeakLifetimeHookWithoutRevivingAttempt`: install rollback restores callback ownership while attempts stay permanently revoked.
- `PreparedStreamExecutor.CancelledCudaSubmissionRetainsWorkspaceThroughRealPendingEvent`: actual CUDA event must first report pending, then cancellation/clear/drop owners preserves workspace until drainage. If no pending event can be observed, skips rather than claiming coverage.
- `PreparedStreamExecutor.LegacyCudaRawCaptureIsCompleteBeforeExternalBufferRelease`: real CUDA stream query before mirror copy or external buffer release.

Root targets: `test_prepared_resource_cache`, `test_prepared_stream_executor`, `test_program_context_contract`. MPI2 selection should include `PreparedResourceCache.*` and the two exact ProgramContextContract tests above. The two CUDA tests report `not_executed` on this CPU-only build.

## Boundaries — not complete C36

No generated PDE currently submits this accelerator executor. The actual context submission API, dispatcher/rollback integration and ADC-757 benchmark are connected; this is not evidence of asynchronous PDE execution. Existing blocking MPI exchanges were not converted to nonblocking requests and acquire no fictitious completion tickets. No GPU/HIP execution or injected native event API failure was available; CUDA/HIP code paths require their native compiler/device campaign. Full required-output consumption accounting, arbitrary DAG cancellation dependencies, concurrent host coordinators, checkpoint serialization of task graphs, and asynchronous multi-level AMR PDE qualification remain outside this patch. Every raw captured allocation must obey the explicit lease contract; functor pointer provenance cannot be inferred by C++.
