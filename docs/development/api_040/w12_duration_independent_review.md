# Independent review of W12 native duration probe

Source reviewed: `3d943097`, test
`ProgramRuntime.NestedChildCommitThenParentRejectRestoresDurationAndExchangeMailbox`
in `tests/cpp/integration/runtime/test_program_runtime.cpp`. This review did not
compile or execute the native test; central reception remains authoritative.

The duration is read from the real `ProgramContext::cache_effective_dt` route to
`CacheManager::effective_dt`, which consumes the accumulated native interval.
The test seeds 0.25 once, then observes 0.35, 0.45 and 0.55 across attempts with
dt 0.1, 0.2 and 0.3. It applies the duration-dependent state change and stages an
actual ExchangeRecord through the qualified ProgramContext boundary. It checks
the integrated exchange amounts, accepted cell values, history, cached field,
clock, macro-step and ledger contents. Child rejection and later parent rollback
must both restore the original cache duration and remove provisional exchanges.
This is not an author-side mock accumulator.

The exchange context checks discriminate all three binary64 dt encodings. The
runtime record formatter actually uses `bit_cast<uint64_t>(point.dt)`;
`ProgramContext::stage_exchange` supplies its authenticated evaluation point.
The current central `build-mpi/CMakeCache.txt` was inspected and specifies
`POPS_REAL_TYPE:STRING=double`. No float32 acceptance follows from the double
precision assertions in this receipt. MultiFab's ordinary copy is deep through
Fab's copy constructor, so constructing the temporary bump does not alias the
accepted state storage.

One test-quality defect was reported before central reception: the deliberate
duplicate append catches any `std::exception` and sets `duplicate_rejected=true`.
An allocation or qualification failure at that call would therefore count as a
successful duplicate rejection. Preserve collective completion, record the
caught message, and assert after the step that it contains
`duplicate accepted exchange occurrence/quadrature contribution`.
`collectively_rethrow_exception` preserves this diagnostic in both serial
(`invalid_argument`) and MPI (`runtime_error` with an authoritative rank), so
requiring a single exception type would be inappropriate. Do not put a fatal
rank-local assertion inside the installed Program body.

The accompanying correction stores `error.what()` without asserting inside
the body and checks the specific duplicate diagnostic after the successful
retry step. Other exception reasons now fail this requirement; the existing
ledger/state/history/cache checks are preserved. No new numerical threshold,
time increment or exception handling was introduced in production.

The initial rejection already requires StepAttemptRejected and observations
after the effective duration and staged exchange. Nonfatal comparisons and
size guards before vector indexing retain participation in subsequent native
collectives. The documented outstanding public `AccumulateDt` lowering gap is
not closed by this manually installed native Program. No AMR or application-wide
W12 qualification follows from the probe.
