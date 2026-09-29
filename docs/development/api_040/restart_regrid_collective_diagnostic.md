# Preserve the actual restart-regrid refusal across MPI ranks

The AND9 C++ reception completes 82 CTest rows: 79 pass, one MPI aggregate
fails, and two single-rank mismatch guards skip. The three-level restart test
passes in serial, including second-resource-refresh injection, exact rollback
and retry. Its MPI2 version reaches all rollback/retry assertions; the only
failure is loss of the injected failure diagnostic at the outer restart boundary.
The diagnostic-only rerun (`f3d3205`) prints the same generic message on both
ranks: `AMR restart regrid failed on at least one MPI rank`.

`AmrSystem::regrid_on_restart` now uses the existing
`collectively_rethrow_exception` helper on the same prepared published lane.
All ranks still vote before leaving the phase; serial execution preserves the
original exception type. MPI selects the lowest failing rank and broadcasts
that rank's actual reason. This retains the nested resource-refresh failure
instead of replacing it with an uninformative string. The helper already
converges diagnostic-allocation failures across ranks.

No scientific criterion, transaction state, rollback order, pending-history
authority, wire format or test expectation changes. The original injection
assertion remains strict. This diagnostic correction requires a native rebuild
and a fresh MPI2 receipt; the prior failed receipts remain evidence, not passes.
