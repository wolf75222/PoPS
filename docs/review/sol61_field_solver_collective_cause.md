# Exact field solver collective exception diagnostics

The real SDK31 table and callback jobs 733032/733033 failed their cause assertions after Native admission. The System solver publication wrapper replaced the provider cause by a generic MPI label. The external component callback/preparation boundaries and the named-field preparation/RHS boundaries also discarded their local exception text. Program cadence and System step already use the common exception transport; they preserved only the generic label they received.

The correction reuses `collectively_rethrow_exception` at these field boundaries. Its existing distributed contract selects the lowest failing rank and broadcasts its diagnostic to every peer; serial execution rethrows the original exception type. External component boundaries retain their previous process-world communicator, while named-field/System boundaries retain their prepared execution lane. Completion destruction/fencing precedes the callback vote, RHS cleanup precedes its vote, and publication rollback precedes the outer exception transport. No callback, report, publication, physics, retry or rollback guard changes. No new ABI/wire contract.

## Source and Host evidence

The C++ bodies in these three paths are byte-identical between the private parent and Native `06b6b190`. The test extracts their actual bodies and compiles them with the actual collective exception and Program rejection headers. A small facade supplies storage-independent cleanup/staging counters. Real MPICH Host processes execute the collective transport; this is not a PoPS engine or provider runtime.

Parent unchanged assertions: 4 PASS / 4 FAIL (1.60 s): serial exception types survived, but MPI dropped original diagnostics and failing-rank identity. Corrected bodies: 8 PASS (1.82 s), including rank-local preflight, callback-after-work, an unrelated named-field dependency and simultaneous failures; all peers receive the same cause, no staging occurs on failure, rollback runs once, and subsequent success remains possible. Earlier harness compiler/path/thread-initialization errors are preserved separately and are not the semantic RED.

Command:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider tests/review/test_sol61_field_solver_collective_cause.py -q --tb=short
```

For the parent comparison add `POPS_CAUSE_BASELINE=06b6b190`. The full `system_fields.cpp` TU also passed Host Clang syntax checking with actual repository headers, MPI and existing ir17 Kokkos includes (Dim2). This does not replace the official rebuild or installed MPI fault rerun. No Native provider/engine, State full-storage rollback, CUDA, scientific calculation or remote job was executed by this test. Native jobs 733032/733033 remain failed receipts.
