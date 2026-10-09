# Independent T5 candidate capture reception, 2026-09-30

Author freeze: `c06485d812e75d5ebdd72bc5153eaaf1465f25c7`, exact parent
`26e5fb04ebb515e83edcd5a7bfe1e933a96cafcb`. Integrated Python source received:
`bb87174d82d95422c3086eea7b0c28cf8390fb15`. This review changes only tests and
this report. It does not install, build, invoke JIT, or write MAIN/shared SDK.

## Received evidence

* Independent public Python authoring/codegen: 9 passes on author freeze, plus
  two intentionally failing rate-first composition witnesses. The legacy
  rate-first refusal also exists in exact parent26e: it is **not attributed to
  T5 as a new regression**. Root's integrated State-priority composition passes
  all 11 independent tests. Its six version8 tests also pass: **17 source passes**.
  Imported package path was explicitly MAIN/python/pops, with bytecode disabled;
  this is not evidence from the older installed SDK506.
* Actual production PreparedIntegralCapture, PreparedResourceCache,
  AcceptedExchangeLedger, and serial ExecutionLane: **19 host checks pass**.
  Private access is exposed only after including dependencies, solely to call
  the actual guard with independently varied data. No System or MPI/Kokkos
  consumer executes in this host probe.
* Independent public native System/provider fixture compiles in the real full
  ProgramRuntime test translation unit, Dim1/Kokkos/MPI headers, syntax only.
  One existing gtest character-conversion warning. It has **not executed**.
* One independently authored historical State-first graph retains exact
  serialized JSON bytes, semantic IR hash, v1 integral identity, and emitted C++
  bytes across parent26e, authorc064, and integrated MAINbb87174. IR hash:
  `590b20efd2b65fc3ff0a5f47a675c41e0380f0ffa0bb7d892d8d9b0f4a15ed0d`.
  JSON SHA256: `328e38aff5f24a44ae670df53085ab560901ca34830a189548da48dce2ece457`;
  C++ SHA256: `5faad83475938090af8717f8544b60ba771cc2d7ac54c0a2efae22fdad8c7871`.
  This is bounded compatibility evidence, not proof for every historical graph.
  Entire accepted_exchange.hpp is unchanged parent/author, SHA256
  `2f9c89c3e873bb270820482e14840aa2fc1d7cc13956a5e9f014aef35ce63731`.

The public witness independently selects the original source `S=-gamma*U`,
retains that physical source node, then composes `U+dt*q*S`. Genuine capture
emission occurs once; its POD read precedes the real for_each_cell kernel.
Mutations of units, scope, version, or integral identity refuse emission;
foreign ownership and mixed points refuse atomically. Unknown units cannot
become dimensionless by capture. An oversized initial integer preserves
OverflowError and leaves declarations unchanged. Direct Equation/FieldProblem
materialization remains unsupported.

Actual native guard checks cover foreign owner, foreign actual cache attempt,
physical point and graph identity, changed units/identity, unrelated ledger
mutation with unchanged scalar, changed scalar, revocation, retry, cache-clear
lease revocation, exact restoration in the same live attempt, and fresh retry.
Serial RuntimeError and OverflowError are preserved exactly, not swallowed.
finish_attempt retains a current capture; begin_attempt/reject/clear revoke it.
Lease memory remains owned after revocation while its authority is invalid.

## Native unit-canonicality defect and public counterprobe

The actual guard accepts `not-json` and reordered JSON
`{"powers":[],"kind":"physical_dimension"}` when their SHA256 is inserted into
a newly declared v2 identity. The public path is unguarded at this freeze:
ProgramContext::declare_integral_state → System::declare_program_integral →
declare_integral_collectively checks the nonempty identity and finite scalar,
but does not parse units. ProgramContext::capture_integral_candidate invokes
the same actual guard, whose require_units_ only checks v2 prefix and digest.
AmrProgramContext uses the same guard and analogous declaration.

`sol61_t5_public_native_capture.inc` supplies a true public System/provider
counterprobe, with no private shim or substitute runtime. Include it after
the helpers in tests/cpp/integration/runtime/test_program_runtime.cpp to run.
It requires malformed/noncanonical units to refuse before POD consumption and
checks State, ledger, time, and macrostep rollback. Serial exceptions retain
invalid_argument; the actual collective exception protocol uses runtime_error
on multi-rank lanes. Syntax reception is green; **native execution is pending**.
The guard-host acceptance plus the direct public forwarding establishes the
missing source invariant; actual System execution remains centrally reserved.

A bounded correction belongs before digest/contract construction in
require_units_, shared by prepare and consume: validate exact canonical
PhysicalDimension bytes (base names, strict sorting/uniqueness, rational
exponents, positive denominator, reduction, explicit zero omission). Do not
convert units or weaken byte identity. Ptolemy owns that separate correction.

Two other private-shim observations have narrower scope: zero stage denominator
is accepted by the guard's injected point getter, but real public set_stage_time
rejects it; +0→-0 point changes compare equal, but no public runtime route that
changes this bit was demonstrated. They are not represented as public native
counterexamples or as passing refusal tests.

## Reproduction and limits

All source commands use env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 and
/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python. Insert the chosen
authentic checkout/python into sys.path, then pytest the independent public
file. Add MAIN/tests/python/unit/codegen/test_program_v8_contract.py for the
17-test integrated reception. The default independent file is intentionally
red on the author freeze for the two composition witnesses described above.

Run the host guard with clang++ -std=c++20 -O0 -Iinclude
tests/review/sol61_t5_capture_guard_host.cpp -o /tmp/pops-sol61-t5-capture-guard-host,
then the executable. The separate-process legacy script accepts a source-root
argument; export the exact parent with git archive before comparing outputs.
--rate-first is a separate support witness and legitimately refuses on parent26e.
Logs and source digests are preserved in
outputs/sol61-t5-independent-review-20260930 outside the review checkout.

Source inspection places collective error voting before exact contract agreement
and before POD return. Actual native Kokkos consumption, MPI rank-local failures,
full public parent/child rollback, accepted external trace delivery exactly once,
retry/restart, and saved-state scientific feedback remain **unexecuted here**.
The author has genuine public FV/exterior fixtures for central reception; their
presence is not converted into an execution claim. No M14 wall/sheath/capacitance
closure or scientific qualification is asserted.
