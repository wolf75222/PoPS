# v1.2 metadata audit and diagnostic receipt

The v1.1 failure is an observer defect: it supplied the pipe-separated generated
artifact key to a semicolon-separated native ABI parser. This is reproduced by
the negative unit test; no permissive parser or omitted comparison repairs it.
The production packages and all earlier protocols remain unchanged.

Changes are confined to new v1.2 timing/resource scripts, their documentation and
tests. Both ABI authorities are recorded and cross-checked at their real boundary.
The resource worker now checks artifact metadata before any binding/profile runs.
Both drivers authenticate two no-JIT metadata workers before either cold compile,
and require their cache/codegen directories to remain empty. The timing worker
purges the same external package/SDK overrides as the resource worker.

A separate diagnostic also found the native loader's private DSO copy in the
external dyld inventory. A random path cannot be compared between processes.
v1.2 excludes it only on exact hash equality with an already authenticated PoPS
DSO/native extension. Every recognized copy path/hash is retained under
`recognized_owned_copies`; all remaining `images` are compared by path and hash.
No prefix-based temporary-directory exemption exists.

Actual checks performed:

- **17 source tests passed**, including malformed/duplicate/missing ABI fields,
  wrong headers, compiler, dimension and standard, exact companion pin, absent
  counters, real property shape, known/unknown/tampered private images, preflight
  ordering, and AST equality of the physical case/initial state/run boundaries
  against frozen v1. Scenario constants and timing budgets are identical.
- Both retained packages were imported in isolated subprocesses and rehashed:
  1079 baseline / 1080 candidate members. Native environment fields agree;
  their different header identities remain authenticated independently.
- The real `_invoke(..., phase="metadata")` path succeeded for both packages;
  cold cache and codegen directories remained empty. It validates class property/
  field/profile API shape, not compiled values or a simulation result.
- A **copy** of the existing baseline v1.1 cache loaded through public compile
  with an audit hook forbidding new DSO compilation. The only compiler operation
  permitted was the existing tiny C++ standard syntax probe. SDK version and
  libomp-prefix queries were read-only. Real artifact properties, DSO hashing,
  `ExecutionContext` and public bind worked. A separate empty profile and then
  two genuine public steps exercised structured profile access: source snapshot,
  enabled true, steps 2, kernels 6, no scratch counters. No comparative timings
  or performance conclusions are drawn from those diagnostic operations.
- Ruff and source syntax checks passed. No installation, shared environment
  modification, native rebuild or benchmark campaign was performed.

The compact [receipt](evidence/joint_reconstruction_v1_2_metadata.json) deliberately
omits diagnostic durations. Full local diagnostic data remain in the author's
`outputs/perf-v12-*`; original failures remain in the parent workspace's
`outputs/performance-t2-run` and `outputs/performance-t2-v1-1-run`.

Limits: candidate compiled-artifact/profile execution was not run; baseline
cache diagnostics are not a cold JIT qualification. The complete twelve-step
ABBA and separate resource campaigns remain root's responsibility. Source
shape tests alone do not establish MPI2/GPU/AMR performance. Compiler flags
remain a required manual causal review, not something inferred from ABI equality.
