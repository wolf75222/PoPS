# Joint reconstruction comparison v1.2

This is a separate protocol revision. The v1 and v1.1 scripts, their receipts,
and the immutable 3d06cab/cf6dace packages are unchanged. The v1.1 failure at
`outputs/performance-t2-v1-1-run/00-baseline-compile.stderr.log` occurred after
JIT but before any timing samples: it parsed `artifact.abi_key` as native ABI.

The actual contracts in both retained SDKs are different:

- `_pops.abi_key()` supplies semicolon-separated compiler version, C++ language
  version, header signature, Kokkos, standard library, MPI ABI and dimension.
- `artifact.abi_key` supplies `headers|JIT compiler|C++ standard|dim=2`.

The new runner stores both complete keys. It strictly checks artifact headers,
compiler and standard against its properties, headers against the authenticated
SDK/native module, dimension 2, and the JIT standard against native `__cplusplus`.
Duplicate, empty or absent native fields fail closed. Across releases only the
header field is omitted from the native environment comparison: every other
field, including future fields, must agree. Artifact keys remain exact within
each lane. SDK and actual loaded external dependencies remain required for
comparison. Existing compiler command/flags provenance remains available for
manual causal review; identical native ABI does not prove identical optimization.

Before either cold JIT, both isolated workers perform a metadata phase: snapshot
rehash, native import/hash, exact SDK, doctor, ABI parsing and metadata API shape.
This phase returns before case construction or compile and must leave both cold
caches empty. Profiling API shape is checked too; this is not proof of execution.
`PYTHONPATH`, `POPS_INCLUDE` and `POPS_NATIVE_VARIANTS_ROOT` are removed from worker
environments; one Python prefix and separate lane caches remain mandatory.

The native loader maps private copies of authenticated DSOs under random paths.
They are excluded from *external* dependency equality only if their SHA256 equals
an authenticated generated DSO or the authenticated native extension. No temporary
path pattern is trusted. Unknown images remain fully compared by path and hash.
Original DSO identities are retained in every worker receipt.

The physical case, exact initial cell averages, numerical method, grid, step
calendar, tolerances, warmups, ABBA order, sample counts, timing boundaries and
budgets are unchanged from the [v1 protocol](joint_reconstruction_benchmark.md).
Use that protocol's commands with `joint_reconstruction_benchmark_v1_2.py` and
fresh output directories. `--worker --phase metadata` is available for isolated
no-JIT diagnostics; it produces no benchmark observation. The result schema is
`pops.api040.joint-reconstruction-comparison.v1.2`.

No new benchmark campaign was run while preparing this version. See
[the metadata audit](joint_reconstruction_v1_2_metadata_audit.md) for checks and
limits. The [v1.2 resource companion](joint_reconstruction_resource_probe_v1_2.md)
pins the exact new timing script and remains separate from timed ABBA workers.
