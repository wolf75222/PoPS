# Joint reconstruction resource probe v1.2

The companion uses the exact SHA256-pinned v1.2 case and authentication helpers.
Its v1 and v1.1 predecessors remain immutable. It performs both isolated metadata
preflights before either compilation, and validates the two ABI representations
before binding or profiling, rather than after all runs.

The [resource protocol](joint_reconstruction_resource_probe.md) remains the
measurement contract: two unprofiled warmups, then three fresh profiled instances
per lane; `System.profile(Profile.Advanced())` surrounds only public `pops.run`.
Public bind/run, initial conditions, twelve steps, mathematical method and final
state checks are unchanged. Every profiled state must equal warmup states exactly.
No timing observations from this companion may enter the ABBA timing report.

Absent counters are unavailable, never zero. `scratch_peak_bytes` denotes the
largest single tracked scratch buffer, not total live or peak process memory.
`kernels` and `kernel_launches` retain their distinct native operation/batch
semantics. All raw snapshots, summary, counters and scopes remain in receipts.
Compiler/MPI/stdlib/Kokkos, SDK, external image identity and numerical equivalence
remain mandatory. Private loader copies are recognized only by authenticated DSO
content, as described in the v1.2 timing protocol.

Use the original companion command with
`joint_reconstruction_resource_probe_v1_2.py` and a fresh output directory.
Schema: `pops.api040.joint-reconstruction-resources.v1.2`. No comparative resource
campaign was performed to prepare this version.
