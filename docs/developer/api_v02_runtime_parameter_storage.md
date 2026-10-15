# Runtime parameter storage

The declared runtime parameter family is finite and indexed by its exact `BindSchema`.
There is no model-independent limit of 32 coefficients. The native count/index ABI
uses a signed `int`; a cardinality outside that representation is rejected before
allocation. Model equations still read `params.get(index)` in the generated kernel.

At bind, `resources={"runtime_parameter_byte_capacity": N}` sets an explicit byte
budget for **one immutable parameter incarnation**. If omitted, the installer derives
and records the smallest sufficient capacity from the exact installed cardinalities
and the native descriptor size. Invalid integers, overflow, and insufficient capacity
are rejected before allocating a carrier. `runtime.inspect()` exposes
`instance.runtime_parameter_storage`: its `initial_storage_bytes` counts each initial
model or Program carrier once, independent of shallow aliases in generated bricks.
This is not a bound on all snapshots an application may retain.

Each carrier owns one read-only HostSpace payload and one read-only payload in the
execution memory space. Its declared storage charge is the two payloads plus the
owning descriptor; allocator-specific bookkeeping is not included. The implementation
performs the copy and fence before publication. Host reads select the host payload;
device reads select the device payload. Rebinding constructs a complete new carrier,
votes on preparation errors, then replaces the live map atomically. Existing kernels
and snapshots retain the old immutable allocation through managed Kokkos views.

A declaration-only generated brick contains the cardinality and no allocation. It
must be bound before launch; model installation and Program access enforce this.
Unknown or differently sized parameter vectors are rejected. No unbound declaration
is a numerical zero/default fallback. Defaults remain in BindSchema.

`DeviceCapture` preserves the existing trivial-value route and adds managed Kokkos
views plus an explicit exhaustive member contract for generated aggregates. It rejects
unmanaged views and unknown nontrivial host containers. A handwritten C++ opt-in is
a trusted representation contract, not a proof that arbitrary methods dereference
only device-accessible pointers. The actual backend compiler must accept every called
method. The RuntimeParams ABI changes, so the native SDK and generated artifacts must
be rebuilt together and authenticated through the normal PoPS identity checks.

Qualification requires the public 33/65/129-coefficient examples to execute in actual
PoPS on CPU and CUDA, with two distinct binds, strict state oracles, restart, immutable
old carriers, and atomic budget/count rejection. Source tests and syntax checks alone
do not establish that qualification.
