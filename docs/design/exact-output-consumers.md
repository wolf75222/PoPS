# Exact scientific output consumers

PoPS scientific output has one resolved boundary. The consumer graph chooses an accepted state and
produces an `OutputRequest`; format writers never search the runtime for a convenient first block or
silently widen a selection. Every `FieldKey` contains the canonical declaration `Handle`, component
manifest identity, layout identity, level and state identity. `OutputSnapshot` adds the exact clock,
plan/bind/run provenance, centering, AMR boxes, coverage mask and metric cell volumes. Every native
piece retains its `global_box_index`, `owner_rank` and explicit `replicated` bit in the snapshot and
format manifest; distributed validation proves the exact indexed box union, not only an equivalent
rectangle union. The current
public contract is deliberately unitless: it neither invents nor persists an opaque unit vocabulary.

The final public formats are typed:

```python
from pops.output import HDF5, NPZ, ParaView, ParallelMode

NPZ(mode=ParallelMode.SERIAL)
HDF5(mode=ParallelMode.ROOT)
HDF5(mode=ParallelMode.COLLECTIVE)
HDF5(mode=ParallelMode.PER_RANK)
ParaView(mode=ParallelMode.ROOT)
```

`ParallelMode` is part of the format identity; it is never inferred from process globals, rank
count, target suffix, or writer availability:

- `SERIAL` requires the proved serial `ExecutionContext` (rank 0, size 1) and one complete snapshot.
- `ROOT` requires a distributed context. Every rank participates in the authenticated native
  gather, but only rank 0 prepares, verifies and atomically publishes the single-file writer.
  Preparation failures and the final receipt are broadcast to every participant.
- `COLLECTIVE` requires a distributed context, an authenticated collective resource plan and the
  native C++ parallel-HDF5 provider. Each rank writes only its exact non-overlapping native
  hyperslabs with exactly one MPIO collective transfer per dataset and rank (including a select-none
  transfer for a rank with no patch). A replicated AMR coarse patch is assigned to rank 0 for this
  mode so it cannot overlap.
- `PER_RANK` requires a distributed context and preserves each rank's exact local pieces, including
  explicitly replicated coarse pieces. Targets are rank-qualified before any file is opened. The
  transaction succeeds only after it aggregates one deterministic receipt per contiguous rank.

`NPZ` supports `SERIAL`, `ROOT` and `PER_RANK`; per-rank NPZ stores each local piece separately with
its global half-open bounds. `HDF5` supports all four modes. ParaView supports `SERIAL`, `ROOT` and
rank-qualified local VTU files in `PER_RANK`; the native external Writer supports `SERIAL`. Unsupported
combinations are refused by the format descriptor, not degraded later.
The resolved mode, request family, snapshot metadata, format identity and canonical target family
must agree across communicator rank order before a writer is entered. Native parallel-HDF5
availability and the collective MPIO transfer API are preflighted before engine construction.
This is a structural extension protocol: every resolved format supplies a writer implementing
`preflight(execution_context) -> dict`. The runtime compares that canonical capability evidence
across ranks; it never branches on an HDF5/NPZ/VTK provider id. A third-party format therefore adds
its own dependency and topology proof without modifying the runtime publisher.
Native Writer components use the equally small optional
`installed_component_requirement() -> dict` writer protocol; the runtime authenticates that
requirement against the format evidence and installed interface without dispatching on a provider
name.

Each implements the public `pops.output.FormatInterface.writer()` contract. A custom provider uses
that same `pops.output` extension seam; it never imports an execution adapter from `pops.runtime`.
After resolution, the private runtime bridge turns an accepted side effect into an exact writer
request. Effect-free `prepare_session()` returns a public structural `WriterSession` on every rank;
its canonical authority and identity are authenticated before any backend I/O. `stage()` then writes
and reopens only a temporary file. A `ROOT` non-authority rank owns a no-op session, `COLLECTIVE`
stages every rank, and `PER_RANK` stages one local target per rank. Any mixed-rank stage failure calls
idempotent, collective-safe `abort_prepare()` on all sessions, so a successful peer cannot leak its
temporary. The accepted-side-effect transaction later calls `publish()` for an exact receipt and an
atomic no-clobber hard-link publication, `abort_prepare()` to remove unpublished residue, or
`rollback()` to compensate an artifact published by the still-open outer transaction. Once that
outer transaction seals, it calls the session's idempotent `finalize()` release phase and permanently
ends compensation before attempting release. `finalize()` must return `None`; a failed or invalid
release is a post-commit diagnostic and remains retryable without reopening rollback. Built-in
sessions retain the exact descriptor returned by `mkstemp` throughout staging and rollback
eligibility; `finalize()` closes that descriptor only after the receipt is sealed. Deterministic
filenames include the consumer, clock, macro-step and exact selection identity;
different bytes targeting the same name are a collision, never an overwrite. Runtime validation is
purely structural: custom sessions need no PoPS base class and the bridge neither checks a built-in
prepared-file type nor dispatches on `provider_id`. Built-in file sessions record the device/inode
created by their hard link; rollback refuses a target that has since been replaced and never deletes
the replacement.

The configured output root and each target parent are stable real directories, not symbolic links.
Staging/quarantine opens the parent and its private quarantine directory without following a leaf
symlink and fails closed if that authority cannot be retained. PoPS does not promise to follow or
retarget an output-root symlink between preparation, publication, rollback or recovery.

The native `ExternalWriter` keeps the same no-clobber boundary without exposing its component's
publication path. The runtime retains a staging-directory descriptor while the component writes its
temporary and private `.component-published` paths. After native verification the runtime
exclusively `os.link()`s that authenticated inode to the deterministic public target and records the
owned device/inode pair. Before invoking a native discard/rollback callback, it first moves every
owned temporary, component and public name into the retained private quarantine; a name replaced by
a third party is preserved and surfaced as an explicit recovery instead of being passed to a
destructive callback. Successful publication also quarantines the private component name.
Finalization closes the retained authority only after the receipt is sealed. This native adapter
remains private and does not weaken the structural Python writer-session extension protocol.

Native discard/rollback callbacks never receive a detached temporary, component-publication or
public-target name. The runtime supplies fresh tombstones inside one authenticated mode-0700
directory, redacts path-bearing metadata, retains that directory only for the callback, and removes
it afterward. Recreating any former public name between detachment and callback therefore cannot
grant the component authority over the replacement.

Every rank-local writer phase, including cancellation-like `BaseException` failures, first emits a
canonical error envelope. All ranks reach the same consensus before cleanup, barriers or failure
re-emission; a rank-local `KeyboardInterrupt`/`SystemExit` cannot split collective control flow.

## Native format contracts

- NPZ is one compressed file containing only selected arrays plus a strict identity-bearing manifest.
  `read_npz()` independently verifies every key, dtype, shape and byte digest.
- HDF5 uses native datasets and `read_hdf5()` verification. Serial/root fields must be complete.
  Collective mode requires the compiled C++ parallel-HDF5 route before preparation; every rank
  writes its declared non-overlapping hyperslabs through `MPI_COMM_WORLD` and the manifest
  authenticates all pieces. Python does not initialize MPI or issue an MPI collective. Partition
  validation scales with piece count rather than global cell count, and shared geometry is written
  once by rank zero.
- ParaView is one native VTK XML UnstructuredGrid (`.vtu`) with inline binary arrays. Selected AMR
  levels retain physical geometry, layout ordinal, valid-box mask, coarse coverage and metric volume;
  cells outside the declared AMR boxes are not emitted. `read_paraview()` parses
  the native XML and authenticates the selected arrays. The currently proved ParaView route is 2D,
  cell-centered data; node/face data and unsupported VTK scalar dtypes are rejected before any
  temporary is created rather than recentered or converted. Supported selected arrays preserve their
  exact dtype; zero padding outside each array's authenticated layout range is structural VTK storage.

Composite reductions multiply by explicit metric volumes and include only valid-box cells that are
not marked as covered on a coarser AMR level. `BalanceTerms` requires storage change, outward boundary flux, sources, reflux and
projection terms. It reports a balance residual and deliberately does not call an open-domain
quantity an invariant. Diagnostic-only outputs remain valid: their owner-qualified diagnostic keys,
terms, layout metadata and provenance are preserved even when no field array is selected. Geometry
origins and spacings use the conventional `(x, y)` and `(dx, dy)` order.

Checkpoint remains a separate restart effect. These consumers do not define a checkpoint schema or
reader and do not call the scientific-output manifest a restart identity. The checkpoint provider
remains the sole owner of sealing, hierarchy/history persistence and strict identity-checked
restore; the consumer transaction only schedules publication after an accepted attempt. Under MPI,
capture reaches consensus on a complete collective-free gather plan before the first native global
accessor, then reaches consensus on the sealed restart identity before rank-zero filesystem I/O.
Final publication is an atomic no-clobber hard link, so a concurrent target creator is preserved.
restart reads/authenticates once on rank zero, broadcasts exact in-memory bytes through the installed
`ExecutionContext`, reaches all-rank preflight consensus before native mutation, and rolls every rank
back if apply or commit diverges. Scientific writers never participate in this protocol.

The manifest-owned MPI acceptance entrypoints are
`tests/python/integration/mpi/test_scientific_output_mpi.py`, which exercises Uniform and AMR through
the final public lifecycle, and `tests/python/integration/io/test_hdf5_parallel.py`, which exercises
the focused writer consensus and failure contracts under the same native MPI world. The AMR witness
proves sparse fine boxes, a replicated coarse authority assigned once to rank zero for collective
I/O, and a participant with no fine-level hyperslab; that empty rank still enters every collective
dataset transfer with `select_none`.
