# Exact scientific output consumers

PoPS scientific output has one resolved boundary. The consumer graph chooses an accepted state and
produces an `OutputRequest`; format writers never search the runtime for a convenient first block or
silently widen a selection. Every `FieldKey` contains the canonical declaration `Handle`, component
manifest identity, layout identity, level and state identity. `OutputSnapshot` adds the exact clock,
plan/bind/run provenance, centering, units, AMR boxes, coverage mask and metric cell volumes.

The final public formats are typed:

```python
pops.output.NPZ()
pops.output.HDF5(parallel=False)
pops.output.HDF5(parallel=True)
pops.output.ParaView()
```

Each implements `FormatInterface.writer()`. The runtime-facing bridge is
`pops.runtime.output_publisher.ConsumerOutputPublisher.prepare(effect)`: ADC-685 resolves an accepted side effect to an
`OutputPreparation`, while the writer returns a verified `PreparedOutputFile`. The bridge wraps it
in `PreparedConsumerOutput`, a nominal ADC-685 `PreparedPublication` carrying the accepted effect and
payload identities. Preparation writes and reopens only a temporary file. The accepted-side-effect
transaction later calls `publish()` for an exact `PublicationReceipt` and an
atomic no-clobber hard-link publication or `discard()` to remove it. Deterministic filenames include
the consumer, clock, macro-step and exact selection identity; different bytes targeting the same name
are a collision, never an overwrite.

## Native format contracts

- NPZ is one compressed file containing only selected arrays plus a strict identity-bearing manifest.
  `read_npz()` independently verifies every key, dtype, shape and byte digest.
- HDF5 uses native datasets and `read_hdf5()` verification. Serial fields must be complete. Collective
  mode requires a resolved communicator and an MPI-enabled h5py build before preparation; every rank
  writes its declared non-overlapping hyperslabs and the manifest authenticates all pieces. Partition
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
reader and do not call the scientific-output manifest a restart identity. ADC-678 remains the sole
owner of checkpoint sealing, hierarchy/history persistence and strict identity-checked restore;
ADC-685 only schedules its publication after an accepted transaction.
