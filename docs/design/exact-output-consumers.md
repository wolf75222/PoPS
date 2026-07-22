# Exact scientific output consumers

PoPS scientific output has one resolved boundary. The consumer graph chooses an accepted state and
produces an `OutputRequest`; format writers never search the runtime for a convenient first block or
silently widen a selection. Every `FieldKey` contains the canonical declaration `Handle`, component
manifest identity, layout identity, level and state identity. `OutputSnapshot` adds the exact clock,
plan/bind/run provenance, centering, AMR boxes, coverage mask and metric cell volumes. Every native
piece retains its `global_box_index`, `owner_rank` and explicit `replicated` bit in the snapshot and
format manifest; distributed validation proves the exact indexed box union, not only an equivalent
rectangle union. The current public contract is deliberately unitless: it neither invents nor
persists an opaque unit vocabulary.

The final public formats are typed:

```python
from pops.output import (
    HDF5,
    NPZ,
    ParaView,
    ParaViewPreset,
    ParallelMode,
    SharedDirectory,
)

NPZ(mode=ParallelMode.SERIAL)
HDF5(mode=ParallelMode.ROOT)
HDF5(mode=ParallelMode.COLLECTIVE)
HDF5(mode=ParallelMode.PER_RANK)
ParaView(
    mode=ParallelMode.ROOT,
    compression=6,
    collection=True,
    preset=ParaViewPreset(
        color_by="temperature",
        color_map="Viridis",
        representation="Surface With Edges",
    ),
)
ParaView(mode=ParallelMode.PER_RANK)  # bounded MPI relay to rank 0 by default
ParaView(mode=ParallelMode.PER_RANK, placement=SharedDirectory())
```

La cible d'un `ScientificOutput` est un nom logique, jamais un nom de fichier :

```python
OUTPUT_FORMAT = ParaView()  # remplacer seulement ParaView par HDF5 ou NPZ

ScientificOutput(
    format=OUTPUT_FORMAT,
    schedule=every(100, clock=program.clock),
    fields=(tracer_U,),
    target="solution/tracer",
)
```

Le fournisseur possède l'extension. Une cible comme `solution/tracer.vtu` est refusée dès
l'authoring, avant le bind ; elle empêcherait le changement de format et entrerait en collision au
deuxième échantillon. Chaque pas accepté dû publie immédiatement un fichier distinct sous le chemin
logique. Pour NPZ et HDF5, une capability structurelle distincte du writer entretient atomiquement
un catalogue `series__f<identité-de-famille><extension>.series`. Cette identité couvre le provider,
la sélection complète et le run ; deux timelines déposées sous le même chemin restent séparées.
`format.reopen(path)` authentifie un fichier et `format.reopen_series(path)` valide le catalogue,
ses chemins, ses temps et leur ordre sans charger tous les champs historiques. `series.latest`
authentifie le dernier fichier ; `series.verify()` rouvre toute la série à la demande, un membre à
la fois. Le répertoire logique suffit tant qu'il ne contient qu'une timeline. Après plusieurs runs
ou familles, cette recherche devient volontairement ambiguë : l'appelant transmet alors le chemin
exact `series__f….<extension>.series` affiché lors du run.

ParaView n'ajoute pas un `.vtu.series` propriétaire en doublon. Avec `collection=True`, sa série
canonique est le dernier catalogue `.pvd` standard ; `ParaView.reopen_series(path_du_pvd)`
authentifie le PVD et toutes ses références VTU ou PVTU. Les catalogues génériques sont activés pour
NPZ en `SERIAL`/`ROOT` et HDF5 en `SERIAL`/`ROOT`/`COLLECTIVE`. Ils sont désactivés en `PER_RANK`,
où le format doit fournir une vraie collection parallèle au lieu de présenter plusieurs morceaux
de rang comme plusieurs instants.

Un provider extensible peut exposer la même petite interface `ScientificSeriesCatalog`
(`catalog_data`, `publish`, `reopen`). Le publisher runtime orchestre cette capability sans branche
sur HDF5, NPZ ou VTK et conserve l'autorité de l'artefact tant que le catalogue n'est pas publié.

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
  ParaView then publishes one standard `.pvtu` index referencing all rank-local `.vtu` leaves.

`NPZ` supports `SERIAL`, `ROOT` and `PER_RANK`; per-rank NPZ stores each local piece separately with
its global half-open bounds. `HDF5` supports all four modes. ParaView supports `SERIAL`, `ROOT` and
rank-qualified local VTU files plus a standard PVTU index in `PER_RANK`; the native external Writer
supports `SERIAL`. Unsupported combinations are refused by the format descriptor, not degraded
later.

Format topology and storage topology are separate. ParaView `PER_RANK` defaults to
`MpiRelayToRoot()`: every rank stages its local VTU, sends bounded byte chunks over the active writer
communicator, and rank 0 authenticates and publishes a complete colocated VTU/PVTU/PVD bundle. The
reader-visible output directory therefore does not have to be shared between compute nodes.
`SharedDirectory()` is the explicit alternative: ranks publish leaves directly and the user asserts
that the same directory and basenames are visible to rank 0 and to ParaView. PoPS does not guess this
filesystem property from paths or mount names.
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

## Physical cadence and public failure actions

`every(n, clock=...)` counts accepted macro-steps. `every_dt(interval, clock=...)` instead owns the
absolute physical-time lattice `interval, 2*interval, ...`. The runtime caps a proposed macro-step
at the next active physical deadline before it prepares an output, so an adaptive CFL step does not
move the sample away from the requested time. Rejected attempts never count as samples. Restart
re-derives the next lattice index from the accepted clock and the persisted consumer cursor prevents
republication of the restored boundary. An `ExternalTimeGrid` must explicitly contain every active
`every_dt` deadline and may land at a binary64-equivalent value only on or after the deadline.

```python
from pops.output import ParaView, Retry, ScientificOutput
from pops.time import every_dt

output = ScientificOutput(
    format=ParaView(),
    schedule=every_dt(0.1, clock=program.clock),
    fields=(fluid.rho, fluid.velocity),
    target="solution",
    failure_action=Retry(max_attempts=3),
)
```

The closed pre-commit file-publication actions are `FailRun()` (the default), `Retry(n)` and
`SkipSampleReported()`. A retry repeats only the compensatable output transaction, never the
accepted numerical step. A skipped sample is recorded in the runtime consumer reports rather than
being silently discarded.

## Temporal catalogues, presentation state and live delivery

With `collection=True`, every accepted ParaView sample publishes a cumulative standard `.pvd`
collection. PoPS versions these catalogues immutably: the newest `.pvd` lists every strictly
increasing physical time published so far, while an older catalogue remains reproducible. Reopen
the newest file to refresh a file-based view; use Catalyst when one already-open visualization must
receive frames while the calculation continues.

With a temporal collection, the default `PortableState()` emits an authenticated JSON recipe and a
small Python driver next to the PVD. Both refer to the PVD by basename, so moving the complete output
directory preserves the bundle. Building this recipe imports no ParaView module and requires no
ParaView installation on the simulation host. The driver can later be run by a real `pvpython`, or
passed through `materialize_paraview_state(...)`, to apply the preset at the last data time.

`MaterializedPVSM(...)` is the explicit stronger request. It selects a real `pvpython`, opens the
portable recipe, applies `ParaViewPreset`, calls `SaveState`, validates the resulting server-manager
XML, and reloads it with `LoadState`. PoPS fails closed if `pvpython`, the requested field/component
or the color preset is unavailable; it never fabricates version-specific PVSM XML.

```python
from pops.output import MaterializedPVSM, PortableState

portable = ParaView(
    collection=True,
    preset=ParaViewPreset(
        color_by="rho",
        representation="Surface With Edges",
        color_map="Viridis",
    ),
    state=PortableState(),
)

with_pvsm = ParaView(
    collection=True,
    preset=ParaViewPreset(color_by="rho"),
    state=MaterializedPVSM(pvpython="/opt/paraview/bin/pvpython"),
)
```

For bounded in-process overlap, `AsyncScientificOutput` runs a complete writer session on a dedicated
non-daemon worker after the numerical step has committed. It returns real writer receipts and can
therefore write NPZ, HDF5 or the complete VTU/PVTU/PVD/state ParaView bundle. `queue_capacity` bounds
retained detached snapshots; a full queue deliberately applies backpressure.

The selected format owns the topology. `SERIAL` uses the sole rank. `ROOT` performs the complete
snapshot gather on the main execution path, then writes from the rank-zero worker without worker
MPI. `PER_RANK` and `COLLECTIVE` run one worker per rank over a run-scoped communicator duplicated
collectively before any worker starts. That private lane has a distinct MPI context from
`MPI_COMM_WORLD`, so numerical and output collective orderings cannot alias. PoPS requires
`MPI_THREAD_MULTIPLE`, authenticates the lane on every worker call and fixes distributed
`max_attempts` to one: retrying after entry into an MPI publication would not be safe. Supported mode
combinations remain those of the format itself; in particular, ParaView has no `COLLECTIVE` mode and
HDF5 owns the collective writer. Within one `RuntimeInstance` run, every post-commit session shares
one process-local FIFO worker. Initialization, frame execution and finalization therefore enter
process-global HDF5/Catalyst state in the same authored order on every rank; synchronous HDF5 drains
that FIFO before entering its own writer.

```python
from pops.output import AsyncScientificOutput, DurableJournal, RaiseOnFlush

async_output = AsyncScientificOutput(
    format=ParaView(collection=True),
    schedule=every_dt(0.1, clock=program.clock),
    fields=(fluid.rho,),
    target="async-solution",
    queue_capacity=2,
    max_attempts=2,
    on_failure=RaiseOnFlush(),
    durability=DurableJournal("results/observer-journal"),
)
```

The bounded in-memory queue alone is process-lifetime state. The versioned `DurableJournal` primitive
adds a different post-commit handoff: it stores the complete detached `ObserverFrame` without pickle,
authenticates its arrays and identities, and moves it through durable `prepared`, `pending` and
`delivered` links. Recovery drops an uncommitted preparation, replays committed pending frames, and
retains delivered evidence. The **at-least-once guarantee begins only after the `pending` handoff has
been durably synced**. A crash after the numerical transaction commits but before that handoff may
still lose the observer frame. After handoff, delivery is not exactly once: a process can crash after
the writer or Catalyst accepts a frame but before the `delivered` link is synced, so recovery may
attempt that frame again.

The journal is also not a numerical checkpoint and does not make the accepted step, checkpoint,
scientific file and external visualization one crash-atomic transaction. It stores observer delivery
work only; restart authority remains with `Checkpoint`. Delivered records retain their complete
authenticated frame as replay evidence and are not automatically pruned. Long-running applications
must therefore budget and manage journal storage outside an active run after retaining the evidence
they need. Both `AsyncScientificOutput` and `LiveVisualization` accept the explicit
`durability=DurableJournal(...)` policy; omitting it keeps the lighter process-lifetime queue.

MPI journals are scoped by consumer and rank and are bound to the resolved delivery target. They do
not require a shared filesystem, but every rank-local root must survive and be presented to the same
logical rank during recovery; node-local scratch is not transparently relocated after a scheduler
remap. Collective recovery authenticates the same temporal event sequence on every rank and can
replay pending frames created under a prior run identity. Replay retains the same authenticated
manifest/consumer and delivery authority. Changing the manifest or its `target_uri` creates a
different identity-scoped journal. Changing only the resolved `output_root` while retaining the same
manifest reaches the existing journal, but its persisted `delivery.json` rejects the different
resolved target authority. Neither case migrates old pending records automatically.

`LiveVisualization` uses the same bounded post-commit boundary with a real Catalyst 2 / Conduit
Blueprint provider. The pipeline path and SHA-256 are frozen into the declaration. `Catalyst` also
freezes an explicit implementation name (`"paraview"` by default), resolved implementation search
directories and script arguments. Initialization sets `catalyst_load/implementation`, optionally
sets `catalyst_load/search_paths`, and forwards `args` through the standard
`catalyst/scripts/pops/args` Blueprint path; a ParaView pipeline can read them with
`paraview.catalyst.get_args()`.

Catalyst and Conduit are imported lazily but the session is opened before numerical advancement, so
missing or incompatible optional dependencies fail before the first step. A successful Catalyst stub
is not accepted as visualization: immediately after `initialize`, PoPS calls `about()`, requires the
reported implementation to match the requested name and a non-empty Catalyst API version, then
retains that implementation/API-version evidence in the delivery receipt as `implementation` and
`catalyst_api_version`; `catalyst/version` is not presented as an implementation version.

The declaration remains authoritative over the loader. PoPS rejects any non-empty
`CATALYST_IMPLEMENTATION_PREFER_ENV`. The built-in provider and `LiveVisualization` currently
support `SERIAL` only. `ROOT`, `PER_RANK` and `COLLECTIVE` are rejected during authoring, and binding
a serial declaration to an MPI execution context is rejected before the first numerical step. PoPS
therefore never passes `catalyst/mpi_comm` and makes no distributed-live claim. MPI simulations use
progressive `AsyncScientificOutput` PVTU/HDF5 artifacts instead.

PoPS owns the only asynchronous layer: it forces `catalyst/async/enabled=0` and refuses an active
inherited `CATALYST_ASYNC_ENABLED`. A delivery receipt therefore follows a completed
`catalyst.execute`. The built-in Catalyst provider admits one consumer/pipeline and one simulation
run per `RuntimeInstance`; multiple pipeline actions must be combined in that script or behind one
custom multiplexing provider. Its one-shot process-global lifecycle reservation is never released,
so even a later `RuntimeInstance` must start in a fresh OS process for another built-in Catalyst
simulation. Distinct concurrent runtimes in one process are outside the contract when asynchronous
HDF5 or Catalyst is present because their run-local FIFOs cannot impose a common order on
process-global library state.

```python
from pathlib import Path

from pops.output import Catalyst, LiveVisualization, ParallelMode, ReportOnly

pipeline = Path("docs/tuto/scalar_advection/catalyst_pipeline.py").resolve()
live = LiveVisualization(
    observer=Catalyst(
        pipeline=str(pipeline),
        implementation="paraview",
        search_paths=(),  # use Catalyst loader defaults, or list implementation directories
        args=("--view=surface",),
    ),
    schedule=every_dt(0.1, clock=program.clock),
    fields=(tracer_U,),
    mode=ParallelMode.SERIAL,
    queue_capacity=2,
    on_failure=ReportOnly(),
)
```

Because live delivery is irreversible, its policies are separate: `RaiseOnFlush()` reports after
draining at the run boundary, while `ReportOnly()` leaves terminal evidence in
`runtime.post_commit_reports` and `runtime.post_commit_diagnostics` without pretending to roll back
the accepted state.

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
  writes its declared non-overlapping hyperslabs through the exact authenticated communicator and
  the manifest authenticates all pieces. A synchronous consumer uses the execution communicator;
  an asynchronous consumer uses its private duplicated worker lane. Python never emulates this
  mode with a gather-to-root writer: the compiled provider owns the MPIO dataset transfers.
  Partition validation scales with piece count rather than global cell count, and shared geometry
  is written once by rank zero. Unlike the default relayed PVTU topology, the single collective HDF5
  target is opened by every rank through parallel HDF5/MPI-IO and must therefore be genuinely
  accessible collectively (normally on a shared parallel filesystem).
- ParaView writes native VTK XML UnstructuredGrid (`.vtu`) leaves with shared mesh vertices and
  standard inline zlib-compressed binary blocks. Array names come from the explicit declaration
  string, for example `"U"` in `model.state("U", ...)`, never from the Python assignment name on the
  left-hand side. Homonymous declarations are block-qualified (then identity-qualified if still
  ambiguous), and declared component names are preserved. Selected AMR levels retain physical
  geometry, layout ordinal, valid-box mask, coarse coverage, VTK ghost flags and metric volume;
  cells outside declared boxes are not emitted.
  Cartesian and polar-annulus coordinates map to physical Cartesian VTK points. `read_paraview()`,
  `read_paraview_parallel()` and `read_paraview_series()` independently parse and authenticate VTU,
  PVTU and PVD artifacts. Every VTU remains self-describing through `TimeValue`; the cumulative PVD
  is the canonical temporal series. The format layer proves Cartesian meshes of spatial rank one,
  two or three with VTK `LINE`, `QUAD` or `HEXAHEDRON` cells, plus the two-dimensional polar-annulus
  mapping. Cell-centered arrays use `CellData`; nodal arrays use `PointData`/`PPointData` without
  recentering when `state=None`. Portable/PVSM presentation state is currently authenticated for
  `CellData` only, so requesting state generation for nodal output fails explicitly. Face-centered
  arrays still require a distinct multi-topology face mesh and are rejected rather than silently
  recentered. Unknown coordinates and unsupported VTK scalar dtypes are likewise rejected before
  publication. Supported selected arrays preserve their exact dtype.

These distribution and presentation features do not widen the native discretization contract. The
current native runtime capture and Catalyst blueprint path remain two-dimensional and cell-centered.
Uniform and AMR hierarchies, including sparse multilevel boxes, are supported inside that native
boundary. The dimension-generic VTU layer can also consume an exact externally constructed 1D/3D or
nodal `OutputSnapshot`; this is a format capability, not a claim that the PoPS solver already owns a
native 1D/3D or nodal state path. Face-centered visualization is rejected rather than silently
projected or recentered.

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
