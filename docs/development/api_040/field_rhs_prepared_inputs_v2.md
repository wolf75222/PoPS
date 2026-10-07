# Field RHS with separately prepared State and Aux inputs

The nonautonomous witness exposed an undeclared Auxiliary in the generated Field RHS.
The previous callback evaluated only `f(State)`; declaring a local name could not provide
arbitrary spatial Auxiliary values, their provenance, or their lifetime. The new common
operation evaluates a scalar density `f(State, ProviderValues, RuntimeParams)` per cell
through C++/Kokkos. Physical State and Auxiliary storage remain distinct inputs.

The input/read contract is version2. `PreparedFieldRhsInputs` is issued privately by
System or AmrSystem and carries the actual Field binding, consumer, source, frame, level,
logical and boundary points, epochs, execution lane, and invocation lifetime. The package
attachment gains a V2 callback, so the native System package ABI becomes8 and the public
Native ABI becomes13. Checkpoint payloads remain Uniform9 and AMR12. All changed headers
require a new header signature, reconstruction and relinking; Native76af/Header773/ABI12
does not qualify this extension.

The frontend selects V2 from the exact nonempty Auxiliary consumer pack. Empty packs keep
the legacy State-only callback and read contract. Explicit C++ V2 with zero provider values
also evaluates the State-only mathematical function through an authentic native request.
No equation, model name, component position, or scientific formula selects a private recipe.

Analytic and Derived providers retain their authored frames, clocks, shapes and dependency
closure. A hidden solved Field in a DerivedAux chain is refused before execution because
this input contract does not carry a FieldContext for that dependency. Pointwise read
effects are certified from exact Auxiliary and parameter registries; unknown expressions
remain uncertified. The previous read-effect classifier retains its default behavior.

`ProgramValueAuthority@1` joins a completed SSA producer to runtime-owned State, scratch
or history storage, its owner, level, slot incarnation, installed immutable Program plan,
and resource attempt. Allocation alone does not publish a value. Write tickets publish
after successful physical evaluation and completion; aliases must be installed edges.
Reset, reuse, regrid, history mutation, rollback and a new attempt revoke prior rights.
The plan and hooks cover the transitive sources of V2 Field nodes. An SSA label or a
matching layout cannot substitute for an actually produced value.

`FieldSolveRequest@2` retains the native source authority and the actual boundary point
through the callback. A missing stage override is refused; only an accepted-only edge
can choose the accepted source automatically. Other AMR levels that use accepted data
retain distinct accepted-source authority. Bootstrap and rematerialization use a native
accepted-source issuer rather than pretending that an SSA stage has executed.

The consuming Field point must equal the active Context point. The producer birth point
may differ according to the installed IR edge. A stage-only scope changes and restores
the stage coordinate while preserving dt, logical intervals and child-clock offsets.
Field lowering uses the actual boundary time, dt, level, tick, substep and fraction.

All patches and input views are prepared before a collective agreement permits kernels.
Candidate finiteness is agreed before publication. Auxiliary candidates needed for
coarse/fine inputs are prepared at the sampling point; publication targets the requested
level. Native owners remain held through normal completion and exception drainage.
A failed fence retains the workspace for recovery. Copies of an input request retain
owners but do not extend invocation rights. In this checkout, Fab and MultiFab copies
are deep copies; retained Kokkos storage views and copied snapshots are distinguished.
Their added allocation, copy and validation costs require comparable measurements.

Current reception before integration build: 56 Source tests pass without loading a Native
module. The independent C++ draft contains13 cases and a real fixture program compiled
and installed through the existing DSO compiler/loader; it has not yet been built or run.
These facts do not receive Native, MPI, GPU, convergence or whole-mission qualification.

Some advanced producers still lack admitted FieldV2 storage hooks: delegated path,
diffusive and principal rates, coupled bundles, retained/spatial solve storage and
remapped loop carriers. They currently refuse a V2 closure; this is unfinished production
coverage, not a permanent language or backend limit. Their actual producer hooks and
independent reception remain mandatory for the full mission.

The next required run is the unchanged public nonautonomous witness after an official
incremental `scripts/build_python.sh --dim 2 --mpi` build in the `pops` environment.
Use `env -u PYTHONPATH` and the installed-package identity driver
`docs/development/api_040/run_installed_checks.py`, selecting
`tests/python/integration/runtime/test_imex_nonautonomous_field.py` in a new output directory.
Its fixed references are `Y=48/25`, `F=144/25`, `U_next=84/25`; evaluating the Field at
the wrong partition gives `87/25`. The test now retains the actual compiler input files.
Only after this run and its independent receipt should the own-checkpoint Retry,
affected non-regression, MPI2 and CUDA variants follow.

The first integration build exposed a stale module-capability ABI constant. Commit
`3ee0180808cfca7c2a93833ce792d83357aff81a` synchronizes it with Native13; the strict
assertion remains in place. The official incremental build then compiled and linked
seven bindings. The installed Dim2 extension has SHA256
`ea0c7bf4e2a752469091030597cedd671291fc74bd9354ab8fbcdb6068eeec1d` and SDK signature
`4b2c57fbdc84b72f8b0c66c7c943eb2f793d519181d7f9032f02313032e8f04a`.

The actual nonautonomous run at `37ec76b04557f50de08683f8545fa7efd28c9bd7` compiled,
bound, accepted one step and wrote a checkpoint, but failed its unchanged numerical
oracles: every accepted State value was `3.374999999999498` instead of `3.36`, and
every Field value was `5.999999999997994` instead of `5.76`. The 25 actual C++, DSO,
array and checkpoint artifacts were retained and independently checked. The intermediate
`Y` was not measured. The Root negative receipt has SHA256
`94dfccb597bde138f918f93ea914294264126399602600e271fd525d36a3992e`.
This receives an executed local CPU/MPI-enabled world1 failure, with no MPI2 or GPU
qualification. The installed package's 2793 entries remained identical in bytes and modes.

The retained Program C++ selects the implicit point before the local linear solve,
then reads its Auxiliary view without preparing that consumer. The library correction
prepares AMR prerequisites through the existing collective Context entry point before
the Fab loop, at the exact RHS SSA state and evaluation identity. Uniform retains its
existing solve-outcome route and authored failure action. Matrix assembly, finite and
singular checks, equations, tableau and numeric bounds are unchanged. This repairs
emission under the existing version2 contract and does not change an ABI.

The correction passes 74 targeted Source tests without Native loading, including distinct
implicit/explicit coordinates, renamed and spatial Auxiliaries, empty provider packs,
solve actions, ARK stages and the Field/SSA cohort. A controlled exact-HEAD baseline
reproduces the three AMR preparation omissions. A separate split-coordinate regression
also fails before emission on that baseline and remains open; this cohort does not
qualify the full split suite. Corrected numerical execution is still pending.
