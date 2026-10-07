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

Historical reception before integration build: 56 Source tests pass without loading a Native
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

The corrected tranche at `ede680fe00746c1bc7e5eeb4afabd9772b92cc33` is now received:
validate, resolve, actual C++ compilation, binding, an accepted step and checkpoint all
pass. All 64 State values are `3.359999999999518` and all 64 Field values are
`5.759999999998074`. Maximum absolute errors are respectively
`4.818367926873179e-13` and `1.9255708139098715e-12`, below the original fixed bounds.
The checkpoint arrays equal the saved NPY arrays exactly. Its native clock is0.25,
macro step1, one level and one rank. The authored physical origin1 is distinct from
that measured native clock. `Y=48/25` remains an independent reference, not a captured
intermediate value.

The actually compiled Program differs from the failed Program by one preparation call
before the provider view; the actual Field model compiler input is identical. Ninja
reported no work for the unchanged C++ core. The new wheel has SHA256
`b961309157ad80a05fc022a249e8f7125185e6754eb0419e2fa62d737371cf30` and contains
the corrected Python emitter with the same Native13 extension. Independent offline
reception verifies the numeric references, 26 retained artifacts, package joins and
checkpoint. ROOT positive receipt SHA256 is
`03e911a86bb8909d0eff29eb3c39cb2c3e6698063b9176654e1344ae733bb37c`.
This receives one CPU Kokkos Dim2/MPI-enabled world1, one-level witness. MPI2, GPU,
Dim3, populated-history lifecycle, convergence and comparable costs remain separate work.

To reproduce this tranche, use the initialized `pops` environment from the integrated
checkout, with a fresh output directory. The actual build and test commands are:

```sh
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 POPS_ENV_NAME=pops Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops CMAKE_PREFIX_PATH=/Users/romaindespoulain/miniforge3/envs/pops POPS_NATIVE_DIM=2 POPS_HEAVY_MODULE_TU_POOL=2 CMAKE_BUILD_PARALLEL_LEVEL=4 OMP_NUM_THREADS=2 OMP_PROC_BIND=false bash scripts/build_python.sh --dim 2 --mpi --wheel-dir /tmp/pops-api040-nonautonomous-wheel
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 POPS_INCLUDE=/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004/include Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops CMAKE_PREFIX_PATH=/Users/romaindespoulain/miniforge3/envs/pops POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 OMP_NUM_THREADS=2 OMP_PROC_BIND=false /Users/romaindespoulain/miniforge3/envs/pops/bin/python docs/development/api_040/run_installed_checks.py --output /tmp/pops-api040-nonautonomous-reception --test tests/python/integration/runtime/test_imex_nonautonomous_field.py
```

The installed-package driver checks the real package origin, extension and shipped source
fingerprints before pytest. It keeps the actual C++, arrays, checkpoint, identity and
test result in the output directory. An existing output directory must be preserved;
choose another directory for a new run. Source tests and successful compilation do not
replace this numeric receipt.

## Added C++ witness specification v2

The first world1 run executed all13 newly added unit cases but refused their setup:
the Uniform direct closure image lacked authoritative block metadata, and the AMR
Program was installed after hierarchy materialization. No Field guard coverage was
received from those failures. The corrected unit fixture uses the public compiled-block
facade, with a model-owned ADL preparer delegating to the real generated storage factory.
It installs the genuine Program before AMR bootstrap, completes bootstrap before the
first step and exports its actual primary clock. Its fixture identity becomes
`tests.field-rhs-value-program/owned-lincomb-v2`; the fixture DSO must be rebuilt.

The earlier Uniform periodic montage also had an inconsistent pure Poisson source:
`rho=u*u+a+b`, `u=2+cos(2*pi*x)`, `a=sin(2*pi*x)+3*t`, `b=2*a+5/4` has mean
`23/4+9*t`. The v2 unit specification explicitly chooses homogeneous Dirichlet faces
and nonperiodic geometry. The source formulas,13 case assertions, finite/ownership
guards and solver tolerances are preserved. No reaction, neutralizing background,
mean removal or silent projection is introduced. The AMR unit witness retains its
periodic screened Poisson equation with reaction one; original scientific examples
retain their authored physical specifications. This changes an added test specification,
not a production ABI, language contract or solver compatibility guard.

The source patch is independently reviewed and the 115 assertion expressions are
byte-identical. Its corrected build, runtime results and backend coverage remain
pending; the earlier13 failures and their actual XML/binary identities are retained.

The first v2 world1 run still refused all13 admission paths. Uniform faces require
the public residual arguments `(alpha,beta,value)=(1,0,0)` for homogeneous Dirichlet;
the earlier all-zero vectors did not express that boundary. The corrected fixture
passes these arguments in the actual API order. The two-level AMR fixture also
declares its2:1 parent/child temporal relation with `integral_only` before materialization,
as required by the frozen hierarchy/checkpoint-capacity authority. Neither boundary
validation nor capacity validation is removed. The three added lines and corrected
argument preserve all115 assertion expressions. This Source review receives no new
runtime success; the earlier v2 binary and13 failures remain preserved.

The f1b4 world1 run reaches five guards successfully, with eight remaining failures.
Four are corrected in the unit setup: V2 requires a nonempty consumer identity even
when its pack has zero values; a foreign owner uses its declared installed Clock;
and a copied Field has separate storage. The previous pointer-equality assertion is
replaced by inequality plus shape and exact valid-cell value checks, while the refusal
to use that copy as an SSA source remains. The test ledger retains114 previous assertion
expressions, replaces one and adds five:120 expressions in the same13 cases.

Three remaining failures establish a Core event mismatch: the Field RHS preparation
used `before_residual` although its providers declared `before_field_solve`. The fourth
is an AMR physical-clock error consistent with the unqualified topology-rematerialization
point in Source; its exact runtime call stack is not captured. Core fixes, default
Derived policy alignment and independent MPI admission review precede the next build.
These failures and their real binaries/XML remain retained; no np2 result is claimed.

## Field event and accepted physical point correction

The Core correction uses a source-private consumer preparation seam. It retains
the same source/layout validation and collective agreement before publication,
with the actual execution event included in the private protocol@1. Numerical
residuals remain `before_residual`; Field RHS uses `before_field_solve`, including
native accepted initial points. The existing input/read2 and public ABI13/package8
contracts, SDK headers and checkpoint formats are unchanged. Core objects and native
libraries must be recompiled and relinked before receiving this implementation.

AMR topology rematerialization distinguishes diagnostic topology labels from
physical Clocks. V2 Field closures use the actual installed primary Clock, or an
exact registered accepted-halo/request Clock, with runtime-owned step/time/dt,
session, epochs, metadata, Field bindings, actual State carriers and bitwise RuntimeParams
validation. Parameter mutation is refused only while this V2 accepted borrow is live.
Initial qualification retains the existing native zero-interval guard. V1 and
unrelated opaque legacy refresh retain their existing path and receive no invented
physical-point authority. The internal rematerialization request is version2.

All output ComponentKeys enter accepted-halo dependencies, Field ordering and the
V2 physical closure. A public potential-plus-gradient Source composition has three
distinct provider identities; its non-first dependency exposed the previous map
error. Setup, point copies, candidate bookkeeping, witness allocation and DAG
serialization now agree collectively before the next collective operation.

The generated default Derived/Analytic policy uses the existing vector policy@2
for `before_residual` and `before_field_solve`, with evaluation freshness. Input and
Field-output initialization/once policies and explicitly authored policies retain
their semantics. A six-event candidate was discarded: unrelated temporal expressions
could become due on the unqualified legacy topology tail. V2 rematerialization already
forces its exact dirty closure; broad default permission for `after_regrid` is unnecessary.

The coherent Source reception is Core R4 review SHA
`c6a294e82c386d56e001602350c73bcd12868e8dfe9fed8a0e70e558a25e49aa`
and policy R2 review SHA
`75927544c21541571ab532968d3d0e00a08ff4ed59602e83eaa8c45a30619d6d`.
Fifty policy tests pass in Main without Native loading. No corrected Native runtime,
MPI2 or GPU result is implied. The earlier AMR failure lacks a captured stack; this
Source correction addresses a demonstrated invalid-point path, pending actual execution.

## Core R4 installed calculation and multilevel counterexample

The official incremental build at `c0f18d50efd454396d6761992a9817adf353f412`
recompiles three Core objects and relinks Native. Seven binding objects are reused.
Installed Native SHA is `b471e01b7d8e669e3d7c11f5cb8925a509edf2a17d9993a5211dc3bc4080f595`;
the header signature remains `4b2c57fbdc84b72f8b0c66c7c943eb2f793d519181d7f9032f02313032e8f04a`.
The unchanged public nonautonomous test passes once, with no skip, from the real
installed package. U and Field meet the original bounds; checkpoint arrays are exact.
Y remains a reference, and effective OpenMP concurrency is not measured in this run.
The [Root receipt](/Users/romaindespoulain/dev/tmp/root-field-rhs-native13-core-r4-nonautonomous-positive-20261007.json)
joins 34 raw artifacts and independent reception. Exact reproduction commands are
retained in [build-command.txt](/Users/romaindespoulain/dev/tmp/PoPS-api040-field-rhs-v2-B-20261007-c0f18d50efd4/build-command.txt)
and [first-command.txt](/Users/romaindespoulain/dev/tmp/PoPS-api040-field-rhs-v2-B-20261007-c0f18d50efd4/first-command.txt).

The corresponding C++ world1 tranche reaches 11 PASS, 2 FAIL and zero skips in
13 cases. The lane expectation now captures an owning string from the live public
System getter; expiry, lane size and release checks remain. This test-only correction
has no new execution receipt yet.

A [separate coherent AMR diagnostic](/Users/romaindespoulain/dev/tmp/root-amr-aux-clock-coherent-diagnostic-reception-20261007.json)
observes three physical `before_field_solve` Analytic calls at levels 0, 0 and 1,
then `after_regrid` at level 0 with the topology diagnostic Clock and no physical
payload. The original guard throws during bootstrap commit. No full native stack
is captured, and the test's mark-bound/step/solve phases are not reached.
All 54 raw artifacts, compilation commands, existing capability values and target
link command are authenticated. Eight extra HDF5 discovery keys are retained;
18 compilation steps are observed. The fixture is regenerated with equal bytes.
The temporary test source is restored exactly and all 2793 package entries remain
unchanged. Prior configuration-drift and pre-run gate attempts retain separate receipts.

The integrated Source correction tracks invalidations and actual accepted publications by
provider and live level, including rollback and the existing POPSAUX3 per-level
metadata. It also gives invalidation cleanup its own registry selection purpose.
The [versioned correction](amr_auxiliary_invalidation_incidence_v1.md) has independent
Source reception; it is not installed or natively qualified.
Its SDK header change requires a new signature and genuine reconstruction before
Cpp13, registry counter-tests, the original retry and MPI2 can receive new results.
