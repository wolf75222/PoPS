# Independent review: M07 prepared physical boundary admission

Baseline examined: the M07 Dim1 installed failure at
`outputs/m07-5d11268-dim1-openmp1/run.log`, and the matching MAIN source in
`system_install.cpp`, `generated_system_block.hpp`, `system_program.cpp`,
`system_block_closures.hpp`, `native_package_capability.hpp` and
`prepared_hyperbolic_boundary.hpp`. This review does not change the case's BC.

The failure is concrete: `materialize_state_block` supplies the real prepared
Path residual, including physical ghost fill, but leaves the legacy full/core,
boundary residual and JVP closures absent. `prepare_block_installation` demanded
all eleven legacy hooks whenever a physical boundary existed. A path residual is
not a boundary linearization and must not publish placeholder callbacks to pass.

The proposed typed `PreparedPhysicalBoundaryRoute` discriminator is appropriate
if its value is authenticated collectively, unknown values fail, the legacy
branch still requires the entire old closure family, and the path branch requires
its actual prepared path callback. A non-Path Program-only storage carrier still
has no physical boundary consumer and must remain refused until implemented.

Existing safeguards independently checked:

- Exact state identity, component count, domain periodicity and sufficient ghost
  depth are checked before candidate publication. Primitive fixed boundary values
  pass through the selected model's real recovery/conversion.
- Path execution compares the System, block, complete evaluation point, stage,
  physical time and exact lane. It rejects input/result aliasing and incompatible
  field contracts. Interfaces and active embedded boundaries remain refused.
- The residual operates on detached state/candidate scratch. Collective preflight
  and publication transaction remain responsible for failure/rollback; no pointer
  to the user's published field is substituted for that candidate.
- Boundary residual/JVP consumers independently require their actual closures.
  Path admission must not make `has_block_boundary_linearization` true.
- External GhostBoundary installation already requires all four compiled
  full/core/flux closures. Therefore Path cannot silently install an unused
  external ghost hook. An initial concern based only on its non-null target was
  withdrawn after reading this preceding guard. External flux/field hooks retain
  their own admission checks.

A separate **real stage-time defect** was found during review: both the Path fill
and the ordinary prepared fill call `fill_physical_model_qualified` without its
last argument, which defaults to zero. The complete point has authenticated
`physical_time`, but an authored analytic inflow would receive t=0 at every stage.
This is invisible to M07's fixed boundary. The author was asked to pass the point's
physical time at both sites and test a nonconstant analytic boundary at two times.

Required discriminating acceptance witnesses: fixed physical BC with a genuine
Path callback accepted; removing that callback rejected; incomplete legacy route
rejected; unknown discriminator rejected; different route encoded in exact package
identity; analytic inflow evaluated at two distinct stage times; failed domain or
foreign point/lane leaves published state/residual unchanged. The author's native
suite and the installed M07 case provide execution qualification; this independent
review itself has not run a native build, JIT, MPI or a scientific trajectory.

Source review of the author diff before the performance freeze is favorable:
`PreparedPhysicalBoundaryRoute` defaults to legacy, Path generation selects
`path_residual`, unknown values and missing Path callbacks are rejected, and the
legacy callback conjunction is preserved. The exact package contract encodes the
route (contract version 5); native package ABI 6 and Python emitter ABI 6 agree.
An independent AST/header source check confirmed that alignment. The two prepared
physical fills now receive `static_cast<Real>(point.physical_time)`.

The new author `SystemInterfaceCoreSession` tests distinguish Path admission from
legacy linearization and test missing Path, incomplete legacy and unknown routes.
Their positive Path callback is instrumented; these are admission tests, not
execution of the real generated Path builder. They were read, not run by this
reviewer. No performance-window CPU tests were started. Final author SHA and native
M07/stage-time evidence remain integration receipt items; this source verdict must
not be reported as native acceptance.
