# M07 prepared physical boundary route

The installed Dim1 M07 build compiled, then `pops.bind()` failed before a step
with `prepared System boundary lacks its complete full/core/residual/JVP
authority` (`outputs/m07-5d11268-dim1-openmp1/run.log`). Its
`CoordinatedFiniteVolume` model is lowered through the generic
`path_conservative_model` state-storage route. That builder supplies
`path_rhs_at_point_prepared`, which fills the prepared halo, invokes the actual
model-qualified physical boundary and evaluates one conservative/path face
operator. It has no legacy spatial residual or boundary JVP. The System
installer previously required all legacy callbacks whenever any physical
boundary was attached, rejecting this complete but different route.

`PreparedSystemBlock` now declares the consumed physical-boundary route.
Legacy spatial packages still require full/core/flux/residual/JVP and external
hooks; path packages require their staged path residual. Unknown route values
and missing path callbacks are refused during block validation. The native
package ABI is 6 and its exact capability contract is version 5, including
the selected route so MPI ranks cannot disagree silently. The path evaluator
still receives the exact `PreparedHyperbolicBoundary` and is called inside the
existing collective preflight and publication transaction. No residual/JVP
stub was added to satisfy the legacy guard.
The distinct staged external `GhostBoundary` component still requires its
legacy full/core closures and is explicitly refused for a path-only block;
this patch does not silently accept that unsupported combination.

Two prepared physical-boundary fills used by staged operators previously
omitted `point.physical_time`, selecting the helper's default `time=0` even for
an analytic inflow at a later stage. They now pass the authenticated point
time. This does not change fixed M07 boundary values. The older no-point
legacy fill retains its initial-time behavior; a general time-dependent
legacy API needs its own authority review.

`SystemInterfaceCoreSession` adds a host test of installation with an
instrumented path callback and an authenticated nonperiodic boundary, plus
negative missing-path, incomplete-legacy and unknown-route cases. This test
checks route admission; the public M07 rerun after rebuilding the native SDK
is still required to prove the actual generated path, physical flux and
hydrostatic lake criteria. Time-dependent analytic inflow also needs a native
two-stage receipt to qualify the staged boundary value. The current installed
native6b and Dim1 f2b packages predate this change and cannot qualify it.
