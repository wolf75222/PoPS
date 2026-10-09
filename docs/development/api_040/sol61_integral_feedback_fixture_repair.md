# Integral feedback native fixture repair, 2026-09-30

Base: `634cba3511fef957e65a21fcae3ab257f164b360`. Only the C++ test fixture
and this report change. Production APIs, headers, equations, tolerances, shared
SDK/environment, MAIN, and JIT are untouched.

The central serial/MPI2 failures are in
`outputs/native-integrated-ir8-cpp-wave4b-discovery-dim1-20260930/ctest.log`:
the fixture selected nonperiodic x topology but installed the periodic-only
`add_scalar` helper. The genuine prepared transport refused it with
`prepared transport faces without a physical boundary require periodic topology`.
The later checkpoint restore also correctly refused a missing native restart
transaction. These are fixture setup defects; guards should remain strict.

The fixture now installs a real scalar State route, then prepares the native
hyperbolic boundary before committing the block. Both x faces use `foextrap`,
the parser's `HyperbolicBoundaryLaw::Extrapolate` token implementing the
first-order Outflow trace. Other dimensions retain complete periodic pairs.
The registry carries exact State/face identities and the scalar component role.
The existing generated ScalarAdvection authority remains `none` limiter,
`rusanov` flux, `conservative`, and `explicit`; no numerical implementation is
replaced. This is the same installation order used by `add_boundary_gas` in
the genuine ProgramRuntime fixture.

The existing values and scientific assertions remain: initial U=2, q=.7,
gamma=.3, dt=.01; S=-gamma*U and q*S; true numerical right exterior face,
unique owner contribution, exactly-once ledger consumption, rejection after
delivery, exact parent rollback, and retry. Constant-state Outflow gives
R=2*(1-.01*.3*.7), unchanged by divergence, and q=.7+.01*R on the unit-area
right face. Both existing `2e-13` guards are unchanged.

Checkpoint restoration now uses
`begin_restart_transaction → restore_checkpoint_program_exchanges →
commit_restart_transaction → finalize_restart_transaction`. It asserts exact
checkpoint/State equality, time=.01, macrostep=1, and transaction completion.
This receives ledger restoration in the real transaction; it does not assert
a fresh System restart or a new scientific checkpoint qualification.

Validation: the full `test_program_runtime.cpp` translation unit compiles with
clang++ C++20 `-fsyntax-only`, Dim1, Kokkos/OpenMP, MPI and parallel-HDF5 defines,
using the authentic review-checkout headers and central gtest headers. Exit 0;
one preexisting gtest char8_t conversion warning. No link, native run, install,
or rebuild performed here. Central serial/MPI2 re-execution is still required
on the rebuilt fixture; earlier successful capture/precision cases are separate
evidence and do not make this repaired feedback case green.

## Central follow-up: explicit advection velocity

The rebuilt wave4c fixture reached rollback, retry and native restart, but its
wall-current assertion still failed: q remained .7 instead of .719958. The
production `ScalarAdvection` default constructor prepares zero velocity on
every axis. The fixture now calls the public `prepare` factory with x velocity
1 and zero transverse velocities, matching its original wall-current oracle.
This changes only the physical test configuration. The true numerical face,
ownership rule, exactly-once guard, rollback checks and `2e-13` tolerances remain.
The MPI CTest selection also receives both new canonical-unit tests. Native
reception of this follow-up is pending the next central build.
