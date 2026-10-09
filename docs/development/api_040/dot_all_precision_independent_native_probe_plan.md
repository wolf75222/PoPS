# Independent native cancellation probe plan

Prepared against exact source `f36176fa` after the offline oracle freeze
`4719480f7c6fcdf162f184aa4538c0b4f1328319`. No compilation or execution is
claimed for this test by the reviewer.

`program_dot_all_precision_independent.inc` adds
`ProgramRuntime.ExplicitVectorPairingPreservesCancellationAcrossComponentsAndOwners`
to the real `test_program_runtime` target through its existing vector pairing
include. The existing MPI2 CTest wrapper selects `ExplicitVectorPairing*`, so
the new case is included without a CMake filter change.

The test constructs a real distributed System, installs its execution lane,
binds valid initial gas state, and contracts actual native scratch vectors.
It distributes the exactly representable products `[1e16,1,-1e16]` either
over different components at one owned cell or over x cells 0,1,4 in one
component. MPI2 partitions x at four: the second route deliberately puts the
high and low on rank zero and the negative high on rank one. The exact global
sum is one, although finalizing each rank to one binary64 scalar loses it.
It rotates the participating component indices to expose component-order
loss. Other cells/components are zero; transverse coordinates are fixed to
zero in dimensions two and three. The contraction must equal one exactly.
The accepted native state must remain byte-identical after the queries.

This fixture therefore distinguishes local compensated accumulation from a
globally transported high/low accumulator. It does not mock a reducer, invent
a physics model in the runtime, alter a temporal target or widen a tolerance.
The test is generic across the compiled dimensions and has no native skip.
Its declared decomposition uses the existing bounded eight-cell fixture,
which requires the MPI rank count to divide eight; that is a test setup
requirement, not a production rank limit.

Central reception must rebuild the real target and run this case in serial
and in the complete MPI2 wrapper on the integrated reducer SHA. Until then
the source fixture is prepared evidence only, not a passed native assertion.
