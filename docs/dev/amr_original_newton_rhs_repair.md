# Original AMR field Newton direction and test discovery

Base: `47a7b2aa4952281091cd2d6e99e0c2151c3c638c`.
Private tree: `work/PoPS-sol61-amr-newton-mpi`. Root owns builds and native reception.

The wave6 MPI wrapper executed the original-field cases and reported
`amr_field_newton_line_search_failed` in the nonlinear manufactured witness and
the publication witness. The serial CTest inventory contained only the four
historical declarations in the main `.cpp`, plus the MPI wrapper. It did not
discover the six tests in the included original-field fragment. There was no
native evidence that those six passed serially.

The source defect is a Newton direction mismatch, not an MPI-only equation.
The workspace solves `J_F * delta = RHS` and forms `trial = q + alpha * delta`.
Its original AMR adapter supplied `RHS=F(q)` while the central-difference JVP
correctly differentiated `+F`. For `F(q)=q-target`, starting at zero gives
`delta=-target`: every positive alpha increases the residual. With the actual
publication witness `R*q + .2*q^3 - forcing`, a constant Neumann field eliminates
diffusion and the same wrong sign makes every admitted line-search step uphill.
Ptolemy established this independently in `018b0332c6701f0d7ce68474e0b50900afefac45`.

The repair negates only the workspace's RHS in
`PreparedAmrFieldResidual::residual`, inside its existing local phase with fence
and collective failure vote. The authored body, positive JVP, original-equation
terminal recheck, captures and candidate authority keep their meanings. The
generic workspace and legacy Uniform/AMR spatial adapters are unchanged; those
already supply `-F`. No tolerance, solver budget, units, identity or ABI changes.
Candidate visibility still depends on the full original residual recheck and
publication still uses the existing collectively authenticated SolveOutcome.

A true native fixture now solves the original identity local equation with the
actual composite diffusion provider, partial refinement, two levels and
partitioned ownership. The constant Neumann mode has `F(q)=q-target`; component
widths 1, 3 and 5 test the same generic path. It checks physical residual, candidate
error and preservation of provider-owned values before acceptance. The existing
two-resolution/permutation witness retains its signed cross-diffusion and cubic
coupling, nonconstant cosine profile, actual discrete forcing and independent
original-equation check. It remains required in native reception.

The CMake helper has an explicit `DISCOVERY_SOURCES` port. Only
`test_composite_general_field` supplies its included fragment through that port;
the four old case names/labels and all other target registrations are retained.
Real CMake GoogleTest source scanning, configured with no compiler or execution,
now finds 11 serial cases (four historical, six original, one new), with target
labels. The existing MPI wrapper runs the same full target.

Checks: before repair, the author sign probes had 6 FAIL / 1 PASS against the
actual callback. After repair, 8 source/math/discovery checks PASS. They include
the identity counterexample, arbitrary widths, signed nonconstant coupled
diffusion/cubic algebra and both permutations. Ptolemy's exact independent
source/math probe reports `sign_convention_compatible` on the changed tree.
Ruff and diff checks PASS. The full composite test TU passes Dim1 syntax checking
against real private headers/Kokkos/MPI includes; GTest macros were type-check
stubs, with one existing nodiscard warning. No executable, library, SDK or
installed environment was built or changed.

Root must rebuild/re-discover this target and run its 11 actual serial cases and
MPI2 wrapper, preserving the previous native campaign's other cases. Native
convergence, publication/rollback and ownership are pending that reception;
the source/math checks are not a native MPI PASS.
