# Optional original AMR spatial-basis Jacobi realization

The authentic non-preconditioned N32/012 first Newton exhausts the unchanged
80-column restart / 240-column budget. Its recomputed norm exceeds the declared
stop; changing the stop or budget would not repair an implementation defect.
Its red native history remains immutable. A distinct native case now expects
this exact N32 identity-realization budget refusal, consumes RejectAttempt and
checks candidate invisibility and unchanged live fields.

The native interfaces expose the actual full-tower original operator, active
coverage, measures, prepared generation and lane. They expose no complete AMR
diagonal. FAC's internal Jacobi smoother uses a local approximate diagonal; its
formula does not authorize the actual matrix diagonal at coarse/fine interfaces.
The existing PreparedLinearPreconditioner sessions describe a single MultiFab
and cannot silently represent this full hierarchy. No such authority is borrowed.

`AmrFieldNewtonKrylovWorkspace::solve_preconditioned` now accepts an explicit
stationary linear right-apply callback over a full tower. It applies JVP to the
preconditioned Arnoldi direction and accumulates the corresponding physical
correction. Existing image storage is reused. The actual complete-correction JVP
recheck, original-equation recheck and atomic Outcome publication remain in force.
The identity `solve` entry point delegates through an identity callback.

`PreparedAmrFieldResidual::prepare` accepts a final optional native realization:
identity remains the default. `kSpatialBasisJacobi` explicitly selects
`pops.amr.original-spatial-jacobi.basis-response@1`, with residual authority @2.
Identity retains authority @1 and contributes no new bytes to its exact contract.
The existing provider and FieldNewton option structures/virtual tables, Python
IR and checkpoint wire are unchanged. This is a C++ native-core realization;
it is not yet a Python authoring/preparation port or a universal default.

The selected realization computes `d_i = [A(e_i)-A(0)]_i` from the true prepared
operator on every globally ordered stored cell/component. Current frozen linear
composite providers thus yield their exact spatial diagonal, including physical
boundaries, restriction, ghost interpolation and conservative reflux. The zero
response removes affine boundary data. No coefficient stencil, local reaction,
source equation, model identity or component count is invented. Inactive/covered
cells receive zero inverse; active finite signed diagonals must be nonzero with
finite reciprocals. A missing spatial diagonal refuses this realization; it
does not establish that the original nonlinear equation is mathematically
unsolvable. A different right-preconditioner realization can be added through
the same native callback port.

Preparation performs one zero-response application plus one operator application
per **stored** DOF, including covered parent storage. Cost is O(stored DOFs)
operator applications, with no artificial count ceiling; overflow is rejected.
It retains one deep-owned inverse field tower. This is an explicit reference
realization, not an efficient universal AMR diagonal assembler. All ranks traverse
the same exact prepared topology and participate even if their local ownership
is empty. Local initialization/extraction failures vote before another operator
collective. Prepared generation, exact equation/capture/point authority and live
attempt leases are checked before preparation and each right apply. The spatial
diagonal depends on frozen spatial coefficients/topology and is independent of
Newton q or local captures; all original residual/JVP evaluations still use the
authored body and its captures at the declared point.

The positive native case keeps its historical name and calls the **same**
manufactured original-equation helper with an explicitly selected spatial-Jacobi
realization. It retains N16/N32, two component permutations, original loads/body,
source/FD step, measures, tolerance, restart, budget and original residual/error
checks. A distinct legacy identity case receives the known N32/012 rejection.
JUnit properties expose realization, linear budget and restart. These are
separate realization receptions, not an unchanged-realization claim. New
native cases refuse unknown/mismatched realizations, stale points and absent
active spatial diagonals before a local body or publication can occur.

Source/host checks cover affine-response removal, signed/nonsymmetric matrices,
explicit selection, legacy authority identity and actual extracted right-GMRES
algebra. Full actual native test-TU syntax is checked with real SDK dependency
headers. Native serial/MPI, higher-dimensional, rollback and provider reception
remain root-owned; no native positive is fabricated by these source checks.
