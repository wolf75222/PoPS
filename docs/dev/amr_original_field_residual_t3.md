# Original coupled field residual on an AMR carrier

Base: `b87b828095069b2a4b07ee2c96fac9ca715cdc78`. This first native
freeze provides an apply-only capability and an attempt-owned original-residual
carrier. Public `FieldProblem` AMR lowering remains unsupported until its actual
Program gather/solve/publication connector is installed. There is no per-level
solve, dense surrogate, recognized physical model name or temporal residual
substitution in this carrier.

`pops.hierarchy.original-field-operator@1` is an optional C++ interface beside
`PreparedHierarchyTensorSolver`, preserving the legacy base virtual table.
`require_original_field_operator` votes on capability, exact prepared contract
and execution-lane binding before entering the provider. The general coupled
provider exposes its real FAC composite application, active coverage masks,
level cell measures, physical scalar product and covered-parent synchronization.
Missing capabilities fail collectively before numerical work.

The original apply path freezes finite diffusion entries. Full matrix entries
use arithmetic face interpolation with their authored signs and the actual
FAC conservative coarse/fine flux mismatch. It invokes no scalar solve.
Nonsymmetric or indefinite finite entries do not inherit a linear-solve SPD
claim: the existing public linear solve still prepares symmetric SPD diffusion.
This distinction preserves the original equation's coefficients. A nonlinear
gauge is an unimplemented realization and is refused explicitly; periodic and
homogeneous Neumann boundaries are the general provider's current realizations.
Dimensions and component width are template/runtime inputs, not witness names.

`PreparedAmrFieldResidual` owns its provider, all Newton/JVP arrays and deep
copies of the actual field captures. It evaluates
`F(q) = A_composite(q) + f(q, captured_fields)` using the original local-body
callback, and differentiates this complete residual with centered differences.
All levels participate in one native `AmrFieldNewtonKrylovWorkspace`; covered
parent cells are excluded from every Krylov norm and the actual cell measures
weight the active physical domain. Replicated levels contribute once, without
division by MPI size. A solved candidate is synchronized through the native
coarse/fine transfer and re-evaluated with the original residual before access.
Neither the provider's accepted solution nor runtime State is published here.

The invocation authority contains owner, parent and per-level attempt leases,
topology/materialization generation, original-equation identity, ordered capture
identities/widths and complete per-level evaluation points. The current v1
invocation requires synchronized physical time, duration and stage fractions.
Stage fractions are revalidated as canonical exact rationals within `[0,1]`,
including when their public numerator/denominator members were mutated.
Preparation and candidate access compare these exact authorities collectively.
Coefficients have a revocable preparation generation: another preparation
invalidates an existing snapshot before any entry changes, even if it then fails.
The provider remains alive after rejection while every new access fails.

Local allocation/copy/body/JVP preparation is caught and voted before the next
collective. The new original route opts into guarded FAC extrusion,
interpolation, restriction and matrix flux phases, and guarded Newton algebra
and scalar products. Legacy routes retain their prior opt-out behavior. Existing
Mutable body outputs are authenticated within the body's vote, including their
level count, exact ranked layout and component width. Projection, copy and dot
validate their structure inside their local votes before any dependent reduction;
the fixture removes a level on rank zero after a body that returns normally.
prepared halo/region-transfer operations keep their native protocols. In
particular, `PartitionedRegionTransfer` still terminates on a failed unpack after
its publication gate; this carrier does not establish retry qualification for
that existing terminal backend failure. Local-body refusal and fresh-attempt
preparation are separately exercised; runtime retry still requires the Program
connector and does not claim to cover terminal transfer failures.

The native fixture closes a discrete manufactured problem with a three-component
signed SPD diffusion matrix, a positive coupled local reaction and cubic terms.
The actual native composite application produces the forcing from a specified
nonconstant cosine field; this is explicitly a discrete witness, not a claimed
continuum convergence campaign. It uses real partial refinement on `[1/4,3/4]`,
ratio 2, split patches and homogeneous Neumann endpoints, coarse resolutions
16/32 and both `(0,1,2)` and `(2,0,1)` component orders. It separately recomputes
the original residual of the solved candidate, checks provider publication is
unchanged, mutates the source storage after capture, and injects rank-local body
failure and point/owner/provenance/width/generation/attempt mutations. The
apply-only signed/non-symmetric case separately confirms that linear SPD solve
acceptance is not broadened.

Remaining public connection is precise: the existing `solve_spatial_field` IR
already preserves physical equation, capture and point identities, but
`program_emit_nonlinear_field.py` builds a Uniform workspace and the AMR phase
scheduler recognizes only temporal `solve_spatial_nonlinear` or linear barriers.
The next connector must identify the original-field barrier, retain the actual
provider through topology refresh, gather deep-owned field/capture/seed towers
inside real `LevelAttemptEnvelope`s, build authority from those actual leases
and points, evaluate the IR local body in the Kokkos callback, consume the real
`SolveOutcome`, and stage every observation/publication before atomic acceptance.
No public AMR claim, observer receipt, checkpoint or restart claim follows from
this first numerical freeze. No M14 wall/circuit or M27 physical data are invented.

Author checks are source and syntax only. The full existing composite-general
test translation unit, including the new fixture, is checked with the real
Kokkos/MPI headers by `clang++ -fsyntax-only` in dimension 1. Central native
build/execution and serial/MPI reception are required before runtime claims.
Both new headers are included in `pops_headers.manifest` for SDK delivery.
