# Diffusive RHS issued-value boundary

`ProgramValueAuthority@1` now lowers the transitive `diffusive_rhs` producer of a
V2 Field source into its actual `RhsScratch` slot `(SSA id, subslot 0)`. This
admits one existing diffusion delegate. Unknown producer kinds and delegated
path RHSs remain refused; no equation, model label, or formula selects a route.

The private `delegated-output-publication@1` emission envelope records the
actual output token, owner, storage identity, allocation-only setup prefix,
pre-write boundary, and end of the emitted evaluation. It does not issue a
runtime value by itself. The caller authenticates this envelope against the
installed immutable SSA storage plan and uses the existing write-ticket issuer.

For top-level diffusion, only `ctx.rhs_scratch(id, 0, input)` is shared between
cadence branches. Stage selection, Cartesian/constitutive preparation, provider
read preparation, kernels, transport assembly, local stability and source guards
stay in the due body. The input read tickets are checked after selecting the
actual stage, before evaluation; completion follows the successful delegate tail.
Allocation failure issues no write ticket. Evaluation refusal reaches no
completion. Hold/Zero use their declared empty-input off-cadence branch with a
real restore/zero write before completion. The previous private delegate return
value and no-hook residual/implicit ordering remain intact.

A completed rate and a successful Field candidate are provisional products
inside an attempt, not an `AcceptedSnapshot`. Independently composed explicit
updates keep their existing deferred convex stability guard at the consuming
state combination. A late update or Field refusal must withdraw candidate rights
through the existing attempt rollback before atomic accepted publication. No
stability phase, threshold, floating-point option, or constitutive arithmetic is
changed by this boundary.

The source-only witness authors a real diffusive rate, an Euler predictor, and a
V2 Field depending on that predictor. The baseline refuses this transitive rate
before an authority plan exists. Candidate tests authenticate the real scratch
identity, stage/ticket/guard ordering, cadence-only setup, foreign-boundary
refusal, unchanged implicit delegate API, and a late rejection before commit.
Cadence/partition observations use a separately authored unchanged endpoint;
they do not claim an unsupported accepted diffusive quadrature.

The existing native test
`PreparedFieldRhsInputs.OneRankNonfiniteCandidateLeavesAcceptedStorageAndAuxiliaryExact`
already retains State, Auxiliary and potential images, refuses a late Field
candidate, rolls back, and checks that produced-value and source validators expire.
This tranche adds source-level ordering coverage; native rebuilding and real
diffusion-to-Field execution remain separate qualification work. Source checks
are not MPI, GPU, or scientific acceptance evidence.
