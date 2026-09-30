# Original FieldProblem through the synchronous AMR Program

This connector follows the numerical carrier freezes `104b0ffb`, `b19e5794`
and `85a79b6f`, independently reviewed in `c8907cd3`. It connects the existing
`solve_spatial_field` IR to the real native composite provider. Model and
FieldProblem declarations precede the temporal Program in the public witness.
No physical model name or component count selects the numerical implementation.

The source contract remains `pops.spatial-field-residual@1`: original equations,
ordered unknown handles, ordered State captures, diffusion coefficients, local
expressions, physical boundaries and declared point are reauthenticated before
lowering. The AMR target adds the typed native component
`pops.hierarchy.original-field-residual`, including the two SDK headers already
listed in `pops_headers.manifest`. Uniform lowering and legacy provider manifests
retain their prior identities. No IR schema or central native ABI change is
introduced by this connector. The field-resource map now holds a shared solver
so that an attempt-owned residual/Outcome keeps its real provider alive when
Context retires that map; the Context object layout itself does not grow.

Gather runs in every actual synchronized LevelAttemptEnvelope. It evaluates the
original constant finite diffusion entries into the genuine assembly fields and
copies actual captured State values and a declared scalar-field seed, if any,
into field-owned storage. The new explicit retained-input scratch access does
not zero gathered captures on consumption; absent input, changed layout/width
or rank-divergent produce/consume policy refuses collectively before a reset.
The legacy scratch default still produces/reset-zeroes its slot.

Context constructs invocation authority by visiting those same envelopes,
including exact graph/equation/application identities, source State owner,
canonical authored stage, actual physical time, duration, level, tick, substep,
topology/materialization generation and live attempt leases. Resource-level
selection alone cannot impersonate a physical evaluation point. The owning
carrier freezes the gathered values before any residual kernel. The emitted
Kokkos body evaluates the genuine local expressions; FAC evaluates the genuine
composite diffusion and its coarse/fine conservative mismatch. Centered JVPs
re-evaluate that complete residual, and one Newton/Krylov workspace solves the
entire tower with active-coverage/owner/cell-measure scalar products.

The original finite signed operator uses arithmetic face interpolation also for
scalar/diagonal packs: a negative coefficient cannot silently become the positive
fallback of a harmonic SPD interpolation. The existing linear/SPD route keeps
its old face policy. The native sign fixture applies +D and -D to a nonconstant
field on partial refinement and requires opposite nonzero composite images.
Candidate synchronization copies the real prepared ghosts into the publication
image; components/observations therefore consume the provider's synchronized
solution after its true collective SolveOutcome is accepted.

Before Accept, the publication validates the actual live authority again and
checks exact layouts and finite grown fields. Staging changes no live solution;
Accept restores all prepared images as one publication action. The Program
observes every level, stages physical auxiliary fields, publishes them through
the existing transaction, and then evaluates the original PDE source with those
auxiliary fields. Parent rejection restores State, auxiliary payloads and accepted
metadata, histories and lifecycle through the existing Program transaction.
Checkpoint/restart uses the accepted State/auxiliary/history protocols. Solver
scratch is transient; the default initial guess is explicitly zero, and a declared
seed is re-evaluated each attempt, so a restart does not depend on a hidden warm
start from uncheckpointed solver storage.

Current realization: synchronous Cartesian AMR, one top-level original-field
barrier, constant finite coupled diffusion, the local reaction/load expressions
accepted by the original source contract, periodic or homogeneous Neumann
boundaries, and no nonlinear gauge. Nested/multiple barriers, combined continuation
scheduling, asynchronous field-time transfer, embedded-boundary geometric
operators and shared-interface field operators require additional declared
realizations and refuse explicitly. These are provider/scheduler limits, not
mathematical limits on the generic carrier. Existing post-publication terminal
region-transfer backend failures retain their native policy; retry qualification
of those backend faults is not asserted here.

The public mechanism witness has a three-component nonsymmetric diffusion matrix,
coupled cubic reactions, spatially varying captured coefficient/load data and a
closed constant solution. Its source consumer uses the observed solution in a
real PDE update. Scalar, ordered three-component and permuted three-component
cases span coarse resolutions 16/32, genuine partial ratio-two refinement,
optional declared seed, accepted histories, saved NPZ state, checkpoint/restart
and parent rejection both before and after an accepted field image. Independent
recalculation from the saved captured data verifies the original reaction residual
and constant target. Constant diffusion vanishes for this exact target; this
witness does not establish nonconstant continuum AMR convergence. The earlier
native discrete manufactured fixture separately exercises nonconstant composite
coarse/fine application. Neither fixture invents a M14/M27 scientific campaign.

Author validation is source-only Python and C++ syntax with real Kokkos/MPI
headers. The installed serial/MPI campaign is deliberately not run in this
private checkout: central integration, rebuild and native reception are required
before public runtime, restart or rollback qualification. The public fixture
asserts installed Python identity, compiles once collectively, converges local
checks between native collectives, writes evidence on rank zero and records
artifact, native dimension, MPI rank/size and evidence path.

Recorded author checks for this freeze: 30 targeted Python tests passed (22
legacy original Uniform tests plus eight AMR source/admission cases). After the
owned invocation/callback allocation change, all eight AMR cases passed again.
The full composite-general native test translation unit passed dimension-1
syntax with two pre-existing warnings; the full scratch/retained-input test
translation unit passed dimension-1 syntax with one pre-existing gtest warning.
The latter used compile-only helper macro strings to parse the existing dynamic
compiler helper; no helper invocation, linking, JIT or test execution occurred.
The emitted public permuted/seeded/guarded program passed dimension-2 syntax
against the actual Kokkos/MPI headers. Ruff and `git diff --check` passed.
