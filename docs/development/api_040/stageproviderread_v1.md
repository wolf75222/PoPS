# Provider reads at an exact forward-Euler stage, version 1

Contract identifier: `pops.stageproviderread@1`.

This contract extends the read-authority boundary of `accepted-update-ssp@1`.
It leaves its exact rational coefficient arithmetic, coefficient-one convex
factorization, accepted expressions and existing spatial guards unchanged.
It also preserves `pops.accepted-static-provider-read@1`; static observations
remain authenticated external storage, with their own mutation refusals.

The original accepted-update contract requires a frozen mathematical closure
for extra Field reads. The static contract explicitly does not make computed
providers static. Neither contract proves that a fresh field solved from the
current evolving State belongs to the exact rate evaluation that reads it.
A stage read is therefore a separate obligation, rather than a new meaning
of "frozen".

For example, an authored composition may solve `-L_h phi = U - B`, consume
that solve, publish phi, and accept `U_next = U + dt R(U, phi)`, while B is
unchanged. Phi depends mathematically on U and cannot have a frozen-State
certificate. The exact accepted expression nevertheless has one coefficient-one
forward-Euler contribution. To use its conditional spatial premise, the compiler
must authenticate the actual solve and provider reads at that contribution.
Reading a solve from another State or a replacing publication can retain the
same algebraic shape while changing the authored mathematical operation.
Names, article identities and specific equations do not supply this authority.

For every contributing RHS independently, the compiler verifies:

1. The actual source `ProgramModelGraph`, exact typed rate declaration and
   instance, formal StateSpace/FieldSpace and complete constitutive read union.
   Resolved operation-plan payloads and issued identity are reauthenticated;
   publication reprojection must agree with the actual emitted ProviderPack.
2. Exact issued SSA objects, clock, mathematical point, region and FieldContext.
   The publication's consumer State must be the RHS State itself, allowing only
   the existing authenticated successful terminal-guard alias. A current stage
   State is a boundary: its earlier-stage ancestry is proved by the existing
   accepted-expression/convex analysis, not mistaken for a simultaneous Field input.
3. Complete component coverage and the original resolved output claim: target,
   producer, solved unknown, observation and selected component. Each physical
   Field equation is authenticated against its issued ResolvedProgramFieldPlan,
   current canonical payload and actual solve graph. Linear load/coefficient
   expressions and reaction payloads are re-encoded from registered equations;
   every executed load/coefficient/apply physical boundary also matches the
   registered boundary definition,
   so changed executable mathematics cannot retain an unchanged identity label.
4. The consumed solve and every real free capture and input/output binder.
   Mathematical State reads must be the exact RHS stage State or a foreign State
   whose accepted value is proved unchanged. Different evolving owners need an
   additional joint-owner premise; this contract does not silently assume one.
5. The publication precedes its RHS and no intervening executed publication
   replaces any selected constitutive component. A later stage may publish its
   own snapshot after an earlier RHS. Producer identity alone cannot distinguish
   two writes of the same physical field, so execution order remains required.

The public `validate -> resolve -> compile` pipeline carries its existing
`resolved.program_field_plans` automatically through the production model graph.
Scientific authoring needs no second assembly of plans, technical override or
manual certificate. An internal caller lacking that resolved authority is refused.
Source/emitter tests may use the same production graph factory as compilation.

There is no production restriction to one RHS, one stage, one observation or
one unknown. Every RHS has its own authenticated read snapshot, and the existing
convex proof composes its arbitrary explicit rational stage/accepted rows.
Joint registered field systems can supply multiple declared observations and
components. Existing language restrictions, such as competing physical producers
for the same component or effects with no proved nonmutation contract, remain
refusals; this contract adds no automatic capability for opaque or mutating effects.

The convex result is conditional. If each actual forward-Euler evaluation
`U_i + dt F_i(U_i; fields_i)` satisfies a common convex spatial invariant under
its existing guards, the exact coefficient-one convex decomposition preserves
that invariant. Authenticating a stage read does not prove that spatial premise,
a positivity or entropy theorem for arbitrary physics, global order, stability
outside its guard, or a scientific benchmark result. Each executed solve,
consumption action, publication, field evaluation and guard still executes.

Durable Source witnesses cover actual dependent-field source emission, joint
one/two-field systems at one/two stages, renaming and automatic production-plan
carry. A separately authored two-screened-field witness also receives its unchanged
SSPRK3 composition at c=(0,1,1/2), with exact b=(1/6,1/6,2/3), frozen foreign donor
and real source-authored Field/State observations. This is Source proof coverage,
not a numerical or installed-runtime result. Refusals cover foreign evolving States, replacing publications, changed
clock/point/region/context/component, changed source/Field/operation-plan identity
or payload, wrong callable binder roles, emitted-pack mismatch and negative
accepted weights. Equations/compositions are unchanged, Native imports are
blocked, and no C++ source, installed runtime, build or numerical guard is changed.
Source admission does not qualify Native, MPI, GPU, 3D or long-time physics.
