# Consumed Field mapping — Source contract

`mapped-consumed-output@1` selects an explicitly declared component or gradient
of a consumed Field solve. Its FieldProblem owns the scalar FieldSpace (support,
units, cell sampling); the compiler infers its layout from all equation inputs.
No user State anchor or first-input fallback is admitted. Conditional declaration
schema 2 and discretization schema 3 retain the old schemas when absent.

Physical coordinate units follow declared support-coordinate order.
`FieldDiscretization(observation_axes=...)` authenticates their embedding into
Native axes. Gradient selection must be an active declared axis; its units are
Field units divided by the corresponding physical coordinate units. Signed
factors are explicit exact integers or rational numbers.

`FieldSolution.mapping_port(unknown, derivative_axis=..., factor=...)` and
`publish_mapped(map, bindings, states=...)` compose the existing public physical
map and publisher. Each invocation produces a private destination-shaped scalar
candidate. The ordinary same-layout publisher still validates the destination
FieldSpace and commits under its existing enclosing transaction. Mapping receipts
carry qualified Field identities, not State block names; width one denotes the
selected observation, not the owning State width.

IR 24 is conditional on this operation. ABI 9 and capability
`mapped_consumed_field_output` are required before provider installation.
Legacy State maps retain their wire and numeric operations. Source authority
validation and consensus-payload allocation now vote before MPI transport or
consensus; communication equivalence with the prior route is not claimed.

This implementation emits Uniform storage only. Existing physical-transfer
HOST memory guards remain; no AMR, GPU or scientific Native qualification is
asserted. Python composes equations and methods; generated C++/Kokkos performs
cell work. Old accepted images are never substituted for consumed stage outputs.
Freshness, owner layout/distribution, Program-point identity, collective failure
and rollback remain mandatory authorities. A real SDK rebuild and independent
Native witnesses are required after the Source freeze.
