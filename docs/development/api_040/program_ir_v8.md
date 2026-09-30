# Program IR 8: original spatial residuals and typed integral candidates

Program IR version 8 discriminates `solve_spatial_field`, `integral_candidate`,
and the declaration metadata `integral_units_v2`. A typed declaration selects
version 8 even without a consumer: its persistent identity includes canonical
physical units and differs from the version 1 identity of a legacy declaration.
The graph scan visits lazy regions and the separate timestep-bound graph.
Version 8 takes precedence over `dot_all`'s version 7, independently of node
insertion order. Transport-only legacy integral declarations retain version 5;
prepared diffusive trace declarations retain version 6; a legacy graph using
only the new vector pairing retains version 7.

These are conditional serialized-program contracts, distinct from semantic IR
version 3 and native ABI 5 in the central release contract. The new prepared
capture header changes the installed SDK signature and requires a fresh native
build. This change adds no runtime-state member or checkpoint wire field.

The pointwise expression integration preserves priority for a typed State as
the cell support, regardless of expression input order. Legacy reduction
scalars retain component `scalar`; typed integral captures use component
`value`. Global scalars cannot define cell support by themselves.

`tests/python/unit/codegen/test_program_v8_contract.py` receives real public
declarations, vector pairings, spatial solves and lazy regions. Native receipt
of their consumers and MPI behavior is recorded separately after rebuilding;
version selection alone is not numerical or backend qualification.
