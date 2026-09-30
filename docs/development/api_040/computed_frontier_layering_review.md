# Temporal dependency repair after computed-frontier integration

The architecture gate exposed two current reverse edges:
`time.expressions -> time._program.value_validation` and
`time.solve_problem -> time._program.local_product`. Its acyclic-import assertion
also requires concrete value algebra and mathematical expressions to remain
independent of each other's implementations.

The single issued-value ownership check now lives in `time.value_support`;
Program-specific validation delegates to it. The same low-level nominal markers
identify coefficients, affine carriers and Program values without importing the
concrete value-algebra module from expressions. Exact Program-issued object
identity, owner identity and optional value kind still gate every captured value.

`LocalResidual` delegates its product construction to the Program protocol. The
Program's local authoring mixin alone imports the private local-product builder.
The problem descriptor remains an inert public protocol and does not import its
implementation layer. No graph test, allowed dependency or cycle check is relaxed.

Evidence: all six time-package architecture checks pass. A coherent 81-test
source selection passes across Scalar frontier authoring/emission, generic
expressions, ownership/provenance guards, LocalResidual products, local operators,
readonly captures and independent product counterchecks. Ruff and diff checks
pass. No C++ header or native ABI changes belong to this repair.
