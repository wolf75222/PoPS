# One-way ownership of symbolic numerical policy authoring

The ten dependency violations reported by `test_import_graph.py` came from
finite linear algebra and source-authored reconstruction/face policies importing
compiler IR classes, plus a numerical policy importing the physics facade's
state class. The architecture gate and its allowlists are unchanged.

The model layer now supplies `scalar_contract`: immutable scalar/vector bodies,
their traversal, and nominal state shape. Registry and physics board states
implement the same abstract `StateShapeHandle` contract owned by `model`.
Numerical policies consume that contract; they import neither `physics` nor
`_ir`, including inside functions. Existing state handles retain their public
classes, owner identities, component order and resolution rules. A duck object
with matching component labels is refused.

`linalg` now has only standard-library dependencies. Finite supports and maps
retain immutable mathematical declarations. Versioned scalar/application plans
preserve every literal, operation, support label and coefficient; they neither
evaluate a map nor fold scalar arithmetic in Python. Registered exact literals
use their existing literal protocol and are captured once, including units and
target spelling. The consuming IR converts declarations through a local memo
and cycle guard into the existing `FiniteApplication`/`FiniteProjection` nodes.
An IR-owned weak identity cache preserves reification across separately authored
mixed Expr operations without retaining declarations or compiler expressions.
All projections of one map share one joint native application. The IR validates
the complete finite contract without importing `linalg` in reverse.

Program component packing and materialization belong to `time`: an explicit
finite-component protocol validates the ordered field labels; the Program
lowers the captured components and authenticates their owned temporal inputs
and output template. Finite declarations also remain legal model equation and
scalar Program expression inputs. There is no new evaluator, native header,
artifact schema, checkpoint format or ABI change. Emitted `finite_linear_v1`,
`finite_projection_v1` and `finite_support_v1` identities are retained.

Reception: 196 source tests pass, including the unchanged global import graph
and temporal package architecture suites, finite W06/M09 source generation,
joint/scalar reconstruction and user face authoring/adversarial tests, physics
board atomicity/multispecies, temporal expressions, computed Scalar frontiers
and exact scalar literals. The new independent counterexamples inspect imports
through every AST scope, exercise exact Fraction/Decimal/registered literals,
verify one joint application and unfurled arithmetic, refuse malformed/cyclic
plans and shape ducks, and capture a finite result in a real model operator.
Ruff and `git diff --check` pass.

Independent follow-up caught a sharing regression in the first dependency repair:
separate `Expr + finite_projection` operations each created an application,
changing the encoded Program DAG. The weak reification cache fixes that defect
without moving compiler ownership into `linalg`. Counterexamples compare the
mixed public Program hash and encoded nodes to the former explicit IR DAG,
verify shared literal operands, and verify both declarations and lowered
expressions are reclaimed. The initial affected selection passes 45 source tests;
the complete prior affected selection plus these counterexamples passes 199.

The test process selected installed Dim2 for loader compatibility and imported
Python sources from this checkout using explicit `PYTHONPATH=$PWD/python` after
`env -u PYTHONPATH`. No shared environment, SDK, installation or native header
was changed. Native compile/bind/run, MPI, GPU and GitHub CI are not qualified
by this source reception.
