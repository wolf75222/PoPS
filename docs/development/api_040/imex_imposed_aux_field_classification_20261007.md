# Imposed auxiliary dependencies and IMEX Field operands

`Model.operator(inputs=None)` infers actual solved FieldSpace operands. An explicit tuple,
including `inputs=()`, declares those operands and cannot hide a solved-field read. The same
rule applies to legacy single-state authoring and owner-qualified multi-state Field symbols.
Imposed AuxSpace reads remain exact provider requirements, rather than positional Field operands.
This corrects the existing authoring contract; it introduces no Native API, ABI, wire format,
or numerical scheme change.

The legacy operator view distinguishes its declared solved-field components from imposed
AuxSpace declarations while retaining the existing auxiliary requirement inference, including
primitive-expression closure. A local-linear operator that reads an imposed clock auxiliary
can therefore remain nullary. A genuinely coupled Field operand remains non-nullary and the
existing IMEX field-independent implicit-operator guard still rejects it.

Producers can be attached after the Program clock is authored. At provider resolution, the
exact selected auxiliary dependency graph is traversed. A DerivedAux chain reaching a solved
FieldSpace absent from the operator's sealed signature is refused; the signature and FieldContext
proofs are not retagged. Existing authenticated Field binding projections determine the allowed
FieldSpace names.

The existing implicit solve and application still consume their implicit stage coordinate
(`c_I = 1` for IMEX Euler). Analytic auxiliary preparation uses the existing consuming solve
path. Explicit Field solve/read composition continues to use `c_E = 0` for IMEX Euler and the
same State identity. No tableau coefficient, equation, solver tolerance, or guard is changed.

Source validation uses the IR17 environment with Main's `python` directory explicitly inserted
before importing PoPS. The six affected test files contain 69 passing cases, including ten new
cases for constant and imposed-auxiliary nullary operators, explicit Field declarations, atomic
refusals, transitive hidden-field dependencies, and multi-state inference. Commands, XML, initial
test-authoring errors, final file hashes and logs are retained in
`/Users/romaindespoulain/dev/tmp/sol61-imex-imposed-aux-field-source-20261007`.

This is Source validation only. The independent nonautonomous witness retains its equations
and rational oracles. Its Uniform GeometricMG and screened CartesianCG resolve refusals are
preserved separately; its public AMR/MG realization and subsequent rebuilt-package execution
are separate reception obligations. No Native or GPU qualification follows from these tests.
