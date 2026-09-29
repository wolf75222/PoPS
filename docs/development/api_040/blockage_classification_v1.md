# C04 blockage classification, schema 1

`pops.blockage.Blockage.to_data()` is the versioned diagnostic record for a
**proven** API 0.4 blockage. It records `classification`, rejecting `phase`,
`source`, generic `cause`, and the exact missing `capability` when one exists.
It is attached to a `ValueError` or `NotImplementedError` subclass, preserving
existing catch behavior and human-readable messages. It is not serialized into
Program IR, native ABI, checkpoints, or compiled artifacts.
`Availability.no` retains the same record when it translates a classified
capability error into pre-runtime inspection.

The classification belongs to the rejecting layer, not to a model name or a
string found in the error message:

| Class | Decision | Current classified seam |
| --- | --- | --- |
| EXPR | Language cannot express an unknown or argument | A typed scalar `ProgramValue` cannot serve as a `SolveUnknown` state/field template. |
| IMPL | Relation is represented, but selected realization/lowering is absent | Retained rate with no chosen balance realization; represented custom solver IR op/reduction without C++ lowering. |
| MATH | Model/method hypotheses demonstrably conflict | A finite negative diagonal diffusion coefficient contradicts the selected nonnegative two-point monotone method. |
| SCOPE | A new mathematical domain must be specified | No current rejection is automatically assigned this class. A future domain request needs its own explicit contract and gate. |

Malformed source, foreign identity, and missing authoring bindings remain their
original validation errors. In particular, a generic `NotImplementedError` or
unexecuted method does not imply EXPR; absence of numerical realization is
IMPL. Missing HLL wave-speed or HLLC/Roe provider declarations are reported as
missing capabilities, without a MATH label: provider absence alone does not
prove that the physical model contradicts the method. Several blockages may
occur successively at different layers, so this
record describes one refusal and Python exception chaining retains its cause.
The schema is intentionally separate from public checkpoint and codegen
versions; changing its fields requires a new diagnostic schema version.
