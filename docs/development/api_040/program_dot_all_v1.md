# Explicit Program vector pairing, version 1

`Program.dot_all(a, b)` selects contract `pops.program.dot-all@1`. It returns
one collective Scalar: the sum of `a[cell,component] * b[cell,component]` over
every component on the exact Program owner's active cells. It uses raw cell
algebra, excludes inactive embedded-boundary cells and AMR cells covered by finer
levels, and applies no physical volume or kappa weights. Physical integrals
remain the existing measured System operations. Both inputs require compatible
field Spaces, component counts and one explicit identical Program block owner.
General field-problem storage without such an owner is refused in this version.

The native providers authenticate the complete field layout/component contract,
loop over every component using the existing active-cell local reduction, then
perform one error-convergence vote and one sum on the prepared execution lane.
Ranks with no owned patches still participate. This does not introduce a second
runtime, a host gamma evaluator, or a numerical tolerance.

The historical `dot`, `norm2` and `norm_inf` component-zero meanings are retained.
Their existing serialized attrs and Program identities are unchanged. New
vector pairing has discriminated IR attrs:

```json
{"kind": "dot_all", "component_contract": "pops.program.dot-all@1"}
```

The emitter refuses a missing/foreign contract and emits `ctx.dot_all(owner,a,b)`.
Canonical selection occurs in `python/pops/time/_program/serialization.py`,
`_ProgramSerialization._serialize`: promote to ProgramIR version 7 only when a
real `reduce(kind=dot_all)` is present, including owned lazy/control-flow/apply/
residual regions and the dt-bound body. Programs without this extension keep
the previously selected schema (v5, or the centrally integrated v6 constitutive
trace profile). Central generated release/ABI/support registries are integrated
by the primary worker; this author checkout does not regenerate them. The
moving-interval profile can use the same central v7 capability envelope.

The computed-frontier rotation fixture changes only its explicit contraction
from `dot` to `dot_all`. Its rate, Heun stages, relaxation expression, guards,
retry budget, durations, endpoint-ULP policy and restart comparison are retained.
For requested duration 1, `increment=(-.5,1)` yields gamma `.8`, candidate
`(.6,.8)` and duration `.8`. Component-zero pairing instead yields gamma 4,
which explains the genuine SDK307 rejection with effective duration
`0x1.0000000000000p+2`; at requested `.5` it yields effective duration 8.
Both rejected attempts restored the accepted clock/state. The correct generic
vector contraction is `-2*dot_all(U,increment)/dot_all(increment,increment)`.

Source counterexamples exercise emitted discrimination, legacy attrs, recursive
schema selection and forged contract refusal. The compiled fixture
`tests/cpp/integration/runtime/program_dot_all_contract.inc` calls the genuine
`ProgramContext` on multicomponent fields, proves both contractions and refuses
wrong component width. Include it in the central `test_program_runtime.cpp`.
The complete existing runtime TU plus this fixture passes Dim2 C++ syntax
checking; the new AMR provider method also passes Dim2 instantiation syntax.
The enlarged source selection initially passed 239 tests and exposed a reserved
sink error-message compatibility regression; the original balance-specific
message is restored while frontier sinks remain refused. The affected final
authoring/Scalar-frontier selection passes 51 tests (three native/compiler cases
deselected). Two additional legacy process-isolated control-flow scripts were
blocked at bootstrap by missing native dimension selection in their subprocess;
they are not claimed as passing. Ruff and diff-check pass.
The final complete enlarged direct-source selection at frozen implementation
`21a56b9` passes **240 tests, three native/compiler cases deselected**, in 50.93s.
It imports this checkout's Python via explicit `PYTHONPATH=$PWD/python` after
`env -u PYTHONPATH`, selecting installed Dim2 only for loader compatibility.
This source receipt does not qualify the changed native provider methods.
These checks do not qualify native execution. Rebuild/relink both installed
dimensions after integrating the changed provider headers, select ProgramIRv7
support centrally, run this C++ fixture and the existing installed computed
frontier rotation/retry/restart plus MPI refusal fixtures on that exact SDK.
