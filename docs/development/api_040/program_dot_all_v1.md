# Explicit Program vector pairing, version 1

`Program.dot_all(a, b)` selects contract `pops.program.dot-all@1`. It returns
one collective Scalar: the sum of `a[cell,component] * b[cell,component]` over
every component on the exact Program owner's active cells. It uses raw cell
algebra, excludes inactive embedded-boundary cells and AMR cells covered by finer
levels, and applies no physical volume or kappa weights. Physical integrals
remain the existing measured System operations. Both inputs require compatible
field Spaces, component counts and one explicit identical Program block owner.
General field-problem storage without such an owner is refused in this version.

The native providers authenticate the complete field layout/component contract.
Their dedicated local kernel intersects the EB active mask with the native AMR
coverage mask (`1` retained, `0` covered), while retaining raw component algebra.
The coverage consumer reuses the existing owner traversal and recovers each
resolved field's actual level from its native owner identity; callback order does
not identify levels because Scratch/Direct inputs can skip them. The coverage
getter is a prepared local lookup, and never materializes hierarchy data or enters
collectives from inside local validation. Legacy traversal and kernels are unchanged.

Every rank validates its local data before one error-convergence vote and one
sum on the prepared execution lane. A distributed field contributes its local
patches. A physically replicated field is validated on every rank and contributes
only on lane rank zero, without division by rank count. Ranks with no owned
patches still participate. Active nonfinite inputs, nonfinite masks, product
overflow and component/patch/level accumulation overflow are refused before
the sum; a nonfinite global sum is refused after that common collective. Field
values excluded by coverage or EB activity do not enter the contraction. No
physical volume weights, numerical tolerance, host gamma evaluator or second
runtime are introduced.

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


The bounded AMR/numeric correction following independent review `f212b9c`
adds a direct-source host fixture
`tests/python/unit/runtime/test_dot_all_native_contract_host.py`. It compiles
and executes the actual local kernel, local accumulator, provider methods,
field-contract checks and full legacy/new owner traversals extracted from the
checkout. Its 24 assertions cover coverage/EB intersection, covered NaN exclusion,
active NaN refusal, local/product/global overflow, five field-contract divergences,
Direct level selection, unique replicated contribution and invalid data on a
noncontributing replica. Storage, ownership classification, iteration and MPI
collectives are explicit host substitutes; this is not native execution evidence.
The old six component-zero methods and complete legacy AMR owner visitor are
byte-identical to `21a56b9`.

The enlarged generic C++ fixture additionally calls real providers for an AMR
regrid with an analytically prescribed two-cell coarse footprint, exclusion of
covered coarse NaNs, rank-local width/ghost-contract refusal, poisoned last-rank
replicas, product/local overflow and distributed global overflow (MPI2 required).
These cases must run on the newly built central SDK. The author only syntax-checks
the complete existing runtime TU plus these fixtures in Dim1 and Dim2; no installed
native state, real MPI collectives, GPU execution or scientific reception is
qualified by these source/host/syntax checks. No JIT, installation, shared environment
or central release/ABI contracts are changed by this correction.

The final correction source selection passes **54 tests** in 12.79s: the host
fixture, computed-frontier contracts, strict import-graph gate, Program expression
and authoring atomicity suites. It explicitly imports this checkout's Python,
selecting installed Dim2 solely for loader compatibility. Ruff and diff-check
pass. The native provider headers pass Dim1/Dim2 syntax with the full existing
runtime TU and generic fixture. The only compiler warning is the pre-existing
Googletest char8_t conversion warning.


The first central real Dim1 reception exposed a fixture initialization error:
`ExplicitVectorPairingUsesNativeAmrCoverageAndIgnoresCoveredNaN` returned 16
against 40, both before and after poisoning covered coarse cells. The receipt is
`outputs/native-integrated-codec-dot-owner-amr-dim1-20260930/ctest.xml` in the
workspace campaign. `AmrSystem::set_conservative_state` stages the vector before
engine construction (`src/runtime/amr/amr_system.cpp:16553`). In
`ensure_engine`'s `materialize_state`, the field is set to zero and
`explicit_bootstrap=true` checks that a source is staged without applying it
(`:10420`); the automatic `write_field` belongs to the other branch. The fixture
never ran the public explicit projection. Thus the six retained coarse cells
were zero and only the four child cells at value two contributed: 16.

The bounded fixture correction selects automatic initialization with
`explicit_bootstrap=false` and asserts collectively, before regrid, that every
actual coarse DOF equals two using a nonnegative squared error. No tagger is
installed, so automatic refinement is absent. The child/regrid/coverage/NaN
checks and expected 40 are retained. `publish_regrid` does not change
`active_level_`; `state(int)` returns the live attempt or facade accepted field,
whose exact pointer is classified `Family::State`. That family traverses all
live levels. No consumer, legacy reduction or header is changed by this fix.
The corrected fixture passes syntax checking; successful real native reception
must be re-established centrally.
