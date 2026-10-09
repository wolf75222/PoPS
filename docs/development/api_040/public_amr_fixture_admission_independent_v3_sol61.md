# Public AMR fixture admission: independent review @3

This corrects a review gap, not production physics. Target is fixture
`b943dd6627550364337d41743cb4efe2f7b33480`, parent
`54dc1c0dd56c8fbc1a8982a259fc894739389f90`, in private worktree
`PoPS-sol61-amr-fixture-admission-review`. This branch changes tests/docs only.
No MAIN, environment, SDK, donor, C++ header, build, native module or JIT is
modified or executed. The source package is explicitly the private checkout.

Historical `c880644af0cc73c6144b18786bbbee93dadcdc7d` received ten AST/math/
extracted-controller checks on fixture `a2f7a2a9`, but did not exercise public
Case/validate/resolve admission. ROOT then received 162 source passes and five
failures in its coherent command: `AMR refine value rule requires strict >
threshold`. The old `<.97` refinement rule was inadmissible. That gap and those
results remain historical; the ten passes are not reinterpreted as public
admission. ROOT did not integrate that old review as an acceptance proof.

The current fixture changes only threshold/layout from `a2f7a2a9`: strict
captured `a>1.035`, buffer1, box maximum8. AST comparison receives every other
builder statement, original diffusion matrix and reaction equations, physical
coefficient `a=1+.04cos`, initial projections/transfers, numerical controls,
capture/history function, saved-residual guards and parent rollback test body
unchanged. No budget, tolerance, temporal method or original residual is
changed to repair the predicate.

## Actual public source admission

The new independent test calls the actual public builder followed by
`pops.validate`, `pops.resolve`, `resolved.verify` and `emit_cpp_program` for:

| Cells | Components/order | Seed | Guard | Declared realization |
|---|---|---|---|---|
| 16² | scalar 0 | default | no | SpatialBasisJacobi@1 |
| 16² | three, 012 | default | no | SpatialBasisJacobi@1 |
| 32² | three, 201 | declared zero | no | SpatialBasisJacobi@1 |
| 16² | three, 201 | default | yes | SpatialBasisJacobi@1 |

Each immutable resolved plan verifies, admits dimension2, emits one real
composite solve and one explicit Jacobi selector, keeps original hierarchy
authority and staged component publication, and selects ProgramIR9. The seeded
case emits its hierarchy seed; the guarded case retains domain rejection. No
local block inverse or condensation shortcut is substituted. No catalogue,
provider or `pops.resolve` function is monkeypatched. This is the same existing
pure Python resolution/catalogue route as the repository's field codegen tests;
it does not load the installed native extension or execute the emitted C++.

A separate positive receives the builder default `right_preconditioner=None`:
its real resolved/emitted legacy branch remains IR8 without a Jacobi selector.
This preserves the distinct default realization, not legacy N32 convergence.
The exact historical `a2f7a2a9` builder is extracted from its Git blob, executed
against the unchanged real public DSL and passed to public validate/resolve.
It refuses with the exact strict-GT diagnostic before C++ emission. This is
a source-authoring negative, not a synthetic native outcome.

The tests assert source package origin before/after actual admission and reject
any loaded `pops._bootstrap`, `pops._pops` or nested `._pops` module. Source
imports, immutable descriptor construction, resolved plans and C++ strings are
the only execution here; `pops.compile`, bind, run and native selection are not
called.

## Tag geometry and remaining science

An independent analytic cell-average calculation of the original coefficient
at N16/N32 receives symmetric narrow extrema bands meeting the periodic seam.
The one-cell buffered bands still leave an untagged interior, with margin to
threshold greater than `1e-4`. This is not the old central strip, and the test
does not fabricate finest-owner masks or a native hierarchy. The max-box8
clustering choice is actually present in each admitted public layout.

The native clustering result, distributed/empty-rank ownership, true partial
coarse/fine masks, actual Jacobi convergence on N16/N32/permutations and strict
original F residual remain ROOT's future reception. Old N16 accepted files have
zero coarse active cells and remain only constant-reaction observations. The
new constant-target witness cannot qualify nonconstant diffusion/reflux
accuracy. No N32 convergence, native rollback/retry/history/cache/restart,
MPI, GPU or complete AMR claim follows from these source checks.

The unchanged guard declares FixedDt10DT once and uses public pops.run with
requested endpoints on the same runtime. Neither source admission nor the
historical extracted-controller mock proves native parent rollback. Actual
large-interval refusals, subsequent safe intervals with the same declared
controller, exact checkpoint/history/carrier restoration and replay must pass
on the coherently rebuilt installation.

## Results and commands

Independent **9 PASS in 75.85 s**, existing public-admission/subcycling
selection **5 PASS in 70.36 s** (three unrelated tests deselected). Ruff and
diff-check pass. Both commands explicitly remove native dimension selection
and use the source package; no environmental installation or setup is run.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_amr_fixture_admission_independent.py --tb=short
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/python/unit/fields/test_amr_original_field_codegen.py -k 'closed_public or refuses_subcycling' --tb=short
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_amr_fixture_admission_independent.py
rtk git diff --check
```

This receives the corrected public **source admission**, with its exact
realization and unchanged physical body, and closes the identified review gap
at that layer. It is not a native scientific replay or GitHub CI gate. The
separate captured-diffusion @2/IR10 extension is outside this fixture-admission
freeze and keeps its own source/native reception requirements.
