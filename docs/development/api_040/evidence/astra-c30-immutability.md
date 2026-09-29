# C30 — Descriptor deletion after public validation

Date: 2026-09-29. Shared checkout `work/PoPS`, branch
`codex/api-040-native-20260928`. Only the two files below were edited for this task;
other workers' edits and the installed environment were preserved.

- `python/pops/_descriptor_protocol.py`
- `tests/python/unit/problem/test_deep_freeze_storage.py`

## Reproduced defect and correction

Public lifecycle reproduction on a scalar transport Case with MUSCL/VanLeer:
`pops.validate(case)` then `pops.resolve(case, layout=...)` produces order 2.
Deleting `method._frozen` succeeded because Descriptor guarded assignment only.
Assigning `method.reconstruction = FirstOrder()` then resolving again silently
produced order 1 without a new validation. The stored `case.snapshot.hash` stayed
unchanged, while the resolved plan identity changed. Deleting public attributes
`method.reconstruction`, `method.positivity_floor` or `numerics.rates` also succeeded
after validation, corrupting frozen declarations rather than rejecting mutation.

Added Descriptor.__delattr__ rejects every attribute deletion after freeze,
including deletion of the lifecycle marker. This fixes the shared production
descriptor mechanism used by numerical methods and plans. It does not introduce
a parallel authority, snapshot, or protocol. Mutable authoring copies continue
to permit edits and deletion before their own validation.

Six new pytest cases use public Case/numerics/validate/resolve/compile APIs:

- Three parameterized attempts to delete frozen method state, followed by stable
  original snapshot/plan identity and successful plan verification.
- Deletion of a frozen numerical-plan family.
- Detached editable copy: change to FirstOrder does not affect the original plan.
- Forced low-level tampering of resolved positivity_floor is refused by
  `pops.compile` before native dimension selection. This independent integrity
  guard already worked; it was verified, not newly implemented.

## Execution evidence

Interpreter: `/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`.
All test invocations used `env -u PYTHONPATH` and inserted repository `python/`
at the start of `sys.path` before import. Observed package source:
`work/PoPS/python/pops/__init__.py`. No native build or package reinstall was run.

Before correction, focused file: **4 failed, 4 passed** (3.50 s).
The four failures are exactly the forbidden deletions above.
Receipt: `outputs/astra-c30-red.xml`.

After correction, focused file: **8 passed** (5.61 s).
Receipt: `outputs/astra-c30-focused-green.xml`.

Broader affected-area run: **228 passed, 1 failed** (42.35 s).
Receipt: `outputs/astra-c30-green.xml` (despite the historical filename, this
receipt is not all-green). Passing groups include deep freeze (8), physics model
freeze (5), Program deep freeze (5), numerical plans (8), resolved operations (17),
and 185 descriptor cases.

The sole failure is
`tests/python/unit/descriptors/test_lib_descriptors.py::test_user_riemann_is_external`.
It imports `_bootstrap` without selecting a native dimension, and raises
`no PoPS native dimension is selected; call pops.compile(resolved_plan) first`.
Rerunning that one case with Descriptor.__delattr__ restored in-process to
object.__delattr__ reproduces the identical failure, before any mutation test.
Receipt: `outputs/astra-c30-native-baseline.xml`.
This is an unresolved limitation of the source-only invocation, not a green
native integration claim. No test was deleted, skipped or weakened.

Focused reproduction command, from repository root (prefix `rtk proxy`):

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0, "python"); import pytest; raise SystemExit(pytest.main(["-q", "tests/python/unit/problem/test_deep_freeze_storage.py", "--junitxml=outputs/astra-c30-focused-green.xml"]))'
```

The broader pytest argument list was the focused file plus:
`tests/python/unit/physics/test_physics_freeze_protocol.py`,
`tests/python/unit/time/test_program_deep_freeze.py`,
`tests/python/unit/numerics/test_discretization_plan.py`,
`tests/python/unit/codegen/test_resolved_operation_pipeline.py`, and
`tests/python/unit/descriptors`.

`git diff --check` passed for both changed files. SHA-256 at validation:

```text
605bcd06d7d020c011f8b660b7efa5985c7c7691363dfb37c03ccfea6f167c8e  python/pops/_descriptor_protocol.py
1bc42f3ae6a1960746ea79aaf0286c0165a19d27e353cf018d2e32b89b7a07f6  tests/python/unit/problem/test_deep_freeze_storage.py
```

## Boundaries

This is concrete ordinary-Python deletion protection plus plan-integrity evidence.
It is not a Python security sandbox: deliberate object.__setattr__, writable
__dict__, or replacement of class methods can bypass Python guards, and must not
be treated as supported mutation APIs. The forced plan-tamper test checks that
compile still refuses changed semantic evidence at its own boundary.

The five existing model-freeze and five Program deep-freeze tests passed, including
stale-storage/detachment and rollback coverage. No comprehensive callable-capture
or arbitrary extension-object proof is claimed. Time expressions, IR application,
root codegen, and version files were not edited. The existing public class in this
checkout is Case; its freeze/validate lifecycle was used for the requested
Problem.freeze investigation. No GitHub CI or native execution was performed here.
