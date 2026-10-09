# Initial accepted checkpoint strategy - 30 September 2026

Base: `5c361200bbfd324ceeefd604446f2bfb55bfa31e`, exclusive worktree
`PoPS-sol61-initial-checkpoint`, branch `codex/api040-sol61-initial-checkpoint`.

The installed Uniform/AMR Program previously bound its clock schedule but left
`TemporalRestartState.strategy` empty until the first run. Consequently a public
initial checkpoint refused a genuinely declared `FixedDt` before any evolution.

Both installation tails now pass the authenticated authored strategy to
`configure_program`, before Uniform runtime publication. Configuration validates
an isolated candidate and publishes its schedule/controller together. The generic
provider contract `pops.step-strategy.initial-runtime-controls@1` supplies
`initial_runtime_controls()`: `{}` is the default; `None` explicitly means that
caller-supplied runtime controls are required. `ExternalTimeGrid` uses `None` and
does not invent grid coordinates. Its initial checkpoint gives a precise missing
runtime-controls refusal. An undeclared Program can still bind and cannot invent
a checkpointable controller.

FixedDt and AdaptiveCFL initial envelopes contain their exact descriptor/default
controls. ErrorControlledDt also retains the existing initial proposal queue.
Checkpoint remains read-only and schema 2 is unchanged. Rebinding a restored
state authenticates the declaration without replacing its controls, queued
decision or `_restored_pending` obligation; the exact next attempt still validates
those controls. Prepared runs cannot replace the installed strategy declaration.
Composite leaves require identical initial declarations, including pending ones.

## Evidence and limits

Authenticated source import: this worktree's `python/pops/__init__.py`; interpreter
`/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`. No installation,
native build, SDK/header change or shared-environment mutation occurred.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/python/unit/runtime/test_initial_checkpoint_strategy.py \
  tests/python/unit/time/test_step_transaction_contract.py -q \
  --junitxml=outputs/initial-checkpoint-source-20260930.xml
```

Result: **23 passed**, no skips, 0.80 s. Six parameterized tests execute the real
Uniform/AMR installation tails with frozen authored Programs and a source storage
probe; they verify initial checkpoint, exact serialized controller state, queued
ErrorControlled proposal and restored rebind. Additional cases cover undeclared
Program installation, composite leaves, required external controls, read-only
checkpoint, custom restored controls, wrong/omitted declaration, invalid default
controls and unknown provider protocol. ErrorControlled Programs carry a real
lowered ERROR_ESTIMATE guard. Existing transaction authoring tests also pass.

An attempted source-only run of `test_step_strategy.py` and
`test_temporal_restart_state.py` stopped during collection: their native imports
require a selected PoPS dimension. This is not a passing regression claim; output
is in `outputs/initial-checkpoint-runtime-regression-20260930.xml`.

Native initial Integral/ND checkpoint-and-restart receptions remain for the
parent's installed Dim2 replay after integration. These checks qualify source
lifecycle/envelope behavior, not native field restoration, MPI/GPU execution,
ABI compatibility or the scientific PDE trajectory.
