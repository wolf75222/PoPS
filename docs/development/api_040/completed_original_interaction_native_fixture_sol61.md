# Prepared IR19 installed-native fixture (not executed)

Baseline: `a19462650cd01f76e0ac5f0a1611851b405a32a7`. Three new files only;
production, MAIN, reception checkout, SDK and ENV are unchanged. ROOT owns
compilation and Serial/MPI execution. This is an observation of a completed
original solve, not a nonlocal term inside its Newton residual or a full M26 PDE.

## Physical problem and realization

The fixture wraps the actual `tests/python/support/evolved_stage_amr.py` builder
without editing its equations, seven Newton controls, load, initial conditions,
partial two-level AMR, ratio two, transfer, constraint, temporal tau or acceptance.
The original equation is `F=Q(T)-tau*(div(D grad T)+forcing)-Qn=0`.
Scalar `Q=T+T^2`; coupled `Q0=T0+T0^2+0.1*T1^2`,
`Q1=T1+T1^2+0.2*T0*T1`, with `z=0.25*T0+0.5*T1`.
The evolved partition is `T1,T0`, the original unknown tuple remains `T0,T1,z`.
The signed nonsymmetric diffusion matrix is retained, `[[.012,.002],[-.001,.014]]`.
Temperature is homogeneous in this bounded witness: it does not qualify a
nonzero composite spatial flux or nonlinear coarse restriction. Partial AMR is
still required and the interaction output varies in space.

Two cases at N8 select the real scalar `T0` or original tuple component one `T1`.
They reuse the **same issued field_component** stored by the original history,
after genuine `solve_spatial_field` and consumed Outcome. Source block/space/
State reference stay absent. `owner_block=Q0` is only the issued allocation route;
Q0 is never relabeled as T. Source proof, original tuple component, owner and
point are retained in the actual IR19 program and native snapshot barrier.

The new observation is

`I(x)=sum_active_source_cells [(.25+x0*y1-2*y0+.5*x1)*T(y)*V(y)]`.

It selects `CellVolumeMeasure`, `CellMidpoint`, `DirectSpatialInteraction` with
32 MiB explicit workspace. The original full-field solve still explicitly selects
`FullResidualBasisLU@1`, 256 MiB dense budget, unchanged FD step and controls.
No SPD assumption or per-model production dispatch is introduced.

Four **native** reductions, raw sum, absolute sum, min and max over the active
composite output quotient, are recorded by the actual Program. The reference
evaluates the written kernel directly in NumPy from saved native temperature,
active masks and native geometry volumes. It never calls a PoPS field action,
source emitter, solve or interaction helper as its expected operator.
The original `3e-8` guard is used as `error <=3e-8*max(1,abs(reference))`, fixed
before execution. Original F, Q projection, constraints and Q conservation retain
the original independent guards; prescribed forcing remains bit-exact.

## What is saved and what remains missing

Five phases are required: initial, accepted, continuous, reloaded and replay.
Every phase has a separate actual checkpoint, hashed and authenticated against
its live creator immediately after capture, before any observation is written.
Checkpoints and observation paths are disjoint; final hashes must match the
initial seals. Continuous/replay compare every payload byte and manifest field
except their two legitimately distinct, independently authenticated run/restart
identity leaves, using the existing exact Stage comparator.

Native Q/forcing arrays, global T/z history, both raw history slots, POPSHID1
sample identities/durations, diagnostic original F and the four interaction
results, clock, carrier inventory, epoch, rank-local auxiliary blobs and state
pieces with their original ownership/replica metadata are archived. The binary
arrays/blobs use the received `pops.spatial-interaction-fixture-array-wire@1`
typed wire, and observation NPZs are reloaded and compared byte-for-byte.
Replicas are preserved as replicas; rank zero writing is not physical ownership.

Native geometry snapshots retain boxes, valid cells, coverage and volumes.
Cell centers are explicitly **derived** from the bound Cartesian metadata; they
are not advertised as an unavailable coordinate getter. Original provider EB
refusal and the non-EB plan give the null-mask volume-fraction authority; no
fabricated kappa array is called a native getter. Native `spatial_shape` and
partial two-level masks must agree with the declared N8 layout.

There is no current Python getter for the native I array, nor a completed-source
`store_history(I)` port. The saved `I_COMPUTED_REFERENCE` arrays are plainly
references; only the four scalar reductions are actual outputs. These do not
qualify every output cell or a missing array/history/gradient consumer. Those
remain real production obligations. Initial T is unavailable before the first
global solve and is never fabricated. Reloaded history/diagnostics are durable;
the private IR19 snapshot itself must be newly minted by the replay solve.

Receipts retain actual binary DSOs, sidecar hashes, same-component exported
CPP/IR, compiler command, program hash, platform/native module identities,
canonical InitialConditionPlan Handle bindings and fixture source hashes.
They do not manufacture ROOT approval seals or certify the live execution before
ROOT runs it. Authentic declared Analytic ICs are used, with no indexed array
initialization bypass.

## Checks performed and commands for ROOT

SOURCE_ONLY: both genuine Case/validate/resolve/emit routes pass, original solve
attributes remain identical to the helper, IR19 is emitted, FullLU and four native
reduction ports appear in both generated routes, and all initial subjects are
canonical Handles. A synthetic **host-only** two-level midpoint test distinguishes
T from nonlinear Q, the transposed kernel and signed values, and exercises typed
wire/NPZ roundtrip. Its synthetic arrays are not a native receipt or evidence.
Canonical receipt/InitialConditionPlan JSON was checked on the coupled source.
No generated CPP compilation, native run, SDK build, JIT or MPI was performed.

Source command (own private tree only):

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/python/integration/runtime/test_public_completed_original_interaction.py \
  -k 'source_uses or reference_kernel'
```

ROOT's installed campaign has two exact native nodes:

```text
tests/python/integration/runtime/test_public_completed_original_interaction.py::test_public_completed_original_interaction_exact_restart_and_saved_reductions[scalar]
tests/python/integration/runtime/test_public_completed_original_interaction.py::test_public_completed_original_interaction_exact_restart_and_saved_reductions[coupled-partition-permuted]
```

Use ROOT's authenticated new installed SDK runner, once Serial and once MPI2;
do not run against the previous package without IR19 headers. Record JUnit
`completed_original_interaction_receipt`, dimension, rank, size, width and artifact
identity. The fixture's collective wrappers agree compilation, paths, errors and
checks before any next collective. Independent M26 source/science readers and
external ROOT source/runtime/launch seals are still required for reception.
