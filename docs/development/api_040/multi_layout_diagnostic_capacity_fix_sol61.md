# Multi-layout compiled diagnostic inventory correction

Baseline is native reception source
135179aa9039101f47e47472979f97434ee2b0c9, old environment SDK2e4/Dim2. The actual
MPI2 nonregression batch `installed-sdk2e4-product-entropy-feedback-nonreg-mpi2-dim2`
closed with 13 cases per rank: six M19 failures and seven entropy/feedback/retry
passes, 1138.74 seconds. Every M19 failure occurs at the initial `_save` checkpoint,
after successful compile, bind, global-state observation and local-box observation.
No initial checkpoint or state NPZ was published; only provenance exists.

The innermost diagnostic is the same on both ranks:
`Program diagnostic checkpoint exceeds its chosen resource capacity; call runtime.configure_checkpoint_diagnostics(capacity_per_rank=larger_bytes) collectively before capture/restart`.
This is not a physical reduce/lift failure, an MPI directory mismatch or a
reason to raise an arbitrary test capacity.

`install_layout_checkpoint_resource_budget` supplies the aggregate InstallPlan
to `_common_budget`. A truly multi-layout artifact deliberately has
`artifact.program=None` but registered `artifact.layout_programs`. The previous
`_compiled_diagnostic_inventory` verified the artifact and then looked only at
the absent scalar Program, yielding unknown inventory/capacity. Capture therefore
refused even the valid empty initial diagnostic image (40 codec bytes).

The correction in `python/pops/runtime/_checkpoint_resource_budget.py` verifies
the same artifact first, then inventories its registered compiled layout Programs
when no scalar Program exists. The union of literal record names is a finite
upper bound for every child diagnostic table. Shared names are counted once.
The existing codec bound remains exactly `40 + sum(16 + UTF8-name-bytes)` per
rank, including the native frontier-duration record. No array size, model name,
M19 dimensions, extra numerical tolerance or fixture constant enters the bound.

Any missing source or nonliteral/dynamic record name in any executable slice
still produces unknown inventory and requires explicit collective configuration.
An external artifact with no executable evidence also remains unknown. Verified
source mutation refuses before recomputing the inventory. Ordinary single-Program
inventory and user-chosen capacity semantics remain unchanged. The existing
`pops.program-diagnostics.checkpoint-capacity@1` resource contract, checkpoint
schema, IR, native headers and ABI are unchanged: this repairs evidence selection,
not the codec or the authority protocol.

The actual failed provenance declares scalar Program null, three layout Programs,
four blocks and three external providers. The corresponding three retained
ProgramCPP files exist in each of the six actual caches and contain no literal
`record_scalar` calls. The real compiler `_compiled_handle` passes `generated_cpp=src`
on both cache/compile paths; keeping a source file is a separate inspection setting.
A read-only calculation on these eighteen real source files gives the previous
inventory None and corrected inventory `pops.frontier.duration`, hence 78 bytes
per rank. This source calculation does not reconstruct an aggregate artifact,
load a DSO, rerun native code or claim successful checkpoints after correction.
Its external report is `/tmp/sol61-m19-diagnostic-capacity-source-comparison.json`,
SHA256 `7a130b454513221de19ce24dc606667f5619b6bf60c78f5d9aff3bb55825d8d2`.

Validation in the private `pops-sol61-layout-diagnostic-capacity` checkout:

```bash
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
 -o pythonpath=python tests/python/unit/runtime/test_multi_layout_checkpoint_diagnostic_capacity.py \
 tests/review/test_sol61_program_diagnostic_checkpoint.py
```

38 source/host tests passed: ten new tests plus 28 existing diagnostic resource,
capture/restart and collective refusal checks. The new tests use genuine typed
aggregate artifacts/verifiers and explicit synthetic transport images; a native
platform-constructor seam is substituted only to avoid native selection. They
cover finite literal union/duplicates, UTF8 names, empty/populated tables on
ranks 0/1/2, codec-derived capacities, absent/dynamic slices, mutated retained
sources, external absence and single-Program compatibility. Synthetic images
are protocol tests, never actual native results. Ruff and diff checks pass.

An additional broad source-only invocation including `test_runtime_planning.py`
produced 47 passes and 18 failures, all requiring an intentionally unselected
native dimension or absent source-package `_pops` export. That run is not claimed
green; it failed before diagnostic-capacity calculation in those cases. No fake
native bootstrap or broad test relaxation was added. ROOT owns authentic rebuilt
native/MPI regression reception and any proof that the corrected initial and
subsequent checkpoints now succeed. No MAIN, native reception checkout, shared
environment, native build or JIT was modified by this worker.
