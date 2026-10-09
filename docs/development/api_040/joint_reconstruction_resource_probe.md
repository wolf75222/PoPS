# Joint reconstruction: separate native resource observation

This is a **second, instrumented phase** after the frozen
[`joint_reconstruction_benchmark.py`](joint_reconstruction_benchmark.py) timing
comparison. It does not alter that protocol or contribute samples to its wall
time ratio. It uses the exact same scalar User reconstruction, five-component
linear advection, 48×48 periodic grid, initial array, `FixedDt=1e-4`, twelve
accepted steps, output checks and numerical tolerances. It compares the
retained installed snapshots `3d06cab` and clean `cf6dace`, never a joint
reconstruction against a scalar one.

The companion authenticates every installed source/header member and native
extension against each snapshot's identity, pins both commits and source
manifest digests, imports each package in its own `python -I` process, checks
doctor/header/ABI and the generated artifact's CPU, float64, MPI size-one
platform. It uses one Python executable and separate JIT/codegen directories,
then compares compiler, SDK, ABI and actually loaded external library images.
Ambient `POPS_INCLUDE` and `POPS_NATIVE_VARIANTS_ROOT` overrides are removed
from both workers, so each resolves only its own authenticated package SDK.
The shared v1 case is imported only after its script SHA-256 matches the pinned
digest. The original benchmark and both installed snapshots are read-only.

Each lane compiles once, runs two unprofiled fresh-instance warmups, then
binds three more fresh instances. For each of those three, the private
`RuntimeInstance._executor` must be a native `System`; only `pops.run` is
enclosed in `System.profile(Profile.Advanced())`. Bind, compilation, initial
conditions and output inspection are outside the profile window. `pops.bind`
and `pops.run` remain the public execution path. The profile session resets
the native counters on entry and snapshots them before disabling them on
exit. This private seam is permitted **only in this observation tool**, and
the identical installed `_system.py`, `_profile.py` and native profiler
interfaces are used in both lanes. Profiling overhead makes these runs
unsuitable for timing comparisons.

The script retains the complete native structured profile snapshot, its
private `PerformanceSummary`, and raw counter and scope mappings for each run.
Selected fields are `scratch_allocs`, `scratch_peak_bytes`, `kernels`,
`kernel_launches`, `mpi_messages`, `mpi_reductions` and `halo_exchanges`.
`scratch_peak_bytes` means the largest **single** scratch buffer reported by
the native profiler; it is not live bytes, total temporary allocation, RSS or
a high-water mark for all buffers. `kernels` counts native Program operations
or batches according to the relevant `count_kernel_` sites; it is not a
universal count of Kokkos launches. `kernel_launches` is kept separate when
the native runtime actually emits it; its current AMR site also counts
instrumented operations or batches rather than every physical Kokkos launch.
Communication counters are reported
only under their native names; this one-rank Uniform case does not qualify
MPI scaling or all communication traffic. Source inspection found no
`scratch_*` or `mpi_*` increment sites in these retained SDKs, so those
fields may remain unavailable. An absent native counter is **unavailable**;
only an explicitly present zero is reported as zero. A metric is comparable
only when present in all three samples of both lanes. Raw scopes are evidence
about instrumentation, not wall-time observations and are not used for the
v1 ratio.

Run a read-only plan, review its pinned identities and budget, then run the
measurement in a *different*, new empty directory after root's build and
first timing campaign finish:

```bash
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I \
  /absolute/path/to/PoPS-numerical-bodies/docs/development/api_040/joint_reconstruction_resource_probe.py \
  --baseline-root /absolute/path/to/outputs/artifacts/3d06cab \
  --candidate-root /absolute/path/to/outputs/artifacts/cf6dace \
  --python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  --output /absolute/path/to/new-empty-resource-plan
```

Repeat with `--execute` and another new empty output directory. The
predeclared budget is 720 seconds per lane, including JIT compilation and
five fresh runs. Timeout kills the worker's process group and preserves logs
and a failed receipt. Both lanes must pass conservation and exact accepted
step/clock checks. Each profiled state must exactly equal an unprofiled
fresh-instance warmup state, and the lanes' final arrays must agree at
`rtol=atol=1e-12` before
the resource result is emitted. Retain both worker JSON files, `.npy` states,
logs, generated code/DSOs, `plan.json` and `result.json`. The result is a
counter observation for these exact artifacts, not a scientific accuracy
claim or a claim that missing instrumentation allocated nothing.
