# Resource companion v1.3: separate preflight and profiling workspaces

The timing v1.2 protocol is unchanged. The resource v1.2 failure is preserved at
`outputs/resource-t2-v1-2-run.log` in the parent workspace. Both metadata workers
completed and created empty `cache/{baseline,candidate}` and
`codegen/{baseline,candidate}`. The first profiling worker then refused to create
the already existing baseline cache. No profiled sample had started.

This version changes only the resource driver orchestration and its result schema
(`pops.api040.joint-reconstruction-resources.v1.3`). Metadata receipts, logs, cache
and codegen now live under `preflight/`; profiling equivalents live under
`profiling/`. The plan records both relative roots. The profiling worker retains
exclusive directory creation, without `exist_ok`; a pre-existing or contaminated
profiling cache is still an error. Both metadata workers finish and agree before
any profiling worker starts. Every execution requires a fresh output directory.
Earlier scripts, timing observations, failed receipts and installed snapshots
remain immutable.

The exact v1.2 timing-script SHA remains pinned. No change was made to the case,
initial state, compiler/native/SDK authentication, two unprofiled warmups, three
profiled fresh instances per lane, twelve-step calendar, public bind/run, profile
boundary, tolerances, counter units, unavailable-counter handling, state equality,
external library comparison or timeout. The worker, profile, invocation, identity
and aggregation function ASTs are checked equal to resource v1.2.

Validation (no performance campaign):

- Six driver/source tests pass. They run the actual driver and both actual invoke
  implementations with real isolated subprocesses, filesystem creation, receipts
  and state files. Only the child payload is a labelled synthetic protocol worker
  and snapshot authentication is substituted with dummy test identities; this is
  orchestration evidence, not a numerical or resource measurement.
- The old v1.2 driver reproduces `FileExistsError` after exactly two metadata
  children. The new driver reaches two metadata then two profiling children,
  four distinct empty-at-entry cache/codegen paths, numerical comparison and
  counter aggregation. Missing scratch counters remain unavailable.
- Adversaries retain a dirty cache unchanged and refuse reuse before a profiling
  child starts; metadata mismatch stops before profiling; state mismatch refuses
  result publication; restarting a completed output launches no child.
- Ruff passes. A plan-only invocation rehashed both real retained snapshots
  (1079/1080 source members) and verified the pinned timing script, without
  importing a solver, JIT or profiling run.

Run the [original resource protocol](joint_reconstruction_resource_probe.md) with
`joint_reconstruction_resource_probe_v1_3.py` and a **new** output directory.
Plan and execute use different new directories. Full results remain root's
responsibility after independent review; these tests provide no comparative
performance conclusion.

Test command:

```sh
env -u PYTHONPATH /path/to/pops-api040/bin/python -m pytest -q -o pythonpath=python \
  tests/python/unit/runtime/test_joint_resource_v1_3_driver.py
```
