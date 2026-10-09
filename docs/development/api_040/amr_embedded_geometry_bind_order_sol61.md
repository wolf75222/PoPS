# AMR embedded geometry at the authenticated pre-materialization boundary

The actual nonlocal serial campaign at source cfd5b01e, Python metadata378 and
SDK9f57 failed its partial-AMR bind before the spatial interaction producer:
`AMR native package requires a pre-staged RuntimeInstance lane`. The other
history-store failures are a separate correction. No scientific positive result
is inferred from this bind refusal.

At baseline dbc9e1cd, `_install_adaptive_native_engine` called
`install_embedded_boundary` before `install_runtime_authorities`. The latter
stages the native assembly lane from the already-authenticated InstallPlan's
ExecutionContext. Native `AmrSystem::set_analytic_level_set` requires that owned
lane, an installed exact generated block provider, and an unmaterialized
hierarchy. Moving only lane creation earlier would therefore reach the next
refusal: the prepared block set is still empty.

The corrected sequence is:

1. Validate/select the original InstallPlan or its exact registered local AMR projection.
2. Construct the engine, bind its exact checkpoint geometry and retain the plan's ExecutionContext.
3. Install the existing runtime/boundary authorities, staging their genuine collective assembly lane.
4. Inside `_AmrSystemInstall._install_compiled`, validate the original install inputs and install every exact generated block through the existing package route.
5. Consume the original selected normalized layout's embedded geometry after those blocks, before field-method runtime installation, auxiliary values, initial bootstrap sources and `_finish_program_install`/hierarchy materialization.

No ExecutionContext, communicator, package, geometry or protocol is inferred
from a live object or manufactured as fallback. The selected normalized layout
is taken from the authenticated artifact; a local AMR projection must contain
exactly one layout, and the geometry object is passed unchanged. Uniform retains
its prior geometry construction order. Low-level installs without an InstallPlan
retain their explicit geometry-authoring responsibility. No C++ header, ABI,
IR, law or native guard changes.

The existing native assembly lane remains the collective authority for EB
preparation, allocation, finite/geometry validation and exact MPI consensus.
Its guards for a missing lane, absent provider and already built hierarchy remain
unchanged. A bind failure leaves no published RuntimeInstance; this patch does
not relax native rollback or package installation checks.

Validation is SOURCE_ONLY. The stdlib host reception executes exact statement
slices of both private installers and substitutes explicitly named authority,
configuration and native lifecycle adapters. It creates no CompiledModel or
native package and does not qualify DSO, MPI or ExecutionContext authentication.
Six checks receive the baseline lane refusal, candidate lane→block→geometry→
materialization ordering, unchanged caller resource/layout handoff, refusal
before geometry on authority failure, exact one-layout selection, and the three
actual current native lifecycle guards. The public source unit also checks the
new adaptive ordering and unchanged Uniform ordering.

```sh
env -u PYTHONPATH /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -B \
  -m unittest discover -s tests/review \
  -p test_sol61_amr_embedded_geometry_order.py -v
```

Result: six PASS, no PoPS import, JIT, native execution, compilation or environment
mutation. Ruff and diff checks pass. Root must separately execute the public
source unit `tests/python/unit/runtime/test_public_embedded_boundary_lowering.py`
and rerun the actual installed partial-AMR witness after rebuilding the Python
package. MPI numerical/empty-rank execution remains pending; the existing typed
world/resource validation and native collective code are unmodified.
