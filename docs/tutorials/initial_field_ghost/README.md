# Public initial Field → Ghost example

`01_public_initial_field_ghost.py` is a linear module-scope script. It defines the physical problem before setup: zero flux, dc/dt=0, dm/dt=m; phi−(1/8)Δphi=m with homogeneous Neumann data. Initial c=cos(2πx) selects AMR; m=2 is uniform. The mesh is8×8, two levels with ratio2, synchronous accepted Halo, dt=1/64. The physical xmin inflow is `(interior_trace(c), phi+1+time)`; no manually assembled component is required.

Run against a freshly built authenticated installed Dim2 ABI8 SDK, from an empty output directory location:

```sh
env -u PYTHONPATH POPS_EXAMPLE_OUTPUT=/absolute/new/output /absolute/SDK/env/bin/python /absolute/checkout/docs/tutorials/initial_field_ghost/01_public_initial_field_ghost.py
```

The script refuses Source imports outside sys.prefix and non-one-rank launches, selects the installed manifest-authenticated native variant, resolves/compiles/binds, saves initial and accepted checkpoints, restores into a fresh bind and saves a reloaded checkpoint. It compares public valid State arrays, topology and accepted time exactly, then writes actual artifact/component/native hashes and checkpoint pins. It refuses to overwrite prior evidence. MPI collective compilation/persistence is intentionally not provided by this Serial example.

Contracts:

- `inferred-boundary-expression-component@1` creates a content-authenticated source package and infers exact dependencies/bindings; existing component compile/install remains authoritative. Unsupported expressions/supports fail at lowering; missing installed libraries/interfaces fail separately at compile/bind.
- `PointwiseGhostValues@1` reads qualified Field/State dependencies from the actual ghost region at cell centers. It is not a FaceValue conversion or automatic nearest-cell reinterpretation.
- Public `InteriorTrace`/`interior_trace` explicitly reads the primary State's clamped interior trace through request.interior, selecting a named component. A read of one's own output Ghost without an upstream producer is rejected. InteriorTrace's State identity is authenticated without creating a circular ghost dependency.
- Ordinary GhostV1 remains positive-dt. `accepted_initial_ghost@1` is interface11, explicitly declared for initial tick0/stage0/substep0/fraction0/1/dt+0 and the actual physical time origin. It borrows the same prepared state with identical prepare/destroy pointers, enforced by Native.
- `native_interface_extensions@1` declares catalog-exact additional tables in the semantic manifest and native loader expectation. Unknown/version-drift/duplicate interfaces fail closed; the declaration alone does not prove compatibility of arbitrary lifecycles.
- `native-boundary-component-values@1` delegates face values to the exact authenticated binding/manifest/unique region, preserving the AST in the component manifest. Dynamic expressions never enter the BindSchema-only key evaluator.

This commit prepares Source only. SDK12 was being rebuilt at authoring time; no Native execution of this script, Field solve, initial freshness, grown Ghost science, restart Field bit preservation, MPI, GPU or universal boundary support is claimed. The dedicated Native integration fixture `tests/python/integration/amr/test_public_initial_field_ghost.py` records full scientific captures and independently checks the Field/Ghost oracle; its actual receipts, including prior SDK11 bind failure, remain distinct from this example.

Source check (no execution tail, no JIT/SDK mutation):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /absolute/Source/env/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_public_initial_field_ghost_example.py -q --tb=short
```

Worktree setup_env.sh was intentionally not invoked: this task explicitly forbids ENV changes and does not install/build; dependencies are existing read-only Source environment.

Actual Source command used `/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python` with the options above: **2 PASS5.43s**, zero skips. The model-only prefix resolves and passes the real detached bind metadata validator; the execution tail is not evaluated.
