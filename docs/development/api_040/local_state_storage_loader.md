# Storage packages for flux-free local States

The installed Dim2 receipt `installed-5d11268-dim2-face-products` rejected all
four LocalResidual-product cases and four H05 cases before JIT: the compiler
asked an empty physical flux for its Cartesian rank. Source-only rate storage
already existed, but an original local residual need not declare any PDE rate.

The private compiler carrier now obtains storage axes from the Model's authored
typed Frame, checked against the selected StateSpace's exact frame, cell
centering, cell layout and MultiFab storage. An applicable grid operator keeps
its own transport realization; this fallback cannot hide a missing physical
flux. An unframed State still fails instead of borrowing the installed native
dimension. Multi-state lowering applies the same admission to the selected
StateSpace and the original facade's Frame.

No flux, wave speed, eigenvalue or zero transport law is inserted. The existing
`program_only_storage` native route supplies state storage. The public Model
and its canonical Module hash remain unchanged.

`test_local_state_storage_loader.py` exercises complete native-loader source
generation, including the actual System/AMR install entry point, in dimensions
1/2/3 and the resolved heterogeneous 2+3 LocalResidual case in both block orders.
Its initial source run failed at `emit_cpp_brick: call set_flux(...) first`.
After this fix the nine cases pass; together with the existing source-storage
and joint-reconstruction-storage tests, **16/16 pass in 10.58 seconds**.

Independent review found an additional genuine boundary error in `3009947`:
multi-StateSpace lowering passes the selection as a name, whereas the new
helper initially treated it as an object. The follow-up resolves that name
through the exact Module registry. An object selection must be that registry's
actual StateSpace, not a structurally equal foreign descriptor. The new
regression reproduced `AttributeError: str.frame` before the repair. Afterward,
complete loaders for electrons/ions/electrons in dimensions 1/2/3 preserve the
frozen Module hash and produce widths 1/2/1; the two electron sources match
exactly. Unknown names and foreign descriptors are rejected without publishing
storage axes. The expanded affected suite passes **20/20 in 10.19 seconds**.

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python -m pytest -q --tb=short -o pythonpath=python tests/python/unit/codegen/test_local_state_storage_loader.py tests/python/unit/codegen/test_source_state_storage.py tests/python/unit/codegen/test_user_joint_reconstruction_storage.py
```

This receipt covers complete source emission, not C++ compilation or installed
execution. The authentic Dim2 T3/H05 failures remain until the central package
rebuild and replay. No AMR local-residual scientific acceptance is inferred.
