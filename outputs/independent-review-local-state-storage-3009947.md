# Independent review of local State storage carrier (3009947)

Scope: source-only review of `3009947` in `PoPS-principal-group`. No native package was rebuilt or run. The installed T3/H05 failures remain the central reception's evidence, not a success claim of this review.

## Reproduced blocker

A framed, purely local two-StateSpace model reaches the new helper with a **string** selected state name. `_module_to_model` resolves that name, but `lower_and_validate` passes the original string to `prepare_local_state_storage_carrier`, which reads `.frame` from it. The new two-fluid local path therefore fails before C++ emission:

```python
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian
from pops.codegen.module_lowering import lower_and_validate

frame = CartesianDomain("d", (0.0,), (1.0,)).frame(Cartesian(1))
model = pops.Model("two_local", frame=frame)
model.species("electrons", state=["ne"])
model.species("ions", state=["ni"])
lower_and_validate(model, state_space="electrons")
```

With source Python injected explicitly from this checkout, this raises `AttributeError: 'str' object has no attribute 'frame'` at `state_storage_lowering.py:26`. This is a production compiler entry: the existing multi-state selection tests call `lower_and_validate(..., state_space=state_name)` with a string, and `_module_to_model` accepts that exact form. The new heterogeneous test can avoid the defect when an earlier source carrier already populated `_program_only_storage_axes`.

Resolve the selected name through the source Module's `state_spaces()` registry, require it to be an exact declared key, and pass the resulting StateSpace value to the local-carrier qualification. A regression should select A, B, A in one model, check the emitted row arity and identity each time, and reject an unknown name. It should cover a **flux-free, source-free** model so the new helper cannot short-circuit through an existing storage authority.

## Source invariants observed

The single-state path uses the authored `Model.frame` and checks `StateSpace.frame`, `layout`, `centering`, and `storage` before assigning canonical x[/y[/z]] axes. No flux, eigensystem, or wave speed is inserted. `_ranked_axes` refuses simultaneous state-only axes and physical flux. The complete loader-emission tests exercise System and AMR declarations in Dim1/2/3 and check `program_only_storage`, row dimension, no emitted flux/speed declarations, and an unchanged Module hash. The native System/AMR builders have storage-only branches, but source generation does not prove their runtime acceptance.

The authored component tuple feeds `conservative_vars(*state.components)` and the generated brick uses its `n_vars` as `StateVec<n_vars>`/`Schema::nvars`; the new helper does not independently certify `value_shape` or `support`. Those metadata remain subject to the existing type/Module authorities. This review found no concrete shape-aliasing counterexample and makes no new shape guarantee beyond ordered component count.

The helper returns on an applicable grid operator so it does not replace that operator with a storage-only route. It also leaves an unframed state without axes, preserving the existing emission failure. The tests in 3009947 do not compile their generated C++ or execute installed System/AMR cases; central native reception is required.

## Follow-up reviewed: 934206a

The follow-up resolves a string selection through `module.state_spaces()` and admits an object only when it is the registry's actual `StateSpace` instance. It adds a frozen two-species model, emits complete loaders for A, B, A in each of Dim1/2/3, checks row widths 1/2/1, immutable Module hash, unchanged authoring emitter, and byte-identical repeated A emission. It also rejects an unknown name and a structurally equal foreign descriptor. This is a direct fix for the reproduced blocker. No additional source blocker was found in this three-file follow-up. The author reports 20/20 affected source tests; this reviewer read the final diff but did not rerun tests during the central performance window. Installed T3/H05 and AMR/System native acceptance remain pending central rebuild and replay.
