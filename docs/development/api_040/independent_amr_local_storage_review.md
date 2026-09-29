# Independent review: AMR exact local StateStorage

Reviewed source commits: `669f748cb4941dc95f86c397f49d889a1aabe162`,
`9d8cd2d94d2496456cc0b7e68281b91b2245367b`, and fixture correction
`f20f9cd7938543f916b0deb5e799becbe96195d1`. No production files were modified
by this reviewer. Verdict: **favorable within the stated local-storage scope**;
no new production blocker established. Native AMR/MPI execution remains required.

## Authority and propagation

* `_phases.resolve` refuses a block selecting anything other than exactly one
  State before constructing `ResolvedBlock` (lines203–210 in the reviewed tree).
  Therefore the new tuple equality `(subject.qualified_id,)` does not silently
  discard an otherwise admitted multi-State block. Multiple independently selected
  States use distinct qualified Case blocks, including read-only capture blocks;
  all are traversed in `problem._blocks` and `initial_subjects`, not only commits.
* `ResolvedAMRStateStorage` requires an exact ResolvedBlock, `numerics is None`,
  exact `StateStorage`, and that block's exact resolved State identity. It does
  not borrow a numerical plan from a neighboring block. AMR context additionally
  checks Case owner and the subject's assigned layout. Unsupported spatial storage
  fails explicitly instead of acquiring this route by default.
* `_hierarchy` constructs a separate identified stencil requirement from the
  State identity, storage descriptor and dimension. Every axis retains the declared
  ghost depth; local storage contributes no invented reflux requirement. Existing
  numerical rows retain their own stencil/reflux requirements.
* Transfer selects local storage by exact qualified State. Competing numerical
  and storage authorities are refused, and a subject without either authority
  cannot silently obtain a coarse/fine order. The local contract requires order1
  and its exact ghost depth; it does not invent a flux or spatial differential
  operator. The depth3 follow-up verifies coarse/fine `(3,3)` and nesting `(3,3)`.
  This test verifies resolution/selection, not installed depth3 buffer execution.
* Direct typed-value tagging remains valid. A discrete gradient indicator still
  requires a real resolved spatial method and stencil; the new storage authority
  does not manufacture one. ProgramModelGraph and ordinary AMR emission retain
  the actual existing local nonlinear provider.

## Independent fixture findings and closure

The original `_states` gathered level0, then converted/reshaped on root without
convergence before entering the level1 gather. A root-only shape/conversion failure
could therefore strand another rank. Initial patch-box capture had the same
local-exception boundary. The bind lambda also combined context creation and bind
instead of converging between them. The primary review found the same boundaries.

`f20f9cd7` addresses all three: per-level local conversion under `collective_check`,
patch-box capture under `collective_check`, and separate `collective_call` boundaries
for context and bind. Expected rejection gathers every Exception and checks the
serialized `isinstance(RuntimeError)` flag, so subclasses remain accepted and
unexpected families fail on every rank before the next native operation.

`level_valid_mask` reads `spatial_shape` and `patch_boxes` inside `_root_check`.
For these fixtures that is local inspection of an already bound hierarchy:
`AmrSystem::patch_boxes` calls `require_inspectable_hierarchy`, which returns
immediately when the engine exists, then iterates published layout metadata.
The fixtures establish successful collective bind and check level count before
these inspections. This does not authorize root-only lazy engine initialization
on an unbound system, nor repair failures inside native collectives.

## Scientific and native boundary

The reference moves four weighted particles by the declared affine velocity map
and recomputes six moments independently. The initial spatial multiplier is
linear in x; its exact cell average is its cell-center value, stored with x on
the last array axis. Averaging commutes with the affine moment map. The residual
uses a distinct1.1×seed and a captured mapped stage; the committed result is the
mapped measure, not a NumPy answer. Two levels must include partial fine coverage,
so they cannot silently qualify a uniform-only execution. The impossible first
residual is identically1 and tests two refusals with unchanged states, levels,
patches, time and macrostep. No post-processing clipping or threshold change occurs.

Limits: synchronous local Program, no physical face transport/reflux, frozen
regrid policy after bootstrap, no checkpoint/restart or regrid-during-step proof,
no GPU claim. The level-valid checks cover every represented cell but do not
independently qualify global composite flux conservation, which this witness has
no spatial flux to exercise.

Independent execution: `tests/python/unit/amr/test_local_state_storage_amr.py`
**6/6 passed in7.24s**, from the author's source tree using source injection and
an existing native support package. This executes validation/resolution/emission
and pure oracle assertions only. No JIT, library rebuild, installed AMR test or
MPI run was executed by this reviewer.
