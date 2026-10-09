# Independent obligations for global field history storage

This is a source-only design reception on the private review checkout based on
Native `0298d696`, after review-only commits through `a3299cf`. The future
`store_history(..., owner_block=BlockHandle)` implementation is not received:
its author SHA is still required. No native solve, AMR execution, build, JIT,
installation, environment change or author-worktree mutation occurred here.

The independent public witness defines three global unknowns T/z/w with original
equations `-laplacian(q_i)+2*q_i-.25*sum(q_j,j!=i)=load_i`, on a common periodic
Cartesian frame, and three distinct physical blocks. Each block has two state
components. Consumed scalar observations have component width one and no block
or state reference. Thus the observation's physical width is demonstrably
different from a storage block's state width. A bounded CG descriptor is authored
only to construct these source nodes; no solver result or convergence is claimed.
This witness is a generic storage test, not a replacement physical Stage model.

Six source tests pass in `tests/review/test_sol61_global_history_storage_owner.py`:
three actual public consumed observations preserve their global provenance,
width, point and missing storage owner, then the authentic AMR checkpoint-shape
emitter refuses them; foreign Case state qualification is refused despite equal
names; an equal detached BlockHandle cannot issue a state port; and a mutable
observation alias cannot select another solved unknown. Those are baseline
authority tests, not qualification of the future owner_block extension.

## Existing seams inspected read-only

- `python/pops/time/_program/history.py`: field_component and field_gradient
  storage obtains width from their consumed-field contracts. They do not assign
  a physical block to global observations. The scalar store node currently
  inherits value.block, which is None for a global field.
- `python/pops/codegen/program_emit_control.py`: the AMR prelude refuses a
  missing `_history_blocks` owner before registration; the store call has another
  owner check. `program_emit_amr.py` repeats this requirement in POPSAND4 shape
  metadata. All paths must consume the same new authority.
- `include/pops/runtime/program/amr_program_context_history_checkpoint_public.inc`:
  registration derives each level's layout/distribution/rank/ghost geometry from
  the selected block prototype, while components come from the history width.
  These roles must remain distinct.
- `amr_program_context_history_checkpoint_services.inc`: history mutation stages
  a candidate, checks the exact field contract and interval, converges its
  descriptor before copying/publication, and preserves rollback ring images.
  The current public `require_history_owner_(program_owner)` only checks that
  sys_block resolves a nonnegative block; the new port must also authenticate
  the particular registered history's owner, not any other valid block.

## Required future contract and discriminating attacks

The supplied block owns **storage**, not the physical FieldProblem, equation,
observation unknown or logical clock. The extension must not assign
observed.block/state_ref, add `0*Q+T`, alter the original coupled F/JVP, substitute
an update/state read for the observation, or derive width from block nvars.

The future reception will exercise these distinctions separately:

| Attack or control | Required consequence |
|---|---|
| Scalar global observation stored on an exact declared co-located block | Width1 ring, correct per-level geometry, physical global provenance unchanged |
| Global gradient stored on a block with unrelated state width | Gradient dimension/components preserved, not block nvars |
| Same names/canonical metadata from another Case or detached/resealed BlockHandle | Refuse live authority before registry/history mutation |
| Valid different block on the same layout substituted for the registered ring owner | Refuse exact ring-owner mismatch, including native store/read fault seam |
| Owner on another layout/frame/domain or incompatible active-level shape/ghost/distribution | Refuse before copying/publishing, converge rank-local errors on prepared lane |
| Value at another clock/point/window, mutable alias, wrong observation component | Refuse original field/temporal authority; no relabelled history cursor |
| Storage-only owner with no prior Program state/update port | Explicitly authenticate and register its resource route, or precise refusal; never fallback block0 |
| Same history name rebound to a different owner/component/space/clock | Refuse rather than merge or mutate an existing ring |
| Invalid rank-local geometry/owner and empty peer | Collective refusal/rollback or valid empty-rank participation, not early rank-local escape |
| Frozen node/descriptor owner/policy/input/body changed and re-digested | Recheck authentic issued binding, not metadata equality as authority |

Live Case/registry authority must be authenticated before detachment. Any frozen
proof must bind the exact detached node/block/space/clock and consumed-field
inputs without retaining a hidden Case or reminting authority from mutable
metadata. History-only owners must enter `_block_indices`, emission/resource
plans and checkpoint budgets explicitly. The native owner index must agree with
the registered ring descriptor and physical provider topology at every level.

A new versioned storage port must promote IR/request/manifest only for actual
users of the extension. Omitted owner_block must preserve the old unowned AMR
refusal and exact old graph/CPP/module/manifest bytes. The physical FieldProblem
contract must remain global. Acceptance/regrid/checkpoint/restart authority must
retain width/owner/state-space/clock/level descriptors and accepted publication
points; owner changes cannot be silently adopted on restart.

These obligations were sent to the author. Testing the future route, its
conditional version and legacy byte parities awaits the stable SHA. True native
AMR levels, shared/empty MPI ownership, history/rollback/regrid/checkpoint/replay
and saved-state preservation remain ROOT's later reception responsibilities.

Source command (no installed-PoPS/native invocation):

```
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/review/test_sol61_global_history_storage_owner.py -q -p no:cacheprovider
```

Ruff passes. The new tests/docs are independent review artifacts; no production
API is implemented or marked received by this baseline.
