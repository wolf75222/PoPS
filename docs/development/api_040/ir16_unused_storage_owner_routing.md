# IR16 declared storage-only owner routing - source receipt

Base: `378f29084c915ae316308481949ea174858197b2` (including the ea13 descriptor alignment). Production correction: `a02e151d4f6c393d1193f442c1a9eb173a825fd8` (one Python file, nine added lines). This is a correction to the existing `pops.program.global-field-history-storage@1` promise, not a new schema or physical evolution rule. Root's Native/SDK/environment are untouched.

## Defect and correction

The authentic public `_storage_only(5)` witness declares a real fourth Case block and its TimeState but never reads its State.n. A global FieldProblem observation has `block=state_ref=space=None`; its scalar history explicitly selects that real storage owner. On the base, `_block_indices()` omits the owner and AMR checkpoint metadata rejects it. The new independent probe first failed at the missing-owner assertion (1 FAIL, 2.21 s). No fake State read or commit remedies this contract.

`serialization._block_indices()` now preserves the existing State-read and readonly-dt routes, authenticates all originally issued global-history declarations, and then appends missing owners in sorted history-name order. Two rings sharing one owner add one route. Other unused declared TimeStates remain absent. Existing owner indices are unchanged. The scan uses the existing immutable issuance validator; route construction changes neither SSA nodes, commits nor declaration counters. Its additional work is a closure scan plus one lookup per issued ring, with no Native allocation or communication.

## Actual source admission

The new tests use real Case/StateSpace/TimeState/FieldProblem declarations. Uniform and AMR `validate → resolve → ProgramModelGraph → emit_cpp_program` materialize four blocks and four real InitialConditionPlan bindings, while keeping the storage-only width-five State unread and uncommitted. The emitted program block-name table maps `storage-only` to index 3. Its IR16 ring remains scalar width one, uses the original clock/point/descriptor, and registers under that exact owner. Only the three pre-existing physical source States are committed in the completed execution graph. No Native catalogue monkeypatch or runtime substitute is used.

The independent source checks also refuse an altered descriptor (including bool width), owner table, ring width and removed store before returning an owner route. Freeze, rebuild, detach and graph boundaries preserve the hash, scalar descriptor and unread route. The authentic Native capacity comparison and metadata from ea13 remain unchanged.

## Legacy images

Four real Uniform/AMR profiles, with no IR16 ring or an already read IR16 owner, were run against the clean base checkout and the correction. Full serialized IR **including provenance**, full emitted C++, and program hashes are exact in every profile. Authoring helpers were imported from the same absolute source path in both processes: comparing different helper paths initially changes provenance JSON while leaving C++ and semantic program hashes identical; that is not a production delta. The common-origin four-image receipt SHA256 is `a3da9ed761527ed005f1d86a5492b381053ca72df70ef2d4ed92049e5b734727`.

Actual receipts (outside the repository):

- `/Users/romaindespoulain/dev/tmp/sol61-storage-owner-parent-common-origin-images.json`
- `/Users/romaindespoulain/dev/tmp/sol61-storage-owner-fixed-images.json`

## Reproduction and limits

From the fixed checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_storage_only_owner_route.py tests/review/test_sol61_amr_stage_checkpoint_capacity.py
```

The initial new suite received 12 SOURCE_ONLY PASS in 69.57 s. The coherent replay also checks the exact emitted block-name table and the ten prior ea13 capacity checks; **22 SOURCE/HOST PASS in 220.77 s**, including the prior small authentic C++ capacity-guard extraction. Ruff and `git diff --check` pass. No SDK or heavy translation unit was built.

For full legacy receipts, run the new test file as a script in each checkout, using `PYTHONPATH=python:/Users/romaindespoulain/dev/tmp/PoPS-sol61-storage-owner-map` to keep the helper's provenance identical. The unchanged file's `legacy_receipts()` hashes complete images rather than a selected IR fragment.

No Native bind/run, actual MPI, regrid execution, restart execution or multilevel numerical qualification is claimed. Root receives those after independent review. This patch does not change layouts, physical initial conditions, transfer rules, temporal publication, C++ capacity guards, IR ordinals or the separate IR19 authoring router.
