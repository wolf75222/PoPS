# Independent moving initial-reference reception

Reviewed production SHA: `470dfb08` (parent `5c8ff509`). The correction is confined to `_layout_mesh`: established AMR runtime-data projection first, then exact `NormalizedGeometry` projection for composed descriptors, then the historical mesh fallback. This review changes tests and this report only.

**69 source tests pass**, including 18 independent tests in `tests/review/test_sol61_moving_initial_reference.py`, the five author tests, and the 46 existing bind-validation tests. Ruff and whitespace checks pass. No production contradiction was demonstrated.

The independent fixture declares the physical transport model before its Program. It uses real `Case`, `Program`, periodic `FiniteVolume`, `GeometryEvolution`, `MovingControlVolumes`, `Reynolds` update, typed `InitialCondition`/`BindArray`, and real public `validate`/`resolve`. Single-layout storage uses 13 reference cells; the multi-layout fixture assigns two moving descriptors with 13 and 7 cells through the real `LayoutPlanBuilder` and `ResolvedRuntimeLayouts`. The same exact arrays pass the actual typed initial-storage preflight after canonicalization through the real `InitialConditionPlan`. Swapping arrays between blocks refuses both, and a foreign independently declared Handle refuses even under equal semantic names. The original parent `_layout_mesh` is extracted intact from Git and evaluated with the true validation module globals; it refuses the same honest arrays on both single and multi-layout fixtures before any native problem is loaded.

The storage probes preserve component-major order, native axis reversal in ranks 1/2/3, exact declared precision, complete typed state, and exclusion of halo-sized payloads. A malformed normalized duck cannot fall back to otherwise correct `.mesh` metadata, and a subclass of `NormalizedGeometry` is refused by the exact-type gate. The existing AMR runtime-data branch is tested with a normalized method that would raise if called, confirming its precedence. Invalid geometry rank consistency, nonpositive counts and reversed bounds refuse at the immutable geometry constructor.

`NormalizedGeometry` is intentionally rank-generic; a rank-four reference geometry is legal. Native realization remains bounded by the distinct `NativeSpatialLayout` contract, which independently rejects rank four and accepts only dimensions 1/2/3. The new storage projection does not qualify new native dimensions or evolved measures. These tests make that distinction explicit rather than asserting an undocumented ceiling on the reference contract.

The preflight consumes host manifest/argument metadata, with real registry-qualified Handles and actual resolved layouts. It does not manufacture an authenticated compiled artifact, run public native installation, execute a Kokkos/MPI kernel, or claim moving-metric scientific accuracy. The root-owned native ALE campaign remains separate evidence.

Reproduction from the private source checkout:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0,"python"); import pops,pytest; print(pops.__file__); raise SystemExit(pytest.main(["tests/review/test_sol61_moving_initial_reference.py","tests/python/unit/runtime/test_initial_reference_geometry.py","tests/python/unit/runtime/test_bind_validation.py","-q","-p","no:cacheprovider"]))'
```

No native installation, JIT, shared environment mutation or production edit occurred in this review.
