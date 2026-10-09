# Independent generic-core composition review

Reviewer: real GPT-6.1 Sol. Frozen source baseline: `3b2b05098e6bf4f23dccc67b43baf4f1556b0e39`.
Read requirements: handoff `context/PRINCIPES_1.1-1.8.md` and goal-objective attachment `0e1884a0-8812-45f3-ac4e-1f68b82a78e3`.
No production changes, environment setup, native/JIT compilation or numerical backend execution.

| Principle | Decision | Source symbol / witness | Status |
|---|---|---|---|
| 1.1 equations and inferred dependencies | Entropy residual is ordinary symbolic Python; normalized moment plan hides method composition | discrete_entropy.py populations/moments/residual; moment_path_kernel.py emit_moment_path_kernel | M18 architecture acceptable; M14–16 full principles reception blocked |
| 1.2 linear scientific scripts | Not fully assessed by this bounded review | api040_m18_entropy.py residual and solve | No blanket receipt |
| 1.3 model-independent core | Physical transfer is weighted reduction/lift; central moment emitter prescribes Hermite edge/path/flux | physical_support_transfer.hpp apply_physical_support_transfer; moment_path_kernel.py lines 58–87 | M19 architecture acceptable; M14–16 blocked |
| 1.4 Python composition | HyQMOM equations exist as Python algebra; Fan–Li plan leaves recipe to compiler | hyqmom15.py _hyqmom15_polynomial; fan_li.py fan_li15_native_plan | Constitutive algebra accepted; method composition incomplete |
| 1.5 methods extensible | affine operation fixes physical names and slots; numerical skew rotation itself is legitimate | affine_moments.py lines 30–34; program_emit_affine_moments.py lines 26–31 | Basis binding defect remains |
| 1.6 actual genericity witness | Same order-2 basis permuted is rejected; central emitter produces Hermite closure without authored expression | test_generic_core_composition_source_only.py | Actual SOURCE_ONLY execution, no native receipt |
| 1.7 preserved performance | Stable compensated transforms and analytic integration should remain | normalized numerical SDK calls | No performance claim or new measurements |
| 1.8 small stable core | Reuse existing symbolic_path route; expose composition in Python library and explicit basis binding | module_emit_path.py lines 21–23 | Proposal only, no new port before mandatory native AMR closure |

## Minimal next action after mandatory AMR native closure

Compose physical flux, numerical path and regularization products in the Python library through existing public symbolic_path mechanisms. Keep stable recovery, normalized transforms, compensated arithmetic and polynomial integration as explicit mathematical primitives. Give affine push-forward an explicit basis/multi-index binding rather than canonical component names. Compare numerical semantics and cost before replacing any qualified stable path. Generic optimized algorithms are legitimate; a compiler-selected physical recipe is not cured by replacing 15 with N.

## Limits

M18/M19 judgments are architectural inspection, not new CPU, GPU, MPI or AMR evidence. M14–16 full principles reception remains blocked while opaque central composition persists. No production freeze or expansion is requested. The witness intentionally records current behavior; it does not certify the desired architecture or mirror a numerical algorithm.

## Reproduction

From this review worktree:

```sh
env -u PYTHONPATH /usr/bin/python3 tests/review/test_generic_core_composition_source_only.py
```

The module is loaded directly from the checkout, its `python` path is explicit, and `_pops` absence is checked. Stdout identity and inspected-file SHA256 values below belong to the frozen baseline before this tests/docs-only commit.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.026s

OK
{"central_closure": "hermite_raw_edge<2, 3>", "checkout": "/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS-generic-core-review", "equivalent_permuted_basis_refusal": "normalized moment path requires the exact q-outer raw moment ordering", "inspected_sha256": {"include/pops/runtime/dynamic/physical_support_transfer.hpp": "2fdff21763eb66013ec11459c9b67c22988df90fcc6b3d62046bd33db66d3afa", "python/pops/codegen/module_emit_path.py": "cc5a8cd31fffc9bf7d242a2d133f79350fea782b2aca99afde69f98a8d796643", "python/pops/codegen/moment_path_kernel.py": "8a9e813cae071b9665e96cfba930945332853f8731595880d26963542b83bc6f", "python/pops/codegen/program_emit_affine_moments.py": "cea8c0418bd3cf9f50f2275086666b31425028a918de0a61248eb8e14285f699", "python/pops/mesh/native_physical_mapping.py": "a7edac72ff1977825e6c75a0283065913067271a68a7edd669c1c249080d486a", "python/pops/mesh/physical_mapping.py": "93b24cdfac40cfb68d1c6a4cb396cbfb490bfe44cc69415324290e8c828694d2", "python/pops/moments/closures/discrete_entropy.py": "01135de70cdaff721f0aff78296d9e953c929e1ac794bd0f323903c79d3a2e30", "python/pops/moments/closures/hyqmom15.py": "3e19a0ca7e54ca3f3d14503c37e2cb9f0adaa6d349ada25e3c01aa4c0ace9dce", "python/pops/moments/fan_li.py": "136c4acb231bc36487982f92d4751b30dd9c4c8634e07f789e66c4ceb36dea82", "python/pops/moments/model_builder.py": "cb625fa6962c2d9d5a2755cfae493851f1c112492191eb50d120d4edde232d28", "python/pops/time/_program/affine_moments.py": "27e5cc30f994c7eaa3d2ad1b7605dde4972c9e906847a1e458557c2a8afa18f5"}, "native_loaded": false, "python_executable": "/Applications/Xcode.app/Contents/Developer/usr/bin/python3", "python_path": "/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS-generic-core-review/python", "source_head": "3b2b05098e6bf4f23dccc67b43baf4f1556b0e39", "status": "SOURCE_ONLY"}
```
