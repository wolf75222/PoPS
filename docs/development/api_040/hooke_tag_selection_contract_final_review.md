# Independent final TagSelection source review

Actual GPT-6.1 Sol reviewer Hooke; author Galileo. Frozen author08b19cee parent606e2a06. Docs-only separate review worktree. Prior archived native refusal receiptc8610dce and causal reviewf492a1c0 preserved. No production edits, native/JIT/build, package/environment mutation.

Decision: no concrete source blocker found. Candidate dilation is solely cfg.tag_selection_buffer. Resolution keeps authored Buffer separate from nesting.minimum_buffer; derived lookahead remains parent coverage and never expands tag predicate. Coverage combines buffer+lookahead under checked nonnegative/ranked int bounds, then max with actual fine-ghost/coarse-stencil need. Periodic wrapping and output scan-bounds semantics remain unchanged. Physics, threshold, original N8 and strict partial-coverage guard are unchanged; only a rebuilt real run can receive their correction.

Verified complete relay: resolved tagging schema2/selection@1 → LayoutInstallProjection.resolved_tagging → real runtime executor → config lowering requiring actual class descriptors → native pybind ranked field/version → immutable native config → collective initial-materialization contract2 including selection fields → candidate selection → readonly native checkpoint report → strict accepted schema8 preflight. All native declaration/definition/binding/explicit instantiation present. ABI6 and generated products changed; hierarchy native lowering schema3/provider4 date changed nesting semantics. Payload12/POPSCAR1@1 unchanged; old accepted schema7/differentBuffer refused by static preflight before restart transaction.

Bounds: exact nonnegative public scalar Buffer is repeated by spatial rank, width product checked against size_t; native ranked support allows anisotropy and checks neighborhood product before hierarchy allocation. Signed int coordinate extents keep delta/proposed calculations within int64. Transition coverage sum checked before nesting use, and corresponding validation already checks nonnegative fields and signed bound. There is no empirical patch/N cutoff or model dispatch.

Actually executed bounded Source oracles against frozen author checkout:22 passed6.49s, with explicit checkout/python and `_pops` absence asserted. Real public resolution tests Buffer0/1/7 over3levels preserve two nesting transitions((2,2),(2,2))/lookahead(1,1), change selection identity, and lower exact Buffer along dim1/2/3. Tests also reject old config classes, negative/overflow and old/different accepted selection provenance; all hierarchy contract tests including anisotropic facts pass. Config test doubles are explicitly authoring projection probes; they do not establish actual pybind setter execution, native clustering geometry or collective behavior. Native C++ guard test is prepared only. Root's known baseline missing-layout-plan source fixture remains separately owned and must pass in integrated suite.

Executed release generator --check with supported pops Python: passed. Initial `/usr/bin/python3` attempt failed because that interpreter lacks zip(strict=), so that attempt is not a project defect or verification success.

Reproduction:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python - <<'PYCODE'
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"python"))
import pytest
rc=pytest.main(["-q","--tb=short",
 "tests/python/unit/amr/test_public_amr_resolution.py::test_authored_tag_buffer_is_separate_from_multilevel_nesting",
 "tests/python/unit/amr/test_public_amr_resolution.py::test_tag_selection_lowering_refuses_old_config_and_overflow",
 "tests/python/unit/runtime/test_amr_checkpoint_contract.py::test_tag_selection_contract_refused_before_native_restart",
 "tests/python/unit/mesh/amr/test_hierarchy_contract.py"])
assert not any(k=="_pops" or k.endswith("._pops") for k in sys.modules)
raise SystemExit(rc)
PYCODE
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python scripts/generate_release_contract.py --check
```

| Principle | Bounded decision |
|---|---|
| 1.1–1.5 | Authored tag policy preserved, numerical coverage separated through typed contracts |
| 1.6 | Real Source oracles passed; rebuilt original native scientific case required |
| 1.7 | No geometry/performance backend receipt from Source tests |
| 1.8 | Common minimal seam, no per-model/N8 treatment |

Inspected SHA256:

```text
0fa66b115c714ac0d9475c21c646f585ca0528105113b665bc2917c2c3264af2  python/pops/amr/_resolution.py
3a20c74853811ff7dea22a94b18931d6d834cb5e119b1dfc1cfa3e22dca04e9d  python/pops/mesh/_amr/hierarchy_native.py
992e7ec592892d4ed3223abb5a2b38facc78cc6e7259acbf73d5e327aaf2e274  python/pops/runtime/_amr_bind_lowering.py
016176341284ec7d85e3ffa0b2dc200e522df7ec54b640bcdd636f74e0efc0c7  python/pops/runtime/_layout_install_projection.py
19f027babb3de6af98709d794945a0106aace864f95e0cf44493327e9ab6e6f1  python/pops/runtime/_runtime_executor.py
6b8e29ba2592b919bad8704c90b4d64b96dc1318345aadafdcdaa99893c5c398  python/pops/runtime/_amr_checkpoint_contract.py
f7e5b94d554541b1a2b7948a4f2fc2d84a4ea5de999b0e48ae96ed37c966fd2f  include/pops/runtime/amr_system.hpp
d372fa562671830ede3f6ac3873022c67a4728055116738c00bda8103592caa6  src/runtime/amr/amr_system.cpp
e17e8eb677f3566218465dc8f5a59f93a4dfe907280077ea4fb45140297724b6  python/bindings/core/init/init_amr.cpp
1ae0da07a6a8d611f6c011f1c37081195a804947b9a6390b8c0e2932a9a1c35f  schemas/release_contract.v2.json
```
