# Explicit tag selection and derived parent coverage — source correction

Base606e2a06; independent author worktree PoPS-sol61-tag-selection-contract. Production change, not a new Native receipt. The earlier genuine SDK a37c run demonstrated complete fine coverage at N8 despite authored Buffer0; ROOT traced the correct mesh_marker component/threshold and transition buffer2/lookahead1. [Independent causal review](hooke_tag_buffer_nesting_causal_review.md) establishes that both latter fields are derived nesting requirements, not authored tag anticipation. No physical equation, threshold, mesh, clipping, acceptance guard or example is changed here.

`ResolvedTaggingAuthority` schema2 authenticates `pops.amr.tag-selection@1` and authored Buffer separately. The resolved hierarchy transition buffer is now only nesting.minimum_buffer; larger authored Buffer cannot inflate that requirement. The shared hierarchy native provider is version4 and its lowered data schema3, explicitly dating the change of transition semantics. The tag graph itself remains independently authenticated.

Native `AmrSystemConfig` carries tag_selection_contract_version1 and tag_selection_buffer: one ranked extent for the global authored selection policy, applied unchanged at every transition. Public Buffer remains scalar/isotropic; its lowering repeats the exact scalar along the selected spatial rank. Native ranked extents retain anisotropic support for low-level providers. Bind consumes the exact resolved tagging authority through LayoutInstallProjection and refuses an old config class lacking the native descriptors; a Python-only dynamic attribute cannot stand in for native storage.

`prepare_cluster_shards` dilates candidates only by tag_selection_buffer. Parent nesting coverage uses transition_buffer+transition_lookahead, with checked signed-coordinate bounds, then max with the existing actual stencil requirement ceil(fine_ghost/ratio)+source_radius. This policy consumes each typed minimum separately. It does not add source_radius or fine_ghost a second time to the already derived minima, and does not call lookahead an authored future forecast. Both derived minima constrain parent support; neither expands the tag mask. Berger nesting_covered remains a temporary parent-required-box proof; clustering still emits its candidate scan bounds. The hierarchy-resolution minimum guards are unchanged.

Config validates selection version, each non-negative signed-coordinate extent and the entire neighborhood product before hierarchy allocation. The Python isotropic lowering rejects the corresponding signed-width and size_t overflow before assigning a config. Native tests are prepared for old version0, negative/extreme ranked buffers and Dim3 neighborhood overflow. No new target or compiler setup is introduced.

Collective initial-materialization contract version2 authenticates tag version/extent along with all existing ranked transition facts. Native readonly checkpoint_tag_selection_contract exposes version, selection extent and each transition's ratio/buffer/lookahead. Strict AMR accepted-state schema8 includes this report among static preflight fields; old schema7/missing report or different Buffer is refused before beginning the restart transaction. The release Native ABI becomes6 because the public native config shape/interface changed; generated Python/C++ products are updated with the repository generator. ROOT must rebuild rather than reuse SDK a37c.

AMR payload12 and POPSCAR1/state-carriers@1 are unchanged: their full-grown byte transport meaning did not change. A payload12 archive from the old selection composition remains authentic historical evidence in its preserved compatible environment, but its weaker accepted schema7 cannot be replayed as a new selection-contract reception. No relaxed hash comparator or compatibility fallback is added.

Source checks actually performed without importing the native extension:

- Four affected source-unit files:147 passed, one explicitly deselected baseline-defective source fixture,22.08s. They cover Buffer0/1/7 independent of two nesting transitions, semantic selection identity changes, dimensions1/2/3, anisotropic hierarchy ratios/buffers, overflow, missing native config contract, and accepted preflight rejection of old/different selection provenance.
- Full first attempt145 passed/3failed: two new test assertions incorrectly required whole layout/provider identities to remain equal even when the authored selection policy changes; these were corrected to require unchanged numerical transitions and derived nesting minima. The third is an existing fixture defect.
- The remaining exact node test_amr_bind_lowering.py::test_implicit_pair_envelope_precedes_program_and_interface_install was independently executed on pristine base606e2a06 and failed identically in1.03s: _amr_system_install.py261 reads install_plan.artifact.layout_plan.layouts, while that test's artifact SimpleNamespace lacks layout_plan. ROOT independently reproduced it and owns its source-fixture correction. It is not caused by this production patch; the full final integrated source suite still must include ROOT's correction and rerun this node.

SOURCE_ONLY reproduction from this worktree, without Conda setup or installation:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python - <<'PYCODE'
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"python"))
import pytest
rc = pytest.main(["-q", "--tb=short", "-k", "not test_implicit_pair_envelope_precedes_program_and_interface_install",
    "tests/python/unit/amr/test_public_amr_resolution.py",
    "tests/python/unit/mesh/amr/test_hierarchy_contract.py",
    "tests/python/unit/runtime/test_amr_bind_lowering.py",
    "tests/python/unit/runtime/test_amr_checkpoint_contract.py"])
assert not any(k == "_pops" or k.endswith("._pops") for k in sys.modules)
raise SystemExit(rc)
PYCODE
python3 scripts/generate_release_contract.py --check
```

No native/JIT/C++ build, native execution, GPU, MPI, ENV mutation, Conda setup or SLURM campaign was performed by this author. The C++ guard test is source-prepared only. Independent frozen review and a rebuilt package must precede the representative original AMR run, strict checkpoint/restart/replay, refusal/nonregression and public tag-buffer geometry campaigns. Source success does not close the scientific corpus or native acceptance.
