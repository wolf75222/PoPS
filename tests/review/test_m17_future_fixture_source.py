"""Authoring and independent mathematics only; no installed Native execution."""
import numpy as np
import pops
from tests.python.support.m17_fan_li_public_native_case import (
    original,make_case,initial,canonical,ssprk2,reference_proof)


def test_original_non_gaussian_mixture_discriminates_active_product():
    module=original();values=canonical(initial(module.INDICES),module.INDICES)[:,0,:]
    proof=reference_proof(values)
    assert proof['h3_max']>1e-8 and proof['h4_max']>1e-8
    assert proof['regularized_face_max']>module.CRITERIA['nonconservative_max_min']
    assert proof['active_vs_distinct_B0_step_gap']>1e-8
    assert proof['quadrature24_48_step_gap']<module.CRITERIA['path_quadrature_gap']
    assert np.max(np.abs(ssprk2(values)-ssprk2(values,nonconservative=False)))>1e-8


def test_permuted_initial_averages_recover_exact_original_storage():
    module=original();reverse=tuple(reversed(module.INDICES))
    assert canonical(initial(reverse),reverse).tobytes()==initial(module.INDICES).tobytes()


def test_actual_public_original_case_resolves_new_method():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.module_codegen import _emit_bricks
    from pops.codegen.program_codegen import emit_cpp_program
    module=original();case,layout=make_case(module.INDICES)
    plan=pops.resolve(pops.validate(case),layout=layout)
    selected=plan.blocks[0]
    emitter,_=lower_and_validate(selected.model,state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations,numerics=selected.numerics)
    brick=_emit_bricks(emitter._m)[1]
    assert 'integrate_normalized_moment_path<4>' in brick
    assert emitter._m._path_conservative['identity'].startswith('pops.numerics.normalized-polynomial-path.v1:sha256:')
    source=emit_cpp_program(plan.time,model=emitter,target='amr_system')
    assert source.count('ctx.stage_path_rhs(')==2
    assert source.count('ctx.publish_staged_path_rhs(')==2


def test_actual_compiled_wrapper_metadata_json_roundtrip(monkeypatch):
    import json
    import pytest
    from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture,CanonicalValue
    from tests.python.integration.runtime.test_fan_li15_public_composition_runtime import select_program,layout_program_json_identity
    # Actual wrapper classes, Source metadata-only components; no DSO/SDK loaded.
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',lambda **kwargs:CanonicalValue('SOURCE_ONLY_NO_NATIVE'))
    artifact=artifact_fixture(target='amr_system',block_names=('gas',))
    row=select_program(artifact,artifact.plan)
    with pytest.raises(TypeError):json.dumps(row.to_data())
    data={'layout_program':layout_program_json_identity(row),'artifact_identity':artifact.artifact_identity.token}
    restored=json.loads(json.dumps(data,sort_keys=True,allow_nan=False))
    assert restored['layout_program']['identity_token']==row.identity.token
    assert restored['layout_program']['block_names']==['gas']
    assert restored['artifact_identity']==artifact.artifact_identity.token
