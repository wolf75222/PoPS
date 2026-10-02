"""Actual public compile routing; compiler-entry sentinels are Source-only."""
import pytest
import pops
from tests.python.support.atomic_cubature_path_case import make_case

@pytest.mark.parametrize('policy',('allow_missing','require','recompile'))
def test_public_compile_keeps_model_policy_out_of_program_compiler(monkeypatch, policy):
    from pops.codegen import _compile_drivers as driver
    from pops.codegen import _orchestration_compile as orchestration
    case,layout=make_case(nonconservative=True)
    resolved=pops.resolve(pops.validate(case),layout=layout,
                          compile_options={'model_source_policy':policy,'debug':True})
    # This is the unchanged public compile phase immediately after the Native
    # selection/bootstrap cut line. No fake Native module or capability is installed.
    from pops.codegen._phases import compile as compile_phase
    model_calls=[]
    def model_entry(*args,**kwargs):
        options=args[4]
        assert options['model_source_policy']==policy
        model_calls.append(args[0])
        return object()  # Source compiler-entry witness; no binary or runtime.
    monkeypatch.setattr(orchestration,'compile_install_model',model_entry)
    class AtProgramCompiler(Exception):pass
    sentinel=AtProgramCompiler('reached authentic public Program compiler entry')
    # Preserve the actual signature: an unexpected keyword must fail before entry.
    def program_entry(so_path=None,*,model_graph,time,backend,target,problem_snapshot,
                      field_plans,balance_due_contract,native_dimension,
                      shared_interface_codegen_evidence,physical_global_sources,
                      libraries,debug=False):
        assert debug is True and native_dimension==2
        assert resolved.compile_options['model_source_policy']==policy
        raise sentinel
    monkeypatch.setattr(driver,'_compile_problem_impl',program_entry)
    with pytest.raises(AtProgramCompiler) as failure:
        compile_phase(resolved)
    assert failure.value is sentinel
    assert model_calls==[block.name for block in resolved.blocks]
    resolved.verify()  # Filtering must not mutate the authenticated public plan.

def test_projection_preserves_authoring_manifest_and_legacy_options():
    from types import SimpleNamespace
    from pops.codegen._compile_drivers import _program_compile_options
    for policy in ('allow_missing','require','recompile'):
        authored={'model_source_policy':policy,'cxx':'compiler','std':'c++20','force':True,'so_path':'program.so'}
        plan=SimpleNamespace(compile_options=authored,libraries=('library-authority',))
        expected={k:v for k,v in authored.items() if k!='model_source_policy'}
        expected['libraries']=plan.libraries
        assert _program_compile_options(plan)==expected
        assert authored['model_source_policy']==policy
    legacy=SimpleNamespace(compile_options={'debug':True},libraries=())
    assert _program_compile_options(legacy)=={'debug':True,'libraries':()}
