"""Non-author Source witnesses; compiler entries are spies, no Native claim."""
import inspect
import pops
from tests.python.support.principal_primitive_case import primitive_case

def test_require_reaches_every_distinct_resolved_block_before_program_projection(monkeypatch):
    from pops.codegen import _orchestration_compile as models
    from pops.codegen._compile_drivers import _program_compile_options
    case,layout,*_=primitive_case((2,3))
    plan=pops.resolve(pops.validate(case),layout=layout,compile_options={"model_source_policy":"require","debug":True})
    calls=[]
    def entry(name,model,backend,target,options,**kwargs):
        calls.append((name,dict(options),kwargs["consumer_owner_qid"]))
        return object()
    monkeypatch.setattr(models,"compile_install_model",entry)
    result=models.compile_install_models(plan,plan.compile_options)
    assert tuple(result)==tuple(b.name for b in plan.blocks)
    assert len(calls)==2 and len({c[2] for c in calls})==2
    assert all(c[1]["model_source_policy"]=="require" for c in calls)
    projected=_program_compile_options(plan)
    assert "model_source_policy" not in projected and projected["debug"] is True
    projected["debug"]=False
    assert plan.compile_options["debug"] is True
    plan.verify()

def test_single_and_sliced_program_routes_use_same_projection():
    from pops.codegen import _compile_drivers as driver, _phases as phases
    assert "options = _program_compile_options(plan)" in inspect.getsource(driver._compile_resolved_problem)
    source=inspect.getsource(phases.compile)
    assert "options = _program_compile_options(plan)" in source
    assert "models = compile_install_models(plan, plan.compile_options)" in source
