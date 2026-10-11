"""Native initial-program AMR planning and ordered execution, without numerical mocks as evidence."""
import pops
from pops.analytic import constant, input as analytic_input
from pops.initial import InitialCondition
from pops.lib.initial import Analytic, BootstrapProgram, FieldSolve, StateMap, ProviderDerivative
from pops.model.provider_pack import ComponentKey
from pops.projection import ConservativeCellAverage
from pops.mesh._amr import NativeProgramSeed
from tests.python.unit.initial.test_initial_authoring import _case, _resolved_amr_authorities


def initial_program_plan():
    case,frame,_decl,state,threshold=_case(components=("rho","mx","my"))
    key=ComponentKey(str(state.block_ref.model_owner_path.canonical()),"field","potential","potential")
    rho=analytic_input(0,"rho")
    dx,dy=analytic_input(1,"dx"),analytic_input(2,"dy")
    profile=BootstrapProgram(seed=Analytic(frame=frame,components=(constant(1),constant(0),constant(0))),
        stages=(FieldSolve(),StateMap(components=(rho,-rho*dy,rho*dx),
            inputs={0:("state",0),1:ProviderDerivative(key,0),2:ProviderDerivative(key,1)},
            consumer_qid="initial-map"),FieldSolve()),consumer_qid="initial-seed")
    initial=InitialCondition(state=state,value=profile,projection=ConservativeCellAverage())
    case.initials.add(initial)
    pops.validate(case)
    layout,transfer,hierarchy,tagging=_resolved_amr_authorities(case,state,threshold,frame=frame)
    resolved=case.initials.resolve_amr(layout_plan=layout,transfers=transfer,hierarchy=hierarchy,tagging=tagging)
    return resolved.bootstrap_plan,resolved.initial_condition_plan.bindings[0]


def test_native_seed_has_truthful_fine_sampling_and_final_field_program_order():
    plan,binding=initial_program_plan()
    assert type(plan.selections[0].method) is NativeProgramSeed
    operations=[action.operation for action in plan.actions]
    assert operations.count("seed_initial_program")==1
    assert "prolong_from_parent" not in operations
    assert "analytic_reprojection" not in operations
    assert "recompute" not in operations
    assert operations[-1]=="execute_initial_program"
    assert operations.index("create_level") < operations.index("seed_initial_program")
    assert operations.index("seed_initial_program") < operations.index("synchronize_covered_cells")
    assert binding.source.options.to_data()["native_route"]=="bootstrap_program"


def test_native_execution_dispatches_two_forced_field_stages_on_complete_hierarchy():
    from pops.runtime._amr_bootstrap_execution import execute_native_bootstrap
    plan,binding=initial_program_plan()
    calls=[]

    class Native:
        def _begin_bootstrap_plan(self): calls.append(("begin",))
        def _materialize_bootstrap_action(self,subject,operation,route,level):
            calls.append((operation,level,route))
            return 10
        def _solve_bootstrap_initial_field(self,subject,slot,ordinal):
            calls.append(("solve",ordinal,slot))
            from types import SimpleNamespace
            return 20, SimpleNamespace(iters=ordinal,evaluations=ordinal+1,
                residual_norm=.25,reference_residual_norm=1.,rel_residual=.25,
                status="solved",action="continue",reason="native test evidence",
                residual_evaluation=None)
        def _map_bootstrap_state_hierarchy(self,subject,opcodes,literals,inputs,consumer,slot):
            calls.append(("map",len(opcodes),slot))
            return 30
        def _synchronize_bootstrap_state(self,subject,level): calls.append(("restrict",level))

    class Engine:
        _s=Native()
        _bootstrap_parameter_bindings={}
        levels=1
        def _bootstrap_next_level(self):
            self.levels+=1
            return True
        def n_levels(self): return self.levels
        def patch_boxes(self): return ((1,0,0,3,3),)
        def _commit_bootstrap_level(self): calls.append(("commit",))
        def _rollback_bootstrap_level(self): calls.append(("rollback",))

    source=binding.source.options.to_data()
    engine=Engine()
    execution=execute_native_bootstrap(engine,plan,
        ((binding.subject.qualified_id,"tracer",None,"cell","cell","initial_program",source),),
        {"Gauss":"exact-qualified-slot"})
    assert [row for row in calls if row[0]=="solve"]==[
        ("solve",1,"exact-qualified-slot"),("solve",2,"exact-qualified-slot")]
    assert calls.index(("seed_initial_program",1,"bootstrap_program")) < calls.index(("solve",1,"exact-qualified-slot"))
    assert calls.index(("map",3,"exact-qualified-slot")) < calls.index(("solve",2,"exact-qualified-slot"))
    assert calls[-1]==("commit",)
    assert execution.receipts[-1].evidence["hierarchy_levels"]==2
    assert [row["ordinal"] for row in engine._initial_program_reports["tracer"]] == [1,2]
    assert execution.receipts[-1].evidence["stages"][2]["solve_report"]["iters"] == 2
