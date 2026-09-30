"""A physical global source body is independent of its temporal capture method."""
import json
import pytest
import pops
from pops.model import Module
from pops._ir.quantity import PhysicalDimension
from pops.model.manifest import ModuleManifest
from pops.model.global_quantity import global_references
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback


@pytest.mark.parametrize("target",("system","amr_system"))
@pytest.mark.parametrize("periodic",(False,True))
def test_original_source_body_reads_authorized_global_pod(target,periodic):
    case,layout,program,_,_ = build_feedback(physical_global=True,periodic=periodic)
    resolved = pops.resolve(pops.validate(case),layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),target=target)
    assert source.count("ctx.capture_integral_candidate(")==1
    assert source.count("ctx.integral_candidate_value(")==1
    assert source.count("ctx.consume_external_trace(")==1
    consume = source.index("ctx.integral_candidate_value(")
    assert source.index("pops::for_each_cell",consume)>consume
    assert "physical_global_" in source
    source_value = next(value for value in program._values if value.op=="source")
    assert source_value.inputs[-1].op=="integral_candidate"
    assert source_value.attrs["physical_global_inputs_v1"][0]["port"].kind=="global_quantity"
    # The method expression contains source only; gamma and Q are in the source body.
    pointwise = next(value for value in program._values if value.op=="pointwise_expression")
    assert all(value.op!="integral_candidate" for value in pointwise.inputs)


def test_module_manifest_extension_and_units_are_authenticated():
    module = Module("physical_ports")
    q = module.global_quantity("q",units=PhysicalDimension((("charge",1),)))
    data = module.manifest().to_dict()
    assert data["schema_version"]==11
    assert ModuleManifest.from_dict(data).to_dict()==data
    bound = module.manifest().with_abi_key("source-host-fixture-abi")
    assert bound.schema_version==11
    assert ModuleManifest.from_dict(bound.to_dict()).to_dict()==bound.to_dict()
    assert module.declaration_index().authenticate(q)==q
    forged = json.loads(json.dumps(data))
    forged["global_quantities"]["q"]["scope"]="cell"
    with pytest.raises(ValueError): ModuleManifest.from_dict(forged)
    legacy = Module("legacy_ports").manifest().to_dict()
    assert legacy["schema_version"]==10 and "global_quantities" not in legacy
    assert ModuleManifest.from_dict(legacy).to_dict()==legacy


def test_two_physical_source_ports_keep_their_own_bodies_and_occurrences():
    from pops.math import ddt
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.layouts import Uniform
    from pops.numerics import DiscretizationPlan, StateStorage
    from pops.time import FixedDt
    frame = Rectangle("two_ports",lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    model = pops.Model("two_source_physics",frame=frame)
    U = model.state("u",components=("density",))
    Q = model.global_quantity("q",units=PhysicalDimension())
    R = model.global_quantity("r",units=PhysicalDimension())
    S = model.source("loss",on=U,value=(-.3*Q*U[0],))
    T = model.source("gain",on=U,value=(.1*R*U[0],))
    rate = model.rate("reaction",equation=ddt(U)==S+T)
    case = pops.Case("two_ports")
    block = case.block("fluid",model)
    spatial = DiscretizationPlan(); spatial.rates.add(rate,StateStorage()); case.numerics(spatial,block=block)
    program = pops.Program("two_source_method")
    temporal = program.state(block[U])
    q = program.integral_state("q",initial=.7,units=PhysicalDimension())
    r = program.integral_state("r",initial=.2,units=PhysicalDimension())
    body = program.evaluate_source(rate,temporal.n,global_inputs={
        block[Q]:program.integral_value(q,at=temporal.n.point,scope="candidate"),
        block[R]:program.integral_value(r,at=temporal.n.point,scope="candidate")})
    program.commit(temporal.next,program.value("accepted",temporal.n+program.dt*body,at=temporal.next.point))
    program.step_strategy(FixedDt(.01)); case.program(program)
    resolved = pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(frame=frame,cells=(8,1),periodic=PeriodicAxes(frame.axes))))
    code = emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert code.count("ctx.capture_integral_candidate(")==2
    assert code.count("ctx.integral_candidate_value(")==2
    sources = [value for value in program._values if value.op=="source"]
    assert len(sources)==2 and all(len(value.attrs["physical_global_inputs_v1"])==1 for value in sources)


def test_operator_first_module_preserves_the_physical_global_source_body():
    from pops.model import Signature, Rate
    module = Module("operator_first_global")
    state = module.state_space("density",("mass",))
    q = module.global_quantity("circuit",units=PhysicalDimension())
    state_expr = module.state_symbols(state)[0]
    module.operator("reaction",kind="local_source",signature=Signature([state],Rate(state)),
                    expr=(-.3*q*state_expr,))
    lowered = module.__pops_compiler_lowering__()
    assert len(global_references(lowered.emit_model._m._source_terms["reaction"]))==1
    assert module.manifest().to_dict()["schema_version"]==11


@pytest.mark.parametrize("mutation",("units","point","scope","index","port_units"))
def test_mutated_physical_source_binding_refuses_before_kernel(mutation):
    baseline_case,baseline_layout,_,_,_ = build_feedback(physical_global=True)
    baseline = pops.resolve(pops.validate(baseline_case),layout=baseline_layout)
    assert "physical_global_" in emit_cpp_program(baseline.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(baseline.blocks))
    case,layout,program,_,temporal = build_feedback(physical_global=True)
    source = next(value for value in program._values if value.op=="source")
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    capture = source.inputs[-1]
    if mutation=="units": rows[0]["units"]+=" "
    elif mutation=="index": rows[0]["input"]=0
    elif mutation=="port_units": object.__setattr__(rows[0]["port"],"units",PhysicalDimension((("charge",1),)))
    elif mutation=="point": program._replace_value(capture,point=temporal.next.point)
    else: program._replace_value(capture,attrs={**capture.attrs,"scope":"accepted"})
    program._replace_value(source,attrs={**source.attrs,"physical_global_inputs_v1":rows})
    with pytest.raises((ValueError,TypeError)):
        resolved = pops.resolve(pops.validate(case),layout=layout)
        emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_source_port_body_has_no_unbound_native_or_field_fallback():
    model = pops.Model("physical")
    state = model.state("u",components=("mass",))
    q = model.global_quantity("q",units=PhysicalDimension())
    physical = model.source("reaction",on=state,value=(-.3*q*state[0],))
    refs = global_references(model.module.operator_registry().get(physical.reg_name).body)
    assert len(refs)==1
    from pops._ir.lowering import diff
    assert diff(q._node,q).eval({})==1
    assert diff(q._node,state[0]).eval({})==0
    with pytest.raises(NotImplementedError,match="FieldProblem/flux"):
        refs[0].to_cpp()


@pytest.mark.parametrize("components",(("mass",),("a","b","c"),("c","a","b")))
def test_component_number_and_permutation_are_not_a_circuit_field(components):
    case,layout,program,_,_ = build_feedback(physical_global=True,components=components)
    resolved = pops.resolve(pops.validate(case),layout=layout)
    code = emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert code.count("ctx.integral_candidate_value(")==1
    assert "integral_capture_1A" not in code
    source = next(value for value in program._values if value.op=="source")
    assert len(source.space.base_space.components)==len(components)


def test_authoring_refuses_missing_foreign_and_mismatched_captures():
    _,_,program,quantity,temporal = build_feedback(physical_global=True)
    source = next(value for value in program._values if value.op=="source")
    operator = source.attrs["operator_handle"]
    port = source.attrs["physical_global_inputs_v1"][0]["port"]
    with pytest.raises(TypeError,match="explicit global_inputs mapping"):
        program.evaluate_source(operator,temporal.n,global_inputs=None)
    with pytest.raises(ValueError,match="explicit bindings"):
        operator(temporal.n)
    with pytest.raises(ValueError,match="same exact point"):
        program.evaluate_source(operator,temporal.n,global_inputs={port:
            program.integral_value(quantity,at=temporal.next.point,scope="candidate")})
    wrong = program.integral_state("wrong_units",initial=.7,units=PhysicalDimension((("charge",1),)))
    with pytest.raises(ValueError,match="units differ"):
        program.evaluate_source(operator,temporal.n,global_inputs={port:
            program.integral_value(wrong,at=temporal.n.point,scope="candidate")})
    cloned = port._with_owner(port.owner_path)
    with pytest.raises(ValueError,match="registry-issued"):
        program.evaluate_source(operator,temporal.n,global_inputs={cloned:
            program.integral_value(quantity,at=temporal.n.point,scope="candidate")})
