"""Independent physical model-first global-source reception, no native/JIT."""

import pops
import pytest
import copy
from pops._ir.quantity import PhysicalDimension
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import FixedDt
from pops.model import ModuleManifest


def authored(gamma=0.47, components=("z", "a"), *, bind=True):
    frame = Rectangle("review-space", (0.0, 0.0), (2.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("review-physical", frame=frame)
    state = model.state("particles", components=components)
    qport = model.global_quantity("charge", units=PhysicalDimension())
    unused = model.global_quantity("unused", units=PhysicalDimension())
    source = model.source(
        "original_loss",
        on=state,
        value=tuple(-gamma * qport * state[i] for i in range(len(components))),
    )
    equation = model.rate("physical_equation", equation=ddt(state) == source)
    selected = equation.select(source)
    # Freeze the original physical declaration before any Program exists.
    physical_body = model.module.operator_registry().get(source.reg_name).body
    case = pops.Case("review-capture")
    block = case.block("matter", model)
    foreign = case.block("foreign", model)
    spatial = DiscretizationPlan()
    spatial.rates.add(selected, StateStorage())
    case.numerics(spatial, block=block)
    case.numerics(spatial, block=foreign)
    program = pops.Program("review-method")
    temporal = program.state(block[state])
    quantity = program.integral_state("budget", initial=0.82, units=PhysicalDimension())
    capture = program.integral_value(quantity, at=temporal.n.point, scope="candidate")
    if bind:
        result = program.evaluate_source(
            selected, temporal.n, global_inputs={block[qport]: capture}
        )
        value = program.value(
            "next",
            tuple(temporal.n[i] + program.dt * result[i] for i in range(len(components))),
            at=temporal.next.point,
        )
        program.commit(temporal.next, value)
        program.step_strategy(FixedDt(0.03))
        case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(5, 3), periodic=PeriodicAxes(frame.axes)))
    return (
        case,
        layout,
        program,
        model,
        block,
        foreign,
        temporal,
        quantity,
        capture,
        selected,
        qport,
        unused,
        physical_body,
    )


def emitted(args, target="system"):
    case, layout = args[:2]
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(
        resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target=target,
    )


def test_resealed_clone_port_refuses_before_kernel():
    args = authored()
    program = args[2]
    source = next(value for value in program._values if value.op == "source")
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    port = rows[0]["port"]
    clone = port._with_owner(port.owner_path)
    assert clone is not port and clone == port
    rows[0]["port"] = clone
    program._replace_value(source, attrs={**source.attrs, "physical_global_inputs_v1": rows})
    with pytest.raises(ValueError, match="registry-issued"):
        emitted(args)


def test_manifest_global_version_requires_integer():
    data = authored()[3].module.manifest().to_dict()
    forged = copy.deepcopy(data)
    forged["global_quantities"]["charge"]["version"] = 1.0
    with pytest.raises((TypeError, ValueError), match="version|authority|canonical"):
        ModuleManifest.from_dict(forged)


def test_manifest_global_units_reject_boolean_exponent():
    forged = copy.deepcopy(authored()[3].module.manifest().to_dict())
    forged["global_quantities"]["charge"]["units"]["powers"] = [["charge", True, 1]]
    with pytest.raises((TypeError, ValueError), match="units|canonical"):
        ModuleManifest.from_dict(forged)


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_original_body_is_physical_and_only_source_kernel_consumes_capture(target):
    args = authored()
    code = emitted(args, target)
    program, body = args[2], args[-1]
    from pops._ir.quantity import QuantityRef
    from pops.model.global_quantity import GlobalQuantityRef
    from pops._ir.visitors import _children

    state_values = (1.3, -2.1)
    q = 0.82
    environment = {}

    def bind(leaf):
        if isinstance(leaf, QuantityRef):
            environment[leaf.qualified_id] = state_values[leaf.index]
        elif isinstance(leaf, GlobalQuantityRef):
            environment[leaf.handle.qualified_id] = q
        for child in _children(leaf):
            bind(child)

    for expression in body:
        bind(expression)
    assert tuple(expression.eval(environment) for expression in body) == pytest.approx(
        tuple(-0.47 * q * u for u in state_values)
    )
    registered = args[3].module.operator_registry().get("original_loss").body
    assert tuple(str(e) for e in registered) == tuple(str(e) for e in body)
    for expression in registered:
        bind(expression)
    assert tuple(e.eval(environment) for e in registered) == pytest.approx(
        tuple(-0.47 * q * u for u in state_values)
    )
    source = next(v for v in program._values if v.op == "source")
    assert source.inputs[-1].op == "integral_candidate"
    pointwise = next(v for v in program._values if v.op == "pointwise_expression")
    assert all(v.op != "integral_candidate" for v in pointwise.inputs)
    assert code.count("ctx.capture_integral_candidate(") == 1
    assert code.count("ctx.integral_candidate_value(") == 1
    assert code.index("ctx.integral_candidate_value(") < code.index(
        "pops::for_each_cell", code.index("ctx.integral_candidate_value(")
    )
    assert "physical_global_" in code


@pytest.mark.parametrize(
    "mode",
    (
        "missing",
        "unused",
        "foreign_block",
        "clone",
        "foreign_program",
        "wrong_units",
        "wrong_point",
    ),
)
def test_unbound_and_foreign_bindings_are_atomic(mode):
    args = authored(bind=False)
    program, block, foreign, temporal, capture, selected, qport, unused = (
        args[i] for i in (2, 4, 5, 6, 8, 9, 10, 11)
    )
    bindings = {block[qport]: capture}
    if mode == "missing":
        bindings = {}
    elif mode == "unused":
        bindings[block[unused]] = capture
    elif mode == "foreign_block":
        bindings = {foreign[qport]: capture}
    elif mode == "clone":
        bindings = {block[qport]._with_owner(block[qport].owner_path): capture}
    elif mode == "foreign_program":
        other = pops.Program("different")
        q = other.integral_state("other", initial=0.4, units=PhysicalDimension())
        from pops.time.points import TimePoint

        bindings = {
            block[qport]: other.integral_value(q, at=TimePoint(other.clock), scope="candidate")
        }
    elif mode == "wrong_units":
        q = program.integral_state("wrong", initial=0.4, units=PhysicalDimension((("charge", 1),)))
        bindings = {block[qport]: program.integral_value(q, at=temporal.n.point, scope="candidate")}
    else:
        bindings = {
            block[qport]: program.integral_value(args[7], at=temporal.next.point, scope="candidate")
        }
    before = (program._next_id, len(program._values))
    with pytest.raises((ValueError, TypeError)):
        program.evaluate_source(selected, temporal.n, global_inputs=bindings)
    assert before == (program._next_id, len(program._values))


@pytest.mark.parametrize("mode", ("point", "scope", "version", "port_owner"))
def test_resealed_binding_mutations_refuse_emission(mode):
    args = authored()
    program = args[2]
    source = next(v for v in program._values if v.op == "source")
    capture = source.inputs[-1]
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    if mode == "point":
        program._replace_value(capture, point=args[6].next.point)
    elif mode == "scope":
        program._replace_value(capture, attrs={**capture.attrs, "scope": "accepted"})
    elif mode == "version":
        rows[0]["version"] = 2
    else:
        rows[0]["port"] = args[5][args[10]]
    program._replace_value(source, attrs={**source.attrs, "physical_global_inputs_v1": rows})
    with pytest.raises((ValueError, TypeError)):
        emitted(args)


def test_physical_body_change_changes_both_model_and_program_identity():
    first = authored(gamma=0.47)
    second = authored(gamma=0.51)
    assert first[3].module.module_hash() != second[3].module.module_hash()
    assert first[2]._ir_hash() != second[2]._ir_hash()
    assert first[7].identity != second[7].identity
    assert emitted(first) != emitted(second)


def test_direct_unbound_global_has_no_field_or_flux_fallback():
    args = authored(bind=False)
    with pytest.raises(NotImplementedError, match="FieldProblem/flux"):
        args[10]._node.to_cpp()
    with pytest.raises(TypeError, match="explicit global_inputs"):
        args[2].evaluate_source(args[9], args[6].n, global_inputs=None)


@pytest.mark.parametrize("kind", ("local_source", "grid_operator", "field_operator"))
def test_unsupported_default_grid_and_field_consumers_refuse_binding(kind):
    from pops.model import Module, Signature, Rate

    module = Module("unsupported-consumer")
    space = module.state_space("u", ("mass",))
    port = module.global_quantity("q", units=PhysicalDimension())
    symbol = module.state_symbols(space)[0]
    output = (
        module.field_space("potential", ("value",)) if kind == "field_operator" else Rate(space)
    )
    lowering = {"source": "default"} if kind == "local_source" else None
    operator = module.operator(
        "consumer",
        kind=kind,
        signature=Signature([space], output),
        lowering=lowering,
        expr=(port * symbol,),
    )
    case = pops.Case("unsupported-port")
    block = case.block("cells", module)
    program = pops.Program("unsupported-method")
    state = program.state(block[module.state_handle(space)])
    quantity = program.integral_state("q", initial=0.7, units=PhysicalDimension())
    capture = program.integral_value(quantity, at=state.n.point, scope="candidate")
    before = (program._next_id, len(program._values))
    with pytest.raises(
        (NotImplementedError, TypeError), match="default-source|source-only|field-provider"
    ):
        program.evaluate_source(operator, state.n, global_inputs={block[port]: capture})
    assert before == (program._next_id, len(program._values))


@pytest.mark.parametrize("name", ("implicit_source", "solve_implicit_source"))
def test_implicit_source_has_no_global_binding_api(name):
    args = authored(bind=False)
    before = (args[2]._next_id, len(args[2]._values))
    with pytest.raises(TypeError, match="global_inputs"):
        getattr(args[2], name)(args[6].n, global_inputs={args[4][args[10]]: args[8]})
    assert before == (args[2]._next_id, len(args[2]._values))
