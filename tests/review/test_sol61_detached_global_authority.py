"""Independent source-only reception of live-issued/detached global authority."""

import gc
from pathlib import Path
import sys
from types import ModuleType
import weakref

import pops
import pytest
from pops._ir.quantity import PhysicalDimension
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from pops.time._program.detach import detach_compiled_program


def require_detached_source_global(*args, **kwargs):
    from pops.time._program.global_source_plan import require_detached_source_global as check

    return check(*args, **kwargs)


def authored(*, primitive=False, global_port=True, runtime=False):
    frame = Rectangle("independent-space", (0.0, 0.0), (2.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("independent-reaction", frame=frame)
    state = model.state("matter", components=("z", "a", "k"))
    port = model.global_quantity("charge", units=PhysicalDimension()) if global_port else None
    if runtime:
        from pops.params import RuntimeParam

        gain = model.value(model.param(RuntimeParam("rate", default=0.47)))
    else:
        gain = 0.47
    loss = (
        model.primitive("loss", gain * port)
        if primitive
        else (gain * port if global_port else gain)
    )
    source = model.source("decay", on=state, value=tuple(-loss * state[c] for c in range(3)))
    selected = model.rate("balance", equation=ddt(state) == source).select(source)
    case = pops.Case("independent-authority")
    block = case.block("material", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(selected, StateStorage())
    case.numerics(numerics, block=block)
    case.initials.add(
        InitialCondition(
            state=block[state], value=BindArray(), projection=ConservativeCellAverage()
        )
    )
    program = pops.Program("explicit-physical-method")
    temporal = program.state(block[state])
    budget = program.integral_state("budget", initial=0.82, units=PhysicalDimension())
    capture = program.integral_value(budget, at=temporal.n.point, scope="candidate")
    result = (
        program.evaluate_source(selected, temporal.n, global_inputs={block[port]: capture})
        if global_port
        else selected(temporal.n)
    )
    value = program.value(
        "accepted",
        tuple(temporal.n[c] + program.dt * result[c] for c in range(3)),
        at=temporal.next.point,
    )
    program.commit(temporal.next, value)
    program.step_strategy(FixedDt(0.03))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(5, 3), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, model


def prepared(**kwargs):
    values = authored(**kwargs)
    plan = pops.resolve(pops.validate(values[0]), layout=values[1])
    return plan, values


class ReachedProblemCompiler(Exception):
    pass


def stop_compiler(monkeypatch, tmp_path):
    # Substitute external/toolchain access only. Resolution, detachment, model lowering,
    # provider planning and problem emission remain the actual package implementation.
    import pops.codegen._compile_drivers as drivers
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    import pops.native_components as dependencies
    import pops._native_selector as selector

    monkeypatch.setitem(sys.modules, "pops._bootstrap", ModuleType("pops._bootstrap"))
    monkeypatch.setattr(selector, "select_native_dimension", lambda dimension: dimension)
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "pops_header_signature", lambda inc: "INDEPENDENT-SOURCE-ONLY")
    for target in (drivers, toolchain):
        monkeypatch.setattr(target, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
        monkeypatch.setattr(target, "_probe_cxx_std", lambda cc, std: "c++23")
        monkeypatch.setattr(target, "pops_header_signature", lambda inc: "INDEPENDENT-SOURCE-ONLY")
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "independent-source-only")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx=None: "c++")
    monkeypatch.setattr(
        dependencies, "verify_prepared_native_dependencies", lambda *args, **kwargs: None
    )
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    emitted = []

    def external(command, what):
        output = Path(command[command.index("-o") + 1])
        source = Path(next(arg for arg in command if arg.endswith(".cpp")))
        if source.name == "problem.cpp":
            emitted.append(source.read_text())
            raise ReachedProblemCompiler
        output.write_bytes(b"NOT-A-NATIVE-DSO")
        if "-MF" in command:
            Path(command[command.index("-MF") + 1]).write_text(f"{output}: {source}\n")

    monkeypatch.setattr(drivers, "_run_compile", external)
    monkeypatch.setattr(toolchain, "_run_compile", external)
    return emitted


@pytest.mark.parametrize("global_port", (False, True))
def test_public_compile_reaches_real_problem_emission(monkeypatch, tmp_path, global_port):
    emitted = stop_compiler(monkeypatch, tmp_path)
    plan, _ = prepared(global_port=global_port)
    with pytest.raises(ReachedProblemCompiler):
        pops.compile(plan)
    assert len(emitted) == 1
    assert ("ctx.integral_candidate_value(" in emitted[0]) == global_port


def test_detached_program_does_not_retain_model_case_or_program():
    def build():
        plan, values = prepared()
        refs = tuple(weakref.ref(obj) for obj in (values[0], values[2], values[3]))
        detached = detach_compiled_program(
            values[2], physical_global_sources=plan._physical_global_sources
        )
        return detached, refs

    detached, refs = build()
    gc.collect()
    assert [ref() for ref in refs] == [None, None, None]
    source = next(value for value in detached._values if value.op == "source")
    require_detached_source_global(source)
    assert source.state_ref.block_ref._instance_registry is None


def test_lowered_primitive_body_is_authenticated():
    plan, values = prepared(primitive=True)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(
        values[2], physical_global_sources=plan._physical_global_sources
    )
    baseline = emit_cpp_program(detached, model_graph=graph)
    assert "physical_global_" in baseline
    lowered = graph.models_by_block["material"]._m
    original_recipes = lowered.prim_defs
    recipes = dict(original_recipes)
    assert "loss" in recipes
    recipes["loss"] = 2 * recipes["loss"]
    module_hash = graph.models_by_block["material"].module.module_hash()
    object.__setattr__(lowered, "prim_defs", recipes)
    assert graph.models_by_block["material"].module.module_hash() == module_hash
    try:
        forged = emit_cpp_program(detached, model_graph=graph)
    except ValueError as error:
        assert any(term in str(error) for term in ("body", "primitive", "authority"))
    else:
        assert forged != baseline, "mutation must actually change the generated kernel"
        pytest.fail(
            "lowered primitive changed the physical kernel under unchanged Module authority"
        )
    finally:
        object.__setattr__(lowered, "prim_defs", original_recipes)


@pytest.mark.parametrize(
    "mutation", ("clone", "unit", "index", "version", "capture_point", "input_identity")
)
def test_detached_rows_cannot_reseal_authority(mutation):
    from pops.identity import make_identity

    plan, values = prepared()
    detached = detach_compiled_program(
        values[2], physical_global_sources=plan._physical_global_sources
    )
    source = next(value for value in detached._values if value.op == "source")
    require_detached_source_global(source)
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    capture = source.inputs[rows[0]["input"]]
    if mutation == "clone":
        port = rows[0]["port"]
        clone = port._with_owner(port.owner_path)
        assert clone is not port and clone.canonical_identity() == port.canonical_identity()
        rows[0]["port"] = clone
    elif mutation == "unit":
        rows[0]["units"] += " "
    elif mutation == "index":
        rows[0]["input"] = 0
    elif mutation == "version":
        rows[0]["version"] = 1.0
    elif mutation == "capture_point":
        object.__setattr__(capture, "point", next(iter(detached._time_states.values())).next.point)
    else:
        object.__setattr__(capture, "id", capture.id + 999)
    object.__setattr__(source, "attrs", {**source.attrs, "physical_global_inputs_v1": rows})
    # Even a freshly computed serialized IR identity cannot mint the issued proof.
    _ = make_identity("independent-resealed-ir", {"ir_hash": detached._ir_hash()})
    with pytest.raises(ValueError, match="authority|capture|metadata|typed|version|changed|bound"):
        require_detached_source_global(source)


def test_detached_proof_is_not_transferable_to_equal_program():
    first_plan, first_values = prepared()
    second_plan, second_values = prepared()
    first = detach_compiled_program(
        first_values[2], physical_global_sources=first_plan._physical_global_sources
    )
    second = detach_compiled_program(
        second_values[2], physical_global_sources=second_plan._physical_global_sources
    )
    assert first._ir_hash() == second._ir_hash()
    source = next(value for value in second._values if value.op == "source")
    object.__setattr__(
        second, "_physical_global_source_bindings", first._physical_global_source_bindings
    )
    with pytest.raises(ValueError, match="program|authority"):
        require_detached_source_global(source)


@pytest.mark.parametrize("mutation", ("source_body", "module_body"))
def test_detached_original_source_body_still_authenticated(mutation):
    plan, values = prepared()
    detached = detach_compiled_program(
        values[2], physical_global_sources=plan._physical_global_sources
    )
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    assert "physical_global_" in emit_cpp_program(detached, model_graph=graph)
    model = graph.models_by_block["material"]
    if mutation == "source_body":
        original = model._m._source_terms
        original_hash = model.module.module_hash()
        object.__setattr__(
            model._m, "_source_terms", {"decay": tuple(2 * term for term in original["decay"])}
        )
        assert model.module.module_hash() == original_hash
        restore = (model._m, "_source_terms", original)
    else:
        operator = model.module.operator_registry().get("decay")
        original = operator.body
        object.__setattr__(operator, "body", tuple(2 * term for term in original))
        restore = (operator, "body", original)
    try:
        with pytest.raises(ValueError, match="body|authority|supplied current Module"):
            emit_cpp_program(detached, model_graph=graph)
    finally:
        object.__setattr__(*restore)


@pytest.mark.parametrize("mutation", ("clone", "unit", "row_index", "capture_point", "version"))
def test_live_plan_reseal_cannot_issue_new_global_authority(mutation):
    from pops.identity import make_identity

    plan, values = prepared()
    program = values[2]
    source = next(value for value in program._values if value.op == "source")
    plan.verify()
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    if mutation == "clone":
        port = rows[0]["port"]
        rows[0]["port"] = port._with_owner(port.owner_path)
    elif mutation == "unit":
        rows[0]["units"] += " "
    elif mutation == "row_index":
        rows[0]["input"] = 0
    elif mutation == "capture_point":
        capture = source.inputs[rows[0]["input"]]
        object.__setattr__(capture, "point", next(iter(program._time_states.values())).next.point)
    else:
        rows[0]["version"] = True
    object.__setattr__(source, "attrs", {**source.attrs, "physical_global_inputs_v1": rows})
    object.__setattr__(plan, "plan_identity", make_identity("resolved-plan", plan._payload()))
    with pytest.raises(
        (ValueError, TypeError),
        match="authority|registry-issued|capture|version|metadata|input index",
    ):
        plan.verify()


def test_parent_ir_plan_manifest_cpp_bytes():
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/development/api_040/sol61_detached_global_independent_parity.json"
    baseline = json.loads(receipt_path.read_text())["cases"]
    # Source provenance records absolute paths and declaration lines. Use the same
    # unchanged fixture file for parent/candidate, and pristine source interpreters.
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    runner = Path(__file__).with_name("sol61_detached_global_parity.py")
    parent = Path(
        environment.get(
            "SOL61_GLOBAL_PARENT_CHECKOUT",
            str(root.parent / "PoPS-sol61-global-ownership-parent-review"),
        )
    )
    revision = subprocess.run(
        ["rtk", "git", "rev-parse", "HEAD"], cwd=parent, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert revision == "368055dbe1f4c1f5fad4a11791508ae2b0520f03"

    def receipt(checkout):
        result = subprocess.run(
            [sys.executable, str(runner), str(checkout)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(result.stdout)

    current = receipt(root)
    original = receipt(parent)
    assert current == original  # All five fields, including full Manifest/plan identity.
    # Preserve the initial receipt. Its provenance path/lines deliberately differ
    # after adding tests or cherry-picking; stable physical identities still match.
    for case, fields in current.items():
        for key in ("program_ir_hash", "module_hash", "cpp_sha256"):
            assert fields[key] == baseline[case][key]


def test_live_port_from_independent_equal_case_refuses():
    plan, values = prepared()
    _, foreign_values = prepared()
    source = next(value for value in values[2]._values if value.op == "source")
    foreign = next(value for value in foreign_values[2]._values if value.op == "source")
    rows = [dict(row) for row in source.attrs["physical_global_inputs_v1"]]
    port = rows[0]["port"]
    other = foreign.attrs["physical_global_inputs_v1"][0]["port"]
    from pops.time.references import canonical_handle

    assert (
        canonical_handle(port).canonical_identity() == canonical_handle(other).canonical_identity()
    )
    assert port.block_ref._instance_registry is not other.block_ref._instance_registry
    initial_ir = values[2]._ir_hash()
    rows[0]["port"] = other
    object.__setattr__(source, "attrs", {**source.attrs, "physical_global_inputs_v1": rows})
    assert values[2]._ir_hash() == initial_ir
    with pytest.raises(ValueError, match="block owner changed"):
        plan.verify()


@pytest.mark.parametrize("mutation", ("default", "alias", "slots", "components"))
def test_runtime_parameter_and_component_binding_match_module(mutation):
    from pops._ir.values import RuntimeParamRef

    plan, values = prepared(primitive=True, runtime=True)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(
        values[2], physical_global_sources=plan._physical_global_sources
    )
    baseline = emit_cpp_program(detached, model_graph=graph)
    assert "params.get(0)" in baseline
    model = graph.models_by_block["material"]
    impl = model._m
    original_recipes = impl.prim_defs
    original_components = impl.cons_names
    recipes = dict(original_recipes)
    node = impl.runtime_param_nodes()[0]
    module_hash = model.module.module_hash()
    if mutation == "default":
        recipes["loss"] = (
            RuntimeParamRef("rate", 0.94, handle=node.handle)
            * model.module._global_quantities["charge"]
        )
    elif mutation == "alias":
        clone = node.handle._with_owner(node.handle.owner_path)
        assert clone is not node.handle
        recipes["loss"] = (
            RuntimeParamRef("rate", 0.47, handle=clone) * model.module._global_quantities["charge"]
        )
    elif mutation == "slots":
        recipes["unused"] = RuntimeParamRef("aaa", 0.2)
    else:
        object.__setattr__(impl, "cons_names", tuple(reversed(original_components)))
    object.__setattr__(impl, "prim_defs", recipes)
    assert model.module.module_hash() == module_hash
    try:
        with pytest.raises(ValueError, match="(primitive|binding|parameter).*authority"):
            emit_cpp_program(detached, model_graph=graph)
        plan.verify()
    finally:
        object.__setattr__(impl, "prim_defs", original_recipes)
        object.__setattr__(impl, "cons_names", original_components)


def test_unused_constant_and_lowering_cache_preserve_emission():
    from pops._ir.expr import Const

    plan, values = prepared(primitive=True)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(
        values[2], physical_global_sources=plan._physical_global_sources
    )
    baseline = emit_cpp_program(detached, model_graph=graph)
    impl = graph.models_by_block["material"]._m
    original = impl.prim_defs
    object.__setattr__(impl, "_independent_unconsumed_cache", object())
    object.__setattr__(impl, "prim_defs", {**original, "unread_literal": Const(17.0)})
    try:
        assert emit_cpp_program(detached, model_graph=graph) == baseline
    finally:
        object.__setattr__(impl, "prim_defs", original)
