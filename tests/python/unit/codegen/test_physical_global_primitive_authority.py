"""Public physical primitive recipes remain authoritative after Program detach."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pops
import pytest
from pops._ir.quantity import PhysicalDimension
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FixedDt
from pops.time._program.detach import detach_compiled_program


def prepared(*, runtime=False):
    frame = Rectangle("primitive-frame", (0., 0.), (2., 1.)).frame(Cartesian2D())
    model = pops.Model("primitive-physics", frame=frame)
    state = model.state("matter", components=("third", "first", "second"))
    port = model.global_quantity("charge", units=PhysicalDimension())
    gain = model.value(model.param(RuntimeParam("gain", default=.47))) if runtime else .47
    # Both recipes are genuinely reached from the source; the second one makes
    # replacing a transitive recipe distinguishable from editing the body itself.
    loss = model.primitive("loss", gain * port)
    nested = model.primitive("nested_loss", 2 * loss)
    source = model.source("decay", on=state, value=tuple(-nested * state[c] for c in range(3)))
    rate = model.rate("balance", equation=ddt(state) == source).select(source)
    case = pops.Case("primitive-authority")
    block = case.block("material", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage())
    case.numerics(numerics, block=block)
    program = pops.Program("primitive-step")
    temporal = program.state(block[state])
    quantity = program.integral_state("charge", initial=.82, units=PhysicalDimension())
    capture = program.integral_value(quantity, at=temporal.n.point, scope="candidate")
    value = program.evaluate_source(rate, temporal.n, global_inputs={block[port]: capture})
    accepted = program.value("accepted", tuple(temporal.n[c] + program.dt * value[c] for c in range(3)),
                             at=temporal.next.point)
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(.03))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(5, 3), periodic=PeriodicAxes(frame.axes)))
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(program, physical_global_sources=plan._physical_global_sources)
    return plan, graph, detached


@pytest.mark.parametrize("runtime", (False, True))
@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_public_primitive_source_keeps_exact_live_cpp(runtime, target):
    plan, graph, detached = prepared(runtime=runtime)
    live = emit_cpp_program(plan.time, model_graph=graph, target=target)
    frozen = emit_cpp_program(detached, model_graph=graph, target=target)
    assert hashlib.sha256(live.encode()).digest() == hashlib.sha256(frozen.encode()).digest()
    assert "physical_global_" in frozen
    if runtime:
        assert "params.get(0)" in frozen


@pytest.mark.parametrize("mutation", ("loss", "nested_loss", "missing", "components", "runtime_recipe", "runtime_slot", "runtime_handle"))
def test_lowered_physical_inputs_cannot_replace_module_authority(mutation):
    plan, graph, detached = prepared(runtime=mutation.startswith("runtime"))
    model = graph.models_by_block["material"]
    impl = model._m
    baseline = emit_cpp_program(detached, model_graph=graph)
    module_hash = model.module.module_hash()
    recipes = dict(impl.prim_defs)
    if mutation in ("loss", "nested_loss"):
        recipes[mutation] = 2 * recipes[mutation]
    elif mutation == "missing":
        del recipes["loss"]
    elif mutation == "components":
        object.__setattr__(impl, "cons_names", tuple(reversed(impl.cons_names)))
    elif mutation == "runtime_recipe":
        from pops._ir.values import RuntimeParamRef
        node = impl.runtime_param_nodes()[0]
        recipes["loss"] = RuntimeParamRef("gain", .94, handle=node.handle) * model.module._global_quantities["charge"]
    elif mutation == "runtime_slot":
        from pops._ir.values import RuntimeParamRef
        # An extra lowering-only parameter sorts before gain and would shift its
        # native slot, even though the selected source body/recipe is untouched.
        recipes["unused"] = RuntimeParamRef("aaa", .2)
    else:
        from pops._ir.values import RuntimeParamRef
        node = impl.runtime_param_nodes()[0]
        recipes["loss"] = RuntimeParamRef("gain", .47, handle=node.handle._with_owner(node.handle.owner_path)) * model.module._global_quantities["charge"]
    object.__setattr__(impl, "prim_defs", recipes)
    assert model.module.module_hash() == module_hash
    with pytest.raises(ValueError, match="(primitive|binding|parameter).*authority"):
        emit_cpp_program(detached, model_graph=graph)
    assert "physical_global_" in baseline
    plan.verify()


def test_unconsumed_lowering_cache_is_not_an_opaque_authority():
    from pops._ir.expr import Const
    _, graph, detached = prepared()
    model = graph.models_by_block["material"]
    before = emit_cpp_program(detached, model_graph=graph)
    object.__setattr__(model._m, "_review_unconsumed_cache", object())
    object.__setattr__(model._m, "prim_defs", {**model._m.prim_defs, "unused_constant": Const(9.)})
    assert emit_cpp_program(detached, model_graph=graph) == before


def test_public_compile_resolved_primitive_reaches_real_emission(monkeypatch, tmp_path):
    from tests.python.unit.codegen.test_module_lowering import _stub_toolchain
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    drivers = _stub_toolchain(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "pops._bootstrap", ModuleType("pops._bootstrap"))
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "pops_header_signature", lambda inc: "TESTSIG")
    monkeypatch.setattr(toolchain, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(toolchain, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx=None: "c++")
    class EmissionReached(Exception):
        pass
    emitted = []
    def no_compile(command, what):
        output = Path(command[command.index("-o") + 1])
        source = Path(next(arg for arg in command if arg.endswith(".cpp")))
        if source.name == "problem.cpp":
            emitted.append(source.read_text())
            raise EmissionReached
        output.write_bytes(b"SOURCE-ONLY-TEST-DSO")
        if "-MF" in command:
            Path(command[command.index("-MF") + 1]).write_text(f"{output}: {source}\n")
    monkeypatch.setattr(toolchain, "_run_compile", no_compile)
    monkeypatch.setattr(drivers, "_run_compile", no_compile)
    plan, _, _ = prepared(runtime=True)
    with pytest.raises(EmissionReached):
        pops.compile(plan)
    assert len(emitted) == 1
    assert "physical_global_" in emitted[0] and "params.get(0)" in emitted[0]
