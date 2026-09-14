"""Installed projections publish only their exact Uniform input dependencies."""
from __future__ import annotations

from dataclasses import replace
import re

import pytest

import pops
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_kernels import ProgramProviderPlans
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.variables import Conservative
from pops.time import FixedDt
from tests.python.support.layout_plan import cartesian_grid


def _resolved(*, providers=True, blocks=1, repeat=False):
    case = pops.Case("projection_input_case")
    program = pops.Program("projection_input_step")
    states = []
    for index in range(blocks):
        frame = Rectangle("square", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
        model = pops.Model("projection_model_%d" % index, frame=frame)
        state = model.state("U", components=("q",))
        q, = state
        imposed = model.aux("imposed")
        model.aux("unrelated")
        flux = model.flux("stationary", frame=frame, state=state,
                          components={frame.x: (0 * q,), frame.y: (0 * q,)},
                          waves={frame.x: (0,), frame.y: (0,)})
        rate = model.rate("transport", equation=ddt(state) == -div(flux))
        model.projection((imposed if providers else q,))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(
            flux=flux, variables=Conservative(state), reconstruction=FirstOrder(), riemann=Rusanov()))
        block = case.block("block_%d" % index, model)
        case.numerics(numerics, block=block)
        current = program.state(block[state])
        states.append((index, current, rate))
    residuals = [(index, current, rate(current.n)) for index, current, rate in states]
    candidates = []
    for index, current, residual in residuals:
        candidate = program.value("candidate_%d" % index,
                                  current.n + program.dt * residual, at=current.next.point)
        candidates.append((current, candidate))
    for current, candidate in candidates:
        projected = program.project(candidate)
        if repeat:
            projected = program.project(projected)
        program.commit(current.next, projected)
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=24, periodic=True)))


def _source(resolved, *, target="system"):
    return emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved), target=target)


def test_projection_prepares_exact_selected_state_and_pack_before_native_closure():
    resolved = _resolved(blocks=2, repeat=True)
    source = _source(resolved)
    preparations = list(re.finditer(
        r'ctx\.prepare_provider_values\("([^"\n]+)", (\d+), ([^,]+), (\d+)\);', source))
    projections = list(re.finditer(r"ctx\.apply_projection\((\d+), ([^)]+)\);", source))
    assert len(preparations) == len(projections) == 4
    assert len({match[1] for match in preparations}) == 4
    assert len({match[4] for match in preparations}) == 4
    for preparation, projection in zip(preparations, projections, strict=True):
        assert preparation.start() < projection.start()
        assert preparation.groups()[1:3] == projection.groups()
        assert preparation[1].startswith(resolved.blocks[int(preparation[2])].instance_owner_qid + "/program/")
        consumer = re.search(
            r'ConsumerPlan\{"' + re.escape(preparation[1]) + r'",\s*\n([^\n]+)', source)
        assert consumer is not None
        assert '"imposed"' in consumer[1]
        assert '"unrelated"' not in consumer[1]


def test_provider_free_projection_emits_no_auxiliary_publication():
    source = _source(_resolved(providers=False))
    assert "ctx.apply_projection(" in source
    assert "ctx.prepare_provider_values(" not in source


def test_amr_projection_retains_existing_preparation_route():
    source = _source(_resolved(), target="amr_system")
    assert "ctx.apply_projection(" in source
    assert "ctx.prepare_provider_values(" not in source
    assert "install_auxiliary_consumer_plan(ConsumerPlan{" not in source


def test_exact_projection_pack_registration_keeps_keys_and_rejects_conflicts():
    resolved = _resolved(blocks=2)
    graph = build_program_model_graph(resolved)
    from pops.codegen.program_models import model_for_node
    from pops.codegen.program_emit_kernels import _model_impl
    nodes = [node for node in resolved.time._values if node.op == "project"]
    packs = [_model_impl(model_for_node(graph, node))._component_operator_provider_packs["projection"]
             for node in nodes]
    plans = ProgramProviderPlans()
    first = plans.bind_pack(packs[0], "exact-consumer")
    assert first["count"] == 1
    key, = packs[0]
    assert plans._plans["exact-consumer"][0] == (key, packs[0].contract(key))
    assert plans.bind_pack(packs[0], "exact-consumer") == first
    with pytest.raises(ValueError, match="conflicting requirements"):
        plans.bind_pack(packs[1], "exact-consumer")
    with pytest.raises(TypeError, match="exact ProviderPack"):
        plans.bind_pack(None, "missing")
    from pops.model.provider_pack import ProviderPack
    bad = ProviderPack([(key, replace(packs[0].contract(key), centering="face", layout="face"),
                         packs[0].declared_entry(key))])
    with pytest.raises(ValueError, match="cell-layout"):
        plans.bind_pack(bad, "unsupported")
