"""Named-flux composition through the final operator-first Module and Program APIs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
import pops.lib.time as libtime
import pops.model as model
from pops.codegen import Production
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import Var, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0e-3


def _named_flux_module():
    module = model.Module("named-flux-module")
    state_space = module.state_space(
        "U",
        ("rho", "mx", "my"),
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    state = module.state_handle(state_space)
    rho = Var("rho", "cons")
    mx = Var("mx", "cons")
    my = Var("my", "cons")
    u = mx / rho
    v = my / rho
    pressure = 0.5 * rho

    whole_body = {
        "x": (mx, mx * u + pressure, my * u),
        "y": (my, mx * v, my * v + pressure),
    }
    convective_body = {
        "x": (mx, mx * u, my * u),
        "y": (my, mx * v, my * v),
    }
    pressure_body = {
        "x": (0.0 * rho, pressure, 0.0 * rho),
        "y": (0.0 * rho, 0.0 * rho, pressure),
    }
    signature = (state_space,) >> model.Rate(state_space)
    default_flux = module.operator(
        name="flux_default", signature=signature, kind="grid_operator", expr=whole_body)
    whole_flux = module.operator(
        name="whole", signature=signature, kind="grid_operator", expr=whole_body)
    convective_flux = module.operator(
        name="convective", signature=signature, kind="grid_operator", expr=convective_body)
    pressure_flux = module.operator(
        name="pressure", signature=signature, kind="grid_operator", expr=pressure_body)
    sound_speed = sqrt(0.5)
    module.eigenvalues(
        x=(u - sound_speed, u, u + sound_speed),
        y=(v - sound_speed, v, v + sound_speed),
    )
    whole_rate = module.rate_operator(
        "whole_rate", state_space=state, fluxes=(whole_flux,))
    split_rate = module.rate_operator(
        "split_rate", state_space=state, fluxes=(convective_flux, pressure_flux))
    return module, state, default_flux, whole_rate, split_rate


def _program(module, state, rate, *, name: str):
    case = pops.Case(name + "-case")
    block = case.block("plasma", module)
    # A hand-authored operator-first Module already carries its exact grid-operator formulas.
    # FiniteVolume is the higher-level board contract for a physical FluxHandle and must not be
    # forged from a grid_operator merely to drive this lower-level composition oracle.
    program = libtime.ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    return case, program


def test_default_flux_route_does_not_reclassify_named_flux_operators() -> None:
    module, state, default_flux, whole_rate, _ = _named_flux_module()
    default_rate = module.rate_operator(
        "default_rate",
        state_space=state,
        fluxes=(default_flux,),
        default_flux=default_flux,
    )
    _, default_program = _program(module, state, default_rate, name="default-route")
    _, named_program = _program(module, state, whole_rate, name="named-route")

    default_rhs = [node for node in default_program.ir_nodes() if node["op"] == "rhs"]
    named_rhs = [node for node in named_program.ir_nodes() if node["op"] == "rhs"]
    assert len(default_rhs) == len(named_rhs) == 1
    assert default_rhs[0]["attrs"]["fluxes"] is None
    assert named_rhs[0]["attrs"]["fluxes"] == ["whole"]

    lowered = module.to_dsl()
    default_source = emit_cpp_program(default_program, model=lowered)
    named_source = emit_cpp_program(named_program, model=lowered)
    assert "ctx.neg_div_flux_default_into(0," in default_source
    assert "ctx.neg_div_flux_into(" not in default_source
    assert "ctx.neg_div_flux_into(" in named_source


def test_named_flux_sum_lowers_to_one_divergence_kernel() -> None:
    module, state, _, whole_rate, split_rate = _named_flux_module()
    _, whole_program = _program(module, state, whole_rate, name="whole-flux")
    _, split_program = _program(module, state, split_rate, name="split-flux")

    whole_source = emit_cpp_program(whole_program, model=module.to_dsl())
    split_source = emit_cpp_program(split_program, model=module.to_dsl())
    assert whole_source.count("ctx.neg_div_flux_into(") == 1
    assert split_source.count("ctx.neg_div_flux_into(") == 1
    assert "ctx.rhs_into(0," not in split_source


@pytest.mark.compiler
@pytest.mark.native_loader
def test_split_named_flux_step_matches_whole_named_flux_step(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    module, state, _, whole_rate, split_rate = _named_flux_module()
    whole_case, _ = _program(module, state, whole_rate, name="whole-flux-runtime")
    split_case, _ = _program(module, state, split_rate, name="split-flux-runtime")
    frame = Rectangle(
        "named-flux-runtime-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(N, N),
        periodic=PeriodicAxes(frame.axes),
    ))
    resolve_options = {
        "layout": layout,
        "backend": Production(),
        "compile_options": {"include": str(ROOT / "include")},
    }
    whole_resolved = pops.resolve(pops.validate(whole_case), **resolve_options)
    split_resolved = pops.resolve(pops.validate(split_case), **resolve_options)
    whole_artifact = pops.compile(whole_resolved)
    split_artifact = pops.compile(split_resolved)

    coordinates = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    initial = np.stack((rho, 0.4 * rho, -0.2 * rho))

    def advance(artifact):
        instance = pops.bind(artifact, initial_state={"plasma": initial})
        report = pops.run(instance, t_end=DT, max_steps=1)
        assert report.accepted_steps == 1
        return np.asarray(instance.get_state("plasma"))

    whole = advance(whole_artifact)
    split = advance(split_artifact)
    np.testing.assert_allclose(split, whole, rtol=0.0, atol=2.0e-13)
    assert float(np.max(np.abs(whole - initial))) > 1.0e-6
