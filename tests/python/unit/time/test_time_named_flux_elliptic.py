"""Named-flux composition through the final operator-first Module and Program APIs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
import pops.lib.time as libtime
import pops.model as model
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, Var, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0e-3
NSTEPS = 1
REFINE_THRESHOLD = 1.12


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


def _program(module, state, rate, *, name: str, numerics=None, bind_initial: bool = False):
    case = pops.Case(name + "-case")
    block = case.block("plasma", module)
    state_instance = block[state]
    # The Program retains only the physical time graph. Runtime tests attach the separate,
    # explicit numerical authority below; source-emission tests do not need to resolve a Case.
    program = libtime.ForwardEuler(state_instance, rate=rate)
    program.step_strategy(FixedDt(DT))
    if numerics is not None:
        case.numerics(numerics, block=block)
    if bind_initial:
        case.initials.add(InitialCondition(
            state=state_instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        ))
    case.program(program)
    return case, program, state_instance


def _numerical_plan(module, state, *rates):
    plan = DiscretizationPlan()
    for rate in rates:
        contract = module.rate_contract(rate)
        plan.rates.add(
            rate,
            FiniteVolume(
                flux=contract["flux"],
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
    return plan


def test_default_flux_route_does_not_reclassify_named_flux_operators() -> None:
    module, state, default_flux, whole_rate, _ = _named_flux_module()
    default_rate = module.rate_operator(
        "default_rate",
        state_space=state,
        fluxes=(default_flux,),
        default_flux=default_flux,
    )
    _, default_program, _ = _program(module, state, default_rate, name="default-route")
    _, named_program, _ = _program(module, state, whole_rate, name="named-route")

    default_rhs = [node for node in default_program.ir_nodes() if node["op"] == "rhs"]
    named_rhs = [node for node in named_program.ir_nodes() if node["op"] == "rhs"]
    assert len(default_rhs) == len(named_rhs) == 1
    assert default_rhs[0]["attrs"]["fluxes"] is None
    assert named_rhs[0]["attrs"]["fluxes"] == ["whole"]

    lowered = module.to_dsl()
    default_source = emit_cpp_program(default_program, model=lowered)
    named_source = emit_cpp_program(named_program, model=lowered)
    assert "ctx.neg_div_flux_default_into(0," in default_source
    assert "ctx.neg_div_named_flux_into(" not in default_source
    assert "ctx.neg_div_named_flux_into(" in named_source


def test_named_flux_sum_lowers_to_one_divergence_kernel() -> None:
    module, state, _, whole_rate, split_rate = _named_flux_module()
    _, whole_program, _ = _program(module, state, whole_rate, name="whole-flux")
    _, split_program, _ = _program(module, state, split_rate, name="split-flux")

    whole_source = emit_cpp_program(whole_program, model=module.to_dsl())
    split_source = emit_cpp_program(split_program, model=module.to_dsl())
    assert whole_source.count("ctx.neg_div_named_flux_into(") == 1
    assert split_source.count("ctx.neg_div_named_flux_into(") == 1
    assert "ctx.rhs_into(0," not in split_source
    assert split_source.rindex("for (int li = 0;") < split_source.index(
        "ctx.neg_div_named_flux_into("
    )


def test_named_flux_rate_contract_retains_the_exact_ordered_operator_pack() -> None:
    module, state, _, whole_rate, split_rate = _named_flux_module()

    whole = module.rate_contract(whole_rate)
    split = module.rate_contract(split_rate)
    assert whole["state"] == split["state"] == state
    assert tuple(handle.local_id for handle in whole["flux"]) == ("whole",)
    assert tuple(handle.local_id for handle in split["flux"]) == (
        "convective", "pressure")
    assert all(handle.kind == "grid_operator" for handle in (*whole["flux"], *split["flux"]))

    with pytest.raises(ValueError, match="non-empty"):
        FiniteVolume(
            flux=(), variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov(),
        )

    foreign = model.Module("foreign-flux")
    foreign_state = foreign.state_space("U", ("rho", "mx", "my"))
    foreign_flux = foreign.operator(
        "foreign",
        signature=(foreign_state,) >> model.Rate(foreign_state),
        kind="grid_operator",
        expr={"x": (0.0, 0.0, 0.0), "y": (0.0, 0.0, 0.0)},
    )
    with pytest.raises(ValueError, match="different Models"):
        FiniteVolume(
            flux=(whole["flux"][0], foreign_flux),
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov(),
        )

    case, _, _ = _program(
        module, state, whole_rate, name="ordered-flux-contract",
        numerics=_numerical_plan(module, state, whole_rate, split_rate),
    )
    frame = Rectangle(
        "ordered-flux-contract-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    resolved = pops.resolve(
        pops.validate(case),
        layout=Uniform(CartesianGrid(
            frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes))),
    )
    methods = {
        row.rate.local_id: row.method.to_data()
        for row in resolved.blocks[0].numerics.rates
    }
    assert [item["local_id"] for item in methods["split_rate"]["flux"]] == [
        "convective", "pressure"]


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("layout_kind", ("uniform", "amr"))
def test_split_named_flux_step_matches_whole_named_flux_step_on_public_layouts(
    layout_kind,
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    module, state, _, whole_rate, split_rate = _named_flux_module()
    whole_case, whole_program, whole_state = _program(
        module, state, whole_rate, name="whole-flux-runtime",
        numerics=_numerical_plan(module, state, whole_rate, split_rate),
        bind_initial=True,
    )
    split_case, split_program, split_state = _program(
        module, state, split_rate, name="split-flux-runtime",
        numerics=_numerical_plan(module, state, whole_rate, split_rate),
        bind_initial=True,
    )
    frame = Rectangle(
        "named-flux-runtime-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    grid = CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes))

    def resolved_layout(state_instance, program):
        if layout_kind == "uniform":
            return Uniform(grid)
        transfer = AMRTransfer()
        transfer.state(state_instance, StateTransfer())
        return AMR(
            grid=grid,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(
                    Tag(ValueExpr(state_instance) > REFINE_THRESHOLD),
                    Buffer(cells=1),
                ),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
        )

    resolve_options = {
        "backend": Production(),
        "compile_options": {"include": str(ROOT / "include")},
    }
    whole_resolved = pops.resolve(
        pops.validate(whole_case),
        layout=resolved_layout(whole_state, whole_program),
        **resolve_options,
    )
    split_resolved = pops.resolve(
        pops.validate(split_case),
        layout=resolved_layout(split_state, split_program),
        **resolve_options,
    )
    whole_artifact = pops.compile(whole_resolved)
    split_artifact = pops.compile(split_resolved)

    coordinates = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    rho = 1.0 + 0.4 * np.exp(-((x - 0.37) ** 2 + (y - 0.53) ** 2) / 0.015)
    velocity_x = 0.2 + 0.3 * np.sin(2.0 * np.pi * y)
    velocity_y = -0.1 + 0.25 * np.cos(2.0 * np.pi * x)
    initial = np.stack((rho, velocity_x * rho, velocity_y * rho))
    assert float(np.ptp(velocity_x)) > 0.5
    assert float(np.ptp(velocity_y)) > 0.4

    def advance(artifact, resolved):
        bindings = tuple(resolved.initial_condition_plan.bindings)
        assert len(bindings) == 1
        instance = pops.bind(
            artifact,
            initial_values={bindings[0].subject: np.ascontiguousarray(initial)},
        )
        if layout_kind == "amr":
            assert instance.n_levels() == 2
            fine_boxes = tuple(
                (tuple(lower), tuple(upper))
                for level, lower, upper in instance.patch_boxes()
                if int(level) == 1
            )
            assert fine_boxes
            fine_cell_count = sum(
                int(np.prod(np.subtract(upper, lower) + 1))
                for lower, upper in fine_boxes
            )
            # A strict partial fine covering proves that this run owns real coarse/fine faces;
            # full-domain refinement would make a conservation assertion vacuous for reflux.
            assert 0 < fine_cell_count < (2 * N) ** 2
            initial_mass = instance.integral("plasma", component=0, levels=(0,))
        report = pops.run(instance, t_end=NSTEPS * DT, max_steps=NSTEPS)
        assert report.accepted_steps == NSTEPS
        if layout_kind == "amr":
            final_mass = instance.integral("plasma", component=0, levels=(0,))
            mass_scale = max(1.0, abs(initial_mass))
            assert abs(final_mass - initial_mass) <= 2.0e-12 * mass_scale
            return np.asarray(instance.block_level_state_global("plasma", 0)).reshape(initial.shape)
        return np.asarray(instance.get_state("plasma")).reshape(initial.shape)

    whole = advance(whole_artifact, whole_resolved)
    split = advance(split_artifact, split_resolved)
    np.testing.assert_allclose(split, whole, rtol=0.0, atol=2.0e-13)
    assert float(np.max(np.abs(whole - initial))) > 1.0e-6
