"""The complete public path method must reach emitted synchronous SSA barriers."""
import pytest

import pops
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FanLi15RawMomentPath
from pops.projection import ConservativeCellAverage
from pops.time import AdaptiveCFL
from tests.python.support.physics_roles import FRAME
from tests.python.unit.numerics.test_fan_li15_path_contract import _declarations, _method


def test_public_ssprk2_path_retains_both_native_hierarchy_barriers():
    model, state, flux, product, covectors = _declarations()
    rate = model.rate("transport", equation=ddt(state) == -div(flux) - product)
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    plan = DiscretizationPlan()
    plan.rates.add(rate, _method(state, flux, path))
    case = pops.Case("two-stage Fan Li path source")
    block = case.block("moments", model)
    case.numerics(plan, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.2, max_dt=0.001))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=FRAME, cells=(8, 8), periodic=PeriodicAxes(FRAME.axes))))
    selected = resolved.blocks[0]
    emitter, _ = lower_and_validate(selected.model, state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    source = emit_cpp_program(resolved.time, model=emitter, target="amr_system")
    brick = _emit_bricks(emitter._m)[1]
    assert source.count("ctx.stage_path_rhs(") == 2
    assert source.count("ctx.publish_staged_path_rhs(") == 2
    assert source.count("HierarchyBarrierKind::spatial_rhs") == 2
    assert source.count("ctx.path_rhs_courant()") == 2
    assert source.count("ctx.capture_rhs_input_trace(") == 2
    assert "ctx.rhs_into(" not in source
    assert "path_conservative = true" in brick
    assert "PathKernel::admissibility(U)" in brick
    assert "path_integral(U, U, g)" in brick
    assert "integrate_normalized_moment_path<4>" in brick
    assert "fan_li15_path.hpp" not in brick
    assert "stability_speed(const State& U, const Providers& a)" in brick
    assert "pops::Real(4) * max_wave_speed<Axis>(U, a)" in brick
    assert "real_eig_minmax" not in brick
    assert emitter._m._path_conservative["identity"].startswith(
        "pops.fan-li15.path-operator.v1:sha256:")
    with pytest.raises(NotImplementedError, match="synchronous AMR"):
        emit_cpp_program(resolved.time, model=emitter, target="system")


def test_cpp_model_fixture_is_generated_from_the_python_constitutive_plan():
    from pathlib import Path
    from pops.codegen.moment_path_kernel import emit_fan_li15_test_header
    root = Path(__file__).resolve().parents[4]
    assert (root / "tests/cpp/support/generated_fan_li15.hpp").read_text() == emit_fan_li15_test_header()
    for directory in ("include/pops/numerics", "include/pops/runtime"):
        for header in (root / directory).rglob("*.hpp"):
            assert "fan_li15" not in header.read_text(), header


def test_moment_kernel_plan_cannot_reinterpret_permuted_raw_storage():
    from pops.codegen.moment_path_kernel import emit_moment_path_kernel
    from pops.moments.fan_li import fan_li15_native_plan
    plan = fan_li15_native_plan()
    plan["indices"] = tuple(reversed(plan["indices"]))
    with pytest.raises(ValueError, match="exact q-outer raw moment ordering"):
        emit_moment_path_kernel(plan, "InvalidKernel")


def test_typed_zero_measure_faces_cross_only_an_immutable_snapshot():
    from dataclasses import FrozenInstanceError
    from pops.problem._detached import detached_frozen
    model, state, flux, product, covectors = _declarations()
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    from pops.numerics import PathConservativeFiniteVolume, reconstruction, riemann, variables
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov(), zero_measure_faces=(FRAME.boundaries.x_min,))
    rate = model.rate("transport", equation=ddt(state) == -div(flux) - product)
    case = pops.Case("immutable pole contract")
    block = case.block("moments", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    selected = plan.resolve_for(case, block).rates[0].method
    detached = detached_frozen(selected)
    assert detached is not method
    assert detached.zero_measure_faces == (FRAME.boundaries.x_min,)
    with pytest.raises(FrozenInstanceError):
        detached.zero_measure_faces[0].coordinate = 1.0
    with pytest.raises(FrozenInstanceError):
        detached.zero_measure_faces[0].axis.direction = None
    with pytest.raises((RuntimeError, AttributeError)):
        detached.zero_measure_faces = ()


@pytest.mark.parametrize("reconstructed", [None, "prolongation", "coarse_fine"])
def test_public_path_amr_requires_parent_injection(reconstructed):
    from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                          Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.layouts import AMR
    from pops.lib.amr import (CoarseFineInjection, CoarseFineGhostInterpolation,
                             ConservativeInjection, ConservativeLinear, StateTransfer)
    from pops.time import always
    from pops.math import ValueExpr
    from pops.params import RuntimeParam

    model, state, flux, product, covectors = _declarations()
    rate = model.rate("transport", equation=ddt(state) == -div(flux) - product)
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, _method(state, flux, path))
    case = pops.Case("path AMR transfer admission")
    block = case.block("moments", model)
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.2, max_dt=0.001))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer(
        prolongation=ConservativeLinear() if reconstructed == "prolongation" else ConservativeInjection(),
        coarse_fine=CoarseFineGhostInterpolation() if reconstructed == "coarse_fine" else CoarseFineInjection()))
    threshold = case.param(RuntimeParam("refine_density", default=0.1))
    layout = AMR(grid=CartesianGrid(frame=FRAME, cells=(8, 8),
                                   periodic=PeriodicAxes(FRAME.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["M00"] > case.value(threshold)), Buffer(2)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(always(clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    if reconstructed is None:
        resolved = pops.resolve(pops.validate(case), layout=layout)
        assert resolved.target == "amr_system"
    else:
        with pytest.raises(NotImplementedError, match="exact conservative injection"):
            pops.resolve(pops.validate(case), layout=layout)
