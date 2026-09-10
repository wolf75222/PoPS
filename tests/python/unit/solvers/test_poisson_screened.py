"""Final screened-Poisson route: typed DSL, resolved install plan, and native MMS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.codegen._orchestration_compile import capture_field_plans
from pops.codegen.lowering_coverage import LoweringRejection
from pops.domain import Rectangle
from pops.fields import CellCenteredSecondOrder, FieldDiscretization, FieldOutput
from pops.fields.bcs import (
    AllPhysicalBoundaries,
    BoundaryCondition,
    Dirichlet,
    Periodic,
)
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import Const, ValueExpr, laplacian, unknown
from pops.mesh import CartesianGrid
from pops.mesh._amr import Above
from pops.params import ConstParam, RuntimeParam
from pops.physics import Model
from pops.solvers.elliptic import FFT, GeometricMG
from pops.time import FailRun, FixedDt
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout
from tests.python.support.native_execution_context import artifact_execution_context


ROOT = Path(__file__).resolve().parents[4]
N = 64
KAPPA = 50.0
DT = 1.0e-3


class _ScalarFieldIndicator(ValueExpr):
    """Test-owned symbolic indicator using the installed scalar-field tag leaf."""
    def __init__(self, handle, layout_subject):
        super().__init__(handle)
        self.layout_subject = layout_subject

    def __pops_ir_key__(self, recurse):
        return ("mms_scalar_field_indicator", self.handle.qualified_id,
                self.layout_subject.qualified_id)

    def resolve_for_amr_tagging(self, context, *, action, comparison, threshold):
        if action != "refine" or comparison != "gt":
            raise ValueError("MMS field indicator requires strict refine-above semantics")
        if self.handle.owner_path.nodes[0] != context.owner.nodes[0]:
            raise ValueError("MMS field indicator belongs to another Case")
        from pops.physics.board_handles import FieldHandle
        if not isinstance(self.handle, FieldHandle) or self.handle.kind != "field":
            raise ValueError("MMS field indicator requires an exact scalar model field")
        context.layout_plan.layout_for(self.layout_subject)
        return Above(self.handle, threshold)


def _screened_registration(*, solver, boundary):
    model = Model("screened-plan-model")
    (forcing,) = model.state("U", components=("forcing",))
    kappa = model.param(RuntimeParam("kappa", default=KAPPA))
    potential = model.field("potential")
    phi = unknown(potential)
    operator = model.field_operator(
        "screened",
        unknown=potential,
        equation=(-laplacian(phi) + model.value(kappa) * phi == forcing),
        outputs=(FieldOutput("screened_phi", potential),),
    )
    case = pops.Case("screened-plan-case")
    case.block("charge", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), boundary),),
            solver=solver,
        ),
    )
    return case, kappa


def test_screened_plan_carries_one_exact_qualified_scalar_and_refuses_fft() -> None:
    case, _ = _screened_registration(
        solver=GeometricMG(), boundary=Dirichlet(0.0))
    plan = capture_field_plans(
        case,
        lambda value: value,
        target="amr_system",
        layout=final_amr_layout(
            cartesian_grid(n=16, periodic=False), max_levels=2, ratio=2),
    )["screened"]

    reaction = plan.native_options["reaction"]
    handles = plan.provider_parameter_handles("native-install")
    assert reaction["kind"] == "scalar_bind_parameter"
    assert reaction["multiplier"] == 1.0
    assert len(handles) == 1
    assert handles[0].param_kind == "runtime"
    assert reaction["parameter"]["qualified_id"] == handles[0].qualified_id
    from pops.fields._prepared_field_nullspace_registry import (
        prepared_field_nullspace_binding_from_data,
    )

    nullspace = prepared_field_nullspace_binding_from_data(
        plan.native_options["nullspace_provider"]
    )
    assert nullspace.facts.kernel_components == 0
    assert nullspace.resolution.singular is False

    fft_case, _ = _screened_registration(solver=FFT(), boundary=Periodic())
    with pytest.raises(LoweringRejection, match="does not implement a screened operator"):
        capture_field_plans(
            fft_case,
            lambda value: value,
            target="system",
            layout=Uniform(cartesian_grid(n=16, periodic=True)),
        )


def _constant_screened_plan(*, coefficient_kind: str, target: str, layout=None):
    model = Model("constant-screening-%s-%s" % (coefficient_kind, target))
    (forcing,) = model.state("U", components=("forcing",))
    potential = model.field("potential")
    phi = unknown(potential)
    if coefficient_kind == "literal":
        coefficient = 0.5
    elif coefficient_kind == "const_param":
        coefficient = model.value(model.param(ConstParam("kappa", 0.5)))
        assert isinstance(coefficient, Const)
        assert coefficient.handle.param_kind == "const"
    else:  # pragma: no cover - test helper contract
        raise ValueError("unknown coefficient kind")
    operator = model.field_operator(
        "screened",
        unknown=potential,
        equation=(-laplacian(phi) + coefficient * phi == forcing),
        outputs=(FieldOutput("screened_phi", potential),),
    )
    case = pops.Case("constant-screening-%s-%s" % (coefficient_kind, target))
    case.block("charge", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(
                BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),
            ),
            solver=GeometricMG(),
        ),
    )
    return capture_field_plans(
        case,
        lambda value: value,
        target=target,
        layout=(
            final_amr_layout(cartesian_grid(n=16, periodic=False), max_levels=2, ratio=2)
            if layout is None else layout
        ),
    )["screened"]


@pytest.mark.parametrize("coefficient_kind", ("literal", "const_param"))
def test_screened_exact_constants_lower_without_a_fictitious_bind_slot(
    coefficient_kind: str,
) -> None:
    plan = _constant_screened_plan(
        coefficient_kind=coefficient_kind, target="amr_system")
    assert plan.native_options["reaction"] == {
        "schema_version": 1,
        "kind": "scalar_constant",
        "value": 0.5,
    }
    assert plan.provider_parameter_handles("native-install") == ()
    assert plan.bind_native_options({}) == {"reaction": 0.5}


@pytest.mark.parametrize(
    ("target", "installer"),
    (
        ("amr_system", "amr"),
    ),
)
def test_constant_reaction_reaches_both_native_install_protocols(
    target: str, installer: str,
) -> None:
    plan = _constant_screened_plan(coefficient_kind="const_param", target=target)
    installed = []

    class Native:
        def set_field_reaction(self, slot, value):
            installed.append((slot, value))

        def register_elliptic_field(self, *args):
            return None

    host = type("InstallHost", (), {"_s": Native()})()
    models = {
        "charge": type("CompiledModel", (), {"provider_components": ("screened_phi",)})()
    }
    if installer == "system":
        from pops.runtime._system_unified_install import _SystemUnifiedInstall
        _SystemUnifiedInstall._install_field_method_runtime(host, plan, models, {})
    else:
        from pops.runtime._amr_system_install import _AmrSystemInstall
        _AmrSystemInstall._install_field_method_runtime(host, plan, models, {})
    assert installed == [(plan.native_options["provider_slot"], 0.5)]


def test_refined_composite_screened_amr_lowers_to_the_reaction_capable_fac() -> None:
    layout = final_amr_layout(
        cartesian_grid(n=16, periodic=False), max_levels=2, ratio=2)
    plan = _constant_screened_plan(
        coefficient_kind="literal", target="amr_system", layout=layout)
    assert plan.native_options["hierarchy_policy"]["policy_id"] == (
        "pops.field-hierarchy.composite"
    )
    assert plan.native_options["reaction"] == {
        "schema_version": 1,
        "kind": "scalar_constant",
        "value": 0.5,
    }


def test_screened_authoring_refuses_an_unlowered_state_dependent_coefficient() -> None:
    model = Model("opaque-screening-model")
    (forcing,) = model.state("U", components=("forcing",))
    potential = model.field("potential")
    phi = unknown(potential)
    with pytest.raises(TypeError, match="exact finite real/ConstParam"):
        model.field_operator(
            "screened",
            unknown=potential,
            equation=(-laplacian(phi) + forcing * phi == forcing),
            outputs=(FieldOutput("screened_phi", potential),),
        )


def _resolved_mms_case():
    from pops.amr import (
        AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
        Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Outflow
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import EllipticRecompute, StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.projection import ConservativeCellAverage

    frame = Rectangle(
        "screened-mms-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = Model("screened-mms-model", frame=frame)
    state = model.state("U", components=("pure_forcing", "screened_forcing"))
    pure_forcing, screened_forcing = state
    x_axis, y_axis = frame.axes
    stationary_flux = model.flux(
        "stationary_forcing",
        frame=frame,
        state=state,
        components={
            x_axis: (0.0 * pure_forcing, 0.0 * screened_forcing),
            y_axis: (0.0 * pure_forcing, 0.0 * screened_forcing),
        },
        waves={
            x_axis: (0.0 * pure_forcing, 0.0 * screened_forcing),
            y_axis: (0.0 * pure_forcing, 0.0 * screened_forcing),
        },
    )
    stationary_rate = model.rate("stationary_rate", equation=ddt(state) == -div(stationary_flux))
    kappa = model.param(RuntimeParam("kappa", default=KAPPA))

    pure_potential = model.field("pure_potential")
    pure_phi = unknown(pure_potential)
    pure_operator = model.field_operator(
        "pure_poisson",
        unknown=pure_potential,
        equation=(-laplacian(pure_phi) == pure_forcing),
        outputs=(FieldOutput("pure_phi", pure_potential),),
    )
    screened_potential = model.field("screened_potential")
    screened_phi = unknown(screened_potential)
    screened_operator = model.field_operator(
        "screened_poisson",
        unknown=screened_potential,
        equation=(
            -laplacian(screened_phi)
            + model.value(kappa) * screened_phi
            == screened_forcing
        ),
        outputs=(FieldOutput("screened_phi", screened_potential),),
    )

    def discretization() -> FieldDiscretization:
        return FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(
                BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),
            ),
            solver=GeometricMG(),
        )
    case = pops.Case("screened-mms-case")
    block = case.block("charge", model)
    state_instance = block[state]
    # The forcing arrays are static physical data: dU/dt = -div(0) = 0.
    # Give that balance its explicit representation, boundary and bootstrap authorities.
    numerics = DiscretizationPlan()
    numerics.rates.add(stationary_rate, FiniteVolume(
        flux=stationary_flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov(),
    ))
    numerics.boundaries.add(TransportBoundarySet({
        face: Outflow(state=state_instance)
        for face in (frame.boundaries.x_min, frame.boundaries.x_max,
                     frame.boundaries.y_min, frame.boundaries.y_max)
    }))
    case.numerics(numerics, block=block)
    case.initials.add(InitialCondition(
        state=state_instance, value=BindArray(), projection=ConservativeCellAverage()))
    pure_field = case.field(pure_operator, discretization())
    screened_field = case.field(screened_operator, discretization())

    program = pops.Program("screened-mms-step")
    current = program.state(block[state])
    pure_field(current.n, name="solve_pure").consume(action=FailRun())
    screened_field(current.n, name="solve_screened").consume(action=FailRun())
    unchanged = program.value("unchanged", current.n, at=current.next.point)
    program.commit(current.next, unchanged)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    validated = pops.validate(case)
    bound_kappa = validated.resolve(kappa)
    transfer = AMRTransfer()
    transfer.state(state_instance, StateTransfer())
    transfer.field(pure_field, EllipticRecompute())
    transfer.field(screened_field, EllipticRecompute())
    predicate = _ScalarFieldIndicator(block[pure_potential], pure_field) > model.value(kappa)
    # AMR's provider contract owns an explicit possible transition. No tags and a
    # frozen hierarchy keep this MMS on exactly one N x N level, checked at runtime.
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(N, N), periodic=None),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(predicate & ~predicate), Buffer(1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    resolved = pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )
    slots = {
        name: resolved.field_plans[name].native_options["provider_slot"]
        for name in ("pure_poisson", "screened_poisson")
    }
    coordinates = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    initial = np.stack((
        2.0 * np.pi**2 * exact,
        (2.0 * np.pi**2 + KAPPA) * exact,
    ))
    return resolved, bound_kappa, slots, initial, exact


def test_mms_tagging_maps_the_qualified_unknown_to_its_scalar_output() -> None:
    from types import SimpleNamespace

    from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

    resolved, parameter, _, _, _ = _resolved_mms_case()
    pure_plan = resolved.field_plans["pure_poisson"]
    assert pure_plan.operator.unknown.local_id == "pure_potential"
    assert pure_plan.operator.outputs[0].name == "pure_phi"
    calls = []

    class Probe:
        def _set_bootstrap_tagging(self, *args):
            calls.append(args)

    flow_bootstrap_tagging(
        Probe(), resolved.bootstrap_plan, {parameter: KAPPA},
        clock_identity="mms-tagging-clock", field_plans=resolved.field_plans,
    )
    (call,) = calls
    assert call[1] == [pure_plan.operator.unknown.qualified_id] * 2
    assert call[3] == ["pure_phi"] * 2
    assert call[4] == [0, 0]

    calls.clear()
    with pytest.raises(ValueError, match="no authenticated field plan"):
        flow_bootstrap_tagging(
            Probe(), resolved.bootstrap_plan, {parameter: KAPPA},
            clock_identity="mms-tagging-clock",
            field_plans={"screened_poisson": resolved.field_plans["screened_poisson"]},
        )
    assert calls == []

    data = resolved.bootstrap_plan.tagging.runtime_tagging_data({parameter: KAPPA})
    data["refine"]["children"][0]["variable"] = "pure_potential"
    explicit_unknown_name = SimpleNamespace(tagging=SimpleNamespace(
        qualified_id=resolved.bootstrap_plan.tagging.qualified_id,
        runtime_tagging_data=lambda params: data,
    ))
    with pytest.raises(ValueError, match="absent from its prepared output route"):
        flow_bootstrap_tagging(
            Probe(), explicit_unknown_name, {parameter: KAPPA},
            clock_identity="mms-tagging-clock", field_plans=resolved.field_plans,
        )
    assert calls == []


@pytest.mark.compiler
@pytest.mark.native_loader
def test_pure_and_screened_public_equations_match_the_native_mms(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    resolved, parameter, slots, initial, exact = _resolved_mms_case()
    artifact = pops.compile(resolved)
    artifact.verify()
    execution_context = artifact_execution_context(artifact)
    (initial_binding,) = resolved.initial_condition_plan.bindings

    def solve(value: float):
        instance = pops.bind(
            artifact,
            initial_values={initial_binding.subject: np.ascontiguousarray(initial)},
            params={parameter: value},
            resources={"execution_context": execution_context},
        )
        assert instance.n_levels() == 1
        np.testing.assert_array_equal(
            np.asarray(instance.block_level_state_global("charge", 0)).reshape(initial.shape),
            initial,
        )
        report = pops.run(instance, t_end=DT, max_steps=1)
        assert report.accepted_steps == 1
        assert instance.n_levels() == 1
        np.testing.assert_array_equal(
            np.asarray(instance.block_level_state_global("charge", 0)).reshape(initial.shape),
            initial,
        )
        return {
            name: np.asarray(instance.field_potential_global(slot)).reshape(N, N)
            for name, slot in slots.items()
        }

    fields = solve(KAPPA)
    amplitude = float(np.max(np.abs(exact)))
    pure_error = float(np.max(np.abs(fields["pure_poisson"] - exact))) / amplitude
    screened_error = (
        float(np.max(np.abs(fields["screened_poisson"] - exact))) / amplitude
    )
    assert pure_error < 5.0e-3
    assert screened_error < 5.0e-3

    weaker = 5.0
    weaker_fields = solve(weaker)
    weaker_exact = exact * (2.0 * np.pi**2 + KAPPA) / (2.0 * np.pi**2 + weaker)
    np.testing.assert_allclose(
        weaker_fields["screened_poisson"], weaker_exact, rtol=5.0e-3, atol=5.0e-3)
    np.testing.assert_allclose(
        weaker_fields["pure_poisson"], fields["pure_poisson"], rtol=0.0, atol=2.0e-11)

    for invalid in (0.0, -1.0):
        with pytest.raises(ValueError, match="strictly positive at bind"):
            pops.bind(
                artifact,
                initial_values={initial_binding.subject: np.ascontiguousarray(initial)},
                params={parameter: invalid},
                resources={"execution_context": execution_context},
            )
