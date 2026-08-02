"""Direct CompositeTensorFAC identity and authoring contract."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pops.identity.scalar import scalar_data, scalar_literal
from pops.fields import ConstantNullspace, MeanValueGauge
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.native_components import PreparedNativeComponent
from pops.solvers import (
    CG,
    CompositeTensorFAC,
    Hierarchy,
    PreparedHierarchyFlatExecution,
    PreparedHierarchySolverUseFacts,
    PreparedHierarchySolverUsePolicy,
    prepared_hierarchy_solver_provider_by_id,
    register_prepared_hierarchy_solver_provider,
    solvers,
)
from pops.solvers.providers import (
    PreparedHierarchySolverNativeEmission,
    prepared_hierarchy_solver_provider_from_attrs,
)
from pops.time import Program
from tests.python.support.native_execution_context import artifact_execution_context


_HIERARCHY_BASE_CELLS = 8
_HIERARCHY_DT = 0.01
_HIERARCHY_ROTATION_RATE = 3.0
_HIERARCHY_DENSITY = 2.0
_HIERARCHY_POTENTIAL_AMPLITUDE = 0.01


@pytest.mark.parametrize(
    "option",
    [
        {"max_iter": True},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": 1 << 31},
        {"rel_tol": True},
        {"rel_tol": 0},
        {"rel_tol": 1},
        {"rel_tol": float("nan")},
        {"abs_tol": True},
        {"abs_tol": -1},
        {"abs_tol": float("nan")},
        {"fine_sweeps": True},
        {"fine_sweeps": 1.5},
        {"fine_sweeps": 0},
        {"fine_sweeps": 1 << 31},
        {"coarse_cycles": False},
        {"coarse_cycles": "4"},
        {"coarse_cycles": -1},
        {"coarse_cycles": 1 << 31},
        {"coarse_rel_tol": True},
        {"coarse_rel_tol": 0},
        {"coarse_rel_tol": 1},
        {"coarse_rel_tol": float("nan")},
        {"coarse_rel_tol": float("inf")},
        {"coarse_abs_tol": True},
        {"coarse_abs_tol": -1},
        {"verbose": 0},
        {"verbose": "true"},
    ],
)
def test_solver_options_are_strict(option):
    with pytest.raises((TypeError, ValueError)):
        CompositeTensorFAC(**option)


def test_identity_owns_complete_flat_and_refined_solve_contract():
    default = CompositeTensorFAC()
    assert solvers.CompositeTensorFAC is CompositeTensorFAC
    assert default.max_iter == 30
    assert default.rel_tol == 1.0e-9
    assert default.abs_tol == 0.0
    assert default.fine_sweeps is None
    assert default.coarse_rel_tol is None
    assert default.coarse_abs_tol is None
    assert default.coarse_cycles is None
    assert default.verbose is None

    configured = CompositeTensorFAC(
        max_iter=23,
        rel_tol=Fraction(3, 100_000_000),
        abs_tol=Fraction(1, 1_000_000_000_000),
        fine_sweeps=7,
        coarse_rel_tol=Fraction(1, 8),
        coarse_abs_tol=Fraction(1, 10_000_000_000_000),
        coarse_cycles=9,
        verbose=False,
    )
    identity = configured.canonical_identity()
    assert set(identity) == {"schema_version", "provider", "options"}
    assert identity["schema_version"] == 1
    authority = identity["provider"]
    assert authority["provider_id"] == "pops.hierarchy.composite-tensor-fac"
    assert authority["interface_version"] == 1
    assert authority["capabilities"] == [
        "pops.hierarchy.composite-tensor-fac.exact-preparation@1",
        "pops.hierarchy.composite-tensor-fac.flat-krylov@1",
        "pops.hierarchy.composite-tensor-fac.mixed-level-distribution@1",
        "pops.hierarchy.composite-tensor-fac.refined-direct@1",
    ]
    assert authority["flat_execution"]["mode"] == "prepared_krylov_fallback"
    assert authority["flat_execution"]["krylov"]["method_provider"]["provider_id"] == (
        "pops.krylov.bicgstab"
    )
    assert authority["native_component"]["component_id"] == (
        "pops.hierarchy.composite-tensor-fac"
    )
    assert identity["options"] == {
        "max_iter": 23,
        "rel_tol": scalar_data(Fraction(3, 100_000_000)),
        "abs_tol": scalar_data(Fraction(1, 1_000_000_000_000)),
        "fine_sweeps": 7,
        "coarse_rel_tol": scalar_data(Fraction(1, 8)),
        "coarse_abs_tol": scalar_data(Fraction(1, 10_000_000_000_000)),
        "coarse_cycles": 9,
        "verbose": False,
    }
    assert configured.identity != default.identity
    prepared = configured.prepare_program_solve()
    assert prepared.identity_data == identity
    assert prepared.identity.token == configured.identity.token


@pytest.mark.parametrize(
    "forbidden_attribute",
    (
        "method_provider",
        "method_options",
        "preconditioner",
        "preconditioner_provider",
        "preconditioner_options",
        "krylov_footprint",
        "krylov_workspace",
    ),
)
def test_flat_direct_execution_contract_has_no_implicit_krylov_authority(
    forbidden_attribute: str,
):
    execution = PreparedHierarchyFlatExecution.direct_provider()
    assert execution.authority() == {
        "interface": "pops.prepared-hierarchy-flat-execution@1",
        "mode": "direct_provider",
        "krylov": None,
    }
    assert execution.ir_attributes(unused=True) == {}
    execution.validate_ir({}, where="flat direct test")
    with pytest.raises(ValueError, match="unexpected Krylov attributes"):
        execution.validate_ir({forbidden_attribute: {}}, where="flat direct test")


def test_external_use_policy_accepts_any_ncomp_and_future_fact_without_core_branch():
    def validate(facts, operator, _where):
        assert facts.components == 37
        assert facts.extensions["tests.future.tensor_rank"] == 4
        assert operator is future_operator
        return facts

    policy = PreparedHierarchySolverUsePolicy(
        policy_id="tests.use-policy.any-component-future-facts",
        interface_version=1,
        capabilities=frozenset(),
        validator=validate,
    )
    facts = PreparedHierarchySolverUseFacts(
        target="amr_system",
        scope="hierarchy",
        problem_kind="tests.future_operator",
        domain="tests.vector-space",
        range="tests.vector-space",
        components=37,
        singular_nullspace=False,
        extensions={"tests.future.tensor_rank": 4},
    )
    future_operator = object()
    assert policy.validate(
        facts, operator=future_operator, where="external provider"
    ) is facts
    assert policy.authority() == {
        "policy_id": "tests.use-policy.any-component-future-facts",
        "interface_version": 1,
        "capabilities": [],
    }


def test_provider_integer_controls_accept_the_complete_native_int_range():
    cpp_int_max = (1 << 31) - 1
    configured = CompositeTensorFAC(
        max_iter=cpp_int_max,
        fine_sweeps=cpp_int_max,
        coarse_cycles=cpp_int_max,
    )
    assert configured.max_iter == cpp_int_max
    assert configured.fine_sweeps == cpp_int_max
    assert configured.coarse_cycles == cpp_int_max


@pytest.mark.parametrize("name", ["max_iter", "fine_sweeps", "coarse_cycles"])
def test_codegen_rejects_forged_composite_fac_integer_overflow(name):
    from test_hierarchy_scoped_solve_emit import _build

    solver = CompositeTensorFAC()
    program, _ = _build(solver)
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    options = dict(attrs["hierarchy_solver_options"])
    options[name] = 1 << 31
    attrs["hierarchy_solver_options"] = options
    node = SimpleNamespace(attrs=attrs, inputs=solve.inputs)
    provider = prepared_hierarchy_solver_provider_by_id(
        "pops.hierarchy.composite-tensor-fac"
    )

    with pytest.raises(ValueError, match=name):
        provider.validate_node(node, target="amr_system")


def test_codegen_rejects_noncanonical_composite_fac_options_with_valid_identity():
    from test_hierarchy_scoped_solve_emit import _build

    program, _ = _build(CompositeTensorFAC())
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    attrs["hierarchy_solver_options"] = {}

    with pytest.raises(ValueError, match="hierarchy solve options are not canonical"):
        prepared_hierarchy_solver_provider_from_attrs(attrs)


def test_codegen_rejects_flat_absolute_tolerance_that_disagrees_with_provider_identity():
    from test_hierarchy_scoped_solve_emit import _build

    solver = CompositeTensorFAC(abs_tol=Fraction(1, 10_000))
    program, _ = _build(solver)
    solve = next(value for value in program._values if value.op == "solve_linear")
    attrs = dict(solve.attrs)
    attrs["abs_tol"] = scalar_literal(0)
    node = SimpleNamespace(attrs=attrs, inputs=solve.inputs)
    provider = prepared_hierarchy_solver_provider_by_id(
        "pops.hierarchy.composite-tensor-fac"
    )

    with pytest.raises(ValueError, match="convergence controls disagree"):
        provider.validate_node(node, target="amr_system")


def test_program_rejects_forged_composite_fac_negative_absolute_tolerance_before_codegen():
    from test_hierarchy_scoped_solve_emit import _build

    prepared = CompositeTensorFAC().prepare_program_solve()
    options = prepared.options
    options["abs_tol"] = scalar_data(Fraction(-1, 10))
    prepared = replace(
        prepared,
        _options_json=json.dumps(options, sort_keys=True, separators=(",", ":")),
    )

    class ForgedDescriptor:
        def prepare_program_solve(self):
            return prepared

    with pytest.raises(ValueError, match="abs_tol"):
        _build(ForgedDescriptor())


def test_krylov_descriptor_rejects_hierarchy_scope_before_codegen():
    program = Program("krylov-hierarchy-rejected")
    operator = program.matrix_free_operator("operator", scope=Hierarchy())
    rhs = program.scalar_field("rhs")
    problem = LinearProblem(operator, rhs, scope=Hierarchy(), nullspace=None)

    with pytest.raises(TypeError, match="prepared hierarchy-solver provider.*Krylov descriptors"):
        program.solve(problem, solver=CG(max_iter=11, rel_tol=1.0e-6))


def test_composite_provider_refuses_constant_nullspace_until_multilevel_gauge_is_wired():
    from test_hierarchy_scoped_solve_emit import _build

    with pytest.raises(NotImplementedError, match="does not support a singular nullspace"):
        _build(
            CompositeTensorFAC(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0),
            properties=(
                LinearOperatorProperties.symmetric_positive_definite_on_nullspace_complement()
            ),
        )


def test_hierarchy_operator_is_provider_neutral_until_solver_selection():
    program = Program("direct-hierarchy-contract")
    with pytest.raises(TypeError, match="unexpected keyword argument 'provider'"):
        program.matrix_free_operator(
            "legacy", scope=Hierarchy(), provider=CompositeTensorFAC()  # type: ignore[call-arg]
        )
    vector = program.matrix_free_operator(
        "vector", domain="vector", range_="vector", ncomp=2, scope=Hierarchy()
    )
    assert vector.attrs["domain"] == "vector"
    assert vector.attrs["ncomp"] == 2
    assert "hierarchy_solver_provider" not in vector.attrs


def test_hierarchy_apply_rejects_any_unproven_operator_shape():
    program = Program("unproven-hierarchy-apply")
    operator = program.matrix_free_operator("operator", scope=Hierarchy())
    operator = program.set_apply(operator, lambda _program, _out, value: value)
    rhs = program.scalar_field("rhs")
    with pytest.raises(ValueError, match="one scalar scratch"):
        program.solve(
            LinearProblem(operator, rhs, scope=Hierarchy(), nullspace=None),
            solver=CompositeTensorFAC(),
        )


def test_hierarchy_provider_registry_is_append_only():
    provider = prepared_hierarchy_solver_provider_by_id(
        "pops.hierarchy.composite-tensor-fac"
    )
    with pytest.raises(ValueError, match="already registered"):
        register_prepared_hierarchy_solver_provider(provider)


def test_hierarchy_program_and_codegen_core_have_no_builtin_backend_branch():
    root = Path(__file__).resolve().parents[4]
    core = (
        root / "python/pops/time/_program/local.py",
        root / "python/pops/codegen/program_emit_solve.py",
        root / "python/pops/codegen/program_emit_control.py",
        root / "python/pops/codegen/program_emit_kernels.py",
    )
    for path in core:
        source = path.read_text(encoding="utf-8")
        assert "composite_tensor_fac" not in source
        assert "CompositeTensorFAC" not in source


def _public_amr_hierarchy_case(
    solver,
    *,
    max_levels=2,
    temporal_ratios=(3,),
    bound_plasma=False,
    manufactured_plasma=False,
    base_cells=_HIERARCHY_BASE_CELLS,
):
    import pops
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
    from pops.domain import Rectangle
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Outflow
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.analytic import coordinates, cos, sin
    from pops.lib.initial import Analytic, BindArray, Constant, Gaussian
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import (
        DiscretizationPlan,
        FiniteVolume,
        reconstruction,
        riemann,
        variables,
    )
    from pops.params import RuntimeParam
    from pops.physics import Density, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import FailRun, FixedDt, StagePoint, TimePoint, every

    frame = Rectangle(
        "external-hierarchy-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    if bound_plasma and manufactured_plasma:
        raise ValueError("the bound and manufactured plasma profiles are exclusive")

    model = pops.Model("external-hierarchy-plasma", frame=frame)
    state = model.state(
        "U",
        components=("density", "east_momentum", "north_momentum"),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={
            "density": Density(),
            "east_momentum": Momentum(axis=x_axis),
            "north_momentum": Momentum(axis=y_axis),
        },
    )
    density, east_momentum, north_momentum = state
    zero_flux = (
        0.0 * density,
        0.0 * east_momentum,
        0.0 * north_momentum,
    )
    flux = model.flux(
        "inert-transport",
        frame=frame,
        state=state,
        components={x_axis: zero_flux, y_axis: zero_flux},
        waves={x_axis: (0.0, 0.0, 0.0), y_axis: (0.0, 0.0, 0.0)},
    )
    rate = model.rate("inert-rate", equation=ddt(state) == -div(flux))
    rotation = model.operator(
        "implicit-rotation",
        returns=model.local_linear_operator(
            "implicit-rotation",
            on=state,
            matrix=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 3.0),
                (0.0, -3.0, 0.0),
            ),
        ),
    )

    marker_model = pops.Model("external-hierarchy-marker", frame=frame)
    marker_state = marker_model.state(
        "U",
        components=("marker",),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={"marker": Density()},
    )
    (marker,) = marker_state
    marker_x_speed = 0.25
    marker_y_speed = -0.1
    marker_flux = marker_model.flux(
        "marker-transport",
        frame=frame,
        state=marker_state,
        components={
            x_axis: (marker_x_speed * marker,),
            y_axis: (marker_y_speed * marker,),
        },
        waves={x_axis: (marker_x_speed,), y_axis: (marker_y_speed,)},
    )
    marker_rate = marker_model.rate(
        "marker-rate", equation=ddt(marker_state) == -div(marker_flux)
    )

    case = pops.Case("public-external-hierarchy-provider")
    block = case.block("plasma", model=model)
    marker_block = case.block("marker", model=marker_model)
    state_instance = block[state]
    marker_instance = marker_block[marker_state]
    for owner, declared_rate, declared_flux, declared_state, declared_instance in (
        (block, rate, flux, state, state_instance),
        (marker_block, marker_rate, marker_flux, marker_state, marker_instance),
    ):
        numerics = DiscretizationPlan()
        numerics.rates.add(
            declared_rate,
            FiniteVolume(
                flux=declared_flux,
                variables=variables.Conservative(declared_state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        if bound_plasma:
            boundaries = frame.boundaries
            numerics.boundaries.add(
                TransportBoundarySet(
                    {
                        boundaries.x_min: Outflow(state=declared_instance),
                        boundaries.x_max: Outflow(state=declared_instance),
                        boundaries.y_min: Outflow(state=declared_instance),
                        boundaries.y_max: Outflow(state=declared_instance),
                    }
                )
            )
        case.numerics(numerics, block=owner)

    program = pops.Program("public-external-hierarchy-step")
    temporal = program.state(state_instance)
    marker_temporal = program.state(marker_instance)
    coefficients = program.condensed_coeffs(
        "tensor-coefficients",
        state=temporal.n,
        linear_operator=rotation,
        subset=(1, 2),
        c=program.dt * program.dt,
        th_dt=program.dt,
        c_rho=0,
    )
    previous = program.history("plasma.tensor-potential", lag=1, ncomp=1, block=block)
    rhs_storage = program.scalar_field("tensor-rhs-storage")
    rhs = program.condensed_rhs(
        rhs_storage,
        previous,
        temporal.n,
        linear_operator=rotation,
        subset=(1, 2),
        th_dt=program.dt,
        g=program.dt,
    )
    operator = program.matrix_free_operator("tensor-operator", scope=Hierarchy())

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("tensor-laplacian")
        return -1 * builder.apply_laplacian_coeff(laplacian, value, coefficients)

    program.set_apply(operator, apply)
    potential = program.solve(
        LinearProblem(
            operator,
            rhs,
            initial_guess=previous,
            scope=Hierarchy(),
            nullspace=None,
        ),
        solver=solver,
        name="tensor-potential",
    ).consume(action=FailRun())
    program.store_history("plasma.tensor-potential", potential)
    reconstructed = program.condensed_reconstruct(
        "reconstructed-state",
        state=temporal.n,
        phi=potential,
        linear_operator=rotation,
        subset=(1, 2),
        th_dt=program.dt,
        c_rho=0,
    )
    # This independent conservative block belongs to the publish region: its explicit flux work is
    # evaluated after the hierarchy barrier and is therefore still present when couple_levels()
    # performs reflux.  It intentionally shares neither state nor scratch with the condensed block.
    explicit_point = StagePoint(
        "transport-stage", {"main": TimePoint(program.clock, 0)}
    )
    explicit_marker_rate = program.value(
        "marker-transport-rate", marker_rate(marker_temporal.n), at=explicit_point
    )
    transported_marker = program.value(
        "transported-marker",
        marker_temporal.n + program.dt * explicit_marker_rate,
        at=marker_temporal.next.point,
    )
    program.commit(
        temporal.next,
        program.value("accepted-state", reconstructed, at=temporal.next.point),
    )
    program.commit(
        marker_temporal.next,
        program.value(
            "accepted-marker", transported_marker, at=marker_temporal.next.point
        ),
    )
    program.step_strategy(FixedDt(_HIERARCHY_DT))
    case.program(program)

    plasma_initial = Constant((2.0, 0.25, -0.5))
    if bound_plasma:
        plasma_initial = BindArray()
    elif manufactured_plasma:
        # Continuous manufactured solution for the first condensed solve.  With
        # M = I - dt*J, A = I + dt^2*rho*M^-1 and phi*=a sin(2pi x)sin(2pi y),
        # choose m = dt^-1 M A grad(phi*).  Then the authored condensed RHS is
        #   -dt div(M^-1 m) = -div(A grad(phi*)).
        # AnalyticReprojection evaluates this profile independently on every AMR level.
        x_coordinate, y_coordinate = coordinates(frame)
        wave_number = 2.0 * np.pi
        potential_x = (
            _HIERARCHY_POTENTIAL_AMPLITUDE
            * wave_number
            * cos(wave_number * x_coordinate)
            * sin(wave_number * y_coordinate)
        )
        potential_y = (
            _HIERARCHY_POTENTIAL_AMPLITUDE
            * wave_number
            * sin(wave_number * x_coordinate)
            * cos(wave_number * y_coordinate)
        )
        density_profile = _HIERARCHY_DENSITY + 0.0 * x_coordinate
        momentum_factor = (
            1.0 / _HIERARCHY_DT + _HIERARCHY_DT * _HIERARCHY_DENSITY
        )
        plasma_initial = Analytic(
            frame=frame,
            components=(
                density_profile,
                momentum_factor * potential_x
                - _HIERARCHY_ROTATION_RATE * potential_y,
                _HIERARCHY_ROTATION_RATE * potential_x
                + momentum_factor * potential_y,
            ),
        )
    case.initials.add(
        InitialCondition(
            state=state_instance,
            value=plasma_initial,
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=marker_instance,
            # A localized marker creates genuinely nested patches at every requested depth.  Full
            # domain fine levels are a degenerate topology and do not exercise coarse/fine FAC joins.
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.5, y_axis: 0.5},
                background=0.0,
                amplitude=2.0,
                inverse_width=100.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("nested-refine-threshold", default=0.5))
    transfer = AMRTransfer()
    transfer.state(state_instance, StateTransfer())
    transfer.state(marker_instance, StateTransfer())
    if len(temporal_ratios) != max_levels - 1:
        raise ValueError("one independent temporal ratio is required per AMR transition")
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(base_cells, base_cells),
            periodic=None if bound_plasma else PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(
            max_levels=max_levels,
            ratios=tuple(2 for _ in range(max_levels - 1)),
        ),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(marker_instance) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled(
            tuple(
                AMRClockRelation(level, level + 1, ratio)
                for level, ratio in enumerate(temporal_ratios)
            )
        ),
    )
    return case, layout, state_instance


def _constant_rotation_oracle(*, dt, rate, density, east, north):
    """Closed-form inverse of ``I - dt*J`` for the zero-spatial-mode guard case."""
    rotation = dt * rate
    denominator = 1.0 + rotation * rotation
    return np.asarray(
        (
            density,
            (east + rotation * north) / denominator,
            (-rotation * east + north) / denominator,
        ),
        dtype=np.float64,
    )


def _external_hierarchy_counters(so_path):
    library = ctypes.CDLL(so_path)
    library.pops_test_hierarchy_register_calls.restype = ctypes.c_uint64
    library.pops_test_hierarchy_prepare_calls.restype = ctypes.c_uint64
    library.pops_test_hierarchy_execution_queries.restype = ctypes.c_uint64
    library.pops_test_hierarchy_solve_calls.restype = ctypes.c_uint64
    return (
        int(library.pops_test_hierarchy_register_calls()),
        int(library.pops_test_hierarchy_prepare_calls()),
        int(library.pops_test_hierarchy_execution_queries()),
        int(library.pops_test_hierarchy_solve_calls()),
    )


def _external_hierarchy_carry_metrics(so_path):
    library = ctypes.CDLL(so_path)
    library.pops_test_hierarchy_first_solution_norm_sum.restype = ctypes.c_double
    library.pops_test_hierarchy_second_guess_norm_sum.restype = ctypes.c_double
    library.pops_test_hierarchy_second_guess_calls.restype = ctypes.c_uint64
    return (
        float(library.pops_test_hierarchy_first_solution_norm_sum()),
        float(library.pops_test_hierarchy_second_guess_norm_sum()),
        int(library.pops_test_hierarchy_second_guess_calls()),
    )


def _nonuniform_plasma_initial():
    coordinate = (
        np.arange(_HIERARCHY_BASE_CELLS, dtype=np.float64) + 0.5
    ) / _HIERARCHY_BASE_CELLS
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = 1.0 + 0.20 * np.exp(
        -80.0 * ((x - 0.40) ** 2 + (y - 0.55) ** 2)
    )
    east = density * (0.25 + 0.08 * np.sin(2.0 * np.pi * y))
    north = density * (-0.15 + 0.06 * np.cos(2.0 * np.pi * x))
    return np.ascontiguousarray(np.stack((density, east, north)))


def _manufactured_plasma_initial(cells):
    """Evaluate the MMS with the runtime's independent four-point Gauss projection."""
    spacing = 1.0 / cells
    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) * spacing
    x_center, y_center = np.meshgrid(coordinate, coordinate, indexing="xy")
    nodes = (
        -0.861136311594052575223946488892809505,
        -0.339981043584856264802665759103244687,
        0.339981043584856264802665759103244687,
        0.861136311594052575223946488892809505,
    )
    weights = (
        0.347854845137453857373063949221999408,
        0.652145154862546142626936050778000593,
        0.652145154862546142626936050778000593,
        0.347854845137453857373063949221999408,
    )
    wave_number = 2.0 * np.pi
    potential_x = np.zeros((cells, cells), dtype=np.float64)
    potential_y = np.zeros((cells, cells), dtype=np.float64)
    for qy, weight_y in zip(nodes, weights, strict=True):
        for qx, weight_x in zip(nodes, weights, strict=True):
            x = x_center + 0.5 * spacing * qx
            y = y_center + 0.5 * spacing * qy
            weight = 0.25 * weight_x * weight_y
            potential_x += (
                weight
                * _HIERARCHY_POTENTIAL_AMPLITUDE
                * wave_number
                * np.cos(wave_number * x)
                * np.sin(wave_number * y)
            )
            potential_y += (
                weight
                * _HIERARCHY_POTENTIAL_AMPLITUDE
                * wave_number
                * np.sin(wave_number * x)
                * np.cos(wave_number * y)
            )
    density = np.full((cells, cells), _HIERARCHY_DENSITY)
    momentum_factor = (
        1.0 / _HIERARCHY_DT + _HIERARCHY_DT * _HIERARCHY_DENSITY
    )
    east = (
        momentum_factor * potential_x
        - _HIERARCHY_ROTATION_RATE * potential_y
    )
    north = (
        _HIERARCHY_ROTATION_RATE * potential_x
        + momentum_factor * potential_y
    )
    return np.stack((density, east, north))


def _periodic_condensed_fourier_oracle(initial, *, dt, rotation_rate):
    """Independent inverse of the constant-coefficient discrete condensed operator."""
    density, east, north = np.asarray(initial, dtype=np.float64)
    cells = density.shape[0]
    assert density.shape == east.shape == north.shape == (cells, cells)
    np.testing.assert_allclose(
        density, np.full_like(density, density[0, 0]), rtol=0.0, atol=5.0e-15
    )
    spacing = 1.0 / cells

    def centered_difference(values, *, axis):
        return (
            np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
        ) / (2.0 * spacing)

    rotation = dt * rotation_rate
    denominator = 1.0 + rotation * rotation
    flux_east = (east + rotation * north) / denominator
    flux_north = (-rotation * east + north) / denominator
    rhs = -dt * (
        centered_difference(flux_east, axis=1)
        + centered_difference(flux_north, axis=0)
    )
    assert np.max(np.abs(rhs)) > 1.0e-2

    coefficient = 1.0 + dt * dt * float(density[0, 0]) / denominator
    modes = np.fft.fftfreq(cells) * cells
    kx, ky = np.meshgrid(modes, modes, indexing="xy")
    eigenvalue = coefficient * 4.0 * cells * cells * (
        np.sin(np.pi * kx / cells) ** 2
        + np.sin(np.pi * ky / cells) ** 2
    )
    rhs_hat = np.fft.fft2(rhs)
    phi_hat = np.zeros_like(rhs_hat)
    nonzero = eigenvalue > 0.0
    phi_hat[nonzero] = rhs_hat[nonzero] / eigenvalue[nonzero]
    potential = np.fft.ifft2(phi_hat).real

    gradient_east = centered_difference(potential, axis=1)
    gradient_north = centered_difference(potential, axis=0)
    velocity_east = east / density - dt * gradient_east
    velocity_north = north / density - dt * gradient_north
    expected_east = density * (
        velocity_east + rotation * velocity_north
    ) / denominator
    expected_north = density * (
        -rotation * velocity_east + velocity_north
    ) / denominator
    return (
        np.stack((density, expected_east, expected_north)),
        potential,
        rhs,
    )


def _amr_history_level(values, *, level, base_cells=8):
    """Decode one scalar level from the public concatenated AMR history convention."""
    flat = np.asarray(values, dtype=np.float64)
    cells = base_cells * 2**level
    offset = sum((base_cells * 2**coarse) ** 2 for coarse in range(level))
    end = offset + cells * cells
    assert flat.size >= end
    return flat[offset:end].reshape(cells, cells)


def _patch_interior(mask, *, guard_cells=1):
    """Exclude the one-cell C/F interpolation band from a periodic patch mask."""
    interior = np.asarray(mask, dtype=bool).copy()
    for _ in range(guard_cells):
        interior &= (
            np.roll(interior, 1, axis=0)
            & np.roll(interior, -1, axis=0)
            & np.roll(interior, 1, axis=1)
            & np.roll(interior, -1, axis=1)
        )
    assert np.count_nonzero(interior) >= 16
    return interior


def test_header_only_hierarchy_extension_compiles_its_own_generic_provider_identity(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, kokkos_root
    from test_hierarchy_scoped_solve_emit import _build

    source_root = tmp_path / "source"
    source_root.mkdir()
    header = source_root / "tests_hierarchy_provider.hpp"
    header.write_text(
        """#pragma once
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/export.hpp>

#include <atomic>
namespace pops_test_hierarchy {
namespace {
// Keep the complete instrumented provider local to this generated Program DSO. A future test may
// load another artifact containing this header in the same process; no inline counter, provider
// method or vtable may then be coalesced into process-global lifecycle evidence.
std::atomic<std::uint64_t> register_calls{0};
std::atomic<std::uint64_t> prepare_calls{0};
std::atomic<std::uint64_t> execution_queries{0};
std::atomic<std::uint64_t> solve_calls{0};
std::atomic<std::uint64_t> second_guess_calls{0};
// Provider orchestration is single-threaded.  The Kokkos reductions complete before these host-only
// witnesses are read or written, so atomics would add no safety and are unavailable for double in
// the project's C++17 contract.
double first_solution_norm_sum = 0.0;
double second_guess_norm_sum = 0.0;

class DelegatingPrepared final
    : public pops::runtime::program::PreparedHierarchyTensorSolver {
 public:
  DelegatingPrepared(
      std::string contract,
      std::vector<bool> level_populated,
      std::unique_ptr<pops::runtime::program::PreparedHierarchyTensorSolver> delegate)
      : contract_(std::move(contract)),
        level_populated_(std::move(level_populated)),
        delegate_(std::move(delegate)) {
    if (!delegate_)
      throw std::invalid_argument("external hierarchy delegate is missing");
  }
  std::string_view provider_identity() const noexcept override {
    return "tests.hierarchy.header-only";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  pops::runtime::program::HierarchyTensorSolverExecutionPath execution_path()
      const noexcept override {
    execution_queries.fetch_add(1, std::memory_order_relaxed);
    return delegate_->execution_path();
  }
  int level_count() const noexcept override {
    return delegate_->level_count();
  }
  pops::MultiFab& assembly_target(std::string_view slot, int level) override {
    return delegate_->assembly_target(slot, level);
  }
  pops::MultiFab& solution(int level) override {
    return delegate_->solution(level);
  }
  void stage_initial_guess(int level, const pops::MultiFab* guess) override {
    if (solve_calls.load(std::memory_order_relaxed) == 1 && guess != nullptr) {
      second_guess_norm_sum += static_cast<double>(pops::norm_inf(*guess));
      second_guess_calls.fetch_add(1, std::memory_order_relaxed);
    }
    delegate_->stage_initial_guess(level, guess);
  }
  pops::SolveReport solve(
      const pops::runtime::program::HierarchyTensorSolveControls& controls) override {
    const std::uint64_t solve_index =
        solve_calls.fetch_add(1, std::memory_order_relaxed);
    auto outcome =
        pops::runtime::program::solve_prepared_hierarchy_tensor_collectively(*delegate_, controls);
    const auto action =
        outcome.report().solved_value_available()
            ? pops::SolveConsumption::kAccept
            : (outcome.report().action == pops::SolveAction::kRejectAttempt
                   ? pops::SolveConsumption::kRejectAttempt
                   : pops::SolveConsumption::kFailRun);
    pops::SolveReport report = outcome.consume(action);
    if (solve_index == 0 && report.solved_value_available()) {
      for (std::size_t level = 0; level < level_populated_.size(); ++level)
        if (level_populated_[level])
          first_solution_norm_sum +=
              static_cast<double>(pops::norm_inf(delegate_->solution(static_cast<int>(level))));
    }
    return report;
  }
 private:
  std::string contract_;
  std::vector<bool> level_populated_;
  std::unique_ptr<pops::runtime::program::PreparedHierarchyTensorSolver> delegate_;
};

class Provider final
    : public pops::runtime::program::HierarchyTensorSolverProvider {
 private:
  static pops::runtime::program::HierarchyTensorSolverBuildRequest delegate_request(
      const pops::runtime::program::HierarchyTensorSolverBuildRequest& request) {
    auto converted = request;
    converted.options = {
        "pops.hierarchy.composite-tensor-fac.options@1", request.options.values};
    return converted;
  }

 public:
  std::string_view identity() const noexcept override {
    return "tests.hierarchy.header-only";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "tests.hierarchy.header-only@1";
  }
  std::vector<std::string> capability_contracts() const override {
    return {};
  }
  pops::PreparedProviderOptions default_options() const override {
    return {"tests.hierarchy.header-only.options@1", {}};
  }
  pops::PreparedProviderSupport accepts_options(
      const pops::PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == "tests.hierarchy.header-only.options@1"
               ? pops::PreparedProviderSupport::accept()
               : pops::PreparedProviderSupport::reject(1, "header-only options are invalid");
  }
  pops::PreparedProviderSupport supports(
      const pops::runtime::program::HierarchyTensorSolverBuildRequest& request)
      const noexcept override {
    if (!accepts_options(request.options).accepted())
      return pops::PreparedProviderSupport::reject(2, "header-only request is invalid");
    try {
      const pops::runtime::program::detail::CompositeTensorFacHierarchyProvider delegate;
      return delegate.supports(delegate_request(request));
    } catch (...) {
      return pops::PreparedProviderSupport::reject(
          2, "header-only delegate request is invalid");
    }
  }
  pops::PreparedProviderSupport accepts_execution(
      const pops::runtime::program::HierarchyTensorSolverBuildRequest& request,
      pops::runtime::program::HierarchyTensorSolverExecutionPath execution)
      const noexcept override {
    if (!supports(request).accepted())
      return pops::PreparedProviderSupport::reject(3, "header-only request is invalid");
    try {
      const pops::runtime::program::detail::CompositeTensorFacHierarchyProvider delegate;
      return delegate.accepts_execution(delegate_request(request), execution);
    } catch (...) {
      return pops::PreparedProviderSupport::reject(
          3, "header-only execution is invalid");
    }
  }
  std::string expected_prepared_contract(
      const pops::runtime::program::HierarchyTensorSolverBuildRequest& request) const override {
    pops::ExactContractBuilder contract;
    contract.text("tests.hierarchy.header-only.prepared")
        .scalar(std::uint32_t{1})
        .text(request.plan_identity)
        .text(request.operator_contract_identity)
        .sequence(request.assembly_field_slots,
                  [](pops::ExactContractBuilder& item, const std::string& slot) {
                    item.text(slot);
                  })
        .text(request.solution_field_slot)
        .sequence(request.level_populated,
                  [](pops::ExactContractBuilder& item, bool populated) {
                    item.scalar(populated);
                  })
        .sequence(request.level_distributions,
                  [](pops::ExactContractBuilder& item,
                     pops::FieldDistribution distribution) { item.scalar(distribution); })
        .bytes(request.options.exact_contract());
    return std::move(contract).release();
  }
  std::unique_ptr<pops::runtime::program::PreparedHierarchyTensorSolver> prepare(
      const pops::runtime::program::HierarchyTensorSolverBuildRequest& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("header-only hierarchy provider rejected the request");
    const pops::runtime::program::detail::CompositeTensorFacHierarchyProvider delegate;
    auto prepared_delegate = delegate.prepare(delegate_request(request));
    prepare_calls.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<DelegatingPrepared>(
        expected_prepared_contract(request), request.level_populated,
        std::move(prepared_delegate));
  }
};

void register_provider(pops::runtime::program::AmrProgramContext& ctx) {
  register_calls.fetch_add(1, std::memory_order_relaxed);
  ctx.register_hierarchy_tensor_solver_provider(std::make_shared<Provider>());
}

pops::SolveOutcome solve(
    pops::runtime::program::AmrProgramContext& ctx, int block, int components,
    pops::Real relative_tolerance, pops::Real absolute_tolerance, int max_iterations) {
  return ctx.solve_hierarchy_tensor(
      block, components, relative_tolerance, absolute_tolerance, max_iterations);
}
}  // namespace
}  // namespace pops_test_hierarchy

extern "C" POPS_EXPORT std::uint64_t pops_test_hierarchy_register_calls() noexcept {
  return pops_test_hierarchy::register_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_hierarchy_prepare_calls() noexcept {
  return pops_test_hierarchy::prepare_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_hierarchy_execution_queries() noexcept {
  return pops_test_hierarchy::execution_queries.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_hierarchy_solve_calls() noexcept {
  return pops_test_hierarchy::solve_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT double pops_test_hierarchy_first_solution_norm_sum() noexcept {
  return pops_test_hierarchy::first_solution_norm_sum;
}

extern "C" POPS_EXPORT double pops_test_hierarchy_second_guess_norm_sum() noexcept {
  return pops_test_hierarchy::second_guess_norm_sum;
}

extern "C" POPS_EXPORT std::uint64_t pops_test_hierarchy_second_guess_calls() noexcept {
  return pops_test_hierarchy::second_guess_calls.load(std::memory_order_relaxed);
}
""",
        encoding="utf-8",
    )
    component = PreparedNativeComponent.header_only(
        "tests.hierarchy.header-only",
        include_root=source_root,
        entry_headers=(header.name,),
    )
    builtin = prepared_hierarchy_solver_provider_by_id(
        "pops.hierarchy.composite-tensor-fac"
    )

    def emit_external(request, provider, options):
        builtin_emission = builtin.emitter(request, provider, options)
        return PreparedHierarchySolverNativeEmission(
            configure=(
                "pops_test_hierarchy::register_provider(ctx);",
                *builtin_emission.configure,
            ),
            solve=(
                "pops::SolveOutcome %s = "
                "pops_test_hierarchy::solve(ctx, %d, %d, %s, %s, %d);"
                % (
                    request.report_name,
                    request.block_index,
                    request.components,
                    request.relative_tolerance_cpp,
                    request.absolute_tolerance_cpp,
                    request.max_iterations,
                ),
            ),
        )

    external = replace(
        builtin,
        provider_id="tests.hierarchy.header-only",
        emitter_id="tests.codegen.hierarchy.header-only@1",
        option_schema="tests.hierarchy.header-only.options@1",
        capabilities=frozenset(),
        native_component=component,
        emitter=emit_external,
    )
    register_prepared_hierarchy_solver_provider(external)

    class ExternalHierarchySolver:
        def prepare_program_solve(self):
            return external.prepare(
                CompositeTensorFAC(
                    max_iter=400,
                    rel_tol=1.0e-8,
                    abs_tol=1.0e-12,
                ).canonical_options()
            )

    program, source = _build(ExternalHierarchySolver())
    assert "#include <tests_hierarchy_provider.hpp>" in source
    assert "pops_test_hierarchy::solve(ctx," in source
    assert "pops_test_hierarchy::register_provider(ctx);" in source
    assert '"tests.hierarchy.header-only"' in source
    amr = source.split('extern "C" void pops_install_program_amr', 1)[1]
    branch = amr.index("if (ctx.uses_prepared_krylov_fallback())")
    gather = amr.index(".gather(hierarchy_dt)", branch)
    solve_once = amr.index("_level_programs->front().solve(hierarchy_dt)", gather)
    publish = amr.index(".publish(hierarchy_dt)", solve_once)
    assert gather < solve_once < publish
    solve = next(value for value in program._values if value.op == "solve_linear")
    assert solve.attrs["hierarchy_solver_provider"]["provider_id"] == external.provider_id
    staged = tmp_path / "staged"
    assert component.stage_verified(staged) == str(staged)
    assert (staged / header.name).read_text(encoding="utf-8") == header.read_text(
        encoding="utf-8"
    )

    # The same descriptor must survive the public AMR lifecycle.  Bind registers and prepares the
    # external provider, while run must query its provider-selected execution path and invoke its real
    # hierarchy solve.  The external provider delegates numerical storage/iterations to the
    # authenticated builtin FAC provider but retains its own identity, exact prepared contract and
    # observable lifecycle.
    import pops
    from pops.codegen import Production

    manufactured_errors = {}
    constant_oracle = _constant_rotation_oracle(
        dt=_HIERARCHY_DT,
        rate=_HIERARCHY_ROTATION_RATE,
        density=2.0,
        east=0.25,
        north=-0.5,
    )
    configurations = (
        # Preserve the two-step outflow/reflux/history-carry composition.
        (2, (3,), 2, True, False, _HIERARCHY_BASE_CELLS),
        # Independently quantify the nonzero two-level solve at h and h/2.
        (2, (3,), 1, False, True, _HIERARCHY_BASE_CELLS),
        (2, (3,), 1, False, True, 2 * _HIERARCHY_BASE_CELLS),
        # Keep the N-level gather/publish and nonbinary temporal-ratio guard.  The nonzero N-level
        # scientific gate remains open because the general FAC currently diverges on the MMS.
        (3, (3, 5), 1, False, False, _HIERARCHY_BASE_CELLS),
    )
    for (
        max_levels,
        temporal_ratios,
        steps,
        bound_plasma,
        manufactured_plasma,
        base_cells,
    ) in configurations:
        case, layout, plasma_state = _public_amr_hierarchy_case(
            ExternalHierarchySolver(),
            max_levels=max_levels,
            temporal_ratios=temporal_ratios,
            bound_plasma=bound_plasma,
            manufactured_plasma=manufactured_plasma,
            base_cells=base_cells,
        )
        resolved = pops.resolve(
            pops.validate(case),
            layout=layout,
            backend=Production(),
            compile_options={
                "include": str(Path(__file__).resolve().parents[4] / "include"),
                "cxx": native_cxx,
            },
        )
        compiled = pops.compile(resolved)
        assert Path(compiled.so_path).is_file()
        before_bind = _external_hierarchy_counters(compiled.so_path)

        simulation = pops.bind(
            compiled,
            initial_values=(
                {plasma_state: _nonuniform_plasma_initial()}
                if bound_plasma
                else None
            ),
            resources={"execution_context": artifact_execution_context(compiled)},
        )
        bound_register, bound_prepare, bound_execution, bound_solve = (
            _external_hierarchy_counters(compiled.so_path)
        )
        assert bound_register > before_bind[0]
        assert bound_prepare > before_bind[1]
        assert bound_execution >= before_bind[2]
        assert bound_solve == before_bind[3]
        assert simulation.n_levels() == max_levels

        manufactured_oracle = None
        if manufactured_plasma:
            finest_cells = base_cells * 2 ** (max_levels - 1)
            finest_initial = np.asarray(
                simulation.block_level_state_global(
                    "plasma", max_levels - 1
                ),
                dtype=np.float64,
            ).reshape(3, finest_cells, finest_cells)
            active = finest_initial[0] > 0.0
            assert np.any(active) and np.any(~active)
            uniform_initial = _manufactured_plasma_initial(finest_cells)
            np.testing.assert_allclose(
                finest_initial[:, active],
                uniform_initial[:, active],
                rtol=2.0e-13,
                atol=2.0e-13,
            )
            manufactured_oracle = _periodic_condensed_fourier_oracle(
                uniform_initial,
                dt=_HIERARCHY_DT,
                rotation_rate=_HIERARCHY_ROTATION_RATE,
            )

        initial_marker_mass = simulation.integral(
            "marker", component=0, levels=(0,)
        )
        report = pops.run(
            simulation, t_end=_HIERARCHY_DT * steps, max_steps=steps
        )
        run_register, run_prepare, run_execution, run_solve = (
            _external_hierarchy_counters(compiled.so_path)
        )
        assert report.accepted_steps == steps
        assert run_register == bound_register
        assert run_prepare == bound_prepare
        assert run_execution > bound_execution
        assert run_solve == bound_solve + steps

        # Each level owns a distinct qualified clock, while the authored temporal ratios remain
        # independent from the spatial ratio two (and from each other in the three-level tower).
        program_report = simulation.program_report()
        level_clocks = [
            row for row in program_report.clocks if row["kind"] == "level"
        ]
        assert {row["level"] for row in level_clocks} == set(range(max_levels))
        assert all(row["macro_step"] == steps for row in level_clocks)
        assert all(row["phase"] == {"numerator": 0, "denominator": 1} for row in level_clocks)
        assert all(
            row["physical_time"] == pytest.approx(_HIERARCHY_DT * steps)
            for row in level_clocks
        )
        assert program_report.level_relations == [
            {
                "parent_level": level,
                "child_level": level + 1,
                "temporal_ratio": {"numerator": ratio, "denominator": 1},
                "remainder_policy": "integral_only",
            }
            for level, ratio in enumerate(temporal_ratios)
        ]

        # This is one combined Program, not two adjacent tests: its explicit finite-volume rate
        # materializes the accepted interface-flux ledger and the same macro-step then executes the
        # hierarchy-scoped condensed solve.  Every refined level participates and synchronization
        # remains conservative reflux followed by average-down.
        assert {row["level"] for row in program_report.flux_ledger} == set(
            range(max_levels)
        )
        assert {row["phase"] for row in program_report.synchronization} == {
            "reflux",
            "average_down",
        }

        if bound_plasma:
            # ADC-639 composition: the Gaussian marker has nontrivial C/F transport fluxes while the
            # sibling plasma block executes its hierarchy solve.  The accepted marker mass remains
            # conservative after both refluxed macro-steps.
            final_marker_mass = simulation.integral(
                "marker", component=0, levels=(0,)
            )
            assert abs(final_marker_mass - initial_marker_mass) < 1.0e-8

            # ADC-427 composition: after solve 1 the nonzero hierarchy potential is stored in the
            # qualified one-component ring.  At gather 2 every active level stages that exact phi^n
            # as the composite initial guess.  The provider-side norm witness observes the hand-off
            # across the generated store/read boundary rather than merely inspecting authored IR.
            first_solution_norm, second_guess_norm, second_guess_calls = (
                _external_hierarchy_carry_metrics(compiled.so_path)
            )
            assert first_solution_norm > 1.0e-8
            assert second_guess_calls == max_levels
            assert second_guess_norm == pytest.approx(
                first_solution_norm, rel=5.0e-15, abs=5.0e-15
            )
            histories = {
                row["name"]: row for row in program_report.histories
            }
            assert histories["plasma.tensor-potential"] == {
                "name": "plasma.tensor-potential",
                "depth": 2,
                "ncomp": 1,
                "initialized": True,
            }
            for level in range(max_levels):
                actual = np.asarray(
                    simulation.block_level_state_global("plasma", level),
                    dtype=np.float64,
                )
                assert actual.size > 0 and np.isfinite(actual).all()
            continue

        if not manufactured_plasma:
            # N-level orchestration guard: every qualified level receives the exact local implicit
            # rotation for the spatial zero mode, while ratios 3 then 5 remain distinct from spatial
            # ratio two.  The independent nonzero scientific proof is the refined MMS pair below.
            for level in range(max_levels):
                actual = np.asarray(
                    simulation.block_level_state_global("plasma", level),
                    dtype=np.float64,
                ).reshape(3, -1)
                assert actual.shape[1] > 0
                assert np.isfinite(actual).all()
                active = actual[0] > 0.0
                assert np.any(active)
                np.testing.assert_array_equal(
                    actual[0, active],
                    np.full(np.count_nonzero(active), constant_oracle[0]),
                )
                np.testing.assert_allclose(
                    actual[1:, active],
                    np.broadcast_to(
                        constant_oracle[1:, None],
                        (2, np.count_nonzero(active)),
                    ),
                    rtol=5.0e-15,
                    atol=5.0e-15,
                )
                np.testing.assert_array_equal(
                    actual[:, ~active],
                    np.zeros((3, np.count_nonzero(~active))),
                )
            continue

        # Two genuinely refined solves execute the same nonzero manufactured problem at h and h/2.
        # NumPy inverts the corresponding uniform discrete operator by FFT, independently of PoPS,
        # FAC, generated C++, and the external provider.  Compare the qualified fine potential away
        # from the exact C/F interpolation band; the measured error must converge under refinement.
        assert manufactured_oracle is not None
        _, expected_potential, manufactured_rhs = manufactured_oracle
        assert float(np.linalg.norm(manufactured_rhs.ravel())) > 1.0
        actual_potential = _amr_history_level(
            # The accepted macro-step rotates the just-stored phi into lag-1.
            simulation.history_global("plasma.tensor-potential", 1),
            level=max_levels - 1,
            base_cells=base_cells,
        )
        finest_cells = base_cells * 2 ** (max_levels - 1)
        final_finest = np.asarray(
            simulation.block_level_state_global("plasma", max_levels - 1),
            dtype=np.float64,
        ).reshape(3, finest_cells, finest_cells)
        active = final_finest[0] > 0.0
        interior = _patch_interior(active)
        difference = actual_potential - expected_potential
        # The periodic operator admits an arbitrary constant. Remove exactly that gauge mode before
        # measuring the scientific error; no spatial error or provider tolerance is hidden.
        difference -= float(difference[interior].mean())
        relative_l2 = float(
            np.linalg.norm(difference[interior])
            / np.linalg.norm(expected_potential[interior])
        )
        assert np.isfinite(relative_l2) and relative_l2 > 0.0
        manufactured_errors[base_cells] = relative_l2

        # Every level remains populated and finite after the provider publishes the nonzero solution;
        # this catches a provider that only solved/published the finest array while leaving qualified
        # coarse state or history undefined.
        for level in range(max_levels):
            actual = np.asarray(
                simulation.block_level_state_global("plasma", level),
                dtype=np.float64,
            ).reshape(3, -1)
            assert actual.shape[1] > 0
            assert np.isfinite(actual).all()
            active = actual[0] > 0.0
            assert np.any(active)
            if level > 0:
                assert np.any(~active)

    # The two-level runs use h and h/2 on their finest patches.  This is an observed
    # convergence requirement, not a loose absolute tolerance: a zero-RHS solve, level-0-only solve,
    # stale publication or arbitrary C/F fill cannot exhibit the expected refined MMS convergence.
    assert set(manufactured_errors) == {
        _HIERARCHY_BASE_CELLS,
        2 * _HIERARCHY_BASE_CELLS,
    }
    observed_order = np.log(
        manufactured_errors[_HIERARCHY_BASE_CELLS]
        / manufactured_errors[2 * _HIERARCHY_BASE_CELLS]
    ) / np.log(2.0)
    assert observed_order >= 1.5, {
        "coarse_8_relative_l2": manufactured_errors[_HIERARCHY_BASE_CELLS],
        "coarse_16_relative_l2": manufactured_errors[
            2 * _HIERARCHY_BASE_CELLS
        ],
        "observed_order": observed_order,
    }
