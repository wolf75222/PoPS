"""Direct CompositeTensorFAC identity and authoring contract."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops._ir.literals import scalar_data, scalar_literal
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
from pops.solvers.providers import PreparedHierarchySolverNativeEmission
from pops.time import Program


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


def test_flat_direct_execution_contract_has_no_implicit_krylov_authority():
    execution = PreparedHierarchyFlatExecution.direct_provider()
    assert execution.authority() == {
        "interface": "pops.prepared-hierarchy-flat-execution@1",
        "mode": "direct_provider",
        "krylov": None,
    }
    assert execution.ir_attributes(unused=True) == {}
    execution.validate_ir({}, where="flat direct test")
    with pytest.raises(ValueError, match="unexpected Krylov attributes"):
        execution.validate_ir(
            {"krylov_footprint": {"components": 1}}, where="flat direct test"
        )


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


def _public_amr_hierarchy_case(solver):
    import pops
    from pops.amr import (
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
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Constant
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
    from pops.time import FailRun, FixedDt, every

    frame = Rectangle(
        "external-hierarchy-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes

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
    marker_flux = marker_model.flux(
        "marker-transport",
        frame=frame,
        state=marker_state,
        components={x_axis: (0.0 * marker,), y_axis: (0.0 * marker,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    marker_rate = marker_model.rate(
        "marker-rate", equation=ddt(marker_state) == -div(marker_flux)
    )

    case = pops.Case("public-external-hierarchy-provider")
    block = case.block("plasma", model=model)
    marker_block = case.block("marker", model=marker_model)
    state_instance = block[state]
    marker_instance = marker_block[marker_state]
    for owner, declared_rate, declared_flux, declared_state in (
        (block, rate, flux, state),
        (marker_block, marker_rate, marker_flux, marker_state),
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
    program.commit(
        temporal.next,
        program.value("accepted-state", reconstructed, at=temporal.next.point),
    )
    program.commit(
        marker_temporal.next,
        program.value(
            "accepted-marker", marker_temporal.n, at=marker_temporal.next.point
        ),
    )
    program.step_strategy(FixedDt(0.01))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=state_instance,
            value=Constant((2.0, 0.25, -0.5)),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=marker_instance,
            # Tag the complete coarse domain so bootstrap materializes the authored fine level.
            value=Constant((2.0,)),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("flat-refine-threshold", default=1.0))
    transfer = AMRTransfer()
    transfer.state(state_instance, StateTransfer())
    transfer.state(marker_instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(4, 4),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
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
        execution=AMRExecution.synchronous(),
    )
    return case, layout


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
inline std::atomic<std::uint64_t> register_calls{0};
inline std::atomic<std::uint64_t> prepare_calls{0};
inline std::atomic<std::uint64_t> execution_queries{0};
inline std::atomic<std::uint64_t> solve_calls{0};

class DelegatingPrepared final
    : public pops::runtime::program::PreparedHierarchyTensorSolver {
 public:
  DelegatingPrepared(
      std::string contract,
      std::unique_ptr<pops::runtime::program::PreparedHierarchyTensorSolver> delegate)
      : contract_(std::move(contract)), delegate_(std::move(delegate)) {
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
  pops::MultiFab& assembly_target(std::string_view slot, int level) override {
    return delegate_->assembly_target(slot, level);
  }
  pops::MultiFab& solution(int level) override {
    return delegate_->solution(level);
  }
  void stage_initial_guess(int level, const pops::MultiFab* guess) override {
    delegate_->stage_initial_guess(level, guess);
  }
  pops::SolveReport solve(
      const pops::runtime::program::HierarchyTensorSolveControls& controls) override {
    solve_calls.fetch_add(1, std::memory_order_relaxed);
    return delegate_->solve(controls);
  }
 private:
  std::string contract_;
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
        expected_prepared_contract(request), std::move(prepared_delegate));
  }
};

inline void register_provider(pops::runtime::program::AmrProgramContext& ctx) {
  register_calls.fetch_add(1, std::memory_order_relaxed);
  ctx.register_hierarchy_tensor_solver_provider(std::make_shared<Provider>());
}

inline pops::SolveReport solve(
    pops::runtime::program::AmrProgramContext& ctx, int block, int components,
    pops::Real relative_tolerance, pops::Real absolute_tolerance, int max_iterations) {
  return ctx.solve_hierarchy_tensor(
      block, components, relative_tolerance, absolute_tolerance, max_iterations);
}
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
                "pops::SolveReport %s = pops_test_hierarchy::solve("
                "ctx, %d, %d, %s, %s, %d);"
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
            return external.prepare(CompositeTensorFAC().canonical_options())

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

    # The same descriptor must survive the public AMR lifecycle.  Merely loading the library leaves
    # every counter at zero; bind must register and prepare the external provider, while run must
    # query its provider-selected execution path and invoke its real hierarchy solve.  The external
    # provider delegates numerical storage/iterations to the authenticated builtin FAC provider but
    # retains its own identity, exact prepared contract and observable lifecycle.
    import numpy as np
    import pops
    from pops.codegen import Production

    case, layout = _public_amr_hierarchy_case(ExternalHierarchySolver())
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
    assert _external_hierarchy_counters(compiled.so_path) == (0, 0, 0, 0)

    simulation = pops.bind(compiled)
    bound_register, bound_prepare, bound_execution, bound_solve = (
        _external_hierarchy_counters(compiled.so_path)
    )
    assert bound_register >= 1
    assert bound_prepare >= 1
    assert bound_solve == 0
    assert simulation.n_levels() == 2

    report = pops.run(simulation, t_end=0.01, max_steps=1)
    run_register, run_prepare, run_execution, run_solve = (
        _external_hierarchy_counters(compiled.so_path)
    )
    assert report.accepted_steps == 1
    assert run_register == bound_register
    assert run_prepare == bound_prepare
    assert run_execution > bound_execution
    assert run_solve > bound_solve

    actual = np.asarray(
        simulation.block_level_state_global("plasma", 0), dtype=np.float64
    ).reshape(3, 4, 4)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[0], np.full((4, 4), 2.0))
    assert np.max(np.abs(actual[1] - 0.25)) > 1.0e-6
    assert np.max(np.abs(actual[2] + 0.5)) > 1.0e-6
    fine = np.asarray(
        simulation.block_level_state_global("plasma", 1), dtype=np.float64
    )
    assert fine.size > 0 and np.isfinite(fine).all()
