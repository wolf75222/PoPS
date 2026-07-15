"""CompositeFAC survives the final public AMR field lifecycle as backend options."""
from __future__ import annotations

import pytest

import pops.lib.time as libtime
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.field_install import (
    ResolvedFieldInstallPlan,
    _NativeFieldSolveCapabilities,
)
from pops.codegen.lowering_coverage import LoweringRejection
from pops.fields.discretization import (
    CompositeHierarchySolve,
    FieldHierarchyPolicy,
    InferHierarchyFromLayout,
    LevelByLevelSolve,
)
from pops.runtime._amr_system_install import _AmrSystemInstall
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.solvers.elliptic import GeometricMG
from pops.solvers.options import CompositeFAC
from pops.time import FailRun
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_advection_field_model,
)


_FAC_OVERRIDES = {
    "schema_version": 1,
    "kind": "composite_fac",
    "max_iters": 11,
    "fine_sweeps": 17,
    "rel_tol": 2.0e-7,
    "abs_tol": 3.0e-12,
    "coarse_rel_tol": 4.0e-9,
    "coarse_abs_tol": 5.0e-14,
    "coarse_cycles": 19,
    "verbose": True,
}


class _UnsupportedBuiltin(GeometricMG):
    def lower_field_solver(self, *, target, layout):
        lowered = super().lower_field_solver(target=target, layout=layout)
        lowered["native_solver"] = "uninstalled_future_solver"
        return lowered


class _UnknownHierarchyPolicy(FieldHierarchyPolicy):
    def options(self):
        return {"policy": "future_hierarchy"}


class _NativeFieldPlanProbe:
    def __init__(self):
        self.field_solver_args = None

    def set_field_solver_plan(self, *args):
        self.field_solver_args = args

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _ResolvedAmrInstallProbe(_AmrSystemInstall):
    def __init__(self):
        self._s = _NativeFieldPlanProbe()


class _ResolvedSystemInstallProbe(_SystemUnifiedInstall):
    def __init__(self):
        self._s = _NativeFieldPlanProbe()


def _field_program(state, rate, field):
    return libtime.ForwardEuler(
        state, rate=rate, fields=field, solve_action=FailRun())


def _resolve(solver: GeometricMG, *, target: str = "amr_system"):
    model = scalar_advection_field_model("composite-fac-carrier-model")
    return resolve_periodic_field_program(
        model,
        _field_program,
        name="composite-fac-carrier",
        block_name="plasma",
        target=target,
        n=16,
        field_solver=solver,
    )


def _only_field_plan(container):
    plans = container.field_plans
    assert len(plans) == 1
    return next(iter(plans.values()))


def test_geometric_mg_lowering_carries_only_typed_fac_backend_overrides() -> None:
    default = GeometricMG().lower_field_solver(target="amr_system", layout=None)
    assert default["fac_options"] is None
    assert default["mg_options"] == {
        "schema_version": 1,
        "kind": "geometric_mg_options",
        "rel_tol": 1.0e-8,
        "abs_tol": 0,
        "max_cycles": 50,
        "min_coarse": 2,
        "pre_smooth": 2,
        "post_smooth": 2,
        "bottom_sweeps": 50,
        "coarse_threshold": 0,
    }

    solver = GeometricMG(fac=CompositeFAC(
        max_iters=11,
        fine_sweeps=17,
        rel_tol=2.0e-7,
        abs_tol=3.0e-12,
        coarse_rel_tol=4.0e-9,
        coarse_abs_tol=5.0e-14,
        coarse_cycles=19,
        verbose=True,
    ))
    lowered = solver.lower_field_solver(target="amr_system", layout=None)
    assert lowered["fac_options"] == _FAC_OVERRIDES
    assert lowered["mg_options"] == default["mg_options"]
    assert "hierarchy" not in lowered
    assert solver.lower().extra["fac_options"] == _FAC_OVERRIDES
    assert solver.lower().extra["mg_options"] == default["mg_options"]


def test_public_amr_resolve_and_compile_record_preserve_fac_options_and_identity() -> None:
    default = _resolve(GeometricMG())
    configured = _resolve(GeometricMG(fac=CompositeFAC(
        max_iters=11,
        fine_sweeps=17,
        rel_tol=2.0e-7,
        abs_tol=3.0e-12,
        coarse_rel_tol=4.0e-9,
        coarse_abs_tol=5.0e-14,
        coarse_cycles=19,
        verbose=True,
    )))

    default_field = _only_field_plan(default)
    configured_field = _only_field_plan(configured)
    assert default.target == configured.target == "amr_system"
    assert default_field.native_options["hierarchy"] == "composite"
    assert configured_field.native_options["hierarchy"] == "composite"
    assert default_field.native_options["fac_options"] is None
    assert configured_field.native_options["fac_options"] == _FAC_OVERRIDES
    assert configured_field.native_options["mg_options"]["kind"] == "geometric_mg_options"
    assert "mg_options" not in configured_field.native_options["solver_capabilities"]
    assert "fac_options" not in configured_field.native_options["solver_capabilities"]
    assert default_field.identity != configured_field.identity

    # Exercise the detached-record constructor used inside pops.compile. This unit assertion covers
    # identity-preserving detachment only; the integration test crosses the real native install seam.
    default_compiled = CompiledPlanRecord.from_resolved(default)
    configured_compiled = CompiledPlanRecord.from_resolved(configured)
    assert _only_field_plan(default_compiled).native_options["fac_options"] is None
    assert _only_field_plan(
        configured_compiled).native_options["fac_options"] == _FAC_OVERRIDES
    assert default_compiled.contract_identity != configured_compiled.contract_identity


def test_resolve_refuses_a_builtin_route_the_amr_target_cannot_install() -> None:
    with pytest.raises(LoweringRejection, match="can install only"):
        _resolve(_UnsupportedBuiltin())


def test_hierarchy_policy_uses_capability_protocol_and_refuses_unknown_modes() -> None:
    amr = _NativeFieldSolveCapabilities("composite", ("composite",))
    uniform = _NativeFieldSolveCapabilities("level_local", ("level_local",))

    assert InferHierarchyFromLayout().resolve(amr).mode == "composite"
    assert CompositeHierarchySolve().resolve(amr).mode == "composite"
    assert InferHierarchyFromLayout().resolve(uniform).mode == "level_local"
    with pytest.raises(ValueError, match="unsupported"):
        LevelByLevelSolve().resolve(amr)
    with pytest.raises(ValueError, match="unknown"):
        _UnknownHierarchyPolicy().resolve(amr)


def test_resolved_plan_reasserts_closed_mg_schema_and_fac_backend() -> None:
    plan = _only_field_plan(_resolve(GeometricMG(fac=CompositeFAC(max_iters=11))))

    with pytest.raises(TypeError):
        plan.native_options["mg_options"]["max_cycles"] = 99
    with pytest.raises(TypeError):
        plan.native_options["fac_options"]["max_iters"] = 99
    assert isinstance(plan.native_options["provider_pack"], tuple)
    detached = plan.to_data()["native_options"]
    assert isinstance(detached, dict)
    assert isinstance(detached["mg_options"], dict)
    assert isinstance(detached["fac_options"], dict)
    assert isinstance(detached["provider_pack"], list)

    bad_options = dict(plan.native_options)
    bad_mg = dict(bad_options["mg_options"])
    bad_mg["schema_version"] = True
    bad_options["mg_options"] = bad_mg
    with pytest.raises(TypeError, match="closed schema-v1"):
        ResolvedFieldInstallPlan(
            plan.name, plan.operator, plan.discretization, plan.target,
            plan.rhs_providers, bad_options, plan.coverage, plan.nonlinear_provider,
            plan.identity,
        )

    bad_options = dict(plan.native_options)
    bad_options["hierarchy"] = "level_local"
    with pytest.raises(ValueError, match="composite AMR hierarchy"):
        ResolvedFieldInstallPlan(
            plan.name, plan.operator, plan.discretization, plan.target,
            plan.rhs_providers, bad_options, plan.coverage, plan.nonlinear_provider,
            plan.identity,
        )


@pytest.mark.parametrize(
    ("probe_type", "target"),
    ((_ResolvedAmrInstallProbe, "amr_system"), (_ResolvedSystemInstallProbe, "system")),
)
def test_native_install_receives_exact_resolved_plan_identity(probe_type, target) -> None:
    plan = _only_field_plan(_resolve(GeometricMG(), target=target))
    probe = probe_type()

    probe._install_field_plan(plan.name, plan)

    assert probe._s.field_solver_args[:3] == (
        plan.native_options["provider_slot"],
        plan.identity.token,
        plan.native_options["provider_identity_text"],
    )


@pytest.mark.parametrize("install_type", (_AmrSystemInstall, _SystemUnifiedInstall))
def test_bind_install_seams_have_no_legacy_solver_adapter(install_type) -> None:
    assert not hasattr(install_type, "_install_solver")
