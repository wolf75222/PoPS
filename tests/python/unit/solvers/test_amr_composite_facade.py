"""CompositeFAC survives the final public AMR field lifecycle as backend options."""
from __future__ import annotations

import pytest

import pops.lib.time as libtime
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.field_install import ResolvedFieldInstallPlan
from pops.codegen.lowering_coverage import LoweringRejection
from pops.fields.discretization import (
    CompositeHierarchySolve,
    FieldHierarchyPolicy,
    InferHierarchyFromLayout,
    LevelByLevelSolve,
)
from pops.fields.solve import ResolvedHierarchyPolicy
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
    def _prepared_field_solver(self):
        return object(), {}


class _ExternalHierarchyPolicy(FieldHierarchyPolicy):
    def options(self):
        return self.resolved_authority().authority()

    def resolved_authority(self):
        return ResolvedHierarchyPolicy(
            "tests.field-hierarchy.overlapping-schwarz",
            7,
            "tests.field-hierarchy.overlapping-schwarz.options@2",
            {"overlap": 3},
        )


class _HierarchyCapabilities:
    """Test-owned open policy-binding context, independent of field lowering."""

    def __init__(self, inferred):
        self.inferred = inferred

    def inferred_hierarchy_policy(self):
        return self.inferred

    def bind_hierarchy_policy(self, policy):
        if not isinstance(policy, ResolvedHierarchyPolicy):
            raise TypeError("foreign hierarchy authority")
        return policy


class _NativeFieldPlanProbe:
    def __init__(self):
        self.field_solver_args = None

    def set_field_solver_plan(self, *args):
        self.field_solver_args = args

    def register_configured_field_solver_provider(self, *_args):
        return "pops.test.exact-provider-identity"

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
    _provider, default = GeometricMG()._prepared_field_solver()
    assert default["fac"] is None
    assert default["mg"] == {
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
    _provider, lowered = solver._prepared_field_solver()
    assert lowered["fac"] == {
        key: value for key, value in _FAC_OVERRIDES.items()
        if key not in {"schema_version", "kind"}
    }
    assert lowered["mg"] == default["mg"]
    assert "hierarchy" not in lowered
    assert solver.lower().extra["fac_options"] == _FAC_OVERRIDES
    assert solver.lower().extra["mg_options"] == {
        "schema_version": 1,
        "kind": "geometric_mg_options",
        **default["mg"],
    }


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
    assert default_field.native_options["hierarchy_policy"]["policy_id"] == (
        "pops.field-hierarchy.composite"
    )
    assert configured_field.native_options["hierarchy_policy"] == (
        default_field.native_options["hierarchy_policy"]
    )
    from pops.fields._prepared_field_solver_registry import (
        prepared_field_solver_binding_from_data,
    )

    default_binding = prepared_field_solver_binding_from_data(
        default_field.native_options["solver_provider"]
    )
    configured_binding = prepared_field_solver_binding_from_data(
        configured_field.native_options["solver_provider"]
    )
    assert default_binding.options["fac"] is None
    assert configured_binding.options["fac"] is not None
    configured_native = configured_binding.resolution.native_contract["options"]
    assert configured_native["fac.max_iters"] == 11
    assert configured_native["fac.fine_sweeps"] == 17
    assert configured_native["fac.rel_tol"] == 2.0e-7
    assert configured_native["mg.rel_tol"] == 1.0e-8
    assert default_field.identity != configured_field.identity

    # Exercise the detached-record constructor used inside pops.compile. This unit assertion covers
    # identity-preserving detachment only; the integration test crosses the real native install seam.
    default_compiled = CompiledPlanRecord.from_resolved(default)
    configured_compiled = CompiledPlanRecord.from_resolved(configured)
    default_compiled_binding = prepared_field_solver_binding_from_data(
        _only_field_plan(default_compiled).native_options["solver_provider"]
    )
    configured_compiled_binding = prepared_field_solver_binding_from_data(
        _only_field_plan(configured_compiled).native_options["solver_provider"]
    )
    assert default_compiled_binding.options["fac"] is None
    assert configured_compiled_binding.options["fac"] is not None
    assert default_compiled.contract_identity != configured_compiled.contract_identity


def test_resolve_refuses_an_unauthenticated_provider_binding() -> None:
    with pytest.raises(LoweringRejection, match="invalid prepared field solver"):
        _resolve(_UnsupportedBuiltin())


def test_hierarchy_policy_uses_an_open_versioned_authority_protocol() -> None:
    composite = CompositeHierarchySolve().resolved_authority()
    level_local = LevelByLevelSolve().resolved_authority()
    amr = _HierarchyCapabilities(composite)
    uniform = _HierarchyCapabilities(level_local)

    assert InferHierarchyFromLayout().resolve(amr).authority() == composite.authority()
    assert CompositeHierarchySolve().resolve(amr).authority() == composite.authority()
    assert InferHierarchyFromLayout().resolve(uniform).authority() == level_local.authority()
    external = _ExternalHierarchyPolicy().resolve(amr)
    assert external.authority() == {
        "policy_id": "tests.field-hierarchy.overlapping-schwarz",
        "interface_version": 7,
        "option_schema": "tests.field-hierarchy.overlapping-schwarz.options@2",
        "options": {"overlap": 3},
    }


def test_resolved_plan_reasserts_closed_provider_schema_and_hierarchy() -> None:
    plan = _only_field_plan(_resolve(GeometricMG(fac=CompositeFAC(max_iters=11))))

    with pytest.raises(TypeError):
        plan.native_options["solver_provider"]["options"]["mg"]["max_cycles"] = 99
    assert isinstance(plan.native_options["provider_pack"], tuple)
    detached = plan.to_data()["native_options"]
    assert isinstance(detached, dict)
    assert isinstance(detached["solver_provider"], dict)
    assert isinstance(detached["provider_pack"], list)

    bad_options = plan.native_install_data()
    bad_options["solver_provider"]["provider"]["resolver_id"] = "pops.test.unregistered"
    with pytest.raises(NotImplementedError, match="not registered"):
        ResolvedFieldInstallPlan(
            plan.name, plan.operator, plan.discretization, plan.target,
            plan.rhs_providers, bad_options, plan.coverage, plan.nonlinear_provider,
            plan.identity,
        )

    bad_options = plan.native_install_data()
    bad_options["hierarchy_policy"] = LevelByLevelSolve().resolved_authority().authority()
    with pytest.raises(ValueError, match="lowering native contract"):
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
    if target == "amr_system":
        hierarchy = plan.native_options["hierarchy_policy"]
        assert probe._s.field_solver_args[11:15] == (
            hierarchy["policy_id"],
            hierarchy["interface_version"],
            hierarchy["option_schema"],
            hierarchy["options"],
        )


@pytest.mark.parametrize("install_type", (_AmrSystemInstall, _SystemUnifiedInstall))
def test_bind_install_seams_have_no_legacy_solver_adapter(install_type) -> None:
    assert not hasattr(install_type, "_install_solver")
