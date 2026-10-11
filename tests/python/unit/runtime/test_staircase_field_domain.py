"""Independent cell selection and numerical V-cycle authority, without a transport mask."""
from dataclasses import FrozenInstanceError
import pytest

from pops.fields.domain import CellCentreBall
from pops.solvers.elliptic import GeometricMG, RedBlackVcycle, RoundoffAwareResidual
from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts


def facts(*, target="system", dimension=2, embedded=False, kind="uniform", levels=1, composite=False):
    return PreparedFieldSolverFacts(
        target=target, operator={"screened": False},
        layout={"kind": kind, "levels": levels, "cells": [8] * dimension,
                "embedded_boundary": embedded, "topology_identity": "test.exact-square@1"},
        hierarchy={"policy_id": "pops.field-hierarchy.composite" if composite else "pops.field-hierarchy.level-local", "interface_version": 1,
                   "option_schema": "pops.field-hierarchy.options.empty@1", "options": {}},
        boundary={"iterate_dependent": False}, nonlinear=False,
    )


def prepared(domain=None, *, target="system", dimension=2, embedded=False):
    provider, options = GeometricMG(cycle=RedBlackVcycle(), field_domain=domain)._prepared_field_solver()
    return provider.prepare(options=options, facts=facts(target=target, dimension=dimension, embedded=embedded), where="test")


def test_domain_is_immutable_and_serializes_strict_per_level_support():
    domain = CellCentreBall(centre=(0.0, 0.0), radius=16)
    assert domain.to_data()["region"] == "strict-cell-centre-ball"
    assert domain.to_data()["exterior"] == {"kind": "dirichlet", "value": 0.0}
    with pytest.raises(FrozenInstanceError):
        domain.radius = 8.0


def test_cycle_and_domain_are_independent_identity_authorities():
    full = prepared()
    ball = prepared(CellCentreBall(centre=(0.0, 0.0), radius=16.0))
    shifted = prepared(CellCentreBall(centre=(1.0, 0.0), radius=16.0))
    assert full.identity != ball.identity != shifted.identity
    assert full.resolution.native_contract["options"]["domain_dimension"] == 0
    assert ball.resolution.native_contract["options"]["domain_dimension"] == 2


def test_domain_does_not_select_a_hidden_smoother():
    with pytest.raises(ValueError, match="explicitly prepared"):
        GeometricMG(field_domain=CellCentreBall(centre=(0.0, 0.0), radius=16.0))


def test_amr_provider_declares_fac_difference_and_keeps_domain_identity():
    provider, options = GeometricMG(cycle=RedBlackVcycle(), field_domain=CellCentreBall(centre=(0.0, 0.0), radius=16.0))._prepared_field_solver()
    binding = provider.prepare(options=options, facts=facts(target="amr_system", kind="amr", levels=2, composite=True), where="test")
    native = binding.resolution.native_contract
    assert native["schema_identity"] == "pops.amr.field-solver-options.red-black-geometric-mg@1"
    assert len(native["options"]) == 24
    assert native["options"]["fac.fine_sweeps"] == 4
    assert native["options"]["fac.rel_tol"] == 1e-8
    assert "partitioned-FAC" in provider.use_policy.capabilities["composite_algorithm"]


@pytest.mark.parametrize("target,dimension,embedded", [("amr_system", 2, False), ("system", 3, False), ("system", 2, True)])
def test_mismatched_or_unqualified_topologies_refuse(target, dimension, embedded):
    with pytest.raises(ValueError):
        prepared(CellCentreBall(centre=(0.0, 0.0), radius=16.0), target=target, dimension=dimension, embedded=embedded)


@pytest.mark.parametrize("centre,radius", [((0.0, 0.0), True), ((0.0, 0.0), 0), ((float("nan"), 0.0), 1), ((0.0,), float("inf"))])
def test_invalid_region_authorities_refuse(centre, radius):
    with pytest.raises((ValueError, TypeError)):
        CellCentreBall(centre=centre, radius=radius)


@pytest.mark.parametrize("target,kind,levels,composite,width", [("system", "uniform", 1, False, 18), ("amr_system", "amr", 2, True, 26)])
def test_roundoff_rule_is_explicit_versioned_and_identity_bearing(target, kind, levels, composite, width):
    plain = GeometricMG(cycle=RedBlackVcycle())
    opted = GeometricMG(cycle=RedBlackVcycle(residual_evaluation=RoundoffAwareResidual()))
    base_provider, base_options = plain._prepared_field_solver()
    provider, options = opted._prepared_field_solver()
    environment = facts(target=target, kind=kind, levels=levels, composite=composite)
    base = base_provider.prepare(options=base_options, facts=environment, where="test")
    binding = provider.prepare(options=options, facts=environment, where="test")
    native = binding.resolution.native_contract
    assert native["schema_identity"].endswith("@2")
    assert len(native["options"]) == width
    assert native["options"]["roundoff_policy"] == 1
    assert native["options"]["roundoff_policy_version"] == 1
    assert base.resolution.native_contract["schema_identity"].endswith("@1")
    assert binding.identity != base.identity
    assert "residual_evaluation" not in RedBlackVcycle().to_data()


def test_roundoff_authority_is_immutable_and_has_no_arbitrary_tolerance_multiplier():
    rule = RoundoffAwareResidual()
    assert rule.to_data() == {"policy": "binary64-five-point-gamma-bound", "version": 1}
    with pytest.raises(TypeError):
        RoundoffAwareResidual(multiplier=1e8)
    with pytest.raises(TypeError):
        RedBlackVcycle(residual_evaluation=True)
