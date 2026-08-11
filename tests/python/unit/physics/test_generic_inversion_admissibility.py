from __future__ import annotations

import pytest

from pops.physics import (
    AdmissibilityConstraint,
    AdmissibleSet,
    EnforcementPhase,
    EnforcementRule,
    EnforcementSchedule,
    InversionProviderCatalog,
    InversionWorkspaceBudget,
    PreparedInversionProvider,
    ProjectionProvider,
    VariableInversionProblem,
)


def _problem(dimension: int = 3) -> VariableInversionProblem:
    return VariableInversionProblem(
        dimension=dimension,
        state_identity="conservative:moments-v2",
        provider_inputs_identity="provider-pack:thermo-v1",
        candidate_identity="primitive:moments-v2",
        failure_identity="moment-inversion-failure:v1",
        failure_codes=((1, "invalid_input"), (2, "no_convergence")),
        workspace=InversionWorkspaceBudget(bytes=4096, alignment=64),
    )


def test_inversion_contract_captures_parameters_and_is_dimensioned() -> None:
    mutable = {"tolerance": 1.0e-12, "iterations": [8, 16]}
    provider = PreparedInversionProvider("moments.newton", 2, _problem(), mutable)
    before = provider.canonical_bytes()
    mutable["iterations"].append(32)
    mutable["tolerance"] = 1.0

    assert provider.canonical_bytes() == before
    assert provider.to_data()["parameters"]["tolerance"] == {"binary64": (1.0e-12).hex()}
    assert _problem(1).identity != _problem(2).identity != _problem(3).identity
    with pytest.raises(TypeError):
        provider.parameters["extra"] = 1  # type: ignore[index]


def test_inversion_authoring_and_direct_construction_are_identical() -> None:
    problem = _problem(2)
    direct = PreparedInversionProvider(
        "closed.form", 1, problem, {"branch": "analytic", "scale": 2.0}
    )
    authored = PreparedInversionProvider.author(
        "closed.form", problem, branch="analytic", scale=2.0
    )
    assert direct.to_data() == authored.to_data()
    assert direct.canonical_bytes() == authored.canonical_bytes()
    assert direct.identity == authored.identity


def test_inversion_and_admissibility_collisions_are_rejected() -> None:
    catalog = InversionProviderCatalog()
    provider = PreparedInversionProvider.author("iterative", _problem(), max_iterations=12)
    catalog.register(provider)
    with pytest.raises(ValueError, match="provider id collision"):
        catalog.register(
            PreparedInversionProvider.author("iterative", _problem(), max_iterations=20)
        )

    finite = AdmissibilityConstraint.finite("finite", 1, components=(0, 1, 2))
    positive = AdmissibilityConstraint.positive("positive", 2, component=0, lower_bound=0.0)
    with pytest.raises(ValueError, match="constraint id collision"):
        AdmissibleSet.declare(
            finite, AdmissibilityConstraint.positive("finite", 3, component=1, lower_bound=0.0)
        )
    with pytest.raises(ValueError, match="diagnostic code collision"):
        AdmissibleSet.declare(
            finite, AdmissibilityConstraint.positive("other", 1, component=1, lower_bound=0.0)
        )
    assert AdmissibleSet.declare(finite, positive).to_data()["constraints"][1]["kind"] == "positive"


def test_admissible_set_authoring_is_generic_immutable_and_canonical() -> None:
    parameters = {"order": 5, "weights": [1, 2, 3]}
    constraints = AdmissibleSet.declare(
        AdmissibilityConstraint.finite("finite", 10, components=(0, 1, 2, 3)),
        AdmissibilityConstraint.positive("component-0-floor", 11, component=0, lower_bound=1.0e-9),
        AdmissibilityConstraint.realizability(
            "hankel-cone", 12, provider="moments.hankel", **parameters
        ),
        AdmissibilityConstraint.custom(
            "bounded-ratio", 13, provider="model.ratio", numerator=2, denominator=0
        ),
    )
    before = constraints.canonical_bytes()
    parameters["weights"].append(4)
    assert constraints.canonical_bytes() == before
    assert "density" not in repr(constraints.to_data()).lower()


def test_projection_and_schedule_authoring_equivalence() -> None:
    direct = ProjectionProvider(
        "moments.nearest-realizable",
        1,
        3,
        "primitive:moments-v2",
        "provider-pack:none",
        {"tol": 0.0},
    )
    authored = ProjectionProvider.author(
        "moments.nearest-realizable",
        dimension=3,
        candidate_identity="primitive:moments-v2",
        inputs_identity="provider-pack:none",
        tol=0.0,
    )
    assert direct.canonical_bytes() == authored.canonical_bytes()

    rules = {
        EnforcementPhase.INITIALIZATION: EnforcementRule(True, True),
        EnforcementPhase.RECONSTRUCTION: EnforcementRule(True, False),
        EnforcementPhase.SOURCE_SOLVE: EnforcementRule(True, True),
        EnforcementPhase.BOUNDARY: EnforcementRule(True, False),
        EnforcementPhase.ACCEPTANCE: EnforcementRule(True, False),
    }
    schedule = EnforcementSchedule.from_mapping(dict(reversed(tuple(rules.items()))))
    direct_schedule = EnforcementSchedule(*(rules[phase] for phase in EnforcementPhase))
    assert schedule.canonical_bytes() == direct_schedule.canonical_bytes()
    assert schedule.at(EnforcementPhase.SOURCE_SOLVE).project_if_invalid
    assert [entry["phase"] for entry in schedule.to_data()["phases"]] == [
        phase.value for phase in EnforcementPhase
    ]

    with pytest.raises(ValueError, match="requires an admissibility check"):
        EnforcementRule(False, True)
    with pytest.raises(ValueError, match="every phase"):
        EnforcementSchedule.from_mapping({EnforcementPhase.ACCEPTANCE: EnforcementRule(True)})
