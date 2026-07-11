"""ADC-660: public phase functions accept only the immediately preceding phase value."""
from __future__ import annotations

import pytest

import pops
from pops.model import Module


def _problem(name="phase-case"):
    problem = pops.Problem(name=name)
    problem.add_block("fluid", Module("physics"))
    return problem


def test_validate_is_the_only_mutable_to_frozen_transition():
    problem = _problem()
    assert not problem.frozen
    assert pops.validate(problem) is problem
    assert problem.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        problem.aux("late", 1.0)


def test_every_public_phase_rejects_wrong_phase_inputs():
    with pytest.raises(TypeError, match="exact pops.Problem"):
        pops.validate(object())
    with pytest.raises(TypeError, match="frozen Problem"):
        pops.resolve(_problem("unfrozen"), layout=object())
    with pytest.raises(TypeError, match="ResolvedSimulationPlan"):
        pops.compile(_problem("wrong-compile"))
    with pytest.raises(TypeError, match="CompiledSimulationArtifact"):
        pops.bind(object(), pops.BindInputs())
    with pytest.raises(TypeError, match="exact InstallPlan"):
        pops.install(object())


def test_bind_has_no_semantic_override_keywords():
    with pytest.raises(TypeError, match="unexpected keyword"):
        pops.bind(object(), pops.BindInputs(), solvers={"phi": object()})
    with pytest.raises(TypeError, match="unexpected keyword"):
        pops.bind(object(), pops.BindInputs(), cadence=object())
