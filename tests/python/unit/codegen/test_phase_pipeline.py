"""Wrong-phase values are rejected at every canonical lifecycle boundary."""
from __future__ import annotations

import pytest

import pops
from pops.codegen import _phases
from pops.codegen._plans import BindInputs
from pops.layouts import Uniform
from tests.python.support.layout_plan import cartesian_grid


def test_every_public_phase_rejects_wrong_phase_inputs():
    with pytest.raises(TypeError, match="exact pops.Case"):
        _phases.validate(object())

    unfrozen = pops.Case("wrong-phase")
    with pytest.raises(TypeError, match="frozen Case"):
        _phases.resolve(unfrozen, layout=Uniform(cartesian_grid(n=8)))

    with pytest.raises(TypeError, match="ResolvedSimulationPlan"):
        _phases.compile(object())

    inputs = BindInputs()
    with pytest.raises(TypeError, match="CompiledSimulationArtifact"):
        _phases.bind(object(), inputs)

    with pytest.raises(TypeError, match="InstallPlan"):
        _phases.install(object())
