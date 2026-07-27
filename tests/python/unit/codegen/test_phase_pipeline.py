"""Wrong-phase values are rejected at every canonical lifecycle boundary."""
from __future__ import annotations

import builtins

import pytest

import pops
from pops.codegen import _phases
from pops.codegen._plans import BindInputs
from pops.layouts import Uniform
from pops.model import Module
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout


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


@pytest.mark.parametrize(
    "layout",
    [
        Uniform(cartesian_grid(n=8, name="programless-uniform")),
        final_amr_layout(cartesian_grid(n=8, name="programless-amr")),
    ],
    ids=("uniform", "amr"),
)
def test_resolve_refuses_programless_layout_before_lowering_or_native_import(
    layout, monkeypatch,
):
    model = Module("programless-model")
    model.state_space("U", ("u",))
    case = pops.Case("programless-case")
    case.block("fluid", model)
    validated = pops.validate(case)
    snapshot_hash = validated.snapshot.artifact_hash

    lowered = []
    monkeypatch.setattr(
        _phases,
        "_resolve_problem_model",
        lambda _model: lowered.append(_model),
    )
    imported_native = []
    original_import = builtins.__import__

    def refuse_native_import(name, *args, **kwargs):
        if name in ("pops._pops", "pops._bootstrap"):
            imported_native.append(name)
            raise AssertionError("resolve attempted a native import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_native_import)
    with pytest.raises(
        ValueError,
        match="requires a whole-system Program for every Uniform or AMR layout",
    ):
        pops.resolve(validated, layout=layout)

    assert lowered == []
    assert imported_native == []
    assert validated.snapshot.artifact_hash == snapshot_hash
