"""ADC-655: a public compiled artifact retains only immutable snapshot/install values."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import compile_drivers, orchestration  # noqa: E402
from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: E402
from pops.codegen._compiled_model_identity import model_compile_identity  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import Uniform  # noqa: E402
from pops.physics import Model  # noqa: E402
from pops.physics.facade import Model as PdeModel  # noqa: E402


def _loader(source, target, so_path):
    compiled = CompiledModel(
        so_path, "production", "add_native_block",
        (), (), (), 0, None, 0, {}, {"cpu": True, "amr": target == "amr_system"},
        "abi", source._model_hash(), "c++", "c++20", target=target,
        definition_identity=model_compile_identity(source))
    compiled.name = source.name
    return compiled


class _Model(Model):
    def __init__(self, name):
        super().__init__(name)
        state = self.state("U", ("u",))
        (u,) = state
        self.flux(
            "transport", on=state, x=[0.0 * u], y=[0.0 * u],
            waves={"x": [0.0 * u], "y": [0.0 * u]},
        )


def _resolved_problem(name, blocks):
    layout = Uniform(CartesianMesh(n=16))
    problem = pops.Problem(name=name, layout=layout)
    for block_name, physics in blocks:
        problem.block(block_name, physics=physics, spatial=pops.FiniteVolume())
    problem.program(pops.Program("stub-time"))
    return orchestration.resolve(orchestration.validate(problem), layout=layout)


def test_public_artifact_has_no_authoring_backdoor_and_plan_containers_are_immutable(
        monkeypatch, tmp_path):
    binary = tmp_path / "compiled-test-artifact.so"
    binary.write_bytes(b"immutable artifact boundary")

    def fake_compile_problem(*, time, problem_snapshot, **kwargs):
        return CompiledProblem(
            str(binary), program=time, model=None, abi_key="test-abi",
            cxx="c++", std="c++20", problem_snapshot=problem_snapshot)

    monkeypatch.setattr(compile_drivers, "compile_problem", fake_compile_problem)
    monkeypatch.setattr(
        PdeModel, "compile",
        lambda source, *, target, **kwargs: _loader(source, target, str(binary)),
    )
    plan = _resolved_problem("artifact-boundary", (
        ("ions", _Model("ions")),
        ("electrons", _Model("electrons")),
    ))
    artifact = orchestration.compile(plan)
    plan = artifact.plan

    assert artifact.authoring_snapshot.hash == plan.snapshot.hash
    assert tuple(block.name for block in plan.blocks) == ("ions", "electrons")
    assert {block.name for block in artifact.blocks} == {"ions", "electrons"}
    for forbidden in (
        "_problem", "_block_specs", "_block_models", "_block_compiled_models",
        "_layout", "_target", "_field_solvers", "_outputs",
    ):
        assert not hasattr(artifact, forbidden)

    with pytest.raises(TypeError):
        plan.field_solvers["phi"] = object()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.blocks = ()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        artifact.plan = None

    assert isinstance(plan.blocks, tuple)
    assert isinstance(plan.outputs, tuple)
    assert isinstance(plan.diagnostics, tuple)


def test_model_compile_must_return_the_standard_immutable_loader(monkeypatch):
    monkeypatch.setattr(
        PdeModel, "compile",
        lambda _source, *, target, **_kwargs: SimpleNamespace(
            so_path="/tmp/mutable.so", target=target, mutable=[]),
    )
    model = _Model("mutable-result")
    plan = _resolved_problem("mutable-result", (("fluid", model),))

    with pytest.raises(TypeError, match="must return exact CompiledModel"):
        orchestration.compile(plan)
