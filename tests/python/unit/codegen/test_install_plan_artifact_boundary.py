"""ADC-655: a public compiled artifact retains only immutable snapshot/install values."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import compile_drivers, orchestration  # noqa: E402
from pops.codegen._artifact_freeze import seal_attributes  # noqa: E402
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.codegen._compiled_model_identity import model_compile_identity  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import Uniform  # noqa: E402
from pops.model import Module  # noqa: E402


class _Loader(CompiledModel):
    def __init__(self, source, target):
        super().__init__(
            source.so_path, "production", "add_native_block",
            (), (), (), 0, None, 0, {}, {"cpu": True, "amr": target == "amr_system"},
            "abi", source._model_hash(), "c++", "c++20", target=target,
            definition_identity=model_compile_identity(source))
        self.name = source.name


class _Dsl:
    def __init__(self, name, so_path="/tmp/nonexistent-test-artifact.so"):
        self.name = name
        self.so_path = str(so_path)

    def compile(self, *, backend, target, **kwargs):
        return _Loader(self, target)

    def _model_hash(self):
        return "model-hash:%s" % self.name


class _Model:
    def __init__(self, name, so_path="/tmp/nonexistent-test-artifact.so"):
        self.name = name
        self.module = Module(name)
        self.module.state_space("U", ("u",))
        self.owner_path = self.module.owner_path
        self.dsl = _Dsl(name, so_path)

    def declaration_index(self):
        return self.module.declaration_index()


class _Time(pops.Program):
    def __init__(self):
        super().__init__("stub-time")


class _Artifact:
    def __init__(self, problem_snapshot, so_path):
        self.so_path = str(so_path)
        self.abi_key = "test-abi"
        self.cxx = "c++"
        self.std = "c++20"
        self.model = None
        self.bind_schema = None
        self.install_plan = None
        self._problem_snapshot = problem_snapshot
        self._sealed = False

    @property
    def authoring_snapshot(self):
        return self._problem_snapshot

    def _seal(self):
        seal_attributes(self)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("compiled artifact is immutable")
        object.__setattr__(self, name, value)


def test_public_artifact_has_no_authoring_backdoor_and_plan_containers_are_immutable(
        monkeypatch, tmp_path):
    binary = tmp_path / "compiled-test-artifact.so"
    binary.write_bytes(b"immutable artifact boundary")

    def fake_compile_problem(*, problem_snapshot, **kwargs):
        return _Artifact(problem_snapshot, binary)

    monkeypatch.setattr(compile_drivers, "compile_problem", fake_compile_problem)
    problem = (pops.Problem(name="artifact-boundary")
               .block("ions", physics=_Model("ions", binary))
               .block("electrons", physics=_Model("electrons", binary)))
    artifact = orchestration.compile(
        problem,
        layout=Uniform(CartesianMesh(n=16)),
        time=_Time(),
    )
    plan = artifact.install_plan

    assert artifact.authoring_snapshot.hash == plan.snapshot_hash
    assert tuple(block.name for block in plan.blocks) == ("ions", "electrons")
    assert set(plan.block_models) == {"ions", "electrons"}
    for forbidden in (
        "_problem", "_block_specs", "_block_models", "_block_compiled_models",
        "_layout", "_target", "_field_solvers", "_outputs",
    ):
        assert not hasattr(artifact, forbidden)

    with pytest.raises(TypeError):
        plan.field_solvers["phi"] = object()
    with pytest.raises(TypeError):
        plan.block_models["late"] = object()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.blocks = ()
    with pytest.raises(AttributeError):
        artifact.install_plan = None

    assert isinstance(plan.blocks, tuple)
    assert isinstance(plan.outputs, tuple)
    assert isinstance(plan.diagnostics, tuple)


def test_model_compile_must_return_the_standard_immutable_loader(monkeypatch):
    class MutableDsl(_Dsl):
        def compile(self, *, backend, target, **kwargs):
            return SimpleNamespace(
                so_path="/tmp/mutable.so", target=target, mutable=[])

    model = _Model("mutable-result")
    model.dsl = MutableDsl("mutable-result")
    problem = pops.Problem(name="mutable-result").block("fluid", physics=model)

    with pytest.raises(TypeError, match="must return pops.codegen.CompiledModel"):
        orchestration.compile(
            problem, layout=Uniform(CartesianMesh(n=16)), time=_Time())
