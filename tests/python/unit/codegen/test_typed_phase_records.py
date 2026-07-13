"""ADC-660: exact immutable resolve/compile/bind/install phase values."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pops.codegen import _plans as plans
from pops.codegen._layout_resolution import layout_lowering_coverage
from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
from pops.identity import make_identity
from pops.mesh import CartesianMesh, normalize_layout_plan
from pops.mesh.layouts import Uniform
from pops.model import OwnerPath
from pops.model.bind_schema import BindSchema
from pops.problem._snapshot import AuthoringSnapshot


class _Canonical:
    def __init__(self, name):
        self.name = name

    def to_data(self):
        return {"name": self.name}


class _Compiled:
    def __init__(self, path, *, target="system", name="compiled"):
        self.so_path = str(path)
        self.target = target
        self.backend = "production"
        self.artifact_identity = make_identity("artifact", {"component": name})

    def inspect(self):
        return "inspect"

    def requirements(self):
        return "requirements"

    def manifest(self):
        return "manifest"

    def arguments(self):
        return "arguments"

    def capability_matrix(self):
        return "capabilities"


def _resolved_plan():
    nested_layout = {"mesh": {"shape": [16, 16]}}
    layout_plan = normalize_layout_plan(
        Uniform(CartesianMesh(n=16)), owner=OwnerPath.case("typed-phases"))
    plan = plans.ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "typed-phases"}),
        target="system",
        backend="production",
        layout=nested_layout,
        layout_plan=layout_plan,
        time=_Canonical("rk2"),
        blocks=(plans.ResolvedBlock(
            "fluid", _Canonical("model"), {"flux": ["hll"]}, "production"),),
        bind_schema=BindSchema(),
        compile_values={},
        field_solvers={"phi": {"algorithm": "mg", "levels": [2, 4]}},
        outputs=({"format": "hdf5"},),
        diagnostics=({"name": "mass"},),
        libraries=({"name": "kernels", "symbols": ["flux"]},),
        requirements={"mpi": False},
        capabilities={"cpu": True, "gpu": False},
        lowering_coverage=layout_lowering_coverage(layout_plan),
        compile_options={"std": "c++20"},
    )
    return plan, nested_layout


def _artifact(tmp_path):
    plan, _ = _resolved_plan()
    program_path = tmp_path / "program.so"
    block_path = tmp_path / "block.so"
    program_path.write_bytes(b"program-v1")
    block_path.write_bytes(b"block-v1")
    program = _Compiled(program_path, name="program")
    block = _Compiled(block_path, name="block")
    artifact = CompiledSimulationArtifact(
        plan,
        program,
        (CompiledBlockArtifact("fluid", block, plan.blocks[0].spatial),),
    )
    return artifact, program_path


def test_resolved_plan_is_exact_deeply_frozen_and_self_authenticating():
    plan, source_layout = _resolved_plan()
    assert not hasattr(plans, "ResolvedPlan")
    assert plan.plan_identity.domain == "resolved-plan"
    assert dict(plan.compile_values) == {}

    source_layout["mesh"]["shape"].append(32)
    assert plan.layout["mesh"]["shape"] == (16, 16)
    with pytest.raises(TypeError):
        plan.field_solvers["phi"]["levels"] = ()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.target = "amr_system"
    plan.verify()

    object.__setattr__(plan, "target", "amr_system")
    with pytest.raises(ValueError, match="identity verification failed"):
        plan.verify()


def test_wrong_phase_and_structural_lookalikes_are_rejected():
    plan, _ = _resolved_plan()
    with pytest.raises(TypeError, match="exact ResolvedSimulationPlan"):
        CompiledSimulationArtifact(object(), object(), ())
    with pytest.raises(TypeError, match="exact InstallPlan"):
        plans.require_install_plan(plan)


def test_compiled_artifact_is_one_exact_wrapper_and_rehashes_binaries(tmp_path):
    artifact, program_path = _artifact(tmp_path)
    assert artifact.so_path == str(program_path)
    assert artifact.inspect.__func__ is CompiledSimulationArtifact.inspect
    assert artifact.manifest.__func__ is CompiledSimulationArtifact.manifest
    artifact.verify()

    program_path.write_bytes(b"program-tampered")
    with pytest.raises(ValueError, match="identity verification failed"):
        artifact.verify()


def test_bind_inputs_preserve_array_references_but_detect_content_mutation():
    state = np.arange(8, dtype=np.float64)
    inputs = plans.BindInputs(
        initial_state={"fluid": state},
        params={"alpha": 2.0},
        aux={"gravity": state},
        resources={"device": _Canonical("cpu:0")},
    )
    assert inputs.initial_state["fluid"] is state
    assert inputs.aux["gravity"] is state
    inputs.verify()

    state[0] = -1.0
    with pytest.raises(ValueError, match="value/resource was mutated"):
        inputs.verify()


@pytest.mark.parametrize(
    "key", ["solver", "solvers", "cadence", "layout", "target", "backend", "spatial",
            "outputs", "diagnostics", "program", "algorithm"],
)
def test_bind_inputs_cannot_override_resolved_semantics(key):
    with pytest.raises(TypeError, match="cannot override resolved semantics"):
        plans.BindInputs(resources={key: _Canonical("forbidden")})


def test_install_plan_is_bind_created_and_authenticates_all_inputs(tmp_path):
    artifact, _ = _artifact(tmp_path)
    state = np.ones((2, 2))
    inputs = plans.BindInputs(initial_state={"fluid": state})
    install = plans.InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={
            "fluid": {
                "model": artifact.blocks[0].model,
                "spatial": artifact.blocks[0].spatial,
                "initial": state,
            }
        },
        params=artifact.bind_schema.resolve_bind(
            {}, compile_values=artifact.plan.compile_values),
        aux={},
    )
    assert install.target == "system"
    assert install.capabilities["cpu"] is True
    assert plans.require_install_plan(install) is install
    assert install.bind_identity.domain == "bind"

    state[0, 0] = 9.0
    with pytest.raises(ValueError, match="mutated"):
        install.verify()
