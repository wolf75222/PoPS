"""AMR no-Program artifacts expose every InstallPlan block through the public protocol."""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen._plans import InstallBlock, InstallPlan  # noqa: E402
from pops.codegen.compiled_artifact import CompiledArtifact  # noqa: E402
from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.amr import RegridEvery  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402
from pops.problem._snapshot import AuthoringSnapshot  # noqa: E402
from pops.runtime import _bind_validation  # noqa: E402


def _model(name, variables, *, aux=(), mpi=False):
    model = CompiledModel(
        so_path="/tmp/%s.so" % name,
        backend="production",
        adder="add_native_block",
        cons_names=variables,
        cons_roles=["Density"] + ["Scalar"] * (len(variables) - 1),
        prim_names=variables,
        n_vars=len(variables),
        gamma=None,
        n_aux=len(aux),
        params={},
        caps={"cpu": True, "amr": True, "mpi": mpi, "gpu": False},
        abi_key="test-headers|clang++|c++23",
        model_hash="hash-" + name,
        cxx="clang++",
        std="c++23",
        target="amr_system",
        aux_extra_names=aux,
    )
    object.__setattr__(model, "name", name)
    return model


def _artifact():
    ions = _model("ions-model", ["rho", "energy"], aux=["temperature"], mpi=True)
    electrons = _model("electrons-model", ["density"], mpi=False)
    layout = AMR(
        base=CartesianMesh(n=16),
        max_levels=3,
        ratio=2,
        regrid=RegridEvery(4),
    )
    snapshot = AuthoringSnapshot({"kind": "amr-no-program-multiblock"})
    ions.install_plan = InstallPlan(
        snapshot_hash=snapshot.hash,
        target="amr_system",
        layout=layout,
        blocks=(
            InstallBlock("ions", ions, None),
            InstallBlock("electrons", electrons, None),
        ),
        bind_schema=None,
        field_solvers={},
        outputs=(),
        diagnostics=(),
        has_program=False,
    )
    ions._problem_snapshot = snapshot
    return ions, layout, snapshot


def test_no_program_arguments_and_manifest_use_every_install_plan_block():
    artifact, _layout, _snapshot = _artifact()

    arguments = artifact.arguments()
    assert tuple(arguments.instances) == ("ions", "electrons")
    assert "block" not in arguments.instances
    assert arguments.instances["ions"]["components"] == 2
    assert arguments.instances["ions"]["conservative"] == ["rho", "energy"]
    assert arguments.instances["electrons"]["components"] == 1
    assert arguments.instances["electrons"]["conservative"] == ["density"]
    assert set(arguments.layout_runtime["ghost_depth_by_block"]) == {"ions", "electrons"}
    assert arguments.layout_runtime["layout"] == "amr"
    assert arguments.layout_runtime["supports_mpi"] is False
    assert set(arguments.aux) == {"temperature"}

    manifest = artifact.manifest()
    assert tuple(manifest.blocks) == ("electrons", "ions")
    assert set(manifest.ghost_depth_by_block) == {"ions", "electrons"}
    assert manifest.supports_amr is True
    assert manifest.supports_mpi is False

    requirements = artifact.requirements()
    assert requirements.constraints["layout"] == "amr"
    assert requirements.constraints["backend"] == "production"
    assert artifact.estimate_memory(CartesianMesh(n=8)).n_cons == 3


def test_no_program_compiled_model_satisfies_protocol_and_runs_bind_gates(monkeypatch):
    artifact, layout, _snapshot = _artifact()
    assert isinstance(artifact, CompiledArtifact)
    assert artifact.inspect().blocks[0]["name"] == "electrons"

    manifest = artifact.manifest()
    monkeypatch.setattr(
        _bind_validation,
        "loaded_runtime_facts",
        lambda: {
            "abi_key": manifest.abi_key,
            "precision": manifest.precision,
            "communicator": manifest.communicator,
            "supports_mpi": manifest.supports_mpi,
            "supports_gpu": manifest.supports_gpu,
        },
    )
    with pytest.raises(ValueError, match="initial state for unknown block 'ghost'"):
        _bind_validation.run_bind_gates(
            artifact, layout, {"ghost": object()}, params={}, aux={})


def test_compiled_problem_inspect_amr_reads_install_plan_layout():
    model, layout, snapshot = _artifact()
    compiled = CompiledProblem(
        "/tmp/program.so", None, None,
        "test-headers|clang++|c++23", "clang++", "c++23",
    )
    compiled.install_plan = InstallPlan(
        snapshot_hash=snapshot.hash,
        target="amr_system",
        layout=layout,
        blocks=(InstallBlock("ions", model, None),),
        bind_schema=None,
        field_solvers={},
        outputs=(),
        diagnostics=(),
        has_program=True,
    )
    compiled._problem_snapshot = snapshot

    assert not hasattr(compiled, "_layout")
    report = compiled.inspect_amr().to_dict()
    assert report["layout"] == "amr"
    assert report["max_levels"] == 3
    assert {row["slot"] for row in report["policies"]} == {"regrid"}

    override = compiled.inspect_amr(AMR(base=CartesianMesh(n=8), max_levels=2))
    assert override.to_dict()["max_levels"] == 2
