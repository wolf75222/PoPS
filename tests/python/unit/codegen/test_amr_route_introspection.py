#!/usr/bin/env python3
"""AMR-route introspection through the exact compile/bind phase records.

The public compile result is always ``CompiledSimulationArtifact``.  AMR still carries one native
``CompiledModel`` per block, but tests must not recreate the retired convention where a model was
mutated with compile- or bind-phase metadata.
"""
import sys

import numpy as np
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.codegen._plans import (  # noqa: E402
    BindInputs,
    InstallPlan,
    ResolvedBlock,
    ResolvedSimulationPlan,
)
from pops.codegen.compiled_artifact import (  # noqa: E402
    CompiledBlockArtifact,
    CompiledSimulationArtifact,
)
from pops.codegen._compiled_model_identity import model_compile_identity  # noqa: E402
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402
from pops.model import Module  # noqa: E402
from pops.model.bind_schema import BindSchema  # noqa: E402
from pops.model.resolved_bindings import ResolvedBindings  # noqa: E402
from pops.params import ConstParam, RuntimeParam  # noqa: E402
from pops.problem import Problem  # noqa: E402
from pops.problem._snapshot import AuthoringSnapshot  # noqa: E402


def _amr_artifact(*, n_aux=2, mpi=True, runtime_param=True):
    """Return an exact AMR compiled artifact without compiling or loading a shared library."""
    source = Module("amr-introspection-source")
    source.state_space("U", ("rho", "mx", "my"))
    params = {}
    if runtime_param:
        params["alpha"] = RuntimeParam("alpha", default=1.0)
    params["gamma"] = ConstParam("gamma", 1.4)
    for declaration in params.values():
        source.param(declaration)
    aux = ["B_z", "phi_bg"][:n_aux]
    component = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"],
        cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=n_aux,
        params=params, caps={"cpu": True, "amr": True, "mpi": mpi}, abi_key="k",
        model_hash="h", cxx="c++", std="c++23", target="amr_system",
        aux_extra_names=aux, definition_identity=model_compile_identity(source),
    )
    layout = AMR(base=CartesianMesh(n=64, periodic=True), max_levels=2, ratio=2)
    schema_problem = Problem(name="amr-introspection-case")
    schema_problem.add_block("block", source)
    schema = BindSchema.from_problem(schema_problem)
    plan = ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"kind": "amr-route-introspection-stub"}),
        target="amr_system",
        backend="production",
        layout=layout,
        time=None,
        blocks=(ResolvedBlock("block", source, None, "production"),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={"amr": True},
        capabilities={"cpu": True, "amr": True, "mpi": mpi},
    )
    artifact = CompiledSimulationArtifact(
        plan=plan,
        program=None,
        blocks=(CompiledBlockArtifact("block", component, None),),
    )
    return artifact


def test_amr_compile_result_is_the_exact_artifact():
    artifact = _amr_artifact()
    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.target == "amr_system"
    assert artifact.blocks[0].model.target == "amr_system"
    artifact.verify()


def test_arguments_report_amr_layout_block_aux_and_typed_params():
    args = _amr_artifact().arguments()
    assert args.layout_runtime["layout"] == "amr"
    assert args.layout_runtime["supports_mpi"] is True
    assert len(args.instances) == 1
    instance = next(iter(args.instances.values()))
    assert instance["components"] == 3
    assert instance["conservative"] == ["rho", "mx", "my"]
    assert set(args.aux) == {"B_z", "phi_bg"}
    params = {row["name"]: row for row in args.params.values()}
    assert set(params) == {"alpha", "gamma"}
    assert params["alpha"]["kind"] == "runtime"
    assert params["alpha"]["required"] is False
    assert params["gamma"]["kind"] == "const"
    assert params["gamma"]["required"] is False


def test_arguments_do_not_fabricate_mpi_capability():
    args = _amr_artifact(mpi=False).arguments()
    assert args.layout_runtime["supports_mpi"] is False


def test_bind_creates_exact_install_plan_without_mutating_compiled_components():
    artifact = _amr_artifact(runtime_param=False)
    initial = np.array([1.0, 0.0, 0.0])
    inputs = BindInputs(initial_state={"block": initial})
    params = artifact.bind_schema.resolve_bind(
        {}, compile_values=artifact.plan.compile_values)
    install = InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={
            "block": {
                "model": artifact.blocks[0].model,
                "spatial": artifact.blocks[0].spatial,
                "initial": inputs.initial_state["block"],
            }
        },
        params=params,
        aux={},
    )
    assert type(install.bind_inputs) is BindInputs
    assert type(install.params) is ResolvedBindings
    assert install.params is params
    assert install.target == "amr_system"
    assert install.block_models["block"] is artifact.blocks[0].model
    install.verify()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
