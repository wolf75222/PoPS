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
)
from pops.codegen._compiled_artifact import (  # noqa: E402
    CompiledBlockArtifact,
    CompiledSimulationArtifact,
)
from pops.codegen._compiled_model_identity import model_compile_identity  # noqa: E402
from pops.codegen._phases import bind as bind_phase  # noqa: E402
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.model import Module  # noqa: E402
from pops.model.resolved_bindings import ResolvedBindings  # noqa: E402
from pops.params import ConstParam, RuntimeParam  # noqa: E402
from tests.python.support.resolved_amr_plan import resolved_amr_plan  # noqa: E402
from tests.python.support.native_execution_context import (  # noqa: E402
    artifact_execution_context,
)


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
    resolved_params = {}
    if runtime_param:
        resolved_params["alpha"] = RuntimeParam("alpha", default=1.0)
    resolved_params["gamma"] = ConstParam("gamma", 1.4)
    aux = ["B_z", "phi_bg"][:n_aux]
    component = CompiledModel(
        so_path="<stub-amr>", backend="production",
        cons_names=["rho", "mx", "my"],
        cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=n_aux,
        params=params, caps={"cpu": True, "amr": True, "mpi": mpi},
        abi_key=pops._pops.abi_key(),
        model_hash="h", cxx="c++", std="c++23", target="amr_system",
        aux_extra_names=aux, definition_identity=model_compile_identity(source),
    )
    plan = resolved_amr_plan(
        block_names=("block",),
        parameters=tuple(resolved_params.values()),
        tag_parameter="alpha" if runtime_param else None,
        cells=64,
        name="amr-route-introspection",
    )
    artifact = CompiledSimulationArtifact(
        plan=plan,
        program=None,
        blocks=(
            CompiledBlockArtifact(
                "block", component, plan.blocks[0].spatial, ("U",)
            ),
        ),
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
        execution_context=artifact_execution_context(artifact),
    )
    assert type(install.bind_inputs) is BindInputs
    assert type(install.params) is ResolvedBindings
    assert install.params is params
    assert install.target == "amr_system"
    assert install.block_models["block"] is artifact.blocks[0].model
    install.verify()


def test_public_amr_bind_refuses_the_retired_initial_state_compatibility_route():
    artifact = _amr_artifact(runtime_param=False)
    with pytest.raises(ValueError, match="single layout initialization authority"):
        bind_phase(
            artifact,
            BindInputs(initial_state={"block": np.ones((3, 8, 8), dtype=np.float64)}),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
