"""Source-only exact phase-record regression; no native loading or compilation."""
from types import MappingProxyType
from dataclasses import replace

import numpy as np
import pytest

from pops._platform_contracts import artifact_platform_manifest
from pops.codegen._plans import BindInputs, InstallPlan
from pops.runtime._layout_install_projection import LayoutInstallProjection
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture


@pytest.fixture
def metadata_platform(monkeypatch):
    # Only the executable-platform discovery seam is metadata-only. The resolved
    # public AMR plan, compiled artifact, local authorities and InstallPlan are real.
    monkeypatch.setattr(
        "pops.codegen._compiled_artifact._common_platform_manifest",
        lambda *, backend, target, blocks, programs, external: artifact_platform_manifest(
            backend=backend, target=target, component=blocks[0].model),
    )


def _install(artifact):
    inputs = BindInputs(initial_state={row.name: np.ones((8, 8))
                                      for row in artifact.blocks})
    return InstallPlan(
        artifact=artifact, bind_inputs=inputs,
        instances={row.name: {"model": row.model, "spatial": row.spatial,
                              "initial": inputs.initial_state[row.name]}
                   for row in artifact.blocks},
        params=artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        aux={},
    )


def _projectable_artifact():
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.codegen._layout_resolution import ResolvedRuntimeLayout, ResolvedRuntimeLayouts
    artifact = artifact_fixture(target="amr_system")
    handle = artifact.layout_plan.layouts[0].handle
    layouts = ResolvedRuntimeLayouts(artifact.layout_plan,
                                    (ResolvedRuntimeLayout(handle, artifact.plan.layout),))
    return CompiledSimulationArtifact(replace(artifact.plan, layout=layouts),
                                      artifact.program, artifact.blocks, artifact.layout_programs)


def test_real_single_layout_install_relays_registered_tagging(metadata_platform):
    install = _install(artifact_fixture(target="amr_system"))
    layout_id = install.artifact.layout_plan.layouts[0].handle.qualified_id
    local = install.layout_amr_authorities[layout_id]
    assert type(install) is InstallPlan
    assert install.resolved_tagging is local.authorities.tagging
    assert install.resolved_tagging.buffer_cells == 1
    projected_install = _install(_projectable_artifact())
    projected_local = next(iter(projected_install.layout_amr_authorities.values()))
    child = LayoutInstallProjection(projected_install,
                                   projected_install.artifact.layout_programs[0], projected_local)
    assert child.resolved_tagging is projected_install.resolved_tagging
    install.verify()


def test_multilayout_aggregate_refuses_unqualified_tagging(metadata_platform):
    from tests.python.unit.runtime.test_runtime_planning import _artifact
    install = _install(_artifact(("a", "b"), heterogeneous=True))
    assert len(install.artifact.layout_plan.layouts) == 2
    with pytest.raises(ValueError, match="single layout"):
        _ = install.resolved_tagging


def test_uniform_layout_has_no_adaptive_tagging_fallback(metadata_platform):
    install = _install(artifact_fixture())
    with pytest.raises(TypeError, match="registered local AMR"):
        _ = install.resolved_tagging


def test_qualified_lookup_refuses_foreign_layout_and_missing_authority(metadata_platform):
    install = _install(_projectable_artifact())
    with pytest.raises(ValueError, match="registered layout identity"):
        install._resolved_tagging_for_layout("foreign::layout")
    local = next(iter(install.layout_amr_authorities.values()))
    child = LayoutInstallProjection(install, install.artifact.layout_programs[0], local)
    object.__setattr__(install.artifact.plan, "layout_amr_authorities", MappingProxyType({}))
    with pytest.raises(TypeError, match="registered local AMR"):
        _ = install.resolved_tagging
    with pytest.raises(ValueError, match="lost its parent"):
        _ = child.resolved_tagging


def test_local_tagging_requires_exact_resolved_authority(metadata_platform):
    install = _install(artifact_fixture(target="amr_system"))
    local = next(iter(install.layout_amr_authorities.values()))
    object.__setattr__(local.authorities, "tagging", object())
    with pytest.raises(TypeError, match="exact ResolvedTaggingAuthority"):
        _ = install.resolved_tagging
