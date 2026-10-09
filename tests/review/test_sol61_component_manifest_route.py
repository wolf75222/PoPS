"""Real detached handles and typed aggregate records; no Native execution."""
import pytest
from pops.external.artifact_manifest import build_compiled_manifest, build_component_manifest
from tests.python.unit.codegen.test_compiled_model_boundary import _SourceModel, _compile
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture


def test_compiled_model_manifest_uses_its_exact_component_metadata():
    compiled = _compile(_SourceModel())
    retained = compiled.module_manifest.to_dict()
    manifest = compiled.manifest()
    assert manifest.variables == ('u',)
    assert manifest.roles == ('scalar',)
    assert manifest.supports_uniform is True
    assert manifest.supports_amr is False
    assert manifest.supports_mpi is None
    assert manifest.required_headers_sig == 'abi'
    assert manifest.ghost_depth is None
    assert not manifest.ghost_depth_by_block
    assert not manifest.field_outputs
    assert compiled.module_manifest.to_dict() == retained
    with pytest.raises(TypeError, match='CompiledSimulationArtifact'):
        build_compiled_manifest(compiled)


@pytest.mark.parametrize("names", [("zeta", "alpha"), ("alpha", "zeta")])
def test_aggregate_manifest_remains_whole_artifact_not_component_fallback(monkeypatch, names):
    from pops._platform_contracts import artifact_platform_manifest
    # Source-only platform metadata seam; no Native module is loaded or fabricated.
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',
        lambda *, backend, target, blocks, programs, external: artifact_platform_manifest(
            backend=backend, target=target, component=blocks[0].model))
    artifact = artifact_fixture(target='amr_system', block_names=names)
    manifest = artifact.manifest()
    assert set(manifest.blocks) == {'zeta', 'alpha'}
    assert manifest.variables == ()  # no representative model for an aggregate
    with pytest.raises(TypeError, match='exact CompiledModel/CompiledProblem'):
        build_component_manifest(artifact)
