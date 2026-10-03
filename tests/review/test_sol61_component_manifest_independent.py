"""Independent exact-handle Source checks; no Native library or runtime."""
import json
from pathlib import Path
import sys
import pytest
import pops
from pops.codegen.loader import CompiledModel, CompiledProblem
from pops.external.artifact_manifest import CompiledArtifactManifest, build_component_manifest
from tests.python.unit.codegen.test_compiled_model_boundary import _compile, _SourceModel


def test_real_source_origin_and_detached_wire_roundtrip(monkeypatch):
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2] / 'python')
    assert 'pops._pops' not in sys.modules
    model = _compile(_SourceModel('different-name'))
    assert type(model) is CompiledModel
    monkeypatch.setattr(CompiledModel, 'arguments', lambda self: (_ for _ in ()).throw(AssertionError('detached arguments forbidden')))
    report = model.manifest()
    assert type(report) is CompiledArtifactManifest
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert CompiledArtifactManifest.from_dict(data).to_dict() == data
    assert report.ghost_depth is None
    assert report.dimension is None and report.native_entrypoints == ()


def test_exact_problem_handle_unique_named_route_and_no_storage_invention():
    model = _compile(_SourceModel())
    problem = CompiledProblem('<source-only>', None, model, 'abi', 'c++', 'c++23')
    # Detached Source route authority, not a runnable Native program.
    problem.program_block_routes = ((0, 'second-distinct-name'),)
    report = problem.manifest()
    assert type(report) is CompiledArtifactManifest
    assert report.blocks == ('second-distinct-name',)
    assert report.variables == ('u',)
    assert report.ghost_depth is None and not report.field_outputs
    problem.program_block_routes = ((0, 'a'), (1, 'b'))
    with pytest.raises(ValueError, match='exactly one'):
        problem.manifest()


def test_component_subclass_and_lookalike_refused():
    class Foreign:
        pass
    with pytest.raises(TypeError, match='exact CompiledModel/CompiledProblem'):
        build_component_manifest(Foreign())
    class Derived(CompiledModel):
        pass
    with pytest.raises(TypeError, match='exact CompiledModel/CompiledProblem'):
        build_component_manifest(object.__new__(Derived))


@pytest.mark.parametrize('names', [('second', 'first'), ('first', 'second')])
def test_attached_aggregate_storage_facts_do_not_leak_to_component(monkeypatch, names):
    from pops._platform_contracts import artifact_platform_manifest
    from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',
        lambda *, backend, target, blocks, programs, external: artifact_platform_manifest(
            backend=backend, target=target, component=blocks[0].model))
    artifact = artifact_fixture(target='amr_system', block_names=names)
    report = artifact.manifest()
    args = artifact.arguments()
    assert report.ghost_depth == args.layout_runtime['ghost_depth']
    assert dict(report.ghost_depth_by_block) == args.layout_runtime['ghost_depth_by_block']
    assert set(report.blocks) == set(names)
    assert report.variables == ()
    assert CompiledArtifactManifest.from_dict(report.to_dict()).to_dict() == report.to_dict()
