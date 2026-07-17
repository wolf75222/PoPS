"""Compiled-artifact manifests have one acyclic implementation owner."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POPS = ROOT / "python" / "pops"
EXTERNAL = POPS / "external"
MANIFEST = EXTERNAL / "artifact_manifest.py"


def test_artifact_manifest_model_and_operations_have_one_owner():
    assert MANIFEST.is_file()
    assert not (EXTERNAL / "_artifact_manifest_ops.py").exists()

    tree = ast.parse(MANIFEST.read_text(), str(MANIFEST))
    classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "CompiledArtifactManifest" in classes
    assert {
        "build_compiled_manifest", "apply_native_manifest", "check_layout_supported",
    } <= functions

    forbidden = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {
                "pops.external.artifact_manifest", "pops.external._artifact_manifest_ops"}:
            forbidden.append((node.lineno, node.module))
    assert not forbidden


def test_production_sources_do_not_reference_the_retired_operations_module():
    offenders = []
    for path in POPS.rglob("*.py"):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = (node.module,)
            elif isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            else:
                names = ()
            if "pops.external._artifact_manifest_ops" in names:
                offenders.append("%s:%d" % (path.relative_to(ROOT), node.lineno))
    assert not offenders


def test_pops_external_public_facade_does_not_expand():
    import pops.external as external
    from pops.external import artifact_manifest

    assert artifact_manifest.CompiledArtifactManifest.__module__ \
        == "pops.external.artifact_manifest"
    assert artifact_manifest.build_compiled_manifest.__module__ \
        == "pops.external.artifact_manifest"
    assert "CompiledArtifactManifest" not in external.__all__
    assert not hasattr(external, "CompiledArtifactManifest")
