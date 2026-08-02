"""ADC-689 source/wheel public API and typing parity proof."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "prove_public_api_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("_public_api_parity_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


proof = _load()


def _synthetic_wheel(path: Path, *, omit: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for source in sorted(proof.SOURCE_PACKAGE.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(proof.SOURCE_PACKAGE).as_posix()
            if relative == omit:
                continue
            archive.write(source, "pops/" + relative)
        archive.writestr(
            "pops-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: PoPS\nVersion: 1.0.0\n",
        )


def test_exact_wheel_and_source_share_public_api_typing_and_lazy_authoring(tmp_path):
    wheel = tmp_path / "pops-1.0.0-py3-none-any.whl"
    _synthetic_wheel(wheel)

    evidence = proof.build_proof(wheel)

    assert evidence["schema_version"] == 1
    assert evidence["public_names"] == list(proof.PUBLIC_ROOT)
    assert evidence["pure_authoring"] is True
    assert evidence["qualified_handles"] is True
    assert evidence["py_typed"] is True
    assert evidence["typed_payload_files"] > 100


def test_wheel_proof_fails_closed_when_typing_payload_is_missing(tmp_path):
    wheel = tmp_path / "pops-1.0.0-py3-none-any.whl"
    _synthetic_wheel(wheel, omit="_pops.pyi")

    with pytest.raises(proof.PublicApiParityError, match="typing payload"):
        proof.build_proof(wheel)


def test_release_workflow_blocks_publication_on_source_wheel_api_parity():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    validate = workflow[workflow.index("  validate:") : workflow.index("  release:")]

    assert "scripts/prove_public_api_parity.py" in validate
    assert '--wheel "${wheels[0]}"' in validate
    assert 'pops-final-evidence-public-api.json' in validate
    assert validate.index("scripts/prove_public_api_parity.py") < validate.index(
        "scripts/run_final_gate.py"
    )
