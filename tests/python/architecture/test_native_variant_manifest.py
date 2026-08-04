"""Architecture contract for the closed native-variant manifest."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRITER = ROOT / "scripts" / "write_native_variant_manifest.py"
PROVER = ROOT / "scripts" / "prove_installed_wheel.py"


def _writer():
    spec = importlib.util.spec_from_file_location("_native_variant_manifest_test", WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prover():
    spec = importlib.util.spec_from_file_location("_native_variant_wheel_proof_test", PROVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extension(native_root: Path, dimension: int, payload: bytes = b"native") -> Path:
    path = native_root / f"dim{dimension}" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _module(extension: Path, dimension: int) -> ModuleType:
    module = ModuleType("pops._pops")
    module.__file__ = str(extension)
    module.__native_dimension__ = dimension
    module.__version__ = "1.2.3"
    module.__has_mpi__ = dimension == 3
    module.__has_kokkos__ = True
    module.abi_key = lambda: f"abi-dim{dimension}"
    return module


def test_writer_extracts_compiled_facts_and_atomically_merges_dimensions(tmp_path, monkeypatch):
    writer = _writer()
    native_root = tmp_path / "pops" / "_native"
    manifest = native_root / "variants.json"
    dim1 = _extension(native_root, 1, b"dim1")
    dim3 = _extension(native_root, 3, b"dim3")
    modules = {dim1.resolve(): _module(dim1, 1), dim3.resolve(): _module(dim3, 3)}
    monkeypatch.setattr(writer, "_load_exact_extension", lambda path: modules[path])

    writer.update_manifest(manifest, dim3, dimension=3, version="1.2.3")
    rows = writer.update_manifest(manifest, dim1, dimension=1, version="1.2.3")

    assert [row["dimension"] for row in rows] == [1, 3]
    assert rows[0] == {
        "dimension": 1,
        "path": f"dim1/{dim1.name}",
        "sha256": hashlib.sha256(b"dim1").hexdigest(),
        "version": "1.2.3",
        "abi_key": "abi-dim1",
        "has_mpi": False,
        "has_kokkos": True,
    }
    assert not tuple(native_root.glob(".variants.*"))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../dim2/_pops.so", "canonical and relative"),
        ("dim1/_pops.so", "for its dimension"),
        ("dim2/not_pops.so", "EXT_SUFFIX"),
    ],
)
def test_manifest_rejects_escaping_or_mislabeled_paths(path, message):
    writer = _writer()
    payload = {
        "schema_version": 1,
        "variants": [{
            "dimension": 2,
            "path": path,
            "sha256": "a" * 64,
            "version": "1.0.0",
            "abi_key": "abi",
            "has_mpi": False,
            "has_kokkos": True,
        }],
    }

    with pytest.raises(writer.NativeVariantManifestError, match=message):
        writer.validate_manifest_payload(payload)


def test_manifest_requires_unique_sorted_rows_and_exact_expected_set():
    writer = _writer()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def row(dimension):
        return {
            "dimension": dimension,
            "path": f"dim{dimension}/_pops{suffix}",
            "sha256": f"{dimension}" * 64,
            "version": "1.0.0",
            "abi_key": f"abi-{dimension}",
            "has_mpi": False,
            "has_kokkos": True,
        }

    with pytest.raises(writer.NativeVariantManifestError, match="unique and sorted"):
        writer.validate_manifest_payload(
            {"schema_version": 1, "variants": [row(3), row(1)]}
        )
    with pytest.raises(writer.NativeVariantManifestError, match="explicit set"):
        writer.validate_manifest_payload(
            {"schema_version": 1, "variants": [row(1), row(3)]},
            expected_dimensions=(1, 2, 3),
        )


def test_writer_cli_is_fully_explicit():
    source = WRITER.read_text(encoding="utf-8")

    for option in ("--extension", "--manifest", "--dimension", "--version"):
        assert option in source
    assert "required=True" in source
    assert 'name = "pops._pops"' in source


def test_wheel_proof_accepts_an_explicit_fat_set_and_rejects_a_hidden_subset(tmp_path):
    prover = _prover()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    wheel = tmp_path / "pops-1.2.3-cp312-cp312-any.whl"
    distribution = tmp_path / "site-packages"
    package = distribution / "pops" / "__init__.py"
    manifest = distribution / "pops" / "_native" / "variants.json"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    rows = []
    members = {}
    for dimension in (1, 2, 3):
        payload = f"native Dim={dimension}".encode()
        relative = f"dim{dimension}/_pops{suffix}"
        rows.append({
            "dimension": dimension,
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "version": "1.2.3",
            "abi_key": f"abi-{dimension}",
            "has_mpi": False,
            "has_kokkos": True,
        })
        members["pops/_native/" + relative] = payload
        installed = distribution / "pops" / "_native" / relative
        installed.parent.mkdir(parents=True)
        installed.write_bytes(payload)
    manifest_payload = {"schema_version": 1, "variants": rows}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    metadata_payload = "Metadata-Version: 2.3\nName: PoPS\nVersion: 1.2.3\n"
    metadata = distribution / "pops-1.2.3.dist-info" / "METADATA"
    metadata.parent.mkdir()
    metadata.write_text(metadata_payload, encoding="utf-8")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pops/__init__.py", "")
        archive.writestr("pops/_native/variants.json", json.dumps(manifest_payload))
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("pops-1.2.3.dist-info/METADATA", metadata_payload)
    direct_url = {
        "archive_info": {"hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
        "url": wheel.as_uri(),
    }

    proof = prover.build_proof(
        wheel,
        package_file=package,
        native_manifest=manifest,
        distribution_root=distribution,
        python_executable=Path(sys.executable),
        installed_version="1.2.3",
        direct_url=direct_url,
        expected_dimensions=(1, 2, 3),
    )

    assert [row["dimension"] for row in proof["native_variants"]] == [1, 2, 3]
    with pytest.raises(prover.NativeVariantManifestError, match="explicit set"):
        prover.build_proof(
            wheel,
            package_file=package,
            native_manifest=manifest,
            distribution_root=distribution,
            python_executable=Path(sys.executable),
            installed_version="1.2.3",
            direct_url=direct_url,
            expected_dimensions=(2,),
        )
