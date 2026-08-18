"""Installed exact-rank capability authentication (plan §5.1 / §5.2)."""
from __future__ import annotations

import hashlib
import importlib.machinery
import json
from pathlib import Path

import pytest


def _write_leaf(tmp_path: Path, *, dimension: int = 1, has_mpi: bool = False) -> Path:
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    root = tmp_path / "native"
    leaf = root / f"dim{dimension}" / f"_pops{suffix}"
    leaf.parent.mkdir(parents=True, exist_ok=True)
    payload = b"fake-exact-rank-leaf"
    leaf.write_bytes(payload)
    row = {
        "dimension": dimension,
        "path": f"dim{dimension}/_pops{suffix}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": "1.0.0",
        "abi_key": "abi-test",
        "has_mpi": has_mpi,
        "has_kokkos": True,
    }
    (root / "variants.json").write_text(
        json.dumps({"schema_version": 1, "variants": [row]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def test_authenticate_exact_rank_leaf_from_variants_manifest(tmp_path: Path):
    from verification.pops_verify.capabilities import authenticate_installed_artifact

    root = _write_leaf(tmp_path, dimension=1, has_mpi=True)
    artifact = authenticate_installed_artifact(
        dimension=1,
        variants_root=root,
        doctor_ok=False,
    )
    assert artifact.dimension == 1
    assert artifact.has_mpi is True
    assert artifact.has_kokkos is True
    assert artifact.sha256 == hashlib.sha256(b"fake-exact-rank-leaf").hexdigest()
    assert artifact.native_variant_manifest_digest
    assert artifact.hdf5_collective is False


def test_authenticate_refuses_digest_mismatch(tmp_path: Path):
    from verification.pops_verify.capabilities import (
        CapabilityError,
        authenticate_installed_artifact,
    )

    root = _write_leaf(tmp_path)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    leaf = root / "dim1" / f"_pops{suffix}"
    leaf.write_bytes(b"tampered")
    with pytest.raises(CapabilityError, match="sha256|digest|differ"):
        authenticate_installed_artifact(dimension=1, variants_root=root, doctor_ok=False)


def test_authenticate_refuses_missing_dimension(tmp_path: Path):
    from verification.pops_verify.capabilities import (
        CapabilityError,
        authenticate_installed_artifact,
    )

    root = _write_leaf(tmp_path, dimension=1)
    with pytest.raises(CapabilityError, match="Dim=2|dimension"):
        authenticate_installed_artifact(dimension=2, variants_root=root, doctor_ok=False)


def test_authenticate_refuses_absent_manifest(tmp_path: Path):
    from verification.pops_verify.capabilities import (
        CapabilityError,
        authenticate_installed_artifact,
    )

    with pytest.raises(CapabilityError):
        authenticate_installed_artifact(dimension=1, variants_root=tmp_path / "missing")
