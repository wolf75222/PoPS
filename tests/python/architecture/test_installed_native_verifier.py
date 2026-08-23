"""Contracts for exact post-install native-dimension verification."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[3]
VERIFIER = ROOT / "scripts" / "verify_installed_native.py"
BUILD_FINGERPRINT = "a" * 64


def _verifier():
    spec = importlib.util.spec_from_file_location("_pops_installed_native_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _installed_native(
    root: Path,
    *,
    dimension: int = 2,
    mpi: bool = True,
    kokkos: bool = True,
    hdf5: bool = True,
    capability=None,
):
    if capability is None:
        capability = {
            "available": True,
            "hdf5_version": "1.14.3",
            "reason": "",
            "communicator": "explicit native MPI communicator",
            "implementation": "C++ HDF5 C API",
        }
    extension = root / "pops" / "_native" / f"dim{dimension}" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(f"native Dim={dimension}".encode())
    abi = f"abi-dim{dimension}"
    row = {
        "dimension": dimension,
        "path": f"dim{dimension}/{extension.name}",
        "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "version": "1.0.0",
        "abi_key": abi,
        "build_fingerprint": BUILD_FINGERPRINT,
        "has_mpi": mpi,
        "has_kokkos": kokkos,
    }
    manifest = extension.parents[1] / "variants.json"
    manifest.write_text(
        json.dumps({"schema_version": 2, "variants": [row]}), encoding="utf-8"
    )
    native = ModuleType("pops._pops")
    native.__file__ = str(extension)
    native.__native_dimension__ = dimension
    native.__version__ = "1.0.0"
    native.__has_mpi__ = mpi
    native.__has_kokkos__ = kokkos
    native.__build_fingerprint__ = BUILD_FINGERPRINT
    native.__has_parallel_hdf5__ = hdf5
    native.abi_key = lambda: abi
    native._parallel_hdf5_capability = lambda: capability
    return extension, native


def test_mpi_hdf5_verification_selects_and_exercises_exact_installed_leaf(tmp_path):
    verifier = _verifier()
    extension, native = _installed_native(tmp_path)
    selected = []

    def selector(dimension):
        selected.append(dimension)
        return native

    assert verifier.verify_installed_native(
        expect_dimension=2,
        expect_mpi=True,
        expect_parallel_hdf5=True,
        selector=selector,
    ) == extension.resolve()
    assert selected == [2]


def test_verifier_rejects_a_selector_returning_the_wrong_dimension(tmp_path):
    verifier = _verifier()
    _, native = _installed_native(tmp_path, dimension=3)

    with pytest.raises(verifier.InstalledNativeVerificationError, match="expected Dim=2"):
        verifier.verify_installed_native(
            expect_dimension=2, selector=lambda _dimension: native
        )


def test_serial_verification_rejects_a_stale_mpi_extension(tmp_path):
    verifier = _verifier()
    _, native = _installed_native(tmp_path, mpi=True, hdf5=True)

    with pytest.raises(verifier.InstalledNativeVerificationError, match="native serial backend"):
        verifier.verify_installed_native(
            expect_dimension=2,
            expect_mpi=False,
            expect_parallel_hdf5=False,
            selector=lambda _dimension: native,
        )

    extension, serial = _installed_native(tmp_path, mpi=False, hdf5=False)
    assert verifier.verify_installed_native(
        expect_dimension=2,
        expect_mpi=False,
        expect_parallel_hdf5=False,
        selector=lambda _dimension: serial,
    ) == extension.resolve()


@pytest.mark.parametrize(
    ("native_kwargs", "message"),
    [
        ({"mpi": False}, "native MPI backend"),
        ({"hdf5": False}, "parallel HDF5 backend"),
        ({"capability": {"available": True}}, "capability report is malformed"),
        ({
            "capability": {
                "available": False,
                "hdf5_version": "1.14.3",
                "reason": "not initialized",
                "communicator": "explicit native MPI communicator",
                "implementation": "C++ HDF5 C API",
            },
        }, "capability is not available"),
    ],
)
def test_requested_native_capabilities_fail_closed(tmp_path, native_kwargs, message):
    verifier = _verifier()
    _, native = _installed_native(tmp_path, **native_kwargs)

    with pytest.raises(verifier.InstalledNativeVerificationError, match=message):
        verifier.verify_installed_native(
            expect_dimension=2,
            expect_mpi=True,
            expect_parallel_hdf5=True,
            selector=lambda _dimension: native,
        )


def test_manifest_digest_and_compiled_facts_are_authenticated(tmp_path):
    verifier = _verifier()
    extension, native = _installed_native(tmp_path)
    extension.write_bytes(b"tampered after manifest")

    with pytest.raises(verifier.InstalledNativeVerificationError, match="disagree"):
        verifier.verify_installed_native(
            expect_dimension=2, selector=lambda _dimension: native
        )


def test_verifier_rejects_a_module_fingerprint_that_disagrees_with_its_manifest(tmp_path):
    verifier = _verifier()
    _, native = _installed_native(tmp_path)
    native.__build_fingerprint__ = "b" * 64

    with pytest.raises(
        verifier.InstalledNativeVerificationError, match="build fingerprint"
    ):
        verifier.verify_installed_native(
            expect_dimension=2, selector=lambda _dimension: native
        )


def test_verifier_cli_requires_an_explicit_dimension():
    source = VERIFIER.read_text(encoding="utf-8")

    assert '"--expect-dim", required=True' in source
    assert "select_native_dimension" in source
    assert 'importer("pops._pops")' not in source
    assert "build_fingerprint" in source
    assert "__build_fingerprint__" in source
