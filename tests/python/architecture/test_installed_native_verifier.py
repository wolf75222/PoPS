"""Source-level contract for the post-install native dependency verifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
VERIFIER = ROOT / "scripts" / "verify_installed_native.py"


def _verifier():
    spec = importlib.util.spec_from_file_location("_pops_installed_native_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native(extension: Path, *, mpi=True, hdf5=True, capability=None):
    if capability is None:
        capability = {
            "available": True,
            "hdf5_version": "1.14.3",
            "reason": "",
            "communicator": "explicit native MPI communicator",
            "implementation": "C++ HDF5 C API",
        }
    return SimpleNamespace(
        __file__=str(extension),
        __has_mpi__=mpi,
        __has_parallel_hdf5__=hdf5,
        _parallel_hdf5_capability=lambda: capability,
    )


def test_mpi_hdf5_verification_imports_and_exercises_the_installed_provider(tmp_path):
    verifier = _verifier()
    extension = tmp_path / "_pops.so"
    extension.touch()
    imported = []

    def importer(name):
        imported.append(name)
        return _native(extension)

    assert verifier.verify_installed_native(
        expect_mpi=True, expect_parallel_hdf5=True, importer=importer,
    ) == extension.resolve()
    assert imported == ["pops._pops"]


def test_serial_verification_rejects_a_stale_mpi_extension(tmp_path):
    verifier = _verifier()
    extension = tmp_path / "_pops.so"
    extension.touch()

    with pytest.raises(verifier.InstalledNativeVerificationError, match="native serial backend"):
        verifier.verify_installed_native(
            expect_mpi=False,
            expect_parallel_hdf5=False,
            importer=lambda _name: _native(extension, mpi=True, hdf5=True),
        )

    assert verifier.verify_installed_native(
        expect_mpi=False,
        expect_parallel_hdf5=False,
        importer=lambda _name: _native(extension, mpi=False, hdf5=False),
    ) == extension.resolve()


@pytest.mark.parametrize(
    ("native", "message"),
    [
        (lambda path: _native(path, mpi=False), "native MPI backend"),
        (lambda path: _native(path, hdf5=False), "parallel HDF5 backend"),
        (
            lambda path: _native(path, capability={"available": True}),
            "capability report is malformed",
        ),
        (
            lambda path: _native(path, capability={
                "available": False,
                "hdf5_version": "1.14.3",
                "reason": "not initialized",
                "communicator": "explicit native MPI communicator",
                "implementation": "C++ HDF5 C API",
            }),
            "capability is not available",
        ),
    ],
)
def test_requested_native_capabilities_fail_closed(tmp_path, native, message):
    verifier = _verifier()
    extension = tmp_path / "_pops.so"
    extension.touch()

    with pytest.raises(verifier.InstalledNativeVerificationError, match=message):
        verifier.verify_installed_native(
            expect_mpi=True,
            expect_parallel_hdf5=True,
            importer=lambda _name: native(extension),
        )
