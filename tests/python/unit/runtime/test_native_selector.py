from __future__ import annotations

import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

import pops
from pops import _native_selector as selector


@pytest.fixture(autouse=True)
def _isolated_selector_state(monkeypatch: pytest.MonkeyPatch):
    previous_module = sys.modules.pop("pops._pops", None)
    previous_attribute = getattr(pops, "_pops", None)
    if hasattr(pops, "_pops"):
        delattr(pops, "_pops")
    monkeypatch.setattr(selector, "_STATE", selector._UNSELECTED)
    monkeypatch.setattr(selector, "_MODULE", None)
    monkeypatch.setattr(selector, "_DIMENSION", None)
    monkeypatch.setattr(selector, "_FAILURE", None)
    yield
    sys.modules.pop("pops._pops", None)
    if hasattr(pops, "_pops"):
        delattr(pops, "_pops")
    if previous_module is not None:
        sys.modules["pops._pops"] = previous_module
    if previous_attribute is not None:
        pops._pops = previous_attribute


def _fake_variant(path: Path, dimension: int = 2) -> selector._Variant:
    return selector._Variant(
        dimension=dimension,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        version="1.0.0",
        abi_key="abi;dim=%d" % dimension,
        has_mpi=False,
        has_kokkos=True,
        kokkos_execution_space="Serial",
    )


def _fake_module(path: Path, dimension: int = 2) -> ModuleType:
    class World:
        size = 1

        @staticmethod
        def allgather_bytes(payload: bytes) -> tuple[bytes, ...]:
            return (payload,)

    module = ModuleType("pops._pops")
    module.__file__ = str(path)
    module.__native_dimension__ = dimension
    module.__version__ = "1.0.0"
    module.__has_mpi__ = False
    module.__has_kokkos__ = True
    module.abi_key = lambda: "abi;dim=%d" % dimension
    module.runtime_environment_report = lambda: {"kokkos_backend": "Serial"}
    module.mpi_world = World
    return module


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_selection_is_single_exact_dimension_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dimension: int
) -> None:
    extension = tmp_path / "_pops.so"
    extension.write_bytes(b"native")
    variant = _fake_variant(extension, dimension=dimension)
    module = _fake_module(extension, dimension=dimension)

    def variant_for(requested_dimension: int) -> selector._Variant:
        assert requested_dimension == dimension
        return variant

    monkeypatch.setattr(selector, "_variant_from_manifest", variant_for)

    def load(path: Path) -> ModuleType:
        assert path == extension
        sys.modules["pops._pops"] = module
        return module

    monkeypatch.setattr(selector, "_load_global", load)

    assert selector.select_native_dimension(dimension) is module
    assert selector.select_native_dimension(dimension) is module
    assert selector.selected_native_dimension() == dimension
    assert selector.selected_native_module(required=True) is module
    assert pops._pops is module
    with pytest.raises(RuntimeError, match="cannot switch"):
        selector.select_native_dimension(1 if dimension != 1 else 2)


def test_failed_post_dlopen_verification_hides_module_and_poisons_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension = tmp_path / "_pops.so"
    extension.write_bytes(b"native")
    variant = _fake_variant(extension)
    module = _fake_module(extension, dimension=3)
    monkeypatch.setattr(selector, "_variant_from_manifest", lambda dimension: variant)

    def load(_path: Path) -> ModuleType:
        sys.modules["pops._pops"] = module
        pops._pops = module
        return module

    monkeypatch.setattr(selector, "_load_global", load)

    with pytest.raises(RuntimeError, match="dimension differs"):
        selector.select_native_dimension(2)
    assert "pops._pops" not in sys.modules
    assert not hasattr(pops, "_pops")
    with pytest.raises(RuntimeError, match="previously failed"):
        selector.select_native_dimension(2)


def test_collective_variant_mismatch_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension = tmp_path / "_pops.so"
    extension.write_bytes(b"native")
    variant = _fake_variant(extension)
    module = _fake_module(extension)

    class DivergentWorld:
        size = 2

        @staticmethod
        def allgather_bytes(payload: bytes) -> tuple[bytes, ...]:
            return payload, b'{"dimension":3}'

    module.mpi_world = DivergentWorld
    monkeypatch.setattr(selector, "_variant_from_manifest", lambda dimension: variant)

    def load(_path: Path) -> ModuleType:
        sys.modules["pops._pops"] = module
        return module

    monkeypatch.setattr(selector, "_load_global", load)

    with pytest.raises(RuntimeError, match="MPI ranks selected different"):
        selector.select_native_dimension(2)
    assert selector._STATE == selector._POISONED
    assert "pops._pops" not in sys.modules


def test_manifest_selects_and_hashes_only_requested_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_root = tmp_path / "_native"
    (native_root / "dim1").mkdir(parents=True)
    (native_root / "dim2").mkdir()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    first = native_root / "dim1" / ("_pops" + suffix)
    second = native_root / "dim2" / ("_pops" + suffix)
    first.write_bytes(b"one")
    second.write_bytes(b"two-corrupt-after-manifest")
    document = {
        "schema_version": 2,
        "variants": [
            {
                "dimension": 1,
                "path": "dim1/" + first.name,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                "version": "1.0.0",
                "abi_key": "abi;dim=1",
                "has_mpi": False,
                "has_kokkos": True,
                "kokkos_execution_space": "Serial",
            },
            {
                "dimension": 2,
                "path": "dim2/" + second.name,
                "sha256": "0" * 64,
                "version": "1.0.0",
                "abi_key": "abi;dim=2",
                "has_mpi": False,
                "has_kokkos": True,
                "kokkos_execution_space": "Serial",
            },
        ],
    }
    (native_root / "variants.json").write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(selector, "_native_roots", lambda: (native_root,))

    assert selector._variant_from_manifest(1).path == first
    with pytest.raises(RuntimeError, match="bytes differ"):
        selector._variant_from_manifest(2)


@pytest.mark.parametrize("dimension", [None, True, 0, 4, 2.0, "2"])
def test_dimension_is_an_exact_supported_integer(dimension: object) -> None:
    with pytest.raises(ValueError, match="exactly 1, 2, or 3"):
        selector.select_native_dimension(dimension)
