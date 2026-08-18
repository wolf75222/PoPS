"""ROMEO validate must require the native communicator, not launcher env."""
from __future__ import annotations

import sys
import types

import pytest


def _patch_native(monkeypatch, *, size: int | None, rank: int = 0, has_mpi: bool = True):
    selector = types.ModuleType("pops._native_selector")
    if size is None:
        selector.selected_native_module = lambda *, required=False: None
    else:

        class _Module:
            __has_mpi__ = has_mpi

            def n_ranks(self):
                return size

            def my_rank(self):
                return rank

        selector.selected_native_module = lambda *, required=False: _Module()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)


def test_require_native_world_size_rejects_missing_module_despite_slurm(monkeypatch):
    from verification.pops_verify.mpi_world import NativeWorldError, require_native_world_size

    _patch_native(monkeypatch, size=None)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    with pytest.raises(NativeWorldError, match="unavailable|native|communicator"):
        require_native_world_size(2)


def test_require_native_world_size_rejects_singleton_695285(monkeypatch):
    from verification.pops_verify.mpi_world import NativeWorldError, require_native_world_size

    _patch_native(monkeypatch, size=1, rank=0)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    with pytest.raises(NativeWorldError, match="1|requested|2"):
        require_native_world_size(2)


def test_require_native_world_size_accepts_native_two(monkeypatch):
    from verification.pops_verify.mpi_world import require_native_world_size

    _patch_native(monkeypatch, size=2, rank=0)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert require_native_world_size(2) == 2


def test_romeo_676_validate_source_requires_native_world_not_slurm() -> None:
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[3]
        / "verification"
        / "machines"
        / "romeo_676_validate.py"
    ).read_text(encoding="utf-8")
    assert "require_native_world_size" in text
    assert "warning SLURM_NTASKS" not in text
