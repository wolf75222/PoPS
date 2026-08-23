"""Contract tests for the one public RuntimeInstance synchronization seam."""

from __future__ import annotations

import pytest

from pops.runtime._runtime_instance import RuntimeInstance


def test_synchronize_calls_selected_native_seam_once(monkeypatch) -> None:
    calls: list[str] = []

    class Native:
        @staticmethod
        def runtime_synchronize() -> None:
            calls.append("fence")

    monkeypatch.setattr(
        "pops._native_selector.selected_native_module", lambda *, required: Native()
    )
    RuntimeInstance.synchronize(object.__new__(RuntimeInstance))
    assert calls == ["fence"]


def test_synchronize_refuses_missing_native_seam(monkeypatch) -> None:
    monkeypatch.setattr(
        "pops._native_selector.selected_native_module", lambda *, required: object()
    )
    with pytest.raises(RuntimeError, match="runtime_synchronize seam"):
        RuntimeInstance.synchronize(object.__new__(RuntimeInstance))
