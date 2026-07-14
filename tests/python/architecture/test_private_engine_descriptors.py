"""ADC-693: native engine descriptors stay private and retired modules stay absent."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "python" / "pops" / "runtime"


@pytest.mark.parametrize("module", ("pops.runtime.bricks", "pops.runtime.integrate"))
def test_retired_runtime_modules_are_not_importable(module: str) -> None:
    with pytest.raises(ModuleNotFoundError, match=module.replace(".", r"\.")):
        importlib.import_module(module)


def test_only_the_private_engine_descriptor_aggregate_remains() -> None:
    from pops import runtime

    assert not (RUNTIME / "bricks.py").exists()
    assert not (RUNTIME / "integrate.py").exists()
    assert (RUNTIME / "_engine_descriptors.py").is_file()
    assert "_engine_descriptors" not in runtime.__all__
    assert not hasattr(runtime, "_EXPORTS")


def test_runtime_sources_use_no_retired_public_descriptor_import() -> None:
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in RUNTIME.rglob("*.py")
        if not path.name.endswith((" 2.py", " 3.py"))
    }
    retired = "pops.runtime." + "bricks"
    assert not [path for path, source in sources.items() if retired in source]
    assert any(
        "pops.runtime._engine_descriptors" in source
        for source in sources.values()
    )
