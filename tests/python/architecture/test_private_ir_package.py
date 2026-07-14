"""The symbolic implementation has one private package and one public authoring facade."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "python" / "pops"


def test_ir_exists_only_at_the_private_package_path() -> None:
    assert (PACKAGE / "_ir" / "__init__.py").is_file()
    assert not (PACKAGE / "ir").exists()
    assert not (PACKAGE / "ir.py").exists()


def test_retired_public_ir_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"pops\.ir"):
        importlib.import_module("pops.ir")


def test_math_is_the_public_front_door_for_private_ir_values() -> None:
    public_math = importlib.import_module("pops.math")
    private_ir = importlib.import_module("pops._ir")

    assert public_math.Expr is private_ir.Expr
    assert public_math.ValueExpr is private_ir.ValueExpr
    assert public_math.ddt is private_ir.ddt
    assert "_ir" not in public_math.__all__
