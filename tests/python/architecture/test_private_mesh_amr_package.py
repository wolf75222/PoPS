"""The adaptive public DSL and its private mesh implementation have one owner each."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pops import amr
from pops.descriptors import Availability


PACKAGE = Path(__file__).resolve().parents[3] / "python" / "pops"


def test_mesh_amr_implementation_has_only_the_private_physical_path() -> None:
    assert not (PACKAGE / "mesh" / "amr").exists()
    assert (PACKAGE / "mesh" / "_amr" / "__init__.py").is_file()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.mesh.amr")

    implementation = importlib.import_module("pops.mesh._amr")
    assert implementation.AMRTransfer is amr.AMRTransfer
    assert implementation.IgnoreAMRCriteria is amr.IgnoreAMRCriteria
    assert hasattr(amr, "PatchLayout")
    assert "PatchLayout" not in implementation.__all__
    assert not hasattr(implementation, "PatchLayout")


def test_availability_has_one_canonical_descriptor_owner() -> None:
    implementation = importlib.import_module("pops.mesh._amr")
    descriptor_module = importlib.import_module("pops.mesh._descriptor")

    assert Availability.__module__ == "pops._descriptor_protocol"
    assert "Availability" not in implementation.__all__
    assert not hasattr(implementation, "Availability")
    assert descriptor_module.__all__ == ["MeshDescriptor"]
    assert not hasattr(descriptor_module, "Availability")
