"""The public inspect bridge is inert, stable and JSON serialisable."""
from __future__ import annotations

import json

import pops
from pops.model import Module


def test_inspect_is_the_single_public_structured_view() -> None:
    assert pops.inspect is not None
    assert "inspect" in pops.__all__


def test_inspect_dispatches_to_a_descriptor() -> None:
    from pops.mesh.cartesian import CartesianMesh

    mesh = CartesianMesh(n=8)
    record = pops.inspect(mesh)
    assert record == mesh.inspect()
    assert record["name"] == "CartesianMesh"


def test_inspect_case_is_json_ready_and_does_not_compile() -> None:
    case = pops.Case("plasma")
    case.block("ne", Module("electron"))

    record = pops.inspect(case)
    assert record["name"] == "plasma"
    assert "blocks" in record
    json.dumps(record)


def test_inspect_plain_object_never_runs_numerics() -> None:
    assert "repr" in pops.inspect(object())
