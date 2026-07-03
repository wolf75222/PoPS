"""ADC-527: available() is an explainable Availability -- never a bare bool -- carrying reason,
missing and alternatives, including the ``partial`` status.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.descriptors import Availability, BrickDescriptor  # noqa: E402


def test_availability_yes_no_partial_classmethods():
    assert Availability.yes().status == "yes"
    assert Availability.yes().ok is True
    no = Availability.no("no native symbol", missing=["native_id"], alternatives=["use HLL()"])
    assert no.status == "no" and not no.ok
    assert no.missing == ["native_id"]
    assert no.alternatives == ["use HLL()"]
    part = Availability.partial("only uniform", missing=["amr"], alternatives=["layout=Uniform"])
    assert part.status == "partial"
    assert not part.ok                 # partial is not fully available
    assert part.reason and part.alternatives


def test_partial_carries_reason_and_alternatives_in_str():
    part = Availability.partial("AMR unsupported on this route",
                                missing=["amr"], alternatives=["compile Uniform"])
    text = str(part)
    assert "partial" in text
    assert "AMR unsupported" in text
    assert "compile Uniform" in text


def test_brick_availability_is_explainable_for_a_planned_brick():
    # A catalogued-but-not-native brick reports an explainable 'no' with the typed alternative.
    planned = BrickDescriptor("Planned", "native", category="riemann", native_id="",
                              available=False)
    status = planned.availability()
    assert isinstance(status, Availability)
    assert status.status == "no"
    assert status.reason
    assert status.alternatives  # points at pops.inspect_capabilities()
    # A native brick is available.
    native = BrickDescriptor("HLL", "native", category="riemann", native_id="pops::HLLFlux")
    assert native.availability().ok


def test_availability_never_returns_a_bare_bool_on_the_descriptor_family():
    from pops.mesh.cartesian import CartesianMesh  # noqa: PLC0415
    status = CartesianMesh(n=8).available()
    assert isinstance(status, Availability)
    assert not isinstance(status, bool)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
