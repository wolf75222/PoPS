"""Public Python descriptors only; no native ownership/publication reception."""

from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.mesh import AxisQuadrature, PhysicalSupportMap
from pops.model import PhysicalDimension, PhysicalSupport


X = ("x", "space-A")
V = ("v", "velocity-A")
VELOCITY = PhysicalDimension((("L", Fraction(1)), ("T", Fraction(-1))))
DENSITY = PhysicalDimension((("M", Fraction(1)), ("L", Fraction(-2))))
MOMENT = PhysicalDimension((("M", Fraction(1)), ("L", Fraction(-1)), ("T", Fraction(-1))))


def port(support, width, unit, **changes):
    space = SimpleNamespace(
        support=support,
        units=(unit,) * width,
        representation="conservative",
        sampling="cell_average",
    )
    for key, value in changes.items():
        setattr(space, key, value)
    return SimpleNamespace(subject=SimpleNamespace(space=space))


@pytest.mark.parametrize("nv", [2, 5, 11])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_public_mapping_retains_signed_weights_components_and_explicit_axis_embeddings(nv, width):
    product = PhysicalSupport((V, X))
    retained = PhysicalSupport((X,))
    weights = tuple(Fraction((-1) ** i * (2 * i + 1), 16) for i in range(nv))
    mapping = PhysicalSupportMap(
        product,
        retained,
        reductions=(AxisQuadrature(1, -3, 2, nv, VELOCITY, weights=weights),),
        source_axes=(1, 0),
        target_axes=(1,),
        native_dimension=2,
    )
    assert mapping.operation_abi == 2
    assert mapping.source_to_target == (1, -1)
    assert mapping.to_data()["kind"] == "support-reduction@2"
    assert mapping.reductions[0].weights == weights
    mapping.validate_ports(port(product, width, DENSITY), port(retained, width, MOMENT))
    extension = PhysicalSupportMap(
        retained, product, source_axes=(1,), target_axes=(1, 0), native_dimension=2
    )
    assert extension.operation_abi == 3
    assert extension.source_to_target == (-1, 0)
    assert extension.to_data()["kind"] == "support-extension@2"
    assert extension.to_data()["storage"]["inverse_closure"] is False
    extension.validate_ports(port(retained, width, MOMENT), port(product, width, MOMENT))


@pytest.mark.parametrize(
    "attack",
    [
        "foreign_domain",
        "component_count",
        "unknown_units",
        "unchanged_units",
        "sampling",
        "representation",
    ],
)
def test_public_mapping_refuses_invalid_support_width_measure_units_and_sampling(attack):
    product, retained = PhysicalSupport((X, V)), PhysicalSupport((X,))
    mapping = PhysicalSupportMap(
        product, retained, reductions=(AxisQuadrature(1, -3, 2, 5, VELOCITY),), target_axes=(0,)
    )
    source, target = port(product, 3, DENSITY), port(retained, 3, MOMENT)
    if attack == "foreign_domain":
        target.subject.space.support = PhysicalSupport((("x", "foreign-space"),))
    elif attack == "component_count":
        target.subject.space.units = (MOMENT,)
    elif attack == "unknown_units":
        source.subject.space.units = (None,) * 3
    elif attack == "unchanged_units":
        target.subject.space.units = (DENSITY,) * 3
    elif attack == "sampling":
        source.subject.space.sampling = "pointwise"
    else:
        target.subject.space.representation = "auxiliary"
    with pytest.raises(ValueError):
        mapping.validate_ports(source, target)
