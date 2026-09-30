"""Actual public M19 product declarations and lowering, with no native execution."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from pops.mesh import AxisQuadrature, PhysicalSupportMap
from pops.model import PhysicalDimension, PhysicalSupport
from pops.runtime._physical_mapping import validate_physical_geometry
from tests.python.integration.runtime.test_m19_product_support_runtime import (
    CASES, input_and_expected, resolve_product_chain,
)


@pytest.mark.parametrize("nx,nv,width", CASES)
def test_public_product_is_spatial_fibres_not_component_packing(tmp_path, nx, nv, width):
    plan, weights = resolve_product_chain(tmp_path, nx, nv, width)
    assert len(plan.layout_plan.layouts) == 3
    assert len(plan.layout_plan.mappings) == 3
    reductions = [row.requirement.physical_map for row in plan.layout_plan.mappings
                  if int(row.requirement.operation) == 2]
    assert len(reductions) == 2
    assert all(len(row.source_support.coordinates) == 2 and len(row.target_support.coordinates) == 1
               for row in reductions)
    assert all(row.source_to_target == (-1, 0) for row in reductions)
    signed, = [row for row in reductions if row.reductions[0].weights == weights]
    assert signed.reductions[0].cells == nv
    assert signed.to_data()["storage"]["inverse_closure"] is False
    shapes = {tuple(plan.layout_plan.normalized(layout.handle).native_spatial_layout.shape)
              for layout in plan.layout_plan.layouts}
    assert shapes == {(nv, nx), (nx, 1)}
    initial, expected = input_and_expected(nx, nv, width, weights)
    assert initial["population"].shape == (width, nx, nv)
    assert expected[2].shape == (width, 1, nx)
    np.testing.assert_array_equal(expected[2][:, 0, :],
        np.einsum("cxv,v->cx", initial["population"], np.asarray(weights)))
    np.testing.assert_array_equal(expected[3], np.broadcast_to(expected[2].transpose(0, 2, 1), (width, nx, nv)))


def test_product_domain_identity_and_units_are_not_inferred_from_shape():
    x, v = ("x", "physical-domain"), ("v", "velocity-domain")
    product, position = PhysicalSupport((x, v)), PhysicalSupport((x,))
    length = PhysicalDimension((("length", Fraction(1)),))
    distribution = PhysicalDimension((("count", Fraction(1)), ("length", Fraction(-1))))
    density = PhysicalDimension((("count", Fraction(1)),))
    mapping = PhysicalSupportMap(product, position,
        reductions=(AxisQuadrature(1, -2, 2, 3, length, weights=(1, -2, 4)),))
    import pops
    source = pops.Model("phase").state("f", components=("a",), support=product, units=(distribution,), sampling="cell_average")
    target = pops.Model("position").state("rho", components=("a",), support=position, units=(density,), sampling="cell_average")
    mapping.validate_ports(SimpleNamespace(subject=source), SimpleNamespace(subject=target))
    wrong = pops.Model("wrong").state("rho", components=("a",), support=position, units=(distribution,), sampling="cell_average")
    with pytest.raises(ValueError, match="units disagree"):
        mapping.validate_ports(SimpleNamespace(subject=source), SimpleNamespace(subject=wrong))
    with pytest.raises(ValueError, match="homonymous"):
        replace(mapping, target_support=PhysicalSupport((("x", "foreign-domain"),)))
    with pytest.raises(ValueError, match="strict subset"):
        replace(mapping, reductions=())


@pytest.mark.parametrize("field", ("shape", "lower", "upper", "periodicity"))
def test_retained_geometry_cannot_be_relabelled(field):
    product = PhysicalSupport((("x", "position"), ("v", "velocity")))
    mapping = PhysicalSupportMap(product, PhysicalSupport((product.coordinates[0],)),
        reductions=(AxisQuadrature(1, -2, 2, 3, PhysicalDimension()),))
    source = SimpleNamespace(shape=(4, 3), lower=(0, -2), upper=(1, 2), periodicity=(True, True), coordinate_system="cartesian")
    target = SimpleNamespace(shape=(4, 1), lower=(0, 0), upper=(1, 1), periodicity=(True, True), coordinate_system="cartesian")
    validate_physical_geometry(SimpleNamespace(physical_map=mapping), source, target)
    changes = {"shape": (5, 1), "lower": (.125, 0), "upper": (2, 1), "periodicity": (False, True)}
    setattr(target, field, changes[field])
    with pytest.raises(ValueError, match="exactly aligned"):
        validate_physical_geometry(SimpleNamespace(physical_map=mapping), source, target)
