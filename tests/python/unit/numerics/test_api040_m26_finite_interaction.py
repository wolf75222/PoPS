"""Closed finite M26 kernel, weighted adjoint, and native Program emission."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.linalg import FiniteMeasure, FiniteSupport, FiniteSymmetricInteraction


EXAMPLES = Path(__file__).resolve().parents[4] / "examples/migration/scientific"


def _modules():
    sys.path.insert(0, str(EXAMPLES))
    try:
        return (importlib.import_module("api040_m26_finite_interaction"),
                importlib.import_module("api040_m26_finite_oracle"))
    finally:
        sys.path.remove(str(EXAMPLES))


@pytest.mark.parametrize("variation", (False, True))
def test_independent_fourier_oracle_equals_direct_periodic_quadrature(variation):
    _example, oracle = _modules()
    first, second = oracle.density_pair(variation=variation)
    nodes = oracle.COORDINATES
    kernel = np.cos(2 * math.pi * (nodes[:, None] - nodes[None, :]))
    direct_first = oracle.WEIGHT * kernel @ first
    direct_second = oracle.WEIGHT * kernel @ second
    expected = oracle.metrics(first, second)
    np.testing.assert_allclose(expected["potential_first"], direct_first, atol=4e-16)
    np.testing.assert_allclose(expected["potential_second"], direct_second, atol=4e-16)
    assert min(first.min(), second.min()) > 0
    assert abs(expected["pair_first_adjoint_second"]
               - expected["pair_action_first_second"]) < 3e-16
    assert abs(expected["energy_second"] - expected["energy_first"]
               - oracle.exact_quadratic_increment(first, second)) < 4e-16
    assert abs(expected["mass_first"] - np.mean(first)) < 2e-16


def test_measure_and_kernel_authorities_reject_bad_weights_and_asymmetry():
    support = FiniteSupport("nodes", ("a", "b", "c"))
    with pytest.raises(ValueError, match="strictly positive"):
        FiniteMeasure(support, (1., 0., 1.))
    with pytest.raises(ValueError, match="strictly positive"):
        FiniteMeasure(support, (1., float("nan"), 1.))
    with pytest.raises(ValueError, match="weight"):
        FiniteMeasure(support, (1., 1.))
    measure = FiniteMeasure(support, (1., 2., 3.))
    with pytest.raises(ValueError, match="exactly symmetric"):
        FiniteSymmetricInteraction(measure, ((1., 2., 0.), (3., 1., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match="finite"):
        FiniteSymmetricInteraction(measure, ((1., 2., 0.), (2., float("inf"), 0.),
                                            (0., 0., 1.)))
    interaction = FiniteSymmetricInteraction(
        measure, ((1., 2., 0.), (2., 1., 3.), (0., 3., 1.)))
    assert interaction.map.coefficients == ((1., 4., 0.), (2., 2., 9.), (0., 6., 3.))
    with pytest.raises(ValueError, match="ordered support"):
        measure.pair(support.bind((1, 2, 3)), FiniteSupport(
            "other", support.dofs).bind((1, 2, 3)))


@pytest.mark.parametrize("permuted", (False, True))
def test_complete_program_uses_native_finite_map_and_preserves_two_captures(permuted):
    example, oracle = _modules()
    case, layout, subjects, order = example.build_case(permuted=permuted)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(
        resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert resolved.resolved_dimension == 1
    assert set(subjects) == {"density_first", "density_second", "potential_first",
                             "potential_second", "measured_pairing"}
    assert len(order) == oracle.NODES
    assert code.count("pops::detail::finite_linear_apply<12, 12>") == 2
    assert "ctx.commit_many(" in code
    assert "measured_interaction" in code


def test_two_rebinds_keep_ordered_support_and_distinct_positive_densities():
    example, oracle = _modules()
    _case, _layout, subjects, order = example.build_case()
    first = example._initial_values(subjects, order, variation=False)
    second = example._initial_values(subjects, order, variation=True)
    assert first.keys() == second.keys()
    for values in (first, second):
        for name in ("density_first", "density_second"):
            np.testing.assert_array_less(0, values[subjects[name]])
    assert not np.array_equal(first[subjects["density_first"]],
                              second[subjects["density_first"]])
    assert len(oracle.PERMUTATION) == len(set(oracle.PERMUTATION)) == oracle.NODES
