"""Pure oracle and public authoring checks for the installed T3 v2 reception."""

import numpy as np
import pops
import pytest

from tests.python.support.local_product_operator_case import make_case
from tests.python.support.local_product_operator_oracle import (
    GAINS, WIDTHS, equation_without_capture, incompatible_sum_lower_bound,
    manufactured_solution, original_residual,
)


@pytest.mark.parametrize("gains", GAINS)
def test_manufactured_oracle_uses_every_cell_component_and_original_equation(gains):
    values = manufactured_solution()
    old = equation_without_capture(values, gains)
    assert tuple(value.shape for value in values) == ((1, 4, 4), (2, 4, 4), (4, 4, 4))
    assert all(np.ptp(value) > 0 for value in values)
    assert max(np.max(np.abs(r)) for r in original_residual(values, old, gains)) == 0.
    wrong_source = tuple(gain * value**2 + np.arange(1, value.shape[0] + 1)[:, None, None]
                         * value + .1 * sum(np.sum(v, axis=0) for v in values) - capture
                         for value, capture, gain in zip(values, old, gains, strict=True))
    assert max(np.max(np.abs(r)) for r in wrong_source) > .1


def test_incompatible_cell_has_positive_lower_bound_for_every_real_candidate():
    old = list(equation_without_capture(manufactured_solution(), GAINS[0]))
    for value in old:
        value[:, 1, 2] = -100.
    assert incompatible_sum_lower_bound(old, GAINS[0])[1, 2] > 600.
    count = sum(WIDTHS)
    minimizer = tuple(np.broadcast_to(
        (-(np.arange(1, width + 1) + .1 * count) / (8. * gain))[:, None, None],
        (width, 4, 4)) for width, gain in zip(WIDTHS, GAINS[0], strict=True))
    total = sum(np.sum(row, axis=0) for row in original_residual(minimizer, old, GAINS[0]))
    np.testing.assert_allclose(total, incompatible_sum_lower_bound(old, GAINS[0]),
                               rtol=0., atol=2.e-13)


@pytest.mark.parametrize("reverse", (False, True))
def test_public_product_helper_keeps_legacy_arity_and_returns_qualified_parameters(reverse):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    assert len(make_case()) == 3
    case, layout, subjects, parameters = make_case(
        widths=WIDTHS, derived=True, reverse=reverse, return_parameters=True)
    assert len(subjects) == len(parameters) == 3
    assert len(set(parameters)) == 3
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rendered = emit_cpp_program(resolved.time, model_graph=graph)
    assert "prepare_local_nonlinear_problem<7>" in rendered
    assert "coupled_implicit failed:" in rendered
