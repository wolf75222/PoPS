"""MMS fixture admission; independent FV law, not installed-native evidence."""
from __future__ import annotations

import numpy as np
import pops
import pytest

from tests.python.support.captured_diffusion_mms import (
    CANDIDATE_BETA, RESIDUAL_TOL, build, initial_data, matrices,
    original_action,
)


@pytest.mark.parametrize("width,order", [(1, (0,)), (3, (2, 0, 1))])
def test_candidate_mms_preserves_full_unknown_dependent_law(width, order):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    for cells in (8, 16):
        initial, target, spatial = initial_data(cells, width, candidate_diffusion=True)
        action, measured = original_action(
            target, initial["material"][0], candidate_diffusion=True)
        np.testing.assert_array_equal(action, initial["forcing"])
        np.testing.assert_array_equal(spatial, measured)
        np.testing.assert_allclose(spatial.sum(axis=(1, 2)), 0, atol=3e-13)
        frozen_action, _ = original_action(target, initial["material"][0])
        assert np.max(np.abs(frozen_action-action)) > 10000*RESIDUAL_TOL
        diffusion, _ = matrices(width)
        assert CANDIDATE_BETA > 0 and np.any(diffusion != 0)
    case, layout, program, token = build(
        16, width, order, candidate_diffusion=True)
    validate_nonlinear_field_request(program, token)
    assert token.attrs["contract"] == "pops.spatial-field-residual@3"
    coefficient = token.inputs[1]
    assert coefficient.attrs["storage_role"] == "deferred_candidate_coefficient"
    assert coefficient.attrs["coefficient_evaluation"] == "pops.field.coefficients.per-candidate@1"
    assert token.attrs["source_contract"]["linear_residual_verification"] == (
        "pops.field.linear.true-correction-residual@1")
    assert program._serialize()["version"] == 11
    _, _, _, captured_token = build(16, width, order)
    assert token.attrs["newton_controls"] == captured_token.attrs["newton_controls"]
    assert token.attrs["finite_difference_step"] == captured_token.attrs["finite_difference_step"]
    resolved = pops.resolve(pops.validate(case), layout=layout)
    emitted = emit_cpp_program(
        resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="system")
    assert "nonfinite_original_field_residual" in emitted
    assert "nonfinite_candidate_diffusion" in emitted
