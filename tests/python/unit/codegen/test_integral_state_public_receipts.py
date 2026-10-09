"""Pre-native checks of the independently decoded receipt and its public model."""
import struct

import numpy as np
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_integral_state_public_restart import build_case
from tests.python.support.integral_state_receipts import decode_exchange_image, ssprk_upwind_step


def _word(value):
    return struct.pack("<Q", value % (1 << 64))


def _text(value):
    value = value.encode()
    return _word(len(value)) + value


def _double(value):
    return struct.pack("<d", value)


def test_independent_popsex_reader_preserves_signed_incidence_and_consumption():
    identities = ("operation", "occurrence", "context", "cell:7/axis:0/side:1")
    image = b"POPSEX02" + _word(1) + b"".join(map(_text, identities))
    image += _word(-1) + _double(1.) + _double(1.2) + _double(.005) + _word(1)
    image += _word(1) + _word(2) + _word(1) + _word(1) + _text("evaluation:4")
    image += _word(1) + _text("q") + _double(.7) + _double(.706)
    image += _word(1) + b"".join(map(_text, identities))
    actual = decode_exchange_image(image)
    record, = actual["records"]
    assert (record["axis"], record["side"], record["component"]) == (0, 1, 0)
    assert record["orientation"] == -1 and record["exterior"]
    assert record["flux"] * record["weight"] == .006
    assert actual["quantities"] == {"q": (.7, .706)}
    assert actual["consumed"] == [identities]
    with pytest.raises(AssertionError, match="exact exchange image"):
        decode_exchange_image(image + b"extra")
    with pytest.raises(AssertionError, match="POPSEX02"):
        decode_exchange_image(b"POPSEX01" + image[8:])


def test_upwind_oracle_distinguishes_stages_and_conserves_mass_with_boundary_current():
    initial = 1. + .2 * (np.arange(8) + .5) / 8
    dt = .01
    accepted, increments = ssprk_upwind_step(initial, dt)
    assert abs(increments[0] - increments[1]) > 1e-6
    # With outflow on x-, that boundary flux remains the unchanged first cell value.
    assert abs((accepted.mean() - initial.mean()) - (dt * initial[0] - sum(increments))) < 3e-16
    constant, constant_increments = ssprk_upwind_step(np.ones(8), dt)
    np.testing.assert_array_equal(constant, np.ones(8))
    assert constant_increments == (.005, .005)


@pytest.mark.parametrize("proposed_dt,axis", ((.01, 0), (.5, 0), (.01, 1)))
def test_public_native_receipt_cases_resolve_and_emit_without_jit(proposed_dt, axis):
    case, layout, quantity, _ = build_case(proposed_dt=proposed_dt, selected_axis=axis)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
                             model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.consume_external_trace(") == 2
    assert quantity.identity in source
    assert "user_face_numerical_stability" in source
