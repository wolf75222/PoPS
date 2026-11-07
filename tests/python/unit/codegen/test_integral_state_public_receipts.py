"""Pre-native checks of the independently decoded receipt and its public model."""
import struct

import numpy as np
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_integral_state_public_restart import build_case
from tests.python.support.integral_state_receipts import (
    checkpoint_exchange_images, decode_exchange_image, split_history_field_envelope, ssprk_upwind_step)


def _word(value):
    return struct.pack("<Q", value % (1 << 64))


def _text(value):
    value = value.encode()
    return _word(len(value)) + value


def _double(value):
    return struct.pack("<d", value)


def _blob(value):
    return _word(len(value)) + value


_IDENTITIES = ("operation", "occurrence", "context", "cell:7/axis:0/side:1")


def _exchange_image():
    image = b"POPSEX02" + _word(1) + b"".join(map(_text, _IDENTITIES))
    image += _word(-1) + _double(1.) + _double(1.2) + _double(.005) + _word(1)
    image += _word(1) + _word(2) + _word(1) + _word(1) + _text("evaluation:4")
    image += _word(1) + _text("q") + _double(.7) + _double(.706)
    return image + _word(1) + b"".join(map(_text, _IDENTITIES))


def test_independent_popsex_reader_preserves_signed_incidence_and_consumption():
    identities = _IDENTITIES
    image = _exchange_image()
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


def test_checkpoint_reader_opens_the_history_field_envelope_and_keeps_each_image_whole(tmp_path):
    # With History, the plan default Field metadata budget wraps each rank's exchange image in
    # POPSEX06: the image, one (ring, sample ledger) pair per ring, then the metadata table.
    image = _exchange_image()
    rings = _word(1) + _blob(b"retained_rhs") + _blob(b"ledger")
    envelope = b"POPSEX06" + _blob(image) + rings + _blob(b"table")
    raw = image + envelope
    np.savez(tmp_path / "checkpoint.npz", program_exchange_state=np.frombuffer(raw, dtype=np.uint8),
             program_exchange_offsets=np.array([0, len(image), len(raw)], dtype=np.int64))
    images, ledgers = checkpoint_exchange_images(tmp_path / "checkpoint.npz")
    assert images == (image, envelope)
    assert ledgers[0] == ledgers[1] == decode_exchange_image(image)
    assert split_history_field_envelope(envelope) == (image, {b"retained_rhs": b"ledger"}, b"table")
    for damaged, message in ((envelope + b"x", "exact History Field envelope"),
                             (envelope[:-1], "truncated History Field envelope"),
                             (b"POPSEX06" + _blob(image) + _word(1 << 40), "ring count exceeds"),
                             (b"POPSEX06" + _blob(image) + _word(2) + 2 * (_blob(b"r") + _blob(b"s"))
                              + _blob(b""), "duplicate History ring"),
                             (b"POPSEX05" + envelope[8:], "requires POPSEX06")):
        with pytest.raises(AssertionError, match=message):
            split_history_field_envelope(damaged)
    nested = b"POPSEX06" + _blob(b"POPSEX01" + image[8:]) + _word(0) + _blob(b"")
    with pytest.raises(AssertionError, match="POPSEX02"):
        decode_exchange_image(split_history_field_envelope(nested)[0])


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
