"""Exact native coefficient source and an independent arithmetic counterexample."""
from fractions import Fraction

import pytest

from examples.api_v02.method_history_transfer import regular_quadratic_flux_transfer
from pops.codegen.program_emit_history_transfer import emit_history_transfer_callback
from pops.time import HistoryTransfer


def test_lagrange_ieee_inputs_need_rational_evaluation_before_rounding():
    times = (.08, .16, .12)
    target = .10
    floating = tuple((target-times[(i+1)%3])*(target-times[(i+2)%3])
                     /((times[i]-times[(i+1)%3])*(times[i]-times[(i+2)%3]))
                     for i in range(3))
    assert sum(map(Fraction, floating)) != 1
    sources = tuple(map(Fraction, times))
    point = Fraction(target)
    exact = tuple((point-sources[(i+1)%3])*(point-sources[(i+2)%3])
                  /((sources[i]-sources[(i+1)%3])*(sources[i]-sources[(i+2)%3]))
                  for i in range(3))
    assert sum(exact) == 1
    assert sum(w*t for w, t in zip(exact, sources, strict=True)) == point
    assert sum(w*t*t for w, t in zip(exact, sources, strict=True)) == point*point
    # Rounding may discard exact unity again: retaining the rational vector is
    # essential; it is not reconstructed from these application coefficients.
    assert tuple(map(float, exact)) != floating


def test_callback_keeps_exact_weights_and_only_rounds_application_values():
    source = "\n".join(emit_history_transfer_callback(regular_quadratic_flux_transfer().to_data()))
    assert "source.exact_coefficient_bytes" in source
    assert "pops::ExactWeight::from_double(source.times[" in source
    assert "row.exact_weights = {" in source
    assert "row.weights.push_back(weight.value())" in source
    assert "validate_history_transfer_weights(row)" in source
    assert "from_double(row.weights" not in source
    assert "normalize" not in source
    assert source.index("rejected its temporal validity domain") < source.index("std::optional<pops::ExactWeight>")


def test_rational_literals_integer_powers_and_lazy_branch_have_native_algebra():
    from pops import math as pm
    def body(samples, time, ratio):
        x = samples[0].time
        weight = pm.where(x == 0, lambda: Fraction(1, 3), lambda: x**-3)
        return ratio > 0, ((time, 1., (weight, 1-weight)),
                           (samples[1].time, 1., (0., 1.)))
    source = "\n".join(emit_history_transfer_callback(HistoryTransfer(slots=2, body=body).to_data()))
    assert 'from_hex("0x1", "0x3", source.exact_coefficient_bytes)' in source
    assert 'std::string_view("11")' in source
    assert ".zero() ? transfer_exact_" in source
    assert "std::pow" not in source


@pytest.mark.parametrize("method", ("sqrt", "exp", "sin", "cos", "log", "expm1"))
def test_transcendental_is_an_explicit_rounded_boundary(method):
    from pops import math as pm
    def body(samples, time, ratio):
        weight = getattr(pm, method)(samples[0].time)
        return ratio > 0, ((time, 1., (weight, 1-weight)),
                           (samples[1].time, 1., (0., 1.)))
    source = "\n".join(emit_history_transfer_callback(HistoryTransfer(slots=2, body=body).to_data()))
    assert "ExactWeight::from_double(std::"+method+"(" in source
