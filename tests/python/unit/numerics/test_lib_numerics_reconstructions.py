"""N3 (ROM-87): the b6 Reconstruct, weno5, muscl and limiters, the parity oracles of the surface.

Since N3 lots 2 and 3 these b6 bodies are the parity oracles of
``examples/api_v02/b6_reconstructions.py`` (D5), imported from there; the words of the surface
are tested by ``tests/python/unit/surface/test_n3_lot2_muscl.py`` and ``test_n3_lot3_weno5.py``.

* CF4, IR25: ``muscl(U, limiter)`` is the same construction as the one written by hand, with
  ``Reconstruct`` and with ``reconstruction.User``: the three have one source identity, so they
  generate the same C++.
* The builder derives the stencil, the ghost depth and the order bound from the body (D5a), and
  refuses the reads that a per-axis reconstruction cannot realize.
* The generated face policy of ``muscl(U, limiter)``, emitted by the compiler's own lowering, is
  compiled next to the native policy of the same limiter (``include/pops/numerics/fv/
  reconstruction.hpp``) and evaluated on both faces of a cell over a grid of differences that
  covers zeros of both signs, subnormal, tiny, ordinary, huge and non-finite values. Where the
  native trace is finite, the generated one is the same bits; where it is not, the generated
  policy refuses (NaN), as the face guard then does. The one other refusal is vanleer's product
  ab beyond the largest double, which the generated policy, refusing any non-finite intermediate
  value, does not form into a finite slope as the native policy does: it is counted and bounded
  here. The bits differ in one other place, counted too: a cell value -0 with a zero slope, whose
  native trace on the negative face is -0 + (-1/2) 0 = -0, while the mirrored body gives
  -0 + (1/2) 0 = +0; the two traces compare equal.
* The generated policy of ``weno5(U)`` is compiled beside the native ``pops::Weno5`` (``weno5z``)
  with the default regulariser and with 1e-6, over every five-sample stencil of a grid that
  holds both sides of the native rescaling bound 1e37, and over smooth and jumping profiles from
  1e-30 to 1e49 in magnitude: where the native trace is finite the bits are equal, and where it
  is not the generated policy refuses.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import reconstruction

from examples.api_v02.b6_reconstructions import (Reconstruct, mc, minmod, muscl, superbee,
                                                 vanleer, weno5)

REPO = Path(__file__).resolve().parents[4]
LIMITERS = {"minmod": (minmod, "Minmod"), "vanleer": (vanleer, "VanLeer"),
            "mc": (mc, "MC"), "superbee": (superbee, "Superbee")}


def _state(components=("u",)):
    frame = Rectangle("n3_square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("n3_reconstruction", frame=frame)
    return model.state("U", components=components)


@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_muscl_is_the_construction_written_by_hand(name):
    limiter = LIMITERS[name][0]
    U = _state()

    def by_hand_sample(sample):
        u = sample(0)
        return u + .5*limiter(u - sample(-1), sample(1) - u)

    def by_hand_cell(c, a):
        u = c.value(U)
        return u + .5*limiter(u - c.shift(a, -1).value(U), c.shift(a, 1).value(U) - u)

    library = muscl(U, limiter)
    identities = {library.options["source_identity"],
                  reconstruction.User(by_hand_sample, formal_order=2).options["source_identity"],
                  Reconstruct(U, by_hand_cell, formal_order=2).options["source_identity"]}
    assert len(identities) == 1
    assert library.brick_type == "generated" and library.scheme == "source_stencil"
    assert library.options["sample_offsets"] == (-1, 0, 1)
    assert library.options["ghost_depth"] == 2 and library.options["formal_order"] == 2
    assert reconstruction.authenticated_user_reconstruction(library) is library


def test_limiters_differ_and_a_record_state_is_componentwise():
    U = _state(("rho", "m"))
    identities = {muscl(U, limiter).options["source_identity"]
                  for limiter, _ in LIMITERS.values()}
    assert len(identities) == len(LIMITERS)
    # The body sees one component: the same policy serves a scalar and a record state.
    assert muscl(U, vanleer).options["source_identity"] == \
        muscl(_state(), vanleer).options["source_identity"]


def test_reconstruct_derives_its_stencil_and_bounds_its_order():
    U = _state()
    wide = Reconstruct(U, lambda c, a: (2*c.shift(a, -2).value(U) - 13*c.shift(a, -1).value(U)
                                        + 47*c.value(U) + 27*c.shift(a, 1).value(U)
                                        - 3*c.shift(a, 2).value(U))/60)
    assert wide.options["sample_offsets"] == (-2, -1, 0, 1, 2)
    assert wide.options["ghost_depth"] == 3 and wide.options["formal_order"] == 5
    downwind = Reconstruct(U, lambda c, a: c.shift(a, 1).value(U))
    assert downwind.options["sample_offsets"] == (1,)
    assert downwind.options["ghost_depth"] == 2 and downwind.options["formal_order"] == 1
    claimed = Reconstruct(U, lambda c, a: c.shift(a, 1).value(U), formal_order=2)
    assert claimed.options["formal_order"] == 2


@pytest.mark.parametrize("body, message", [
    (lambda U: lambda c, a: c.value(U) if a == 0 else c.shift(a, 1).value(U), "axis of the face"),
    (lambda U: lambda c, a: c.shift(0, 1).value(U), "along the axis a of the face only"),
    (lambda U: lambda c, a: c.shift(a, 1.).value(U), "exact integer"),
    (lambda U: lambda c, a: c.value(object()), "only the representation"),
])
def test_reconstruct_refuses_reads_it_cannot_realize(body, message):
    U = _state()
    with pytest.raises(TypeError, match=message):
        Reconstruct(U, body(U))


def test_weno5_derives_its_stencil_and_checks_its_regulariser():
    U = _state()
    default = weno5(U)
    assert default.options["sample_offsets"] == (-2, -1, 0, 1, 2)
    assert default.options["ghost_depth"] == 3 and default.options["formal_order"] == 5
    assert weno5(U, epsilon=1e-40).options["source_identity"] == \
        default.options["source_identity"]
    assert weno5(U, epsilon=1e-6).options["source_identity"] != \
        default.options["source_identity"]
    for bad in (0., -1e-6, math.inf, math.nan):
        with pytest.raises(ValueError, match="epsilon"):
            weno5(U, epsilon=bad)
    for bad in (True, "1e-6"):
        with pytest.raises(TypeError, match="epsilon"):
            weno5(U, epsilon=bad)


def test_builders_refuse_values_that_are_not_functions():
    U = _state()
    with pytest.raises(TypeError, match="limiter"):
        muscl(U, None)
    with pytest.raises(TypeError, match="value"):
        Reconstruct(U, 1.)
    with pytest.raises(ValueError, match="formal_order"):
        Reconstruct(U, lambda c, a: c.value(U), formal_order=0)


# Differences and values met by a face trace: zeros of both signs, subnormal, tiny, ordinary,
# huge and non-finite. The cells (0, 1e-323, 2.5e-323) give subnormal differences of 2 and 3
# units, whose halves |a|/2 + |b|/2 round up where mc's centring on the smaller one is exact.
_VALUES = (0., -0., 5e-324, 1e-323, 2.5e-323, -3e-310, 1e-160, -2e-158, 0.3, -0.7, 1., 2.5, -4.,
           1e150, -3e153, 1e200, -1e300, 1.7e308, -1.7e308, math.inf, -math.inf, math.nan)


def _cpp_double(value):
    if math.isnan(value):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(value):
        return "%sstd::numeric_limits<double>::infinity()" % ("-" if value < 0 else "")
    return float.hex(value)


def _program(name):
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy

    limiter, native = LIMITERS[name]
    policy = emit_user_reconstruction_policy(
        SimpleNamespace(_user_reconstruction=muscl(_state(), limiter)))
    values = ", ".join(_cpp_double(v) for v in _VALUES)
    return "\n".join((
        "#include <cmath>", "#include <cstdint>", "#include <cstdio>", "#include <cstring>",
        "#include <limits>", "#include <pops/numerics/fv/reconstruction.hpp>",
        policy,
        "struct Sampler { const double* centre; int orientation;",
        "  double operator()(int offset) const { return centre[orientation*offset]; } };",
        "static std::uint64_t bits(double x) { std::uint64_t b; std::memcpy(&b, &x, 8); return b; }",
        "int main() {",
        "  const double values[] = {%s};" % values,
        "  const int n = sizeof(values)/sizeof(values[0]);",
        "  const pops_generated::UserReconstructionPolicy generated{};",
        "  const pops::%s native{};" % native,
        "  long equal = 0, refused = 0, product_refusals = 0, signed_zeros = 0;",
        "  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {",
        "    const double cells[3] = {values[i], values[j], values[k]};",
        "    for (int orientation : {1, -1}) {",
        "      const double centre = cells[1];",
        "      const double slope = native.limited_slope(centre - cells[0], cells[2] - centre);",
        "      const double expected = centre + double(0.5)*double(orientation)*slope;",
        "      const double got = generated.stencil_face_value(Sampler{cells + 1, orientation});",
        "      if (std::isfinite(expected)) {",
        "        if (bits(got) == bits(expected)) { ++equal; continue; }",
        "        if (orientation == -1 && got == 0. && expected == 0. && centre == 0. &&",
        "            std::signbit(centre)) {",
        "          ++signed_zeros; continue; }",
        "        const double a = centre - cells[0], b = cells[2] - centre;",
        "        if (std::isnan(got) && std::isfinite(a) && std::isfinite(b) &&",
        "            !std::isfinite(a*b)) { ++product_refusals; continue; }",
        "        std::printf(\"MISMATCH %a %a %a o=%d native=%a generated=%a\\n\",",
        "                    cells[0], cells[1], cells[2], orientation, expected, got);",
        "        return 1;",
        "      }",
        "      if (!std::isnan(got)) {",
        "        std::printf(\"ACCEPTED %a %a %a o=%d native=%a generated=%a\\n\",",
        "                    cells[0], cells[1], cells[2], orientation, expected, got);",
        "        return 1;",
        "      }",
        "      ++refused;",
        "    }",
        "  }",
        "  std::printf(\"%ld %ld %ld %ld\\n\", equal, refused, product_refusals, signed_zeros);",
        "  return 0;",
        "}",
    ))


@pytest.mark.compiler
@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_generated_muscl_policy_is_the_native_limiter_bit_for_bit(name, native_cxx, tmp_path,
                                                                   record_property):
    source = tmp_path / ("muscl_%s.cpp" % name)
    source.write_text(_program(name))
    binary = tmp_path / ("muscl_%s" % name)
    compiled = subprocess.run(
        [native_cxx, "-std=c++20", "-O2", "-I", str(REPO / "include"), str(source),
         "-o", str(binary)], capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stderr[-4000:]
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout[-2000:]
    equal, refused, product_refusals, signed_zeros = map(int, ran.stdout.split())
    record_property("bitwise_equal_traces", equal)
    record_property("signed_zero_traces", signed_zeros)
    record_property("refused_by_both", refused)
    record_property("vanleer_product_refusals", product_refusals)
    assert equal + refused + product_refusals + signed_zeros == 2*len(_VALUES)**3
    assert equal > len(_VALUES)**3
    if name != "vanleer":
        assert product_refusals == 0
    else:
        # Only differences whose product passes the largest double: |a|, |b| >= 1e154 here.
        assert 0 < product_refusals < refused


# WENO5: five-sample stencils over zeros of both signs, subnormal, tiny, ordinary, values on both
# sides of the native rescaling bound 1e37, huge and non-finite samples.
_WENO_VALUES = (0., -0., 5e-324, -1e-300, 0.3, 1., -1., 2.5, 1e30, 9e36, -2e37, 1e300,
                math.inf, math.nan)


def _weno_program(epsilon):
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy

    policy = emit_user_reconstruction_policy(
        SimpleNamespace(_user_reconstruction=weno5(_state(), epsilon=epsilon)))
    values = ", ".join(_cpp_double(v) for v in _WENO_VALUES)
    native_eps = "" if epsilon is None else "  native.eps = %s;" % float.hex(float(epsilon))
    return "\n".join((
        "#include <cmath>", "#include <cstdint>", "#include <cstdio>", "#include <cstring>",
        "#include <limits>", "#include <pops/numerics/fv/reconstruction.hpp>",
        policy,
        "struct Sampler { const double* centre; int orientation;",
        "  double operator()(int offset) const { return centre[orientation*offset]; } };",
        "static std::uint64_t bits(double x) { std::uint64_t b; std::memcpy(&b, &x, 8); return b; }",
        "static long equal = 0, refused = 0;",
        "static bool compare(const pops::Weno5& native,",
        "                    const pops_generated::UserReconstructionPolicy& generated,",
        "                    const double* cells) {",
        "  for (int orientation : {1, -1}) {",
        "    const double expected = native.stencil_face_value(Sampler{cells + 2, orientation});",
        "    const double got = generated.stencil_face_value(Sampler{cells + 2, orientation});",
        "    if (std::isfinite(expected) ? bits(got) == bits(expected) : std::isnan(got)) {",
        "      ++(std::isfinite(expected) ? equal : refused);",
        "      continue;",
        "    }",
        "    std::printf(\"MISMATCH %a %a %a %a %a o=%d native=%a generated=%a\\n\", cells[0],",
        "                cells[1], cells[2], cells[3], cells[4], orientation, expected, got);",
        "    return false;",
        "  }",
        "  return true;",
        "}",
        "int main() {",
        "  const double values[] = {%s};" % values,
        "  const int n = sizeof(values)/sizeof(values[0]);",
        "  const pops_generated::UserReconstructionPolicy generated{};",
        "  pops::Weno5 native{};",
        native_eps,
        "  double cells[5];",
        "  for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int c = 0; c < n; ++c)",
        "  for (int d = 0; d < n; ++d) for (int e = 0; e < n; ++e) {",
        "    cells[0] = values[a]; cells[1] = values[b]; cells[2] = values[c];",
        "    cells[3] = values[d]; cells[4] = values[e];",
        "    if (!compare(native, generated, cells)) return 1;",
        "  }",
        "  // Smooth and rough profiles at magnitudes from 1e-30 to 1e36 and beyond the rescaling",
        "  // bound: a sine with a shift and a jump, sampled at a deterministic sequence of phases.",
        "  std::uint64_t state = 88172645463325252ull;",
        "  for (int trial = 0; trial < 200000; ++trial) {",
        "    state ^= state << 13; state ^= state >> 7; state ^= state << 17;",
        "    const double phase = double(state % 100000) / 100000.0 * 6.283185307179586;",
        "    const double magnitude = std::pow(10.0, double(int(state >> 20) % 80) - 30.0);",
        "    const int jump = int((state >> 40) % 6);",
        "    for (int k = 0; k < 5; ++k)",
        "      cells[k] = magnitude * (std::sin(phase + 0.3 * k) + (k >= jump ? 0.5 : 0.0));",
        "    if (!compare(native, generated, cells)) return 1;",
        "  }",
        "  std::printf(\"%ld %ld\\n\", equal, refused);",
        "  return 0;",
        "}",
    ))


@pytest.mark.compiler
@pytest.mark.parametrize("epsilon", (None, 1e-6))
def test_generated_weno5_policy_is_the_native_weno5z_bit_for_bit(epsilon, native_cxx, tmp_path,
                                                                 record_property):
    source = tmp_path / "weno5.cpp"
    source.write_text(_weno_program(epsilon))
    binary = tmp_path / "weno5"
    compiled = subprocess.run(
        [native_cxx, "-std=c++20", "-O2", "-I", str(REPO / "include"), str(source),
         "-o", str(binary)], capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stderr[-4000:]
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout[-2000:]
    equal, refused = map(int, ran.stdout.split())
    record_property("bitwise_equal_traces", equal)
    record_property("refused_by_both", refused)
    assert equal + refused == 2*(len(_WENO_VALUES)**5 + 200000)
    assert equal > len(_WENO_VALUES)**5
