"""The generated face pass of a total on a moving configuration (G03 lot M1b, ROM-21; note §7.2).

``moving_total_rate(U, M, k_M)`` (``pops.time._program.moving_total_rate``) is lowered without a
native change, in two generated cell kernels of rank one:

1. the decode ū_K = U_K / V_K, V_K the volume member of the same stage, into a stage field whose
   halos ``ctx.prepare_generated_state`` fills by the boundary laws of the block of U;
2. the rate R_K = -(F_{K+1/2} - F_{K-1/2}) of the total, undivided, where F is the Rusanov flux of
   the relative flux f(ū) - w ū at the speed |λ - w| (``program_emit_moving.relative_struct``).
   The lower face of K reads w on the copy x_left of K, the upper face on its copy x_right, in the
   members' rate k_M of the stage. The two copies of a node are bitwise equal, so both cells of a
   face evaluate the same face, bit for bit.

The Rusanov flux is kept in its two parts until the difference of the cell: the physical part
P = (f(ū_L) + f(ū_R))/2 - a (ū_R - ū_L)/2, a = max |λ - w| over the two traces, and the swept part
S = w (ū_L + ū_R)/2, with F = P - S and R_K = -(P_{K+1/2} - P_{K-1/2}) + (S_{K+1/2} - S_{K-1/2}).
On a constant state ū = u0 the two P cancel bit for bit and the S give u0 (w_right - w_left): when
u0 is a power of two, that is u0 times the rate of V_K bit for bit, so U = u0 V holds exactly along
the run (CF11). Formed as one F per face, f(u0) - w u0 would round and break it.

A face that fails (an inadmissible trace, a non-finite flux or speed) writes NaN into the rate of
its cell, as the b6 interval producer does; the candidate then fails its own checks.
"""
from __future__ import annotations

import json
from typing import Any

# The scratch subslot of the decoded density: the slot R-b's decode takes (program_emit_totals).
_DENSITY_SUBSLOT = 7
_VOLUME = 2   # the component of V in (x_left, x_right, volume)


def moving_total_rates(program: Any) -> tuple[Any, ...]:
    from .program_lowerability import all_ops
    from pops.time._program.moving_total_rate import OPERATION
    return tuple(value for value in all_ops(program) if value.op == OPERATION)


def moving_total_entry(value: Any, authority: Any) -> dict:
    """The face entry of ``rate=``: its FiniteVolume flux and wave speed, first order, Rusanov."""
    from pops.time._program.moving_total_rate import validate_moving_total_rate
    from pops.time.references import canonical_handle
    from .program_emit_moving import check_relative_face, fv_rate_entry
    from .program_models import model_for_node

    validate_moving_total_rate(value)
    state = value.inputs[0]
    entry = fv_rate_entry(authority, model_for_node(authority, value), value.block,
                          value.attrs["operator_handle"], canonical_handle(state.state_ref))
    check_relative_face(entry)
    return entry


def emit_moving_total_rate(value: Any, var: dict, lines: list, *, authority: Any,
                           blocks: Any) -> None:
    from pops.time._evaluation_point import evaluation_stage_fraction

    entry = moving_total_entry(value, authority)
    count, base = entry["group"].component_count, entry["cpp_name"]
    state, members, member_rate = value.inputs
    block = blocks[value.block]
    total, geometry, rate = var[state.id], var[members.id], var[member_rate.id]
    token = "moving_total_%d" % value.id
    density, out = token + "_density", token + "_rate"
    stage = evaluation_stage_fraction(value)
    index = "pops::Index<pops::kNativeDimension>"
    lines += [
        "// G03 M1b: the rate of the totals of block %d on the members of the stage" % block,
        'static_assert(pops::kNativeDimension == 1, "a moving total rate is served in rank one");',
        "ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator),
        "ctx.require_cartesian_generated_operator(%d, %s);" % (block, json.dumps("moving_total_rate")),
        "pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%d, %d, %s);"
        % (density, value.id, _DENSITY_SUBSLOT, total),
        "for (int li = 0; li < %s.local_size(); ++li) {" % total,
        "  const auto stored = std::as_const(%s).fab(li).view();" % total,
        "  const auto members = std::as_const(%s).fab(li).view();" % geometry,
        "  const auto decoded = %s.fab(li).view();" % density,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& cell) {"
        % total,
        "    const pops::Real volume = members(cell, %d);" % _VOLUME,
        "    for (int c = 0; c < %d; ++c) decoded(cell, c) = stored(cell, c) / volume;" % count,
        "  });",
        "}",
        "ctx.prepare_generated_state(%d, %s, %d);" % (block, density, value.id),
        "pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%d, 0, %s);"
        % (out, value.id, total),
        "PoPSMovingTotal_%dRelative %s_model;" % (value.id, token),
        "%s_model.parameter_sets[0] = ctx.program_params(%d);" % (token, block),
        "for (int li = 0; li < %s.local_size(); ++li) {" % total,
        "  const auto decoded = std::as_const(%s).fab(li).view();" % density,
        "  const auto node_rates = std::as_const(%s).fab(li).view();" % rate,
        "  const auto result = %s.fab(li).view();" % out,
        "  const auto model = %s_model;" % token,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& cell) {"
        % total,
        "    using State = typename PoPSMovingTotal_%dRelative::State;" % value.id,
        "    State lower{}, centre{}, upper{};",
        "    for (int c = 0; c < %d; ++c) {" % count,
        "      lower[c] = decoded(%s(cell[0] - 1), c);" % index,
        "      centre[c] = decoded(cell, c);",
        "      upper[c] = decoded(%s(cell[0] + 1), c);" % index,
        "    }",
        "    // Each face of K reads w on the copy of K at that face; the copy of the neighbour is",
        "    // the same node, bitwise, so both cells evaluate the same face.",
        "    auto face = [&](const State& left, const State& right, pops::Real w, State& physical,",
        "                    State& swept) {",
        "      if (model.admissibility(left) != pops::nd::StateConversionStatus::Success ||",
        "          model.admissibility(right) != pops::nd::StateConversionStatus::Success) return false;",
        "      auto relative = model; relative.mesh_speed = w;",
        "      const pops::Real bound = Kokkos::max(relative.template max_wave_speed<0>(left),",
        "                                           relative.template max_wave_speed<0>(right));",
        "      if (!Kokkos::isfinite(bound) || !(bound >= pops::Real(0))) return false;",
        "      const State fl = model.%s::template flux<0>(left);" % base,
        "      const State fr = model.%s::template flux<0>(right);" % base,
        "      for (int c = 0; c < %d; ++c) {" % count,
        "        physical[c] = pops::Real(0.5) * (fl[c] + fr[c]) -",
        "                      pops::Real(0.5) * bound * (right[c] - left[c]);",
        "        swept[c] = w * (pops::Real(0.5) * (left[c] + right[c]));",
        "        if (!Kokkos::isfinite(physical[c]) || !Kokkos::isfinite(swept[c])) return false;",
        "      }",
        "      return true;",
        "    };",
        "    State physical_below{}, physical_above{}, swept_below{}, swept_above{};",
        "    const bool valid =",
        "        face(lower, centre, node_rates(cell, 0), physical_below, swept_below) &&",
        "        face(centre, upper, node_rates(cell, 1), physical_above, swept_above);",
        "    for (int c = 0; c < %d; ++c)" % count,
        "      result(cell, c) = valid ? -(physical_above[c] - physical_below[c]) +",
        "                                    (swept_above[c] - swept_below[c])",
        "                              : std::numeric_limits<pops::Real>::quiet_NaN();",
        "  });",
        "}",
    ]
    var[value.id] = out


__all__ = ["emit_moving_total_rate", "moving_total_entry", "moving_total_rates"]
