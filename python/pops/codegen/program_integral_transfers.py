"""Accepted physical-trace delivery into authored persistent scalar states."""
from __future__ import annotations

import json


def integral_identity(program, name: str) -> str:
    from pops.time._program.integrals import integral_identity as identity
    return identity(program, name)


def emit_integral_declarations(program) -> list[str]:
    return [
        "ctx.declare_integral_state(%s, pops::Real(%s));" %
        (json.dumps(integral_identity(program, name)), float(initial).hex())
        for name, initial in sorted(program._integral_states.items())
    ]


def emit_integral_transfers(program, var, *, region=None) -> list[str]:
    from pops.codegen.program_transport_quadrature import accepted_transport_quadrature

    captured = {key[1]: item for key, item in var.items()
                if isinstance(key, tuple) and len(key) == 2 and key[0] == "accepted_transport"}
    regions = var.get(("accepted_transport_regions",))
    selected = (set(regions.get(region, {})) if regions is not None else
                {rate_id for rate_id, _ in accepted_transport_quadrature(
                    program, {rate_id: row[0] for rate_id, row in captured.items()})})
    requested = {row[1] for row in program._integral_transfers}
    root_diffusive = {value.id for value in program._values
                      if value.op == "diffusive_rhs" and value.id in requested}
    if region is None and root_diffusive:
        from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
        from pops.codegen.program_emit_diffusion import _resolved_diffusive_trace_selection

        for value, _ in accepted_diffusive_quadrature(
                program, partition_stability_checked=var.get(("partition_stability_checked",), ())):
            if value.id in requested and value.id in var:
                captured[value.id] = (value, None, *_resolved_diffusive_trace_selection(value))
                selected.add(value.id)
    lines = []
    for name, rate_id, axis, side, component, scale in program._integral_transfers:
        if region is not None and rate_id in root_diffusive:
            continue  # Top-level constitutive faces are staged by the parent diffusion phase.
        if regions is not None and rate_id not in selected:
            if any(rate_id in values for values in regions.values()):
                continue  # Its faces and duration belong to a different executable region.
        if rate_id not in selected or rate_id not in captured:
            raise ValueError(
                "integral transfer requires an accepted, exact conservative face occurrence"
            )
        rate, _, operation, occurrence = captured[rate_id]
        evaluation = "stage:" + str(rate.point) + "/evaluation:" + str(rate.id)
        lines.append(
            "ctx.consume_external_trace(%s, "
            "pops::runtime::program::AcceptedExchangeLedger::TraceSelection{%s,%s,%d,%d,%d,%s}, "
            "pops::Real(%s));" % (
                json.dumps(integral_identity(program, name)), json.dumps(operation),
                json.dumps(occurrence), axis, side, component, json.dumps(evaluation),
                float(scale).hex(),
            )
        )
    return lines


__all__ = ["emit_integral_declarations", "emit_integral_transfers", "integral_identity"]
