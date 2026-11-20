"""Accepted physical-trace delivery into authored persistent scalar states."""
from __future__ import annotations

import json
from dataclasses import dataclass


def integral_identity(program, name: str) -> str:
    from pops.time._program.integrals import integral_identity as identity
    return identity(program, name)


def emit_integral_declarations(program) -> list[str]:
    return [
        "ctx.declare_integral_state(%s, pops::Real(%s));" %
        (json.dumps(integral_identity(program, name)), float(initial).hex())
        for name, initial in sorted(program._integral_states.items())
    ]



@dataclass(frozen=True)
class IntegralTransfer:
    name: str
    block: int
    operation: str
    occurrence: str
    point: str
    evaluation: str
    axis: int
    side: int
    component: int
    scale: float

    def cpp(self, program) -> str:
        return (
            "ctx.consume_external_trace(%s, "
            "pops::runtime::program::AcceptedExchangeLedger::TraceSelection{%s,%s,%d,%d,%d,%s}, "
            "pops::Real(%s));" % (
                json.dumps(integral_identity(program, self.name)), json.dumps(self.operation),
                json.dumps(self.occurrence), self.axis, self.side, self.component,
                json.dumps(self.evaluation), self.scale.hex(),
            )
        )


def integral_transfers(program, var, *, region=None, model_authority=None) -> list[IntegralTransfer]:
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
                program, model_authority=model_authority,
                partition_stability_checked=var.get(("partition_stability_checked",), ())):
            if value.id in requested and value.id in var:
                captured[value.id] = (value, None, *_resolved_diffusive_trace_selection(value))
                selected.add(value.id)
    rows = []
    block_indices = program._block_indices()
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
        point = str(rate.point)
        if rate.block not in block_indices:
            # This transfer still emits for supported legacy routes, but cannot prove
            # an installed incidence without a Program block to authenticate.
            block = -1
        else:
            block = block_indices[rate.block]
        rows.append(IntegralTransfer(
            name, block, operation, occurrence, point, "stage:" + point + "/evaluation:" + str(rate.id),
            axis, side, component, float(scale),
        ))
    return rows


def emit_integral_transfers(program, var, *, region=None, model_authority=None) -> list[str]:
    return [row.cpp(program) for row in
            integral_transfers(program, var, region=region, model_authority=model_authority)]


def emit_uniform_exchange_incidence(program, rows) -> str:
    from pops.codegen.cpp_strings import cpp_string_literal
    from pops.time._program.integrals import integral_units_bytes

    if rows is None or not rows or any(row.block < 0 for row in rows):
        return ""
    consumers = set()
    for row in rows:
        # Repeated physical selections have no statically established distinct invocation.
        key = (row.operation, row.occurrence, row.evaluation, row.axis, row.side, row.component)
        if key in consumers:
            return ""
        consumers.add(key)
    row_entries = []
    for ordinal, row in enumerate(rows):
        name = row.name
        units = program._integral_units.get(name)
        text = [integral_identity(program, name), row.operation, row.occurrence, row.point,
                row.evaluation, integral_units_bytes(units) if units is not None else ""]
        row_entries.append(
            "  {%s,%d,%d,%d,%d,%d,%s,%s}" % (
                ",".join(map(cpp_string_literal, text)), ordinal, row.block, row.axis, row.side,
                row.component, row.scale.hex(), float(program._integral_states[name]).hex(),
            )
        )
    return (
        '#include <pops/runtime/program/module_metadata.hpp>\n'
        'extern "C" const pops::runtime::program::ProgramExchangeIncidenceRow* '
        'pops_program_exchange_incidence_v1(int* count) {\n'
        '  static const pops::runtime::program::ProgramExchangeIncidenceRow rows[] = {\n'
        + ",\n".join(row_entries) + '\n  };\n'
        f"  *count = {len(rows)};\n"
        '  return rows;\n}\n'
    )


__all__ = ["emit_integral_declarations", "emit_integral_transfers", "integral_identity"]
