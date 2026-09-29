"""Closed original residuals over co-located, owner-qualified State products."""
from collections.abc import Mapping
from types import MappingProxyType

from pops.time.expressions import ProgramExpression, component_names, encode_expressions
from pops.time.values import _resolve_handle
from .value_validation import require_top_level


def build_local_product(program, problem, prepared, *, name=None):
    from .local import _prepared_local_nonlinear_controls
    from pops.time.solve_outcome import SolveOutcome
    from pops.time.value_collections import _CoupledResult
    from pops.time.field_context import merge_field_provenance
    if program._recording:
        raise ValueError("LocalResidual product cannot be nested in another residual region")
    seeds = {key: _resolve_handle(value) for key, value in problem.initial.items()}
    captures = {key: _resolve_handle(value)
                for key, value in sorted((problem.captures or {}).items())}
    values = (*seeds.values(), *captures.values())
    for value in values:
        require_top_level(program, value, "LocalResidual product")
        if value.vtype != "state" or not component_names(value):
            raise ValueError("LocalResidual product requires complete State values")
    if len({value.block for value in seeds.values()}) != len(seeds):
        raise ValueError("LocalResidual product requires one unknown per exact BlockHandle")
    if len({value.point for value in seeds.values()}) != 1:
        raise ValueError("LocalResidual product unknowns require one exact evaluation point")
    # Value shape/units may differ; local geometry and positions must not. Native
    # layout/distribution checks discharge the remaining co-location obligation.
    first = values[0].space
    for value in values:
        for attribute in ("layout", "centering", "support", "sampling", "frame", "clock"):
            if getattr(value.space, attribute) != getattr(first, attribute):
                raise ValueError("LocalResidual product co-location obligation: different " + attribute)
    controls = _prepared_local_nonlinear_controls(prepared, where="LocalNewton product")
    sub = []
    program._recording.append(sub)
    try:
        placeholders = [program._new(
            "state", "state", (), {}, "product_argument_%d" % i, value.block,
            space=value.space, point=value.point, state_ref=value.state_ref,
            field_context=value.field_context) for i, value in enumerate(values)]
        unknowns = MappingProxyType(dict(zip(seeds, placeholders[:len(seeds)], strict=True)))
        frozen = dict(zip(captures, placeholders[len(seeds):], strict=True))
        result = problem.residual(program, unknowns, **frozen)
        if len(sub) != len(placeholders):
            raise NotImplementedError(
                "LocalResidual product currently requires direct component expressions; "
                "source/apply and other Program nodes in the product body are not implemented")
        if not isinstance(result, Mapping) or set(result) != set(seeds):
            raise ValueError("LocalResidual product residual keys must exactly match unknown keys")
        expressions = []
        for key, seed in seeds.items():
            row = result[key]
            if isinstance(row, ProgramExpression):
                row = row.components
            if not isinstance(row, (tuple, list)) or len(row) != len(component_names(seed)):
                raise ValueError("LocalResidual product residual width differs for " + key)
            expressions.extend(row)
        roots, nodes, reads = encode_expressions(expressions, program)
        by_id = {value.id: i for i, value in enumerate(placeholders)}
        if any(value.id not in by_id for value in reads):
            raise ValueError("LocalResidual product body reads an undeclared capture")
        roles = tuple(by_id[value.id] for value in reads)
    finally:
        program._recording.pop()
    blocks = tuple(value.block for value in seeds.values())
    provenance = merge_field_provenance(*(value.field_context for value in values))
    token_name = name or "local_product"
    token = program._new(
        "coupled_solution", "solve_coupled_implicit", values,
        {"blocks": blocks, "method": "newton", "solver_identity": prepared.identity.token,
         "problem_kind": "local_residual_product", "product_version": 1,
         "unknown_names": tuple(seeds), "capture_names": tuple(captures),
         "product_widths": tuple(len(component_names(value)) for value in values),
         "product_reads": roles, "expressions": roots, "expression_nodes": nodes,
         "output_count": len(seeds), **controls}, token_name, blocks[0], point=values[0].point,
        field_context=provenance)

    def project(outcome):
        return _CoupledResult({value.block: program._new(
            "state", "solve_outcome_component", (outcome,),
            {"index": i, "out_block": value.block}, token_name + "_" + key,
            value.block, space=value.space, point=value.point, state_ref=value.state_ref,
            field_context=provenance)
            for i, (key, value) in enumerate(seeds.items())})

    return SolveOutcome(program, token, project, token_name)
