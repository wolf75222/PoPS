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
    from pops.time.field_context import merge_field_provenance, remap_field_provenance
    from pops.time.values import ProgramValue, _Affine
    if program._recording:
        raise ValueError("LocalResidual product cannot be nested in another residual region")
    seeds = {key: _resolve_handle(value) for key, value in problem.initial.items()}
    captures = {key: _resolve_handle(value)
                for key, value in sorted((problem.captures or {}).items())}
    values = (*seeds.values(), *captures.values())
    for value in values:
        require_top_level(program, value, "LocalResidual product")
        if value.vtype == "fields" and any(value is item for item in captures.values()):
            if value.field_context is None:
                raise ValueError("LocalResidual product field capture requires exact provenance")
        elif value.vtype != "state" or not component_names(value):
            raise ValueError("LocalResidual product requires complete States or captured fields")
    if any(value.vtype != "state" for value in seeds.values()):
        raise ValueError("LocalResidual product unknowns must be complete States")
    if len({value.block for value in seeds.values()}) != len(seeds):
        raise ValueError("LocalResidual product requires one unknown per exact BlockHandle")
    if len({value.point for value in seeds.values()}) != 1:
        raise ValueError("LocalResidual product unknowns require one exact evaluation point")
    # Value shape/units may differ; local geometry and positions must not. Native
    # layout/distribution checks discharge the remaining co-location obligation.
    first = values[0].space
    for value in values:
        if value.vtype == "fields":
            continue
        for attribute in ("layout", "centering", "support", "sampling", "frame", "clock"):
            if getattr(value.space, attribute) != getattr(first, attribute):
                raise ValueError("LocalResidual product co-location obligation: different " + attribute)
    controls = _prepared_local_nonlinear_controls(prepared, where="LocalNewton product")
    sub = []
    program._recording.append(sub)
    try:
        placeholders = [None] * len(values)
        frozen_placeholders = {}
        for i, value in enumerate(values):
            if value.vtype == "state":
                placeholder = frozen_placeholders.get(value.id) if i >= len(seeds) else None
                if placeholder is None:
                    placeholder = program._new(
                        "state", "state", (), {}, "product_argument_%d" % i, value.block,
                        space=value.space, point=value.point, state_ref=value.state_ref,
                        field_context=value.field_context)
                placeholders[i] = placeholder
                if i >= len(seeds):
                    frozen_placeholders[value.id] = placeholder
        # A frozen field remains attached to its exact frozen State occurrence.
        # It is not a solve at a Newton candidate, even when that candidate's seed
        # happens to be the same outer value.
        captured_states = {value.id: placeholders[i].id
                           for i, value in enumerate(values) if i >= len(seeds)
                           and value.vtype == "state"}
        for i, value in enumerate(values):
            if value.vtype == "fields":
                context = remap_field_provenance(
                    value.field_context, lambda source: captured_states.get(source, source))
                placeholders[i] = program._new(
                    "fields", "input_fields", (), dict(value.attrs), "product_argument_%d" % i,
                    value.block, space=value.space, point=value.point, field_context=context)
        unknowns = MappingProxyType(dict(zip(seeds, placeholders[:len(seeds)], strict=True)))
        frozen = dict(zip(captures, placeholders[len(seeds):], strict=True))
        result = problem.residual(program, unknowns, **frozen)
        if not isinstance(result, Mapping) or set(result) != set(seeds):
            raise ValueError("LocalResidual product residual keys must exactly match unknown keys")
        expressions = []
        for key, seed in seeds.items():
            row = result[key]
            if isinstance(row, _Affine):
                row = program.value("product_residual_" + key, row, at=seed.point)
            if isinstance(row, ProgramValue):
                if row.vtype not in ("state", "rhs") or row.block != seed.block:
                    raise ValueError("LocalResidual product row must retain its exact unknown block")
                row = tuple(row[c] for c in range(len(component_names(row))))
            if isinstance(row, ProgramExpression):
                row = row.components
            if not isinstance(row, (tuple, list)) or len(row) != len(component_names(seed)):
                raise ValueError("LocalResidual product residual width differs for " + key)
            expressions.extend(row)
        roots, nodes, reads = encode_expressions(expressions, program)
        allowed = {"source", "apply", "linear_source", "linear_combine", "pointwise_expression"}
        seen = set()
        for node in sub:
            if not any(node is item for item in placeholders) and node.op not in allowed:
                raise ValueError("LocalResidual product operation %r is not a supported LOCAL operator" % node.op)
            if any(item.id not in seen for item in node.inputs):
                raise ValueError("LocalResidual product body reads an undeclared capture")
            seen.add(node.id)
        argument_ids = {item.id for item in placeholders}
        extended = (any(node.id not in argument_ids for node in sub)
                    or len(argument_ids) != len(placeholders)
                    or any(v.vtype == "fields" for v in values))
        by_id = {value.id: i for i, value in enumerate(sub if extended else placeholders)}
        if any(value.id not in by_id for value in reads):
            raise ValueError("LocalResidual product body reads an undeclared capture")
        roles = tuple(by_id[value.id] for value in reads)
    finally:
        program._recording.pop()
    blocks = tuple(value.block for value in seeds.values())
    provenance = merge_field_provenance(*(value.field_context for value in (*values, *sub)))
    # Internal remapped field contexts are witnesses for residual evaluation,
    # while public outputs retain the external capture authority.
    if any(v.vtype == "fields" for v in values):
        external = {placeholder.id: value.id for placeholder, value in zip(placeholders, values)}
        provenance = remap_field_provenance(provenance, lambda source: external.get(source, source))
    token_name = name or "local_product"
    token = program._new(
        "coupled_solution", "solve_coupled_implicit", values,
        {"blocks": blocks, "method": "newton", "solver_identity": prepared.identity.token,
         "problem_kind": "local_residual_product", "product_version": 2 if extended else 1,
         "unknown_names": tuple(seeds), "capture_names": tuple(captures),
         "product_widths": tuple(len(component_names(value)) if value.vtype == "state" else 0
                                 for value in values),
         "product_reads": roles, "expressions": roots, "expression_nodes": nodes,
         "output_count": len(seeds), **controls,
         **({"residual_block": sub, "residual_region": program._region_for_block(sub),
             "product_argument_positions": tuple(
                 next(i for i, node in enumerate(sub) if node is item) for item in placeholders)}
            if extended else {})}, token_name, blocks[0], point=values[0].point,
        field_context=provenance)

    def project(outcome):
        return _CoupledResult({value.block: program._new(
            "state", "solve_outcome_component", (outcome,),
            {"index": i, "out_block": value.block}, token_name + "_" + key,
            value.block, space=value.space, point=value.point, state_ref=value.state_ref,
            field_context=provenance)
            for i, (key, value) in enumerate(seeds.items())})

    return SolveOutcome(program, token, project, token_name)
