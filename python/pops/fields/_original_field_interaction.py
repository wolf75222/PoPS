"""Exact physical-to-numerical binding for interactions inside the original residual."""
from collections.abc import Mapping

from pops.identity import canonical_bytes
from pops.identity.scalar import scalar_literal
from pops.time._program.serialization import _json_ready

CONTRACT = "pops.spatial-field-residual@4"


def _primitive_terms(data):
    node = data.get("field_expression") if isinstance(data, Mapping) else None
    if node is None:
        return ()
    if node["type"] == "pops._ir.elliptic.EllipticSum":
        return tuple(term for part in node["data"]["terms"] for term in _primitive_terms(part))
    return (node,)


def _validate_realization(data):
    from pops.time._program.spatial_interaction import _dimension

    data = _json_ready(data)
    if set(data) != {"contract", "measure", "quadrature", "method", "max_workspace_bytes"} or data["contract"] != "pops.original-field-interaction-realization@1" or data["quadrature"] != "pops.cell-midpoint@1" or data["method"] != "pops.direct-spatial-interaction@1":
        raise ValueError("invalid original interaction realization")
    budget = data["max_workspace_bytes"]
    if type(budget) is not int or not 0 < budget < 2**64:
        raise ValueError("original interaction budget requires a positive exact uint64")
    measure = data["measure"]
    if set(measure) != {"contract", "coordinate_units"} or measure["contract"] != "pops.cell-volume-eb@1":
        raise ValueError("invalid original interaction measure")
    for unit in measure["coordinate_units"]:
        _dimension(unit, allow_unknown=False)
    return data


def compile_interactions(problem, realization):
    from pops.math import elliptic_terms, SpatialInteraction
    from ._identity import strict_field_data
    from ._program_problem import _unknown_index

    rows = []
    for row, equation in enumerate(problem.equations):
        for term in elliptic_terms(equation.lhs):
            if type(term) is SpatialInteraction:
                rows.append({"row": row, "column": _unknown_index(problem, term),
                             "scale": scalar_literal(term.scale).to_data(),
                             "kernel": _json_ready(term.kernel),
                             "physical_term": strict_field_data(term)["field_expression"]})
    if rows and realization is None:
        raise ValueError("physical interactions require explicit FieldInteractionQuadrature")
    if realization is not None and not rows:
        raise ValueError("interaction realization has no physical interaction in this FieldProblem")
    if not rows:
        return None
    result = {"contract": "pops.original-field-interactions@1", "realization": realization, "terms": rows}
    validate_interactions(result, problem.to_data(), tuple(u.canonical_identity() for u in problem.unknowns))
    return result


def validate_interactions(data, physical, unknowns):
    from .spatial_interaction import kernel_cpp
    from pops.time._program.spatial_interaction import _dimension

    data, physical, unknowns = map(_json_ready, (data, physical, unknowns))
    if set(data) != {"contract", "realization", "terms"} or data["contract"] != "pops.original-field-interactions@1":
        raise ValueError("invalid original field interaction contract")
    realization = _validate_realization(data["realization"])
    terms = [(row, term) for row, eq in enumerate(physical["equations"])
             for term in _primitive_terms(eq["equation"]["lhs"])
             if term["type"] == "pops._ir.elliptic.SpatialInteraction"]
    if not terms or len(terms) != len(data["terms"]):
        raise ValueError("original interaction lost a physical term")
    for (row, physical_term), term in zip(terms, data["terms"], strict=True):
        if set(term) != {"row", "column", "scale", "kernel", "physical_term"} or type(term["row"]) is not int or term["row"] != row or type(term["column"]) is not int or not 0 <= term["column"] < len(unknowns):
            raise ValueError("original interaction row/source component changed")
        expected_field = {"handle": unknowns[term["column"]]}
        if canonical_bytes(physical_term) != canonical_bytes(term["physical_term"]) or canonical_bytes(physical_term["data"]["field"]) != canonical_bytes(expected_field):
            raise ValueError("original interaction physical source changed")
        if canonical_bytes(physical_term["data"]["kernel"]) != canonical_bytes(term["kernel"]):
            raise ValueError("original interaction kernel changed")
        # strict_field_data represents exact scalars through the scalar v1 protocol.
        from ._program_expression import decode_field_literal
        from ._identity import strict_field_data
        if canonical_bytes(strict_field_data(decode_field_literal(term["scale"]).to_python())) != canonical_bytes(physical_term["data"]["scale"]):
            raise ValueError("original interaction physical sign/scale changed")
        kernel = term["kernel"]
        if set(kernel) != {"contract", "dimension", "tree", "units"} or kernel["contract"] != "pops.spatial-interaction-kernel@1" or type(kernel["dimension"]) is not int or not 1 <= kernel["dimension"] <= 3:
            raise ValueError("invalid original interaction kernel")
        kernel_cpp(kernel["tree"], kernel["dimension"])
        _dimension(kernel["units"])
        axes = realization["measure"]["coordinate_units"]
        if axes and len(axes) != kernel["dimension"]:
            raise ValueError("original interaction measure/kernel axes differ")
    return data
