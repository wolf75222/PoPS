"""Original mixed field equations lowered to the existing spatial Newton workspace."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.identity import make_identity, canonical_bytes
from pops.identity.scalar import scalar_literal, scalar_data
from pops.model import Handle
from pops.time.points import StagePoint, TimePoint
from pops.time.solve_problem import _SpatialFieldResidual
from pops.time.solve_request import DerivativeStrategy, SolveRequest, SolveUnknown, SolveRequestError
from pops.time.values import ProgramValue
from pops.time._program.serialization import _json_ready

from .problem import FieldProblemError
from ._program_problem import _identity, _physical_boundary, _unknown_index
from ._program_expression import encode_field_expression, field_expression_cpp

CONTRACT = "pops.spatial-field-residual@1"


def _product_width(value: Any) -> int | None:
    """Authenticate scalar-product arithmetic without rewriting legacy SSA data."""
    if value.vtype != "scalar_field":
        return None
    if type(value.attrs.get("ncomp")) is int and value.attrs["ncomp"] > 0:
        return value.attrs["ncomp"]
    if value.op == "linear_combine" and value.inputs:
        widths = {_product_width(part) for part in value.inputs}
        if len(widths) == 1 and None not in widths:
            return widths.pop()
    return None


def compile_equations(problem: Any, states: tuple) -> tuple[tuple, tuple]:
    from pops.math import elliptic_terms
    from pops._ir.expr import Laplacian
    from fractions import Fraction
    from pops._ir.elliptic import DivCoeffGrad, Reaction, constant_reaction_scalar
    from pops._ir.handle_expr import ValueExpr

    width = len(problem.unknowns)
    diffusion = [Fraction(0) for _ in range(width * width)]
    local = []
    for row, equation in enumerate(problem.equations):
        expression = -equation.rhs
        for term in elliptic_terms(equation.lhs):
            column = _unknown_index(problem, term)
            if type(term) in (Laplacian, DivCoeffGrad):
                coefficient = 1 if type(term) is Laplacian else constant_reaction_scalar(term.coeff)
                if coefficient is NotImplemented:
                    raise FieldProblemError("field.nonlinear.diffusion", "v1 requires constant diffusion; candidate-dependent spatial coefficients require another realization")
                slot = row * width + column
                diffusion[slot] -= Fraction(term.scale) * Fraction(coefficient)
            elif type(term) is Reaction:
                expression = expression + term.scale * term.coeff * ValueExpr(problem.unknowns[column])
            else:
                raise FieldProblemError("field.nonlinear.operation", "original field equation has an unprepared operation")
        local.append(encode_field_expression(expression, states, unknowns=problem.unknowns))
    # Encoding literals authenticates finite/exact numbers rather than C++ strings.
    encoded_diffusion = tuple(("literal", scalar_literal(value).to_data()) for value in diffusion)
    for expression in local:
        field_expression_cpp(expression, states, views=tuple("capture%d" % i for i in range(len(states))),
                             unknowns=problem.unknowns)
    return encoded_diffusion, tuple(local)


def bind_nonlinear_field_problem(program: Any, field: Handle, registration: Any, *,
                                 values: Any, at: Any, solver: Any) -> SolveRequest:
    from pops.solvers import Newton
    from pops.time._program.value_validation import require_top_level
    from pops.time.stencil import StencilAccess

    problem, numerical = registration.operator, registration.discretization
    if type(solver) is not Newton:
        raise FieldProblemError("field.nonlinear.solver", "original mixed field residual requires explicit Newton/GMRES")
    if numerical.preconditioner is not None or numerical.nonlinear is not None:
        raise FieldProblemError("field.nonlinear.numerics", "v1 has no additional preconditioner or outer solver realization")
    if numerical.boundaries or numerical.nullspace is not None or numerical.gauge is not None or problem.gauge is not None:
        raise FieldProblemError("field.nonlinear.authority", "v1 supports explicit physical boundaries and no gauge/kernel projection")
    if type(at) not in (StagePoint, TimePoint) or not isinstance(values, Mapping):
        raise TypeError("field residual requires exact values mapping and explicit evaluation point")
    dependencies = problem.dependencies()
    if any(handle.kind != "state" for handle in dependencies):
        raise FieldProblemError("field.nonlinear.capture", "captures must be exact State quantities; auxiliary providers are not candidate-dependent closures")
    captures = []
    for handle in dependencies:
        matches = [(key, getattr(value, "_value", value)) for key, value in values.items()
                   if isinstance(key, Handle) and _identity(key) == _identity(handle)]
        if len(matches) != 1:
            raise FieldProblemError("field.nonlinear.capture", "missing exact qualified equation capture")
        key, value = matches[0]
        if not isinstance(value, ProgramValue) or value.vtype != "state" or value.state_ref is None or _identity(value.state_ref) != _identity(key):
            raise FieldProblemError("field.nonlinear.capture", "capture does not preserve its exact State owner")
        require_top_level(program, value, "field residual capture")
        captures.append(value)
    if len(values) != len(dependencies):
        raise FieldProblemError("field.nonlinear.capture", "undeclared or duplicate field residual captures")
    captures = tuple(captures)
    if not captures:
        raise FieldProblemError("field.nonlinear.layout", "v1 requires an exact State capture witnessing the physical Cartesian layout")
    diffusion, local = compile_equations(problem, captures)
    width = len(problem.unknowns)
    boundary = _physical_boundary(problem)
    prototype = program._new("scalar_field", "scalar_field", (), {"ncomp": width},
                             problem.name + "_unknown_product", None, point=at,
                             inherit_state_ref=False)
    coefficients = program._new("scalar_field", "field_problem_coefficients", (),
        {"ncomp": width * width, "unknown_ncomp": width, "field_problem_identity": problem.identity.token,
         "field_dependencies": (), "field_handle": field.canonical_identity(), "physical_boundary": boundary,
         "scope": "level", "coefficient_admissibility": "finite_general", "expressions": diffusion,
         "stencil_access": StencilAccess.pointwise()}, problem.name + "_diffusion", None, point=at,
        inherit_state_ref=False)
    from pops._frozen_data import freeze_containers
    from pops.time._program.equation_identity import _equation_value
    source = freeze_containers({"field_problem": problem.to_data(), "field_handle": field.canonical_identity(),
        "unknown_components": tuple(row.canonical_identity() for row in problem.unknowns),
        "field_problem_identity": problem.identity.token, "diffusion": diffusion, "local_expressions": local,
        "physical_boundary": boundary, "finite_difference_step": scalar_data(numerical.method.finite_difference_step),
        "captures": tuple(_equation_value(program, value) for value in captures)})
    residual = _SpatialFieldResidual(field, prototype, coefficients, captures, local,
                                     numerical.method.finite_difference_step, boundary, source)
    inputs = {"prototype": prototype, "coefficients": coefficients,
              **{"capture_%d" % i: value for i, value in enumerate(captures)}}
    return SolveRequest(problem=residual, unknowns=(SolveUnknown("field_tuple", prototype),),
        equation_inputs=inputs, seeds={"field_tuple": None}, outputs=("field_tuple",),
        derivative=DerivativeStrategy("finite_difference"), residual_interpretation="original_field_equations",
        error_interpretation="spatial_residual_l2", problem_metadata={"field_problem": problem.to_data(),
            "field_handle": field.canonical_identity(),
            "unknown_components": tuple(row.canonical_identity() for row in problem.unknowns)})


def _request_data(program: Any, token: Any, unknown: Any, physical: Any) -> dict:
    from pops.time._program.equation_identity import _equation_value

    captures = token.inputs[2:2 + token.attrs["capture_count"]]
    seed = None if token.attrs["seed_index"] is None else _equation_value(program, token.inputs[-1])
    equation = {"contract": CONTRACT, "local_expressions": _json_ready(token.attrs["local_expressions"]),
                "coefficients": _equation_value(program, token.inputs[1]),
                "captures": [_equation_value(program, value) for value in captures],
                "physical_problem": physical, "boundary": token.attrs["physical_boundary"]}
    equation_id = make_identity("solve-equation", equation).token
    problem = {"equation": equation_id, "physical_problem": physical, "unknowns": [unknown.to_data()],
               "residual_interpretation": "original_field_equations", "error_interpretation": "spatial_residual_l2"}
    return {"schema_version": 1, **problem, "equation_identity": equation_id, "equation_inputs": equation,
            "problem_identity": make_identity("solve-problem", problem).token, "seed": seed,
            "initialization_identity": make_identity("solve-initialization", seed).token,
            "outputs": [unknown.name], "solver_identity": token.attrs["solver_identity"],
            "derivative": {"route": "finite_difference", "scheme": "central_full_residual",
                           "step": token.attrs["finite_difference_step"]},
            "lowering": {"disposition": "native", "adapter": CONTRACT}}


def build_nonlinear_field_request(program: Any, request: Any, prepared: Any, *, name: Any) -> Any:
    from pops.time._program.spatial_solve import PreparedSpatialNewton
    from pops.time._program.value_validation import require_top_level
    from pops.time.solve_outcome import SolveOutcome, ResidualSolution

    residual = request.problem
    if type(prepared) is not PreparedSpatialNewton or len(request.unknowns) != 1 or program._recording:
        raise SolveRequestError("unsupported_lowering", "field residual requires top-level prepared spatial Newton")
    unknown = request.unknowns[0]
    expected = {"prototype": residual.prototype, "coefficients": residual.coefficients,
                **{"capture_%d" % i: value for i, value in enumerate(residual.captures)}}
    if dict(request.equation_inputs) != expected or unknown.template is not residual.prototype or request.outputs != (unknown.name,):
        raise SolveRequestError("equation_input_mismatch", "field residual bindings changed")
    if request.derivative.route != "finite_difference" or request.residual_interpretation != "original_field_equations" or request.error_interpretation != "spatial_residual_l2":
        raise SolveRequestError("unsupported_derivative", "select central full-residual finite differences and original spatial residual explicitly")
    seed = request.seeds[unknown.name]
    inputs = (residual.prototype, residual.coefficients, *residual.captures)
    for value in (*inputs, *(() if seed is None else (seed,))):
        require_top_level(program, value, "field residual binding")
    if seed is not None and _product_width(seed) != residual.prototype.attrs["ncomp"]:
        raise SolveRequestError("unknown_type_mismatch", "field product seed width/type differs")
    attrs = {"contract": CONTRACT, "problem_kind": "original_field_equations", "ncomp": residual.prototype.attrs["ncomp"],
             "field": residual.field, "field_problem_identity": residual.source_contract["field_problem_identity"],
             "source_contract": residual.source_contract,
             "capture_count": len(residual.captures), "seed_index": None if seed is None else len(inputs),
             "local_expressions": residual.local_expressions, "physical_boundary": residual.physical_boundary,
             "finite_difference_step": scalar_data(residual.finite_difference_step),
             "newton_controls": prepared.controls.to_data(), "solver_identity": prepared.identity.token}
    token = program._new("scalar_field", "solve_spatial_field", inputs if seed is None else (*inputs, seed), attrs,
                         name, None, point=residual.coefficients.point, inherit_state_ref=False)
    data = _request_data(program, token, unknown, _json_ready(request.problem_metadata.to_data()))
    token = program._replace_value(token, attrs={**token.attrs, "solve_request": data})
    validate_nonlinear_field_request(program, token)
    def project(outcome: Any) -> Any:
        value = program._new("scalar_field", "solve_outcome_component", (outcome,),
            {"index": 0, "ncomp": attrs["ncomp"], "unknown_identity": unknown.identity,
             "problem_identity": data["problem_identity"]}, (name or token.name) + "_value", None,
            point=token.point, inherit_state_ref=False)
        return ResidualSolution((value,), (unknown.name,), (unknown,), data["problem_identity"])
    return SolveOutcome(program, token, project, name or token.name)


def validate_nonlinear_field_request(program: Any, token: Any) -> None:
    from pops.time._program.spatial_solve import spatial_newton_options
    from pops.time.canonical_data import CanonicalData
    from pops.time._program.equation_identity import _equation_value

    actual = _json_ready(token.attrs["solve_request"])
    source = token.attrs["source_contract"]
    if token.attrs["contract"] != CONTRACT or token.attrs["problem_kind"] != "original_field_equations":
        raise SolveRequestError("equation_identity_drift", "field residual realization changed")
    count, seed_index = token.attrs["capture_count"], token.attrs["seed_index"]
    if type(count) is not int or count <= 0 or len(token.inputs) != 2 + count + (seed_index is not None) \
            or (seed_index is not None and (type(seed_index) is not int or seed_index != 2 + count)):
        raise SolveRequestError("equation_identity_drift", "field capture/seed input slots changed")
    captures = token.inputs[2:2 + count]
    expected_physical = CanonicalData({key: source[key] for key in (
        "field_problem", "field_handle", "unknown_components")}, where="original field authority").to_data()
    def same(lhs: Any, rhs: Any) -> bool:
        return canonical_bytes(_json_ready(lhs)) == canonical_bytes(_json_ready(rhs))
    checks = {"physical_problem": same(actual["physical_problem"], expected_physical),
              "local_expressions": same(source["local_expressions"], token.attrs["local_expressions"]),
              "diffusion": same(source["diffusion"], token.inputs[1].attrs["expressions"]),
              "width": type(token.attrs["ncomp"]) is int and token.attrs["ncomp"] == len(source["unknown_components"])
                       and _product_width(token.inputs[0]) == token.attrs["ncomp"]
                       and (seed_index is None or _product_width(token.inputs[-1]) == token.attrs["ncomp"]),
              "coefficient_contract": token.inputs[1].op == "field_problem_coefficients"
                       and token.inputs[1].attrs["ncomp"] == token.attrs["ncomp"]**2
                       and token.inputs[1].attrs["field_problem_identity"] == source["field_problem_identity"]
                       and token.inputs[1].attrs["coefficient_admissibility"] == "finite_general",
              "point": token.point == token.inputs[0].point == token.inputs[1].point,
              "boundary": token.attrs["physical_boundary"] == source["physical_boundary"],
              "field_identity": token.attrs["field_problem_identity"] == source["field_problem_identity"],
              "FD_step": same(token.attrs["finite_difference_step"], source["finite_difference_step"]),
              "captures": same(source["captures"], tuple(_equation_value(program, value) for value in captures))}
    if not all(checks.values()):
        raise SolveRequestError("equation_identity_drift", "original field contract changed: " +
                                ",".join(key for key, valid in checks.items() if not valid))
    unknown_handles = tuple(Handle.from_canonical_identity(_json_ready(item)) for item in source["unknown_components"])
    for expression in token.attrs["local_expressions"]:
        field_expression_cpp(expression, captures, views=tuple("capture%d" % i for i in range(len(captures))), unknowns=unknown_handles)
    unknowns = actual["unknowns"]
    if len(unknowns) != 1:
        raise SolveRequestError("invalid_unknown", "field solve must retain its full ordered unknown product")
    unknown = SolveUnknown(unknowns[0]["name"], token.inputs[0])
    expected = _request_data(program, token, unknown, actual["physical_problem"])
    if not same(actual, expected):
        raise SolveRequestError("equation_identity_drift", "field residual/seed/solver contract changed")
    controls = _json_ready(token.attrs["newton_controls"])
    spatial_newton_options(controls)
    if token.attrs["solver_identity"] != make_identity("prepared-spatial-newton", controls).token:
        raise SolveRequestError("solver_identity_drift", "field Newton controls changed")
