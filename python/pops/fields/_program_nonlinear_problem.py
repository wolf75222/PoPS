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
from ._program_expression import (encode_field_expression, field_expression_cpp,
                                  field_expression_dependencies)

CONTRACT = "pops.spatial-field-residual@1"
CAPTURED_DIFFUSION_CONTRACT = "pops.spatial-field-residual@2"
CANDIDATE_DIFFUSION_CONTRACT = "pops.spatial-field-residual@3"
PER_CANDIDATE = "pops.field.coefficients.per-candidate@1"
ARITHMETIC_FACES = "pops.field.face-mean.arithmetic@1"
SEED_PRODUCT = "pops.original-field.typed-product-seed@1"


def build_field_seed_product(request: Any, program: Any, *, expressions: Any, space: Any, name: str) -> SolveRequest:
    from dataclasses import replace
    from pops.model.spaces import FieldSpace
    from pops.time._authoring import authoring_transaction
    from pops.time._program.value_validation import require_top_level

    residual = request.problem
    if type(residual) is not _SpatialFieldResidual or len(request.unknowns) != 1:
        raise SolveRequestError("unsupported_lowering", "seed_product currently requires one original field unknown product")
    for value in (residual.prototype, residual.coefficients, *residual.captures):
        require_top_level(program, value, "original field seed product request")
    unknowns = residual.source_contract["unknown_components"]
    names = tuple(row["local_id"] for row in unknowns)
    if type(space) is not FieldSpace or space.components != names or space.centering != "cell" or \
            space.sampling != "cell_value" or any(unit is None for unit in space.units):
        raise SolveRequestError("unknown_type_mismatch", "seed product requires exact ordered physical components, cell values and explicit units")
    if type(expressions) is not tuple or len(expressions) != len(names):
        raise SolveRequestError("unknown_type_mismatch", "seed product expressions differ from the complete unknown tuple")
    point = residual.prototype.point
    authority = {"contract": SEED_PRODUCT, "problem_identity": residual.source_contract["field_problem_identity"],
                 "unknown_components": unknowns, "space": space.to_data(), "point": point.to_data(),
                 "program_owner": program.owner_path.canonical().to_data(), "layout_scope_only": True}
    with authoring_transaction(program):
        seed = program._pointwise_expression(name, expressions, at=point,
                    field_product_space=space, field_product_authority=authority)
        validate_field_seed_product(program, seed, residual.source_contract, point)
        return replace(request, seeds={request.unknowns[0].name: seed})


def validate_field_seed_product(program: Any, seed: Any, source: Any, point: Any) -> None:
    from pops.model.spaces import FieldSpace
    from pops.codegen.program_emit_expressions import pointwise_output_template

    metadata = seed.attrs.get("field_product_seed")
    if metadata is None:
        raise SolveRequestError("unknown_type_mismatch", "typed field seed lost its original product authority")
    if program._issued_values.get(id(seed)) is not seed or program._canonical_value(seed) is not seed:
        raise SolveRequestError("unknown_type_mismatch", "typed field seed is not the current Program-issued product")
    if type(seed.space) is not FieldSpace or seed.vtype != "scalar_field" or seed.state_ref is not None or seed.point != point:
        raise SolveRequestError("unknown_type_mismatch", "typed field seed changed its physical space or exact point")
    expected = {"contract": SEED_PRODUCT, "problem_identity": source["field_problem_identity"],
                "unknown_components": source["unknown_components"], "space": seed.space.to_data(),
                "point": point.to_data(), "program_owner": program.owner_path.canonical().to_data(), "layout_scope_only": True}
    if canonical_bytes(_json_ready(metadata)) != canonical_bytes(_json_ready(expected)) or \
            seed.space.components != tuple(row["local_id"] for row in source["unknown_components"]) or \
            any(unit is None for unit in seed.space.units) or seed.space.sampling != "cell_value":
        raise SolveRequestError("unknown_type_mismatch", "typed field seed changed its original problem, components, units or issuer")
    template = pointwise_output_template(seed)
    for attribute in ("frame", "clock", "support", "layout", "centering"):
        if getattr(seed.space, attribute) != getattr(template.space, attribute):
            raise SolveRequestError("unknown_type_mismatch", "typed field seed changed its physical co-location: " + attribute)
    if any(value.point != point for value in seed.inputs if value.vtype != "scalar"):
        raise SolveRequestError("unknown_type_mismatch", "typed field seed reads a stale physical point")


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


def compile_equations(problem: Any, states: tuple, *, per_candidate: bool = False) -> tuple[tuple, tuple]:
    from pops.math import elliptic_terms
    from pops._ir.expr import Laplacian
    from fractions import Fraction
    from pops._ir.elliptic import DivCoeffGrad, Reaction, SpatialInteraction, constant_reaction_scalar
    from pops._ir.handle_expr import ValueExpr

    width = len(problem.unknowns)
    diffusion = [Fraction(0) for _ in range(width * width)]
    captured_diffusion = {}
    local = []
    for row, equation in enumerate(problem.equations):
        expression = -equation.rhs
        for term in elliptic_terms(equation.lhs):
            column = _unknown_index(problem, term)
            if type(term) in (Laplacian, DivCoeffGrad):
                coefficient = 1 if type(term) is Laplacian else constant_reaction_scalar(term.coeff)
                slot = row * width + column
                if coefficient is NotImplemented:
                    # Unknown reads are authorized only by the explicit @3 realization.
                    value = -term.scale * term.coeff
                    captured_diffusion[slot] = value if slot not in captured_diffusion else captured_diffusion[slot] + value
                else:
                    diffusion[slot] -= Fraction(term.scale) * Fraction(coefficient)
            elif type(term) is SpatialInteraction:
                # Produced collectively from the simultaneous candidate inside F(q).
                continue
            elif type(term) is Reaction:
                expression = expression + term.scale * term.coeff * ValueExpr(problem.unknowns[column])
            else:
                raise FieldProblemError("field.nonlinear.operation", "original field equation has an unprepared operation")
        local.append(encode_field_expression(expression, states, unknowns=problem.unknowns))
    # Encoding literals authenticates finite/exact numbers rather than C++ strings.
    encoded_diffusion = []
    for slot, value in enumerate(diffusion):
        if slot not in captured_diffusion:
            encoded_diffusion.append(("literal", scalar_literal(value).to_data()))
        else:
            try:
                encoded_diffusion.append(encode_field_expression(captured_diffusion[slot] + value, states,
                    unknowns=problem.unknowns if per_candidate else ()))
            except (ValueError, TypeError, NotImplementedError) as error:
                raise FieldProblemError("field.nonlinear.diffusion",
                    "diffusion requires exact State captures; unknown-dependent D has no full spatial nonlinear JVP realization") from error
    encoded_diffusion = tuple(encoded_diffusion)
    field_expression_dependencies(encoded_diffusion, states, unknowns=problem.unknowns if per_candidate else ())
    for expression in local:
        field_expression_cpp(expression, states, views=tuple("capture%d" % i for i in range(len(states))),
                             unknowns=problem.unknowns, duration_name="issued_frame_duration")
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
    method_data = numerical.method.options()
    candidate_policy = method_data.get("coefficient_evaluation")
    from ._evolved_stage_contract import stage_projection, validate_encoded_tau
    projection = stage_projection(problem, program, at)
    temporal_tau = None if projection is None else projection.tau.to_data()
    if projection is not None and projection.to_data()["schema_version"] in (2, 3):
        from ._evolved_stage_contract import validate_additive_capture_reads

        validate_additive_capture_reads(captures)
    if projection is not None:
        for previous in projection.previous:
            witnesses = [value for value in captures if _identity(value.state_ref) == _identity(previous.handle)]
            if len(witnesses) != 1:
                raise FieldProblemError("field.evolution.previous", "original evolution requires the exact accepted previous frame endpoint")
            from ._evolved_stage_contract import validate_previous_capture, issued_previous
            validate_previous_capture(program, witnesses[0], point=at, issued=issued_previous(projection.to_data()))
    from ._original_field_interaction import compile_interactions
    interactions = compile_interactions(problem, method_data.get("interaction_realization"))
    diffusion, local = compile_equations(problem, captures, per_candidate=candidate_policy is not None)
    validate_encoded_tau((*diffusion, *local), temporal_tau)
    width = len(problem.unknowns)
    coefficient_dependencies = field_expression_dependencies(diffusion, captures, unknowns=problem.unknowns if candidate_policy else ())
    contract = method_data["contract"]
    face_policy = method_data.get("coefficient_face_policy")
    if coefficient_dependencies and face_policy is None:
        raise FieldProblemError("field.nonlinear.face_policy",
            "captured diffusion requires explicit CellCenteredNonlinearCoupled(face_policy=Arithmetic@1)")
    boundary = _physical_boundary(problem)
    prototype = program._new("scalar_field", "scalar_field", (), {"ncomp": width},
                             problem.name + "_unknown_product", None, point=at,
                             inherit_state_ref=False)
    coefficient_attrs = {"ncomp": width * width, "unknown_ncomp": width, "field_problem_identity": problem.identity.token,
         "field_dependencies": coefficient_dependencies, "field_handle": field.canonical_identity(), "physical_boundary": boundary,
         "scope": "level", "coefficient_admissibility": "finite_general", "expressions": diffusion,
         "stencil_access": StencilAccess.pointwise()}
    if face_policy is not None:
        coefficient_attrs["coefficient_face_policy"] = face_policy
    if candidate_policy is not None:
        # Deferred descriptor: physical D is evaluated only inside the complete F(q).
        coefficient_attrs.update(coefficient_evaluation=candidate_policy,
            storage_role="deferred_candidate_coefficient",
            unknown_components=tuple(row.canonical_identity() for row in problem.unknowns))
    if temporal_tau is not None:
        coefficient_attrs["temporal_tau"] = temporal_tau
    coefficients = program._new("scalar_field", "field_problem_coefficients",
        captures if coefficient_dependencies else (), coefficient_attrs, problem.name + "_diffusion",
        None, point=at, inherit_state_ref=False)
    from pops._frozen_data import freeze_containers
    from pops.time._program.equation_identity import _equation_value
    source_data = {"field_problem": problem.to_data(), "field_handle": field.canonical_identity(),
        "unknown_components": tuple(row.canonical_identity() for row in problem.unknowns),
        "field_problem_identity": problem.identity.token, "diffusion": diffusion, "local_expressions": local,
        "physical_boundary": boundary, "finite_difference_step": scalar_data(numerical.method.finite_difference_step),
        "captures": tuple(_equation_value(program, value) for value in captures)}
    if interactions is not None:
        source_data["interactions"] = interactions
    if face_policy is not None:
        source_data["coefficient_face_policy"] = face_policy
    if candidate_policy is not None:
        source_data["coefficient_evaluation"] = candidate_policy
        source_data["linear_residual_verification"] = method_data["linear_residual_verification"]
    if projection is not None:
        source_data["evolved_stage"] = projection.to_data()
        source_data["temporal_tau"] = temporal_tau
        from ._evolved_stage_contract import compile_accumulation
        source_data["accumulation"] = compile_accumulation(projection, captures, problem.unknowns)
    source = freeze_containers(source_data)
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
    equation = {"contract": token.attrs["contract"], "local_expressions": _json_ready(token.attrs["local_expressions"]),
                "coefficients": _equation_value(program, token.inputs[1]),
                "captures": [_equation_value(program, value) for value in captures],
                "physical_problem": physical, "boundary": token.attrs["physical_boundary"]}
    if "coefficient_evaluation" in token.attrs:
        equation["diffusion_body"] = _json_ready(token.attrs["source_contract"]["diffusion"])
        equation["coefficient_evaluation"] = token.attrs["coefficient_evaluation"]
    if "interactions" in token.attrs["source_contract"]:
        from ._original_field_interaction import interaction_identity_data
        equation["interactions"] = interaction_identity_data(token.attrs["source_contract"]["interactions"])
    equation_id = make_identity("solve-equation", equation).token
    problem = {"equation": equation_id, "physical_problem": physical, "unknowns": [unknown.to_data()],
               "residual_interpretation": "original_field_equations", "error_interpretation": "spatial_residual_l2"}
    result = {"schema_version": 1, **problem, "equation_identity": equation_id, "equation_inputs": equation,
            "problem_identity": make_identity("solve-problem", problem).token, "seed": seed,
            "initialization_identity": make_identity("solve-initialization", seed).token,
            "outputs": [unknown.name], "solver_identity": token.attrs["solver_identity"],
            "derivative": {"route": "finite_difference", "scheme": "central_full_residual",
                           "step": token.attrs["finite_difference_step"]},
            "lowering": {"disposition": "native", "adapter": token.attrs["contract"]}}
    realization = {}
    if "right_preconditioner" in token.attrs:
        realization["right_preconditioner"] = token.attrs["right_preconditioner"]
    if "right_preconditioner_resources" in token.attrs:
        realization["right_preconditioner_resources"] = _json_ready(token.attrs["right_preconditioner_resources"])
    if token.attrs.get("coefficient_face_policy") is not None:
        realization["coefficient_face_policy"] = token.attrs["coefficient_face_policy"]
    if "coefficient_evaluation" in token.attrs:
        realization["coefficient_evaluation"] = token.attrs["coefficient_evaluation"]
        realization["linear_residual_verification"] = token.attrs["linear_residual_verification"]
    if "seed_product_contract" in token.attrs:
        realization["seed_product_contract"] = token.attrs["seed_product_contract"]
    if realization:
        result["schema_version"] = 2
        result["realization"] = realization
    return result


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
    if set(request.equation_inputs) != set(expected) or \
            any(request.equation_inputs[key] is not value for key, value in expected.items()) or \
            unknown.template is not residual.prototype or request.outputs != (unknown.name,):
        raise SolveRequestError("equation_input_mismatch", "field residual bindings changed")
    if request.derivative.route != "finite_difference" or request.residual_interpretation != "original_field_equations" or request.error_interpretation != "spatial_residual_l2":
        raise SolveRequestError("unsupported_derivative", "select central full-residual finite differences and original spatial residual explicitly")
    seed = request.seeds[unknown.name]
    inputs = (residual.prototype, residual.coefficients, *residual.captures)
    for value in (*inputs, *(() if seed is None else (seed,))):
        require_top_level(program, value, "field residual binding")
    if seed is not None and _product_width(seed) != residual.prototype.attrs["ncomp"]:
        raise SolveRequestError("unknown_type_mismatch", "field product seed width/type differs")
    if seed is not None and seed.op == "pointwise_expression" and seed.vtype == "scalar_field" and "field_product_seed" not in seed.attrs:
        raise SolveRequestError("unknown_type_mismatch", "typed field seed lost its original product authority")
    if seed is not None and "field_product_seed" in seed.attrs:
        validate_field_seed_product(program, seed, residual.source_contract, residual.prototype.point)
    face_policy = residual.source_contract.get("coefficient_face_policy")
    candidate_policy = residual.source_contract.get("coefficient_evaluation")
    contract = "pops.spatial-field-residual@4" if "interactions" in residual.source_contract else CANDIDATE_DIFFUSION_CONTRACT if candidate_policy else CAPTURED_DIFFUSION_CONTRACT if face_policy is not None else CONTRACT
    attrs = {"contract": contract, "problem_kind": "original_field_equations", "ncomp": residual.prototype.attrs["ncomp"],
             "field": residual.field, "field_problem_identity": residual.source_contract["field_problem_identity"],
             "source_contract": residual.source_contract,
             "capture_count": len(residual.captures), "seed_index": None if seed is None else len(inputs),
             "local_expressions": residual.local_expressions, "physical_boundary": residual.physical_boundary,
             "finite_difference_step": scalar_data(residual.finite_difference_step),
             "newton_controls": prepared.controls.to_data(), "solver_identity": prepared.identity.token}
    if face_policy is not None:
        attrs["coefficient_face_policy"] = face_policy
    if seed is not None and "field_product_seed" in seed.attrs:
        attrs["seed_product_contract"] = SEED_PRODUCT
    if candidate_policy is not None:
        attrs["coefficient_evaluation"] = candidate_policy
        attrs["linear_residual_verification"] = residual.source_contract["linear_residual_verification"]
        if prepared.right_preconditioner == "pops.amr.original-spatial-jacobi.basis-response@1":
            raise SolveRequestError("unsupported_realization", "SpatialBasisJacobi@1 requires a frozen linear spatial operator; PerCandidate@1 needs a separately declared Jacobian preconditioner")
    prepared.__post_init__()
    if prepared.convergence is not None:
        attrs["convergence"] = prepared.convergence.to_data()
    if prepared.right_preconditioner is not None:
        attrs["right_preconditioner"] = prepared.right_preconditioner
    if prepared.max_dense_bytes is not None:
        from pops.time._program.spatial_solve import dense_resource_contract
        attrs["right_preconditioner_resources"] = dense_resource_contract(prepared.max_dense_bytes)
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
    from pops.time._program.spatial_solve import spatial_newton_options, spatial_solver_identity
    from pops.time.canonical_data import CanonicalData
    from pops.time._program.equation_identity import _equation_value

    actual = _json_ready(token.attrs["solve_request"])
    source = token.attrs["source_contract"]
    from ._evolved_stage_contract import validate_tau_data, validate_encoded_tau
    temporal_tau = source.get("temporal_tau")
    if temporal_tau is not None:
        validate_tau_data(temporal_tau, program=program, point=token.point)
        stage = source.get("evolved_stage")
        if stage is None or not same_stage_tau(stage, temporal_tau, source):
            raise SolveRequestError("equation_identity_drift", "original stage accumulation/duration authority changed")
    if token.inputs[1].attrs.get("temporal_tau") != temporal_tau:
        raise SolveRequestError("equation_identity_drift", "original coefficient duration changed")
    validate_encoded_tau((*source["diffusion"], *source["local_expressions"]), temporal_tau)
    face_policy = source.get("coefficient_face_policy")
    candidate_policy = source.get("coefficient_evaluation")
    if candidate_policy is not None and (type(candidate_policy) is not str or candidate_policy != PER_CANDIDATE or face_policy != ARITHMETIC_FACES):
        raise SolveRequestError("equation_identity_drift", "candidate coefficient realization changed")
    if token.attrs.get("coefficient_evaluation") != candidate_policy or (candidate_policy and token.attrs.get("right_preconditioner") == "pops.amr.original-spatial-jacobi.basis-response@1"):
        raise SolveRequestError("unsupported_realization", "candidate coefficient policy/preconditioner mismatch")
    criterion = "pops.field.linear.true-correction-residual@1" if candidate_policy else None
    if source.get("linear_residual_verification") != criterion or token.attrs.get("linear_residual_verification") != criterion:
        raise SolveRequestError("unsupported_realization", "true correction residual realization changed")
    from ._original_field_interaction import validate_interactions
    if "interactions" in source:
        validate_interactions(source["interactions"], source["field_problem"], source["unknown_components"])
    expected_contract = "pops.spatial-field-residual@4" if "interactions" in source else CANDIDATE_DIFFUSION_CONTRACT if candidate_policy else CAPTURED_DIFFUSION_CONTRACT if face_policy is not None else CONTRACT
    if (face_policy is not None and (type(face_policy) is not str or face_policy != ARITHMETIC_FACES)) or \
            token.attrs["contract"] != expected_contract or token.attrs["problem_kind"] != "original_field_equations":
        raise SolveRequestError("equation_identity_drift", "field residual realization changed")
    count, seed_index = token.attrs["capture_count"], token.attrs["seed_index"]
    if type(count) is not int or count <= 0 or len(token.inputs) != 2 + count + (seed_index is not None) \
            or (seed_index is not None and (type(seed_index) is not int or seed_index != 2 + count)):
        raise SolveRequestError("equation_identity_drift", "field capture/seed input slots changed")
    captures = token.inputs[2:2 + count]
    seed = None if seed_index is None else token.inputs[-1]
    product_seed = seed is not None and "field_product_seed" in seed.attrs
    if seed is not None and seed.op == "pointwise_expression" and seed.vtype == "scalar_field" and not product_seed:
        raise SolveRequestError("equation_identity_drift", "typed field seed lost its original product authority")
    if token.attrs.get("seed_product_contract") != (SEED_PRODUCT if product_seed else None) or \
            (not product_seed and "seed_product_contract" in token.attrs):
        raise SolveRequestError("equation_identity_drift", "typed field product seed changed its exact contract")
    if product_seed:
        validate_field_seed_product(program, seed, source, token.point)
    if source.get("evolved_stage", {}).get("schema_version") in (2, 3):
        from ._evolved_stage_contract import validate_additive_capture_reads

        validate_additive_capture_reads(captures)
    if source.get("evolved_stage") is not None:
        from ._evolved_stage_contract import issued_previous, validate_previous_capture
        stage = _json_ready(source["evolved_stage"])
        for previous in stage["previous"]:
            witnesses = [value for value in captures if _identity(value.state_ref) == _identity(Handle.from_canonical_identity(previous["handle"]))]
            if len(witnesses) != 1:
                raise SolveRequestError("equation_identity_drift", "original previous capture owner changed")
            validate_previous_capture(program, witnesses[0], point=token.point, issued=issued_previous(stage))
    unknown_handles = tuple(Handle.from_canonical_identity(_json_ready(item)) for item in source["unknown_components"])
    diffusion_dependencies = field_expression_dependencies(source["diffusion"], captures, unknowns=unknown_handles if candidate_policy else ())
    expected_coefficient_inputs = captures if diffusion_dependencies else ()
    expected_diffusion = source["diffusion"]
    if diffusion_dependencies and face_policy is None:
        raise SolveRequestError("equation_identity_drift", "captured diffusion lost its explicit arithmetic realization")
    if (token.attrs.get("coefficient_face_policy") != face_policy or
            token.inputs[1].attrs.get("coefficient_face_policy") != face_policy):
        raise SolveRequestError("equation_identity_drift", "original coefficient face policy changed")
    expected_physical = CanonicalData({key: source[key] for key in (
        "field_problem", "field_handle", "unknown_components")}, where="original field authority").to_data()
    def same(lhs: Any, rhs: Any) -> bool:
        return canonical_bytes(_json_ready(lhs)) == canonical_bytes(_json_ready(rhs))
    checks = {"physical_problem": same(actual["physical_problem"], expected_physical),
              "local_expressions": same(source["local_expressions"], token.attrs["local_expressions"]),
              "diffusion": same(expected_diffusion, token.inputs[1].attrs["expressions"]),
              "width": type(token.attrs["ncomp"]) is int and token.attrs["ncomp"] == len(source["unknown_components"])
                       and _product_width(token.inputs[0]) == token.attrs["ncomp"]
                       and (seed_index is None or _product_width(token.inputs[-1]) == token.attrs["ncomp"]),
              "coefficient_contract": token.inputs[1].op == "field_problem_coefficients"
                       and token.inputs[1].attrs["ncomp"] == token.attrs["ncomp"]**2
                       and token.inputs[1].attrs["field_problem_identity"] == source["field_problem_identity"]
                       and token.inputs[1].attrs["coefficient_admissibility"] == "finite_general"
                       and len(token.inputs[1].inputs) == len(expected_coefficient_inputs)
                       and all(a is b for a, b in zip(token.inputs[1].inputs, expected_coefficient_inputs, strict=True))
                       and same(token.inputs[1].attrs["field_dependencies"], diffusion_dependencies)
                       and token.inputs[1].attrs.get("coefficient_evaluation") == candidate_policy
                       and (not candidate_policy or token.inputs[1].attrs.get("storage_role") == "deferred_candidate_coefficient"
                            and same(token.inputs[1].attrs.get("unknown_components"), source["unknown_components"])),
              "point": token.point == token.inputs[0].point == token.inputs[1].point,
              "boundary": token.attrs["physical_boundary"] == source["physical_boundary"],
              "field_identity": token.attrs["field_problem_identity"] == source["field_problem_identity"],
              "FD_step": same(token.attrs["finite_difference_step"], source["finite_difference_step"]),
              "captures": same(source["captures"], tuple(_equation_value(program, value) for value in captures))}
    if not all(checks.values()):
        raise SolveRequestError("equation_identity_drift", "original field contract changed: " +
                                ",".join(key for key, valid in checks.items() if not valid))
    if candidate_policy:
        from pops.codegen.program_emit_field_routes import _walk_program_nodes
        descriptor = token.inputs[1]
        if any(node.id != token.id and any(part.id == descriptor.id for part in node.inputs)
               for node in _walk_program_nodes(tuple(program._values))):
            raise SolveRequestError("unsupported_lowering", "deferred candidate coefficient is a body descriptor, not a material field value")
    unknown_handles = tuple(Handle.from_canonical_identity(_json_ready(item)) for item in source["unknown_components"])
    for expression in token.attrs["local_expressions"]:
        field_expression_cpp(expression, captures, views=tuple("capture%d" % i for i in range(len(captures))), unknowns=unknown_handles, duration_name="issued_frame_duration" if temporal_tau is not None else None)
    unknowns = actual["unknowns"]
    if len(unknowns) != 1:
        raise SolveRequestError("invalid_unknown", "field solve must retain its full ordered unknown product")
    unknown = SolveUnknown(unknowns[0]["name"], token.inputs[0])
    expected = _request_data(program, token, unknown, actual["physical_problem"])
    if not same(actual, expected):
        raise SolveRequestError("equation_identity_drift", "field residual/seed/solver contract changed")
    controls = _json_ready(token.attrs["newton_controls"])
    spatial_newton_options(controls)
    if "right_preconditioner" in token.attrs and token.attrs["right_preconditioner"] is None:
        raise SolveRequestError("unsupported_realization", "identity realization must retain legacy omission")
    from pops.time._program.spatial_solve import FULL_RESIDUAL_BASIS_LU, validate_dense_resource_contract
    budget = None
    if token.attrs.get("right_preconditioner") == FULL_RESIDUAL_BASIS_LU:
        budget = validate_dense_resource_contract(token.attrs.get("right_preconditioner_resources"))
    elif "right_preconditioner_resources" in token.attrs:
        raise SolveRequestError("invalid_resource_budget", "dense budget belongs only to FullResidualBasisLU@1")
    if token.attrs["solver_identity"] != spatial_solver_identity(controls, token.attrs.get("right_preconditioner"), budget, token.attrs.get("convergence")).token:
        raise SolveRequestError("solver_identity_drift", "field Newton controls changed")


def same_stage_tau(stage: Any, tau: Any, source: Any) -> bool:
    data = _json_ready(stage)
    additive = any(isinstance(row, dict) and row.get("contract") ==
                   "pops.evolved-field-rate.spatial-additive@1" for row in data.get("spatial_rhs", ()))
    from ._evolved_stage_contract import issued_previous
    return (type(data.get("schema_version")) is int and data.get("schema_version") == (3 if issued_previous(stage) else 2 if additive else 1)
            and data.get("tau") == _json_ready(tau)
            and data.get("unknowns") == _json_ready(source["unknown_components"])
            and data in _json_ready(source["field_problem"])["outputs"])
