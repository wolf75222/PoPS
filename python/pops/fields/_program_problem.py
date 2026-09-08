"""Lower one physical field tuple to the existing authenticated linear solve."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops._ir.elliptic import DivCoeffGrad, Reaction, constant_reaction_scalar
from pops._ir.expr import Const, Laplacian
from pops.math import elliptic_terms
from pops.model import Handle
from pops.time.points import StagePoint, TimePoint
from pops.time.references import canonical_handle
from pops.time.stencil import StencilAccess
from pops.time.values import ProgramValue

from . import bcs
from .problem import FieldProblem, FieldProblemError


def _identity(handle: Handle) -> Any:
    return canonical_handle(handle).canonical_identity()


def _unknown_index(problem: FieldProblem, term: Any) -> int:
    from .operator import _field_targets_unknown
    matches = [i for i, unknown in enumerate(problem.unknowns)
               if _field_targets_unknown(term.field, unknown)]
    if len(matches) != 1:
        raise FieldProblemError("field.native.unknown", "native field term has no exact unknown")
    return matches[0]


def _physical_boundary(problem: FieldProblem) -> str:
    laws = []
    for unknown in problem.unknowns:
        rows = [row.relation for row in problem.boundaries if row.unknown == unknown]
        if len(rows) != 1 or type(rows[0].selector) is not bcs.AllPhysicalBoundaries:
            raise FieldProblemError("field.native.boundary", "this native field route requires one explicit all-boundaries relation per unknown")
        condition = rows[0].condition
        if type(condition) is bcs.Periodic:
            laws.append("periodic")
        elif type(condition) is bcs.Neumann and constant_reaction_scalar(condition.flux) == 0:
            laws.append("homogeneous_neumann")
        else:
            raise FieldProblemError("field.native.boundary", "native general fields support periodic or homogeneous Neumann relations")
    if len(set(laws)) != 1:
        raise FieldProblemError("field.native.boundary", "joint unknowns must share one supported boundary topology")
    return laws[0]


def _physical_coefficients(problem: FieldProblem) -> tuple[Any, tuple[Any, ...]]:
    """Authenticate the scalar principal part and complete constant reaction matrix."""
    from fractions import Fraction
    size = len(problem.unknowns)
    if size not in (1, 2):
        raise FieldProblemError("field.native.arity", "native general fields support one or two scalar unknowns")
    diffusion, reaction = [], [Fraction(0)] * (size * size)
    for row, equation in enumerate(problem.equations):
        principals = []
        for term in elliptic_terms(equation.lhs):
            column = _unknown_index(problem, term)
            if type(term) in (Laplacian, DivCoeffGrad):
                if column != row:
                    raise FieldProblemError("field.native.cross_diffusion", "off-diagonal diffusion requires a distinct native realization")
                coefficient = Const(1) if type(term) is Laplacian else term.coeff
                principals.append(-term.scale * coefficient)
            elif type(term) is Reaction:
                coefficient = constant_reaction_scalar(term.coeff)
                if coefficient is NotImplemented:
                    raise FieldProblemError("field.native.reaction", "joint native reaction coefficients must be exact compile-time scalars")
                reaction[row * size + column] += Fraction(coefficient) * Fraction(term.scale)
            else:
                raise FieldProblemError("field.native.principal", "field principal term has no native realization")
        if len(principals) != 1:
            raise FieldProblemError("field.native.principal", "one diagonal scalar diffusion term per equation is required")
        diffusion.append(principals[0])
    if size == 2 and reaction[1] != reaction[2]:
        raise FieldProblemError("field.native.symmetry", "the selected CG route requires a symmetric reaction matrix")
    if problem.gauge is not None:
        if any(sum(reaction[row * size:(row + 1) * size]) != 0 for row in range(size)):
            raise FieldProblemError("field.native.kernel", "the declared shared constant mode is not a kernel of the physical reaction matrix")
        if size == 2 and not (reaction[0] > 0 and reaction[3] > 0 and reaction[1] < 0):
            raise FieldProblemError("field.native.kernel", "two-field shared gauge requires positive lambda coupling; independent kernels need separate explicit constraints")
    elif (size == 1 and reaction[0] <= 0) or (size == 2 and (reaction[0] <= 0 or reaction[3] <= 0 or reaction[0] * reaction[3] <= reaction[1] * reaction[2])):
        raise FieldProblemError("field.native.kernel", "a singular field requires its explicit physical shared gauge")
    return tuple(diffusion), tuple(reaction)


def bind_field_problem(program: Any, field: Handle, registration: Any, *, values: Any, at: Any) -> Any:
    """Freeze explicit equation inputs; field storage never borrows a species owner."""
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.time import DerivativeStrategy, SolveRequest, SolveUnknown
    from pops.time._program.value_validation import require_top_level
    from pops.identity.scalar import scalar_literal
    from ._program_expression import encode_field_expression, field_expression_dependencies
    from .gauges import MeanValueGauge
    from .methods import CellCenteredSecondOrder
    from .nullspace import ConstantNullspace
    from .operator import FieldOperator

    problem = registration.operator
    if type(problem) is not FieldProblem or isinstance(problem, FieldOperator):
        raise TypeError("general field binding requires FieldProblem; scalar provider adapters retain their existing field solve route")
    if type(registration.discretization.method) is not CellCenteredSecondOrder:
        raise FieldProblemError("field.native.method", "general field native realization requires CellCenteredSecondOrder")
    if registration.discretization.preconditioner is not None or registration.discretization.nonlinear is not None:
        raise FieldProblemError("field.native.numerics", "general field numerical options must be realized explicitly by the selected linear solver")
    if registration.discretization.boundaries or registration.discretization.gauge is not None or registration.discretization.nullspace is not None:
        raise FieldProblemError("field.native.authority", "general field physical boundaries and gauge belong to FieldProblem")
    if not isinstance(values, Mapping):
        raise TypeError("field solve values must map exact qualified state Handles to Program state values")
    if type(at) not in (StagePoint, TimePoint):
        raise TypeError("field solve at must be an explicit StagePoint or TimePoint")
    dependencies = problem.dependencies()
    if any(handle.kind != "state" for handle in dependencies):
        raise FieldProblemError("field.native.input", "native general field coefficients and loads currently consume exact state quantities")
    states = []
    for dependency in dependencies:
        candidates = [(key, value) for key, value in values.items()
                      if isinstance(key, Handle) and _identity(key) == _identity(dependency)]
        if len(candidates) != 1:
            raise FieldProblemError("field.native.input", "field solve is missing an exact qualified state input")
        key, value = candidates[0]
        value = getattr(value, "_value", value)
        if not isinstance(value, ProgramValue) or value.vtype != "state":
            raise TypeError("field equation inputs must be Program state values")
        require_top_level(program, value, "field solve input")
        if value.state_ref is None or _identity(value.state_ref) != _identity(key):
            raise FieldProblemError("field.native.input", "field input binding changes its exact state owner")
        states.append(value)
    if len(values) != len(dependencies):
        raise FieldProblemError("field.native.input", "field solve has undeclared or duplicate equation inputs")
    states = tuple(states)
    diffusion, reaction = _physical_coefficients(problem)
    physical_boundary = _physical_boundary(problem)
    size = len(problem.unknowns)
    common = {"ncomp": size, "field_problem_identity": problem.identity.token,
              "field_dependencies": tuple(_identity(row) for row in dependencies),
              "field_handle": field.canonical_identity(), "physical_boundary": physical_boundary}
    rhs_expressions = tuple(encode_field_expression(eq.rhs, states) for eq in problem.equations)
    coefficient_expressions = tuple(encode_field_expression(eq, states) for eq in diffusion)
    rhs_dependencies = field_expression_dependencies(rhs_expressions, states)
    coefficient_dependencies = field_expression_dependencies(coefficient_expressions, states)
    rhs = program._new("scalar_field", "field_problem_load", states,
        {**common, "expressions": rhs_expressions, "field_dependencies": rhs_dependencies,
         "stencil_access": StencilAccess.pointwise()}, problem.name + "_load", None, point=at, inherit_state_ref=False)
    coefficients = program._new("scalar_field", "field_problem_coefficients", states,
        {**common, "expressions": coefficient_expressions, "field_dependencies": coefficient_dependencies,
         "stencil_access": StencilAccess.pointwise()}, problem.name + "_coefficients", None, point=at, inherit_state_ref=False)
    operator = program.matrix_free_operator(problem.name + "_operator",
        domain="scalar" if size == 1 else "vector", range_="scalar" if size == 1 else "vector", ncomp=size)
    def apply(p: Any, out: Any, value: Any) -> Any:
        return p._new("scalar_field", "field_problem_apply", (out, value, coefficients),
            {**common, "field_dependencies": coefficient_dependencies,
             "reaction": tuple(scalar_literal(row).to_data() for row in reaction),
             "stencil_access": StencilAccess.nearest_neighbour()}, problem.name + "_apply", None, inherit_state_ref=False)
    program.set_apply(operator, apply)
    if problem.gauge is None:
        nullspace, gauge = None, None
        properties = LinearOperatorProperties.symmetric_positive_definite()
    else:
        if size == 1:
            nullspace, gauge = ConstantNullspace(), MeanValueGauge(problem.gauge.value)
        else:
            from ._joint_nullspace import shared_constant_nullspace
            nullspace, gauge = shared_constant_nullspace(), problem.gauge
        properties = LinearOperatorProperties.symmetric_positive_definite_on_nullspace_complement()
    linear = LinearProblem(operator, rhs, nullspace=nullspace, gauge=gauge, properties=properties)
    return SolveRequest(problem=linear, unknowns=(SolveUnknown("field_tuple", template=rhs),),
        equation_inputs={"operator": operator, "rhs": rhs}, seeds={"field_tuple": None},
        outputs=("field_tuple",), derivative=DerivativeStrategy("exact"),
        problem_metadata={"field_problem": problem.to_data(), "field_handle": field.canonical_identity(),
                          "unknown_components": tuple(row.canonical_identity() for row in problem.unknowns)})


@dataclass(frozen=True, slots=True)
class FieldSolution:
    """Physical per-unknown observations of one consumed packed native solve."""
    field: Handle
    packed: ProgramValue
    unknowns: tuple[Any, ...]
    problem_identity: str

    def publish(self, bindings: Any, *, states: Any = None) -> ProgramValue:
        """Publish consumed scalar/gradient components to exact physics field inputs."""
        from ._program_publication import publish_field_solution

        return publish_field_solution(self, bindings, states=states)

    def __getitem__(self, unknown: Handle) -> ProgramValue:
        if not isinstance(unknown, Handle):
            raise TypeError("field observations require an exact field unknown Handle")
        matches = [index for index, row in enumerate(self.unknowns) if row == _identity(unknown)]
        if len(matches) != 1:
            raise FieldProblemError("field.observation.unknown", "observation selects a foreign field unknown")
        return self.packed.prog._new("scalar_field", "field_component", (self.packed,),
            {"ncomp": 1, "component": matches[0], "field_problem_identity": self.problem_identity,
             "field_unknown": self.unknowns[matches[0]], "stencil_access": StencilAccess.pointwise()},
            unknown.local_id + "_value", None, point=self.packed.point, inherit_state_ref=False)

    def gradient(self, unknown: Handle, *, dimension: int) -> ProgramValue:
        if type(dimension) is not int or dimension not in (1, 2, 3):
            raise TypeError("field gradient requires an explicit physical dimension")
        value = self[unknown]
        return value.prog._new("scalar_field", "field_gradient", (value,),
            {"ncomp": dimension, "spatial_dimension": dimension,
             "differentiation": "cell_centered_second_order",
             "sampling": "cell", "field_problem_identity": self.problem_identity,
             "stencil_access": StencilAccess.nearest_neighbour()},
            unknown.local_id + "_gradient", None, point=value.point, inherit_state_ref=False)


def observe_field_solution(field: Handle, solution: Any, *, unknown: Handle | None = None) -> Any:
    from pops.time.solve_outcome import ResidualSolution
    if type(solution) is not ResidualSolution or len(solution.values) != 1:
        raise TypeError("field observation requires the consumed ResidualSolution of one field solve")
    packed = solution.values[0]
    if packed.op != "solve_outcome_component":
        raise FieldProblemError("field.observation.unconsumed", "field observation requires the explicit solve outcome projection")
    solve = packed.inputs[0].inputs[0]
    from pops.time._program.serialization import _json_ready
    request = _json_ready(solve.attrs.get("solve_request", {}))
    physical = request.get("physical_problem", {})
    from pops.time._graph.base import CanonicalData
    expected = CanonicalData(_identity(field), where="field observation").to_data()
    if physical.get("field_handle") != expected or solution.problem_identity != request.get("problem_identity"):
        raise FieldProblemError("field.observation.authority", "field observation belongs to a different physical problem")
    registry = getattr(field, "_field_registry", None)
    if registry is None:
        raise FieldProblemError("field.observation.authority", "field observation requires its registered Case field authority")
    registered = registry.resolved_registration(field).operator
    unknowns = tuple(_identity(row) for row in registered.unknowns)
    if CanonicalData(registered.to_data(), where="field observation equations").to_data() != physical["field_problem"] \
            or CanonicalData(unknowns, where="field observation unknowns").to_data() != physical["unknown_components"]:
        raise FieldProblemError("field.observation.authority", "field observation physical tuple changed after solve")
    result = FieldSolution(field, packed, unknowns, registered.identity.token)
    return result if unknown is None else result[unknown]


def validate_field_apply(node: Any) -> None:
    """Authenticate the linear field apply against its coefficient producer before emission."""
    from ._program_expression import decode_field_literal
    if node.op != "field_problem_apply" or len(node.inputs) != 3:
        raise FieldProblemError("field.native.apply", "invalid field apply input contract")
    out, value, coefficient = node.inputs
    size = node.attrs.get("ncomp")
    if type(size) is not int or size not in (1, 2) or any(
            part.vtype != "scalar_field" or part.attrs.get("ncomp") != size
            for part in (out, value, coefficient)):
        raise FieldProblemError("field.native.apply", "field apply changes its exact unknown width")
    if coefficient.op != "field_problem_coefficients" or any(
            coefficient.attrs.get(key) != node.attrs.get(key) for key in
            ("field_problem_identity", "field_dependencies", "field_handle", "physical_boundary")):
        raise FieldProblemError("field.native.apply", "field apply coefficient authority differs from its physical problem")
    if node.attrs.get("physical_boundary") not in ("periodic", "homogeneous_neumann"):
        raise FieldProblemError("field.native.boundary", "field apply has no supported physical boundary")
    reaction = node.attrs.get("reaction")
    if not isinstance(reaction, (tuple, list)) or len(reaction) != size * size:
        raise FieldProblemError("field.native.reaction", "field reaction tuple changes its physical arity")
    for item in reaction:
        decode_field_literal(item)


_NATIVE_FIELD_COMPONENT = None


def native_field_component() -> Any:
    """One packaged native authority for the executed scalar/joint field stencil."""
    global _NATIVE_FIELD_COMPONENT
    if _NATIVE_FIELD_COMPONENT is None:
        from pops.native_components import PreparedNativeComponent
        _NATIVE_FIELD_COMPONENT = PreparedNativeComponent.pops_builtin(
            "pops.general-field.scalar-joint",
            entry_headers=("pops/numerics/elliptic/nd/general_field_operator.hpp",))
    return _NATIVE_FIELD_COMPONENT
