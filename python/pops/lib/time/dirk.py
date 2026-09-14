"""Sequential implicit RK stages through the existing solve/consume protocol."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.solvers import DenseLU
from pops.time import LocalLinear, LocalResidual
from pops.time._methods.tableau import DiagonallyImplicitRungeKuttaTableau

from ._factory import call_at, instance_state, operator_handle, program_factory, resolve_solve_action
from ._helpers import _op_space_arity, _stage_point

BACKWARD_EULER_TABLEAU = DiagonallyImplicitRungeKuttaTableau([[1]], [1], name="backward_euler")
IMPLICIT_MIDPOINT_TABLEAU = DiagonallyImplicitRungeKuttaTableau(
    [[Fraction(1, 2)]], [1], name="implicit_midpoint")


def _build_dirk(program: Any, state: Any, implicit: Any, tableau: Any,
                solver: Any, action: Any) -> None:
    if type(tableau) is not DiagonallyImplicitRungeKuttaTableau:
        raise TypeError("DIRK tableau must be an exact DiagonallyImplicitRungeKuttaTableau")
    implicit = operator_handle(implicit, "DIRK implicit_operator")
    temporal = instance_state(program, state, "DIRK")
    from pops.time.operator_resolution import resolve_operator_handle
    operator = resolve_operator_handle(program, implicit, where="DIRK implicit_operator",
                                      expected_kinds=("local_linear_operator", "local_source"))
    nonlinear = operator.kind == "local_source"
    arity = _op_space_arity(program, implicit)
    if arity != int(nonlinear):
        raise ValueError("DIRK preset requires a field-independent local operator; "
                         "coupled fields and spatial stages require an authored SolveRequest")
    if nonlinear and solver is None:
        raise ValueError("DIRK nonlinear implicit_operator requires an explicit implicit_solver")
    if not nonlinear and solver is not None:
        raise ValueError("DIRK local-linear stages use DenseLU; implicit_solver is nonlinear only")
    initial = temporal.n
    rates = []
    for i in range(tableau.stages):
        point = _stage_point(program, "dirk_stage_%d" % i, tableau.c[i])
        predictor = 1 * initial
        for j in range(i):
            if tableau.A[i][j] != 0:
                predictor = predictor + program.dt * tableau.A[i][j] * rates[j]
        predictor = program.value("dirk_predictor_%d" % i, predictor, at=point)
        diagonal = tableau.A[i][i]
        stage = predictor
        if nonlinear:
            if diagonal != 0:
                def residual(owner: Any, iterate: Any, frozen: Any,
                             coefficient: Any = diagonal) -> Any:
                    return owner.value("dirk_residual", iterate - frozen
                                       - owner.dt * coefficient * owner.source(implicit, state=iterate),
                                       at=iterate.point)
                stage = program.solve(LocalResidual(residual, predictor), solver=solver,
                                      name="dirk_solve_%d" % i).consume(action=action)
            rate = program.source(implicit, state=stage)
        else:
            linear = call_at(program, implicit, name="dirk_linear_%d" % i, point=point)
            if diagonal != 0:
                stage = program.solve(LocalLinear(program.I - program.dt * diagonal * linear,
                                                  predictor), solver=DenseLU(),
                                      name="dirk_solve_%d" % i).consume(action=action)
            rate = program.apply(linear, stage)
        rates.append(program.value("dirk_rate_%d" % i, rate, at=point))
    result = initial
    for coefficient, rate in zip(tableau.b, rates, strict=True):
        if coefficient != 0:
            result = result + program.dt * coefficient * rate
    endpoint = program.value("dirk_step", result, at=temporal.next.point)
    program.commit(temporal.next, endpoint)


def DIRK(state: Any, *, implicit_operator: Any,
         tableau: Any = BACKWARD_EULER_TABLEAU, implicit_solver: Any = None,
         solve_action: Any = None) -> Any:
    """Expand a lower-triangular RK tableau into ordinary sequential solve regions.

    Local linear stages use DenseLU and nonlinear local sources require an explicit
    solver. Every stage result is consumed with the requested failure disposition;
    the endpoint is committed once after all stages. Abscissae sample the implicit
    operator at the current stage. Spatial/coupled field equations use SolveRequest.
    """
    return program_factory("DIRK", _build_dirk, state, implicit_operator, tableau,
                           implicit_solver, resolve_solve_action(solve_action, "DIRK"))


__all__ = ["BACKWARD_EULER_TABLEAU", "DIRK", "DiagonallyImplicitRungeKuttaTableau",
           "IMPLICIT_MIDPOINT_TABLEAU"]
