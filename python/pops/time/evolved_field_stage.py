"""Explicit temporal authorities for original spatial field evolution stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops._ir.expr import Const, Expr
from pops._frozen_data import thaw_data
from pops.identity import canonical_sha256
from pops.identity.scalar import scalar_literal
from pops.time.points import StagePoint, TimePoint, point_clock
from pops.time.values import _Coeff


_TAU_PROGRAMS: dict = {}


class TemporalTau(Expr):
    """An exact multiple of the issued frame duration, bound to one Program/point.

    Legacy ``Program.dt`` coefficients are context-free polynomials. This explicit
    leaf supplies their Program, clock and native window authority when they occur
    in an original FieldProblem. It never snapshots the requested controller dt.
    """

    def __init__(self, program: Any, coefficient: Any, *, at: Any) -> None:
        from pops.time._program.api import Program

        if type(program) is not Program:
            raise TypeError("TemporalTau requires an exact Program")
        if type(coefficient) is not _Coeff or set(coefficient.powers) != {1}:
            raise TypeError("TemporalTau requires an exact multiple of Program.dt")
        factor = scalar_literal(coefficient.powers[1])
        if factor.to_python() <= 0:
            raise ValueError("TemporalTau factor must be positive")
        if type(at) not in (TimePoint, StagePoint):
            raise TypeError("TemporalTau requires an exact evaluation point")
        clock = point_clock(at, "TemporalTau")
        clocks = {program.clock, *(state.clock for state in program._time_states.values())}
        if clock not in clocks:
            raise ValueError("TemporalTau point has no clock declared by this Program")
        # The process-local capability is outside the scientific snapshot; serialized
        # authority consists solely of owner, clock/point and exact duration factor.
        from weakref import ref
        from pops.time import evolved_field_stage as authority

        key = id(self)
        authority._TAU_PROGRAMS[key] = (
            ref(self, lambda _: authority._TAU_PROGRAMS.pop(key, None)),
            ref(program),
        )
        self.program_owner = program.owner_path.canonical().to_data()
        self.factor = factor
        self.point = at

    @property
    def prog(self) -> Any:
        from pops.time import evolved_field_stage as authority

        leaf, owner = authority._TAU_PROGRAMS[id(self)]
        if leaf() is not self:
            raise ValueError("TemporalTau authoring capability changed")
        program = owner()
        if program is None:
            raise ValueError("TemporalTau authoring Program no longer exists")
        return program

    def __deepcopy__(self, memo: Any) -> TemporalTau:
        return self

    def require_program(self, program: Any, *, at: Any = None) -> None:
        if program is not self.prog or program.owner_path.canonical().to_data() != thaw_data(
            self.program_owner
        ):
            raise ValueError("TemporalTau belongs to another or relabelled Program")
        if at is not None and at != self.point:
            raise ValueError("TemporalTau evaluation point differs from the original stage")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "issued_frame_duration",
            "program_owner": thaw_data(self.program_owner),
            "point": self.point.to_data(),
            "factor": self.factor.to_data(),
            "window_authority": "native_issued_cadence_frame@1",
        }

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> tuple:
        return ("temporal_tau@1", canonical_sha256(self.to_data()))

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Any:
        return Const(0)

    def declaration_references(self) -> tuple:
        return ()

    def resolve_references(self, resolver: Any) -> TemporalTau:
        self.require_program(self.prog)
        return self


@dataclass(frozen=True)
class EvolvedOriginalFieldRate:
    """Spatial operator plus an explicit additive captured source (contract @1).

    The additive expression is evaluated by the original RHS provider, with
    exact State captures and issued duration. Unknown-dependent additive sources
    require a distinct per-candidate realization of that RHS port.
    """

    __pops_ir_immutable__ = True
    spatial: Any
    additive: Any

    def __post_init__(self) -> None:
        from pops.fields._references import collect_references
        from pops.math import elliptic_terms

        if self.spatial is not None and not elliptic_terms(self.spatial):
            if type(self.spatial) is bool or scalar_literal(self.spatial).to_python() != 0:
                raise TypeError("original field rate requires spatial terms or explicit zero/None")
        if not isinstance(self.additive, Expr):
            object.__setattr__(self, "additive", Const(self.additive))
        if any(row.kind != "state" for row in collect_references(self.additive)):
            raise NotImplementedError(
                "additive original field RHS requires exact State captures; "
                "unknown-dependent additive sources need a per-candidate RHS realization"
            )

    def to_data(self) -> dict[str, Any]:
        from pops.fields._identity import strict_field_data

        self.__post_init__()
        return {"contract": "pops.evolved-field-rate.spatial-additive@1",
                "spatial": strict_field_data(self.spatial),
                "additive": strict_field_data(self.additive)}

    def resolve_references(self, resolver: Any) -> EvolvedOriginalFieldRate:
        from pops.fields._references import resolve_value

        return EvolvedOriginalFieldRate(
            resolve_value(self.spatial, resolver, where="original spatial rate"),
            resolve_value(self.additive, resolver, where="original additive source"),
        )

    def declaration_references(self) -> tuple:
        from pops.fields._references import collect_references

        return collect_references((self.spatial, self.additive))


@dataclass(frozen=True)
class EvolvedOriginalFieldProjection:
    """The accumulation declaration carried by the very same original problem."""

    __pops_ir_immutable__ = True

    unknowns: tuple
    evolved_unknowns: tuple
    accumulation: tuple
    spatial_rhs: tuple
    constraints: tuple
    previous: tuple
    tau: TemporalTau

    def to_data(self) -> dict[str, Any]:
        from pops.fields._identity import strict_field_data

        return {
            "schema_version": 2 if any(type(row) is EvolvedOriginalFieldRate
                                       for row in self.spatial_rhs) else 1,
            "kind": "evolved_original_field_accumulation",
            "representation": "piecewise_constant_cell",
            "sampling": "cell_average",
            "measure": "cell_volume",
            "unknowns": [row.canonical_identity() for row in self.unknowns],
            "evolved_unknowns": [row.canonical_identity() for row in self.evolved_unknowns],
            "accumulation": [strict_field_data(row) for row in self.accumulation],
            "spatial_rhs": [strict_field_data(row) for row in self.spatial_rhs],
            "constraints": [
                [handle.canonical_identity(), strict_field_data(equation)]
                for handle, equation in self.constraints
            ],
            "previous": [strict_field_data(row) for row in self.previous],
            "tau": self.tau.to_data(),
        }

    def resolve_references(self, resolver: Any) -> EvolvedOriginalFieldProjection:
        from dataclasses import replace
        from pops.fields._references import resolve_handle, resolve_value

        return replace(
            self,
            unknowns=tuple(
                resolve_handle(row, resolver, where="evolved field unknown")
                for row in self.unknowns
            ),
            evolved_unknowns=tuple(
                resolve_handle(row, resolver, where="evolved field unknown")
                for row in self.evolved_unknowns
            ),
            accumulation=tuple(
                resolve_value(row, resolver, where="evolved field accumulation")
                for row in self.accumulation
            ),
            spatial_rhs=tuple(
                resolve_value(row, resolver, where="evolved spatial rate")
                for row in self.spatial_rhs
            ),
            constraints=tuple(
                (
                    resolve_handle(handle, resolver, where="evolved auxiliary unknown"),
                    resolve_value(equation, resolver, where="evolved auxiliary equation"),
                )
                for handle, equation in self.constraints
            ),
            previous=tuple(
                resolve_value(row, resolver, where="evolved field previous")
                for row in self.previous
            ),
        )

    def declaration_references(self) -> tuple:
        from pops.fields._references import collect_references

        return collect_references(
            (self.unknowns, self.accumulation, self.spatial_rhs, self.constraints, self.previous)
        )


@dataclass(frozen=True, init=False)
class EvolvedOriginalFieldStage:
    """Declare Q(q+) - tau R(q+) = Qn and explicit auxiliary field constraints.

    The original FieldProblem and its conserved publication share one sealed local
    accumulation declaration. A physical field remains observable independently.
    Numerical choices belong to the FieldDiscretization supplied at registration.
    """

    problem: Any
    projection: EvolvedOriginalFieldProjection

    def __init__(
        self,
        name: str,
        *,
        unknowns: tuple,
        evolved_unknowns: tuple,
        accumulation: tuple,
        spatial_rhs: tuple,
        previous: tuple,
        tau: TemporalTau,
        constraints: Mapping | None = None,
        boundaries: tuple = (),
    ) -> None:
        from pops._ir.elliptic import DivCoeffGrad, EllipticSum, Reaction
        from pops._ir.expr import Laplacian
        from pops._ir.quantity import QuantityRef
        from pops.fields import FieldProblem
        from pops.math import elliptic_terms
        from pops.model import Handle

        if type(name) is not str or not name:
            raise TypeError("evolved original field stage requires a nonempty name")
        if (
            type(unknowns) is not tuple
            or not unknowns
            or any(not isinstance(row, Handle) or row.kind != "field" for row in unknowns)
        ):
            raise TypeError("original stage requires an explicit physical field unknown tuple")
        if len(set(unknowns)) != len(unknowns):
            raise ValueError("original stage repeats a physical unknown")
        if (
            type(evolved_unknowns) is not tuple
            or not evolved_unknowns
            or len(set(evolved_unknowns)) != len(evolved_unknowns)
            or any(row not in unknowns for row in evolved_unknowns)
        ):
            raise ValueError("evolved field unknowns must be a unique subset of the original tuple")
        width = len(evolved_unknowns)
        if any(
            type(rows) is not tuple or len(rows) != width
            for rows in (accumulation, spatial_rhs, previous)
        ):
            raise ValueError(
                "one accumulation, spatial rate and previous value per evolved unknown is required"
            )
        if type(tau) is not TemporalTau:
            raise TypeError("original stage requires explicit TemporalTau authority")
        if any(type(row) is not QuantityRef or row.handle.kind != "state" for row in previous):
            raise TypeError(
                "original stage previous values must be declared State component quantities"
            )
        constraints = {} if constraints is None else dict(constraints)
        auxiliary = set(unknowns) - set(evolved_unknowns)
        if set(constraints) != auxiliary:
            raise ValueError("original stage requires exact equations for every auxiliary unknown")

        equations = dict(constraints)
        for unknown, q, rate, old in zip(
            evolved_unknowns, accumulation, spatial_rhs, previous, strict=True
        ):
            local = tuple(elliptic_terms(q))
            if not local or any(type(term) is not Reaction for term in local):
                raise TypeError(
                    "conserved accumulation must be an explicit local reaction expression"
                )
            terms = list(local)
            spatial = rate.spatial if type(rate) is EvolvedOriginalFieldRate else rate
            for term in elliptic_terms(spatial):
                if type(term) is Laplacian:
                    terms.append(DivCoeffGrad(term.field, tau, scale=-term.scale))
                elif type(term) in (DivCoeffGrad, Reaction):
                    terms.append(type(term)(term.field, tau * term.coeff, scale=-term.scale))
                else:
                    raise TypeError("original spatial stage rate has an undeclared realization")
            rhs = old + tau * rate.additive if type(rate) is EvolvedOriginalFieldRate else old
            equations[unknown] = EllipticSum(tuple(terms)) == rhs
        projection = EvolvedOriginalFieldProjection(
            unknowns,
            evolved_unknowns,
            accumulation,
            spatial_rhs,
            tuple((row, constraints[row]) for row in unknowns if row in constraints),
            previous,
            tau,
        )
        object.__setattr__(
            self,
            "problem",
            FieldProblem(
                name,
                unknowns=unknowns,
                equations=tuple(equations[row] for row in unknowns),
                boundaries=boundaries,
                outputs=(projection,),
            ),
        )
        object.__setattr__(self, "projection", projection)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "evolved_original_field_stage",
            "field_problem": self.problem.to_data(),
        }


__all__ = ["EvolvedOriginalFieldStage", "EvolvedOriginalFieldRate", "TemporalTau"]


def evolved_equations(projection: EvolvedOriginalFieldProjection) -> tuple:
    """Rebuild the declared original equations; publication cannot choose another Q."""
    from pops._ir.elliptic import DivCoeffGrad, EllipticSum, Reaction
    from pops._ir.expr import Laplacian
    from pops.math import elliptic_terms

    equations = dict(projection.constraints)
    for unknown, q, rate, previous in zip(
        projection.evolved_unknowns,
        projection.accumulation,
        projection.spatial_rhs,
        projection.previous,
        strict=True,
    ):
        terms = list(elliptic_terms(q))
        spatial = rate.spatial if type(rate) is EvolvedOriginalFieldRate else rate
        for term in elliptic_terms(spatial):
            if type(term) is Laplacian:
                terms.append(DivCoeffGrad(term.field, projection.tau, scale=-term.scale))
            elif type(term) in (DivCoeffGrad, Reaction):
                terms.append(type(term)(term.field, projection.tau * term.coeff, scale=-term.scale))
            else:
                raise TypeError("original evolution rate has an undeclared realization")
        rhs = (previous + projection.tau * rate.additive
               if type(rate) is EvolvedOriginalFieldRate else previous)
        equations[unknown] = EllipticSum(tuple(terms)) == rhs
    return tuple(equations[row] for row in projection.unknowns)
