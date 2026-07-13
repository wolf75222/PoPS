"""Canonical HyQMOM15 composition over generic Model and closure contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.frames import Cartesian2D
from pops.math import ddt, div
from pops.moments.closures import HyQMOM15Closure
from pops.moments.model_builder import (
    moment_flux_expressions,
    moment_indices,
    moment_names,
)
from pops.moments.projection import RealizabilityProjection, RealizableSet
from pops.moments.sources import lorentz_sources
from pops.params import ParameterDeclaration, RuntimeParam
from pops.physics import Model


_HYQMOM15_ORDER = 4


def _parameter(value: Any, *, name: str, default: float) -> ParameterDeclaration:
    if value is None:
        return RuntimeParam(name, default=default)
    if not isinstance(value, ParameterDeclaration):
        raise TypeError(
            "%s must be a typed ParameterDeclaration; use RuntimeParam or ConstParam" % name)
    return value


def _magnetic_matrix(indices: tuple[tuple[int, int], ...], omega_c: Any) -> tuple[tuple[Any, ...], ...]:
    """Return the generic linear Lorentz rotation matrix for any closed hierarchy."""
    position = {index: column for column, index in enumerate(indices)}
    rows = []
    for p, q in indices:
        row: list[Any] = [0.0] * len(indices)
        if p:
            row[position[(p - 1, q + 1)]] = float(p) * omega_c
        if q:
            column = position[(p + 1, q - 1)]
            row[column] = row[column] - float(q) * omega_c
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True)
class HyQMOM15Definition:
    """The authored Model plus the exact handles consumers need to compose a Case."""

    model: Model
    state: Any
    flux: Any
    explicit_rate: Any
    electric_source: Any
    implicit_source: Any
    closure: Any
    projection: RealizabilityProjection
    realizable_set: RealizableSet

    @property
    def order(self) -> int:
        return _HYQMOM15_ORDER

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(moment_names(self.order))


class HyQMOM15:
    """Pre-implemented 15-moment physics, assembled from ordinary generic contracts."""

    @staticmethod
    def vlasov_lorentz(
        *,
        name: str = "hyqmom15",
        closure: Any = None,
        projection: Any = None,
        q_over_m: Any = None,
        omega_c: Any = None,
        exact_speeds: bool = True,
        roe: bool = False,
    ) -> HyQMOM15Definition:
        """Author transport, electric forcing and an implicit magnetic local operator.

        The closure is evaluated once through ``LocalClosure`` on symbolic values.  The
        returned object contains a real blackboard :class:`pops.physics.Model`; no HyQMOM
        token enters the operator registry or native lowering.
        """
        selected_closure = HyQMOM15Closure() if closure is None else closure
        selected_projection = (
            RealizabilityProjection() if projection is None else projection)
        if not isinstance(selected_projection, RealizabilityProjection):
            raise TypeError("projection must be a RealizabilityProjection")
        q_decl = _parameter(q_over_m, name="q_over_m", default=-1.0)
        omega_decl = _parameter(omega_c, name="omega_c", default=1.0)

        frame = Cartesian2D()
        model = Model(name, frame=frame)
        names = tuple(moment_names(_HYQMOM15_ORDER))
        state = model.state(
            "U", components=names, roles={"M00": "density"})
        variables = tuple(state)
        expressions = moment_flux_expressions(
            model,
            variables,
            _HYQMOM15_ORDER,
            selected_closure,
            robust=selected_projection.robust,
            eps_m00=selected_projection.eps_m00,
            eps_cov=selected_projection.eps_cov,
        )
        flux = model.flux(
            "transport",
            frame=frame,
            state=state,
            components={frame.x: expressions.x, frame.y: expressions.y},
        )
        if exact_speeds:
            # The blackboard Model owns this PdeModel implementation detail.  The factory
            # merely requests the generic Jacobian provider; no model-specific lowering exists.
            model.dsl.wave_speeds_from_jacobian()
        if roe:
            model.dsl.roe_from_jacobian()

        q_handle = model.param(q_decl)
        omega_handle = model.param(omega_decl)
        q_value = model.value(q_handle)
        omega_value = model.value(omega_handle)
        electric = model.source(
            "electric",
            on=state,
            value=lorentz_sources(
                expressions.moments,
                model.aux("grad_x"),
                model.aux("grad_y"),
                q_value,
                0.0,
            ),
        )
        indices = tuple(moment_indices(_HYQMOM15_ORDER))
        magnetic_math = model.local_linear_operator(
            "magnetic_rotation",
            on=state,
            matrix=_magnetic_matrix(indices, omega_value),
        )
        magnetic = model.operator(
            "magnetic_rotation", returns=magnetic_math)
        explicit_rate = model.rate(
            "transport",
            equation=ddt(state) == -div(flux),
        )
        # Materialize the final declaration-owned manifest after every operator and
        # parameter exists.  Case registration must observe the settled definition
        # fingerprint, never the intermediate fingerprint issued while the registry was
        # still being assembled.
        model.module.manifest()
        return HyQMOM15Definition(
            model=model,
            state=state,
            flux=flux,
            explicit_rate=explicit_rate,
            electric_source=electric,
            implicit_source=magnetic,
            closure=selected_closure,
            projection=selected_projection,
            realizable_set=RealizableSet(_HYQMOM15_ORDER),
        )

    @staticmethod
    def vlasov_poisson_magnetic(order: Any = _HYQMOM15_ORDER, **options: Any) -> Model:
        """Return the canonical Model for callers that do not need its composition handles."""
        if order != _HYQMOM15_ORDER:
            raise ValueError(
                "HyQMOM15 has exactly order 4 (15 moments); use Gaussian for generic orders")
        return HyQMOM15.vlasov_lorentz(**options).model


__all__ = ["HyQMOM15", "HyQMOM15Definition"]
