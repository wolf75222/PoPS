"""Canonical Gaussian-closure moment presets built from ordinary Model contracts."""
from __future__ import annotations

import math
from typing import Any

from pops.fields import FieldOutput, GradientOutput
from pops.math import ddt, div, laplacian
from pops.moments.closures import gaussian_closure
from pops.moments.model_builder import (
    moment_flux_expressions,
    moment_names,
    moment_transport_blocks,
)
from pops.moments.projection import RealizabilityProjection
from pops.moments.sources import MOMENT_VELOCITY_DIMENSION, lorentz_sources
from pops.params import ParameterDeclaration, RuntimeParam as _RuntimeParam
from pops.physics import Density, Model


def _order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("Gaussian order must be an int >= 2")
    return value


def _flag(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError("Gaussian %s must be bool" % name)
    return value


def _parameter(value: Any, *, name: str, default: float) -> ParameterDeclaration:
    if value is None:
        return _RuntimeParam(name, default=default)
    if not isinstance(value, ParameterDeclaration):
        raise TypeError(
            "%s must be RuntimeParam, ConstParam, or DerivedParam; strings are not parameters"
            % name)
    return value


def _author(
    order: Any,
    *,
    name: str,
    robust: Any,
    exact_speeds: Any,
    roe: Any,
    frame: Any = None,
    q_over_m: Any = None,
    poisson_scale: Any = None,
) -> Model:
    order = _order(order)
    robust = _flag(robust, name="robust")
    exact_speeds = _flag(exact_speeds, name="exact_speeds")
    roe = _flag(roe, name="roe")
    if not isinstance(name, str) or not name:
        raise TypeError("Gaussian name must be a non-empty string")

    if frame is None:
        raise TypeError("Gaussian moment models require an explicit two-axis Cartesian frame")
    selected_frame = frame
    axes = getattr(selected_frame, "axes", None)
    if (not isinstance(axes, tuple) or len(axes) != MOMENT_VELOCITY_DIMENSION
            or tuple(getattr(axis, "name", None) for axis in axes) != ("x", "y")):
        raise TypeError("Gaussian frame must expose exactly the Cartesian axes x and y")
    x_axis, y_axis = axes
    model = Model(name, frame=selected_frame)
    state = model.state(
        "U", components=tuple(moment_names(order)), roles={"M00": Density()})
    projection = RealizabilityProjection(robust=robust)
    expressions = moment_flux_expressions(
        model,
        tuple(state),
        order,
        gaussian_closure(order),
        robust=projection.robust,
        eps_m00=projection.eps_m00,
        eps_cov=projection.eps_cov,
    )
    flux = model.flux(
        "transport",
        frame=selected_frame,
        state=state,
        components={x_axis: expressions.x, y_axis: expressions.y},
    )
    if exact_speeds:
        model.wave_speeds_from_jacobian(
            blocks=moment_transport_blocks(order),
            im_tol=0,
        )
    if roe:
        model.roe_from_jacobian()

    sources = []
    if q_over_m is not None or poisson_scale is not None:
        if poisson_scale is None:
            raise ValueError("Gaussian electric forcing requires a Poisson scale")
        if isinstance(poisson_scale, bool) or not isinstance(poisson_scale, (int, float)) \
                or not math.isfinite(float(poisson_scale)) or float(poisson_scale) <= 0.0:
            raise ValueError("Gaussian Poisson scale must be a finite number > 0")
        phi = model.field("phi")
        model.field_operator(
            "electrostatic",
            unknown=phi,
            equation=-laplacian(phi) == float(poisson_scale) * expressions.moments[(0, 0)],
            outputs=(
                FieldOutput("potential", phi),
                GradientOutput("electric", phi, sign=-1),
            ),
        )
        q_handle = model.param(
            _parameter(q_over_m, name="q_over_m", default=-1.0))
        electric = model.source(
            "electric",
            on=state,
            value=lorentz_sources(
                expressions.moments,
                electric_components=tuple(
                    model.aux("electric_%s" % axis.name) for axis in axes),
                q_over_m=model.value(q_handle),
                magnetic_rotation=0.0,
            ),
        )
        sources.append(electric)

    equation = -div(flux)
    for source in sources:
        equation = equation + source
    model.rate("transport", equation=ddt(state) == equation)
    model.module.manifest()
    return model


class Gaussian:
    """Explicit 2V/2D Gaussian-closure models returning canonical Model values."""

    velocity_dimension = MOMENT_VELOCITY_DIMENSION
    supported_spatial_dimensions = (MOMENT_VELOCITY_DIMENSION,)

    @staticmethod
    def transport(
        order: Any = 2,
        *,
        name: str = "gaussian",
        robust: Any = True,
        exact_speeds: Any = True,
        roe: Any = False,
        frame: Any = None,
    ) -> Model:
        """Return a transport-only Gaussian moment Model of arbitrary order."""
        return _author(
            order,
            name=name,
            robust=robust,
            exact_speeds=exact_speeds,
            roe=roe,
            frame=frame,
        )

    @staticmethod
    def vlasov_poisson(
        order: Any = 2,
        *,
        name: str = "gaussian",
        robust: Any = True,
        exact_speeds: Any = True,
        roe: Any = False,
        q_over_m: Any = None,
        eps: Any = 1.0,
        frame: Any = None,
    ) -> Model:
        """Return a Vlasov-Poisson Gaussian moment Model with typed parameter storage."""
        return _author(
            order,
            name=name,
            robust=robust,
            exact_speeds=exact_speeds,
            roe=roe,
            frame=frame,
            q_over_m=_parameter(q_over_m, name="q_over_m", default=-1.0),
            poisson_scale=eps,
        )


__all__ = ["Gaussian"]
