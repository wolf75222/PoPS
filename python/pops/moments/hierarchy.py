"""pops.moments.hierarchy -- the moment-model facade (CartesianVelocityMoments / MomentModel).

A thin, numerics-free facade over the generic generator in
:mod:`pops.moments.model_builder` (``build_moment_model``) and
:mod:`pops.moments.sources` (``lorentz_sources`` / ``maxwellian_moments`` /
``bgk_source``).

The facade carries NO per-cell numeric Python: chainable methods RECORD options on a small dict
and return ``self``; only :meth:`MomentModel.build` touches the engine, mapping the
recorded options literally onto ``build_moment_model``'s existing signature. The Poisson
coupling is authored as an ordinary blackboard field operator on the returned model.
"""
from __future__ import annotations

from typing import Any

from .model_builder import build_moment_model, moment_indices, moment_names
from .sources import MOMENT_VELOCITY_DIMENSION, lorentz_sources
from .closures import gaussian_closure
from .ordering import MomentOrdering
from .basis import MomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection


class CompositeMean:
    """Neutralizing background equal to the live composite mean of M00.

    The authored elliptic RHS is ``eps * M00``.  The AMR field host then subtracts
    ``eps * composite_mean(M00)`` using the same active-coverage reduction as the
    nullspace compatibility check.  That is the neutralizing term, not a projection
    of an arbitrary assembled RHS.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "CompositeMean()"

    def __eq__(self, other: object) -> bool:
        return type(other) is CompositeMean


def _order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("MomentModel order must be an int >= 2 (got %r)" % (value,))
    return value


def _flag(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError("MomentModel %s must be bool" % name)
    return value


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise TypeError("MomentModel %s must be a non-empty identifier" % name)
    return value


def _component_names(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("MomentModel %s must be an ordered component vector" % name)
    try:
        components = tuple(values)
    except TypeError as exc:
        raise TypeError("MomentModel %s must be an ordered component vector" % name) from exc
    if len(components) != MOMENT_VELOCITY_DIMENSION:
        raise ValueError(
            "MomentModel %s requires exactly %d components; got %d"
            % (name, MOMENT_VELOCITY_DIMENSION, len(components)))
    return tuple(_identifier(component, name="%s component" % name) for component in components)


def _coefficient(value: Any, *, name: str) -> Any:
    """Return one explicit typed parameter declaration for a physical coefficient."""
    from pops.params import ConstParam, ParameterDeclaration, RuntimeParam

    if isinstance(value, (ConstParam, RuntimeParam)):
        if value.is_owned:
            raise ValueError(
                "MomentModel %s parameter %r is already owned by %s; sharing requires an "
                "explicit shared owner or tie"
                % (name, value.name, value.owner_identity)
            )
        return value
    if isinstance(value, ParameterDeclaration):
        raise TypeError(
            "MomentModel %s supports ConstParam or RuntimeParam coefficients; "
            "DerivedParam dependencies belong to a concrete model owner" % name
        )
    if isinstance(value, bool):
        raise TypeError(
            "MomentModel %s must be a numeric scalar or typed ParameterDeclaration, not bool"
            % name)
    try:
        return ConstParam(name, value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            "MomentModel %s must be a finite numeric scalar or typed "
            "ParameterDeclaration: %s" % (name, exc)) from None


def _fresh_coefficient(declaration: Any) -> Any:
    """Copy ownerless coefficient metadata so every build gets its own registry authority."""
    from pops.params import ConstParam, RuntimeParam

    # A declaration can be ownerless when recorded and claimed by another model before build().
    # Recheck at the clone boundary so deferred construction cannot bypass the single-owner
    # authority enforced by ParameterDeclaration._claim_owner().
    if declaration.is_owned:
        raise ValueError(
            "MomentModel parameter %r is already owned by %s; sharing requires an explicit "
            "shared owner or tie" % (declaration.name, declaration.owner_identity)
        )
    common = {
        "dtype": declaration.dtype,
        "domain": declaration.domain,
        "unit": declaration.unit,
        "provenance": declaration.provenance,
    }
    if isinstance(declaration, ConstParam):
        return ConstParam(declaration.name, declaration.value, **common)
    if isinstance(declaration, RuntimeParam):
        return RuntimeParam(declaration.name, default=declaration.default, **common)
    raise TypeError("unsupported MomentModel coefficient declaration %r" % type(declaration).__name__)


def _parameter_value(model: Any, declaration: Any, registered: dict[str, Any]) -> Any:
    """Register one declaration once per built model and return its symbolic value."""
    existing = registered.get(declaration.name)
    if existing is not None:
        owner, value = existing
        if owner is not declaration:
            raise ValueError(
                "MomentModel parameters reuse name %r for distinct declarations"
                % declaration.name)
        return value
    handle = model.param(_fresh_coefficient(declaration))
    value = model.value(handle)
    registered[declaration.name] = (declaration, value)
    return value


def CartesianVelocityMoments(order: Any, *, closure: Any = None, robust: bool = True,
                             sources: Any = None, exact_speeds: bool = True,
                             roe: bool = False) -> Any:
    """Construct a 2D Cartesian-velocity moment model facade (records options; no build).

    @p order: max order of the transported moments (order=2 -> 6 vars, order=4 -> 15).
    @p closure: the closure callable (the only physics); ``None`` -> ``gaussian_closure(order)``
       resolved lazily at :meth:`MomentModel.build`.
    @p robust / @p exact_speeds / @p roe: the numerics knobs (see :class:`MomentModel`).
    @p sources: an optional pre-built ``(m, M) -> list`` sources callable (advanced); the
       chainable ``add_*_source`` methods are the usual way to add sources.

    Returns a fresh :class:`MomentModel` recording these options. NOTHING is built yet.
    """
    model = MomentModel(order)
    model._closure = closure
    model._robust = _flag(robust, name="robust")
    model._exact_speeds = _flag(exact_speeds, name="exact_speeds")
    model._roe = _flag(roe, name="roe")
    if sources is not None:
        model._extra_sources = sources
    return model


class MomentModel:
    """A recorded explicit 2V/2D moment specialization; builds a Model on demand.

    Every chainable method mutates a small option dict and returns ``self``. The recorded
    options map literally onto :func:`build_moment_model`'s signature; :meth:`build` is the
    ONLY place that touches the engine. This ``(p, q)`` hierarchy advertises its exact two-axis
    physical scope; it is not a generic-spatial-rank fallback.
    """

    velocity_dimension = MOMENT_VELOCITY_DIMENSION
    supported_spatial_dimensions = (MOMENT_VELOCITY_DIMENSION,)

    def __init__(self, order: Any) -> None:
        self._order = _order(order)
        self._closure = None
        self._robust = True
        self._exact_speeds = True
        self._roe = False
        self._proj = RealizabilityProjection()
        # Recorded source contributions, assembled into ONE callable at build:
        self._electric: Any = None       # (ordered provider components, q_over_m)
        self._magnetic: Any = None       # omega_c name
        self._extra_sources: Any = None  # an advanced pre-built (m, M) -> list
        # Recorded Poisson coupling (applied to the built model):
        self._poisson: Any = None        # (phi, eps, optional uniform background)
        self._composite_mean_background = False

    # --- chainable recorders -----------------------------------------------
    def add_poisson_coupling(
        self,
        phi: Any = "phi",
        eps: Any = 1.0,
        *,
        background: Any = None,
    ) -> Any:
        """Record a Poisson coupling with an optional explicit uniform background.

        The elliptic RHS is ``eps * M00`` when ``background`` is omitted and
        ``eps * (M00 - background)`` otherwise.  ``background`` is a typed physical
        coefficient (literal, ``ConstParam`` or ``RuntimeParam``) or
        :class:`CompositeMean`.  A frozen coefficient is never inferred.  CompositeMean
        is evaluated from the live composite mass with the same coverage as the
        nullspace check; the field solver does not project an arbitrary RHS.
        """
        if isinstance(background, CompositeMean):
            self._composite_mean_background = True
            background_declaration = None
        else:
            self._composite_mean_background = False
            background_declaration = (
                None if background is None else _coefficient(
                    background, name="Poisson background"
                )
            )
        self._poisson = (
            _identifier(phi, name="Poisson field"),
            _coefficient(eps, name="eps"),
            background_declaration,
        )
        return self

    def add_vlasov_electric_source(
        self, electric_components: Any, q_over_m: Any,
    ) -> Any:
        """Record the exact ordered 2V electric provider vector and charge-to-mass ratio."""
        self._electric = (
            _component_names(electric_components, name="electric_components"),
            _coefficient(q_over_m, name="q_over_m"),
        )
        return self

    def add_magnetic_source(self, omega_c: Any) -> Any:
        """Record the magnetic source: the Lorentz magnetic branch with cyclotron frequency
        @p omega_c (a typed parameter declaration or explicit numeric constant)."""
        self._magnetic = _coefficient(omega_c, name="omega_c")
        return self

    def add_numerics(self, *, robust: Any = None, exact_speeds: Any = None,
                     roe: Any = None) -> Any:
        """Override the numerics knobs (any of ``robust`` / ``exact_speeds`` / ``roe``)."""
        if robust is not None:
            self._robust = _flag(robust, name="robust")
        if exact_speeds is not None:
            self._exact_speeds = _flag(exact_speeds, name="exact_speeds")
        if roe is not None:
            self._roe = _flag(roe, name="roe")
        return self

    def set_realizability(self, projection: Any) -> Any:
        """Set the realizability projection (a :class:`RealizabilityProjection`); its ``robust``
        flag also drives the engine robust path."""
        if not isinstance(projection, RealizabilityProjection):
            raise TypeError("set_realizability expects a RealizabilityProjection; got %r"
                            % (projection,))
        self._proj = projection
        self._robust = projection.robust
        return self

    # --- introspection (no engine call) ------------------------------------
    def hierarchy(self) -> Any:
        """A frozen :class:`MomentHierarchy` snapshot of the recorded structure (no build)."""
        return MomentHierarchy(self)

    # --- the single engine touch -------------------------------------------
    def _resolved_closure(self) -> Any:
        """The closure to build with (the recorded one, or ``gaussian_closure(order)``)."""
        return self._closure if self._closure is not None else gaussian_closure(self._order)

    def _sources_cb(self, registered: dict[str, Any]) -> Any:
        """Assemble the recorded source contributions into ONE ``(m, M) -> list`` callable.

        Returns ``None`` when no source was recorded (the engine then wires no source).
        The electric and magnetic Lorentz branches are summed term-by-term (they are aligned
        lists over ``moment_indices``); an advanced pre-built callable is added on top.
        """
        if not (self._electric or self._magnetic or self._extra_sources):
            return None
        electric, magnetic, extra = self._electric, self._magnetic, self._extra_sources
        order = self._order

        def sources(m: Any, M: Any) -> Any:
            idx = moment_indices(order)
            acc: list[Any] = [None] * len(idx)

            def add(terms: Any) -> None:
                for k, t in enumerate(terms):
                    acc[k] = t if acc[k] is None else (acc[k] + t)

            if electric is not None:
                component_names, qom_declaration = electric
                electric_values = tuple(m.aux(component) for component in component_names)
                qom = _parameter_value(m, qom_declaration, registered)
                add(lorentz_sources(
                    M,
                    electric_components=electric_values,
                    q_over_m=qom,
                    magnetic_rotation=0.0,
                ))
            if magnetic is not None:
                omega_c = _parameter_value(m, magnetic, registered)
                add(lorentz_sources(
                    M,
                    electric_components=(0.0, 0.0),
                    q_over_m=1.0,
                    magnetic_rotation=omega_c,
                ))
            if extra is not None:
                add(extra(m, M))
            return [0.0 if term is None else term for term in acc]

        return sources

    def build(self, name: Any = "moments", *, frame: Any = None) -> Any:
        """Build the recorded specification into the canonical ``physics.Model``.

        Maps the recorded options literally onto :func:`build_moment_model`, then authors the
        recorded Poisson coupling through the same field/operator contracts as user code. ``frame``
        is the explicit domain-owned two-axis Cartesian frame; no spatial rank is inferred.
        """
        registered: dict[str, Any] = {}
        m = build_moment_model(
            name, self._order, self._resolved_closure(),
            exact_speeds=self._exact_speeds, robust=self._robust,
            sources=self._sources_cb(registered), roe=self._roe,
            eps_m00=self._proj.eps_m00, eps_cov=self._proj.eps_cov,
            floor_density=self._proj.floor_density,
            repair_moment_matrix=self._proj.repair_moment_matrix,
            matrix_repair_max_density=self._proj.matrix_repair_max_density,
            frame=frame)
        if self._poisson is not None:
            self._apply_poisson(m, registered)
        return m

    # --- internals ----------------------------------------------------------
    def _apply_poisson(self, m: Any, registered: dict[str, Any]) -> None:
        """Author ``-laplacian(phi) == eps * M00`` and its gradient outputs."""
        from pops.fields import FieldOutput, GradientOutput
        from pops.math import laplacian

        phi_name, eps_declaration, background_declaration = self._poisson
        eps = _parameter_value(m, eps_declaration, registered)
        state = m.states["U"]
        density = state[moment_names(self._order)[0]]
        source_density = density
        if background_declaration is not None:
            background = _parameter_value(m, background_declaration, registered)
            source_density = density - background
        phi = m.field(phi_name)
        # The recorded electric vector is bound to this canonical FieldSpace. The provider must
        # materialize that exact space rather than an isomorphic space under another local name.
        m.field_operator(
            "fields",
            unknown=phi,
            equation=-laplacian(phi) == eps * source_density,
            outputs=(
                FieldOutput(phi_name, phi),
                GradientOutput("grad", phi),
            ),
        )


class MomentHierarchy:
    """An immutable snapshot of a :class:`MomentModel`'s structure (introspection only).

    Built from the model's recorded options plus :func:`moment_indices` / :func:`moment_names`.
    It describes the structure (ordering, basis, transforms, sources, projection, speeds); it
    makes NO engine call and holds no numeric data.
    """

    velocity_dimension = MOMENT_VELOCITY_DIMENSION
    supported_spatial_dimensions = (MOMENT_VELOCITY_DIMENSION,)

    def __init__(self, model: Any) -> None:
        order = model._order
        self.order = order
        self.ordering = MomentOrdering()
        self.basis = MomentBasis(order, ordering=self.ordering)
        self.transforms = (CenteredTransform(order), StandardizedTransform(order))
        self.projection = model._proj
        self.speeds = ExactSpeeds.from_flags(model._exact_speeds, model._roe)
        srcs = []
        if model._electric is not None:
            srcs.append(("electric", model._electric))
        if model._magnetic is not None:
            srcs.append(("magnetic", model._magnetic))
        if model._poisson is not None:
            srcs.append(("poisson", model._poisson))
        self.sources = tuple(srcs)

    def names(self) -> Any:
        """The moment-variable names of this hierarchy (``M{p}{q}``)."""
        return moment_names(self.order)

    def __repr__(self) -> str:
        return "MomentHierarchy(order=%d, sources=%d)" % (self.order, len(self.sources))


__all__ = ["CartesianVelocityMoments", "MomentModel", "MomentHierarchy"]
