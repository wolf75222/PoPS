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
from .sources import lorentz_sources
from .closures import gaussian_closure
from .ordering import MomentOrdering
from .basis import MomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection


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
    """A recorded 2D moment-model specification; builds a ``physics.Model`` on demand.

    Every chainable method mutates a small option dict and returns ``self``. The recorded
    options map literally onto :func:`build_moment_model`'s signature; :meth:`build` is the
    ONLY place that touches the engine.
    """

    def __init__(self, order: Any) -> None:
        self._order = _order(order)
        self._closure = None
        self._robust = True
        self._exact_speeds = True
        self._roe = False
        self._proj = RealizabilityProjection()
        # Recorded source contributions, assembled into ONE callable at build:
        self._transport = False
        self._electric: Any = None       # (ex, ey, q_over_m) names
        self._magnetic: Any = None       # omega_c name
        self._extra_sources: Any = None  # an advanced pre-built (m, M) -> list
        # Recorded Poisson coupling (applied to the built model):
        self._poisson: Any = None        # (phi, eps)

    # --- chainable recorders -----------------------------------------------
    def add_transport(self) -> Any:
        """Record the transport (flux) terms. The flux is ALWAYS generated by the engine; this
        flag documents the intent and is a no-op on the engine call (kept for a fluent chain)."""
        self._transport = True
        return self

    def add_poisson_coupling(self, phi: Any = "phi", eps: Any = 1.0) -> Any:
        """Record a Poisson coupling: the elliptic RHS is the charge density (``eps * M00``) and
        the model reads the field gradient aux (``grad_x`` / ``grad_y``). Applied to the built model."""
        self._poisson = (
            _identifier(phi, name="Poisson field"),
            _coefficient(eps, name="eps"),
        )
        return self

    def add_vlasov_electric_source(self, ex: Any, ey: Any, q_over_m: Any) -> Any:
        """Record the Vlasov electric source: the Lorentz electric branch over the aux fields
        @p ex / @p ey (e.g. ``grad_x`` / ``grad_y``) scaled by the param @p q_over_m."""
        self._electric = (
            _identifier(ex, name="electric x field"),
            _identifier(ey, name="electric y field"),
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
                ex_name, ey_name, qom_declaration = electric
                ex = m.aux(ex_name)
                ey = m.aux(ey_name)
                qom = _parameter_value(m, qom_declaration, registered)
                add(lorentz_sources(M, ex, ey, qom, 0.0))
            if magnetic is not None:
                omega_c = _parameter_value(m, magnetic, registered)
                add(lorentz_sources(M, 0.0, 0.0, 1.0, omega_c))
            if extra is not None:
                add(extra(m, M))
            return [0.0 if term is None else term for term in acc]

        return sources

    def build(self, name: Any = "moments") -> Any:
        """Build the recorded specification into the canonical ``physics.Model``.

        Maps the recorded options literally onto :func:`build_moment_model`, then authors the
        recorded Poisson coupling through the same field/operator contracts as user code.
        """
        registered: dict[str, Any] = {}
        m = build_moment_model(
            name, self._order, self._resolved_closure(),
            exact_speeds=self._exact_speeds, robust=self._robust,
            sources=self._sources_cb(registered), roe=self._roe,
            eps_m00=self._proj.eps_m00, eps_cov=self._proj.eps_cov)
        if self._poisson is not None:
            self._apply_poisson(m, registered)
        return m

    def check(self, name: Any = "moments") -> Any:
        """Alias of :meth:`build` (build + the engine's own validation on construction)."""
        return self.build(name)

    # --- internals ----------------------------------------------------------
    def _apply_poisson(self, m: Any, registered: dict[str, Any]) -> None:
        """Author ``-laplacian(phi) == eps * M00`` and its gradient outputs."""
        from pops.fields import FieldOutput, GradientOutput
        from pops.math import laplacian

        phi_name, eps_declaration = self._poisson
        eps = _parameter_value(m, eps_declaration, registered)
        state = m.states["U"]
        density = state[moment_names(self._order)[0]]
        phi = m.field(phi_name)
        m.field_operator(
            "poisson",
            unknown=phi,
            equation=-laplacian(phi) == eps * density,
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
