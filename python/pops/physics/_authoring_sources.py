"""Authoring mixin: sources, elliptic RHS, named operators, stability hooks.

Methods only; the touched attributes (``_source`` / ``_elliptic`` /
``_source_terms`` / ``_linear_sources`` / ``_elliptic_fields`` / ``_stab_speed``
/ ``_stab_dt`` / ``_src_freq`` / ``_proj`` / ``_src_jac`` / ``_rate_operators``)
are created by ``HyperbolicModel.__init__``. Codegen-free and ``_pops``-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pops._ir import _wrap
from pops._ir.visitors import _expr_uses_cons_or_prim
from pops.model import OperatorHandle

from .aux import AUX_CANONICAL

if TYPE_CHECKING:
    from ._model_contract import _HyperbolicModel
else:
    _HyperbolicModel = object


class _SourceMixin(_HyperbolicModel):
    """Source, elliptic, named operators, projection and stability declarations."""

    def set_source(self, s: Any) -> None: self._source = [_wrap(e) for e in s]
    def set_elliptic_rhs(self, e: Any) -> None: self._elliptic = _wrap(e)

    def elliptic_field(
        self,
        name: Any,
        rhs: Any,
        operator: str = "poisson",
        aux: Any = None,
        *,
        gradient_sign: int = 1,
    ) -> None:
        """Declare a NAMED elliptic field (ADC-419): an elliptic solve ``operator(field) = rhs(U)``
        whose solution + derived quantities populate the NAMED aux fields @p aux (default
        ``["phi", "grad_x", "grad_y"]``, the canonical electrostatic triple). @p rhs is an Expr of
        cons / primitives / aux / params (the elliptic right-hand side assembled from the state, the
        same surface as set_elliptic_rhs). @p operator names the elliptic operator (only ``"poisson"``
        is hosted by the runtime today). A named elliptic field is OPT-IN; the unnamed default stays in
        self._elliptic (m.elliptic_rhs). name must be a valid identifier, unique, and not collide with
        the default.

        The declaration carries physics only. A case-owned callable field operator and its
        ``FieldDiscretization`` select the executable solver, boundaries and output route; none of
        those numerical choices are inferred here."""
        n = self.n_vars
        if n == 0:
            raise ValueError("elliptic_field(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("elliptic_field: name must be a non-empty string")
        if name == "default":
            raise ValueError("elliptic_field('default'): the default elliptic field is m.elliptic_rhs "
                             "(set_elliptic_rhs); pass a distinct name")
        if not name.isidentifier():
            raise ValueError("elliptic_field('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if operator != "poisson":
            raise ValueError("elliptic_field('%s'): operator '%s' is not supported (only 'poisson')"
                             % (name, operator))
        if name in self._elliptic_fields:
            raise ValueError("elliptic_field('%s'): already declared" % name)
        aux = list(aux) if aux is not None else ["phi", "grad_x", "grad_y"]
        if not aux:
            raise ValueError("elliptic_field('%s'): aux must list at least one field" % name)
        if len(aux) == 2 or len(aux) > 3:
            raise ValueError(
                "elliptic_field('%s'): aux outputs must have length 1 or 3; the runtime "
                "cannot register %d outputs yet" % (name, len(aux)))
        if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
            raise ValueError(
                "elliptic_field('%s'): gradient_sign must be exactly -1 or 1" % name)
        if len(aux) == 1 and gradient_sign != 1:
            raise ValueError(
                "elliptic_field('%s'): gradient_sign=-1 requires two gradient outputs" % name)
        for a in aux:
            if not (isinstance(a, str) and a.isidentifier()):
                raise ValueError("elliptic_field('%s'): aux field %r is not a valid identifier"
                                 % (name, a))
        rhs = _wrap(rhs)
        # The elliptic RHS brick (emit_cpp_elliptic_field, like the default emit_cpp_elliptic) reads
        # ONLY the conservative state (+ primitives derived from it), never the aux channel: the System
        # assembles f(U) per cell from the block state, before any aux is solved. An rhs reading an aux
        # field would compile to an undefined local -> reject it loud (the default set_elliptic_rhs has
        # the same surface). A source/flux READING the named field's solved aux is the supported pattern;
        # it is the named-elliptic RHS itself that must be a function of U only.
        rhs_aux = rhs.deps() & (set(AUX_CANONICAL) | set(self.aux_extra_names) | {"phi", "grad_x",
                                                                                 "grad_y", "B_z",
                                                                                 "T_e"})
        if rhs_aux:
            raise ValueError("elliptic_field('%s'): rhs may not read aux fields %s; the elliptic "
                             "right-hand side is a function of the conservative state only (the same "
                             "surface as m.elliptic_rhs). Read the SOLVED field's aux in a source/flux."
                             % (name, sorted(rhs_aux)))
        self._elliptic_fields[name] = {
            "rhs": rhs,
            "operator": operator,
            "aux": aux,
            "gradient_sign": gradient_sign,
        }

    def source_term(self, name: Any, exprs: Any) -> Any:
        """Declare a NAMED local source S_name(U, primitives, aux, params): exactly n_cons
        expressions, free to depend on cons / primitives / aux / aux_field / params / constants. A
        named source is OPT-IN -- it is emitted only when a compiled time Program asks for it
        (ctx.rhs(..., sources=[name]) / ctx.source(name)) and is NEVER summed implicitly into the
        legacy total source. name == "default" is the backward-compatible alias of m.source([...])
        (stored in self._source, hash unchanged). Other names must be valid identifiers, unique, and
        must not collide with a linear_source.

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3): an
        inert typed reference (``.name`` / ``.kind == "local_source"``) a Program can pass to
        ``P.call`` in place of the string name, lowering to the byte-identical IR."""
        n = self.n_vars
        if n == 0:
            raise ValueError("source_term(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("source_term: name must be a non-empty string")
        exprs = [_wrap(e) for e in exprs]
        if len(exprs) != n:
            raise ValueError("source_term('%s'): %d expressions for %d conservative variables"
                             % (name, len(exprs), n))
        if name == "default":
            self._source = exprs   # equivalent to m.source([...]) -- the legacy default source
            return OperatorHandle(
                "default", kind="local_source", owner=self.owner_path,
                registered_operator_name="source_default")
        if not name.isidentifier():
            raise ValueError("source_term('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if name in self._source_terms:
            raise ValueError("source_term('%s'): already declared" % name)
        if name in self._linear_sources:
            raise ValueError("source_term('%s'): name collides with a linear_source" % name)
        self._source_terms[name] = exprs
        return OperatorHandle(name, kind="local_source", owner=self.owner_path)

    def linear_source(self, name: Any, matrix: Any) -> Any:
        """Declare a NAMED local linear operator L_name(aux, params): an n_cons x n_cons matrix whose
        coefficients may depend on constants / params / aux / aux_field ONLY -- NOT on conservative or
        primitive variables (otherwise S(U) = L U is not linear in U and could not be treated as a
        local linear source by solve_local_linear). The operator is OPT-IN: never folded into m.source
        or ctx.rhs; a Program uses it explicitly via ctx.linear_source(name) / ctx.apply /
        ctx.solve_local_linear. Name must be a valid identifier, unique, and must not collide with a
        source_term.

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3): an
        inert typed reference (``.name`` / ``.kind == "local_linear_operator"``) a Program can pass to
        ``P.call`` in place of the string name, lowering to the byte-identical IR."""
        n = self.n_vars
        if n == 0:
            raise ValueError("linear_source(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("linear_source: name must be a non-empty string")
        if not name.isidentifier():
            raise ValueError("linear_source('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        rows = [list(r) for r in matrix]
        if len(rows) != n or any(len(r) != n for r in rows):
            raise ValueError("linear_source('%s'): expected a %dx%d matrix (n_cons x n_cons)"
                             % (name, n, n))
        wrapped = [[_wrap(c) for c in row] for row in rows]
        for row in wrapped:
            for coeff in row:
                if _expr_uses_cons_or_prim(coeff):
                    raise ValueError("linear_source '%s' coefficients must not depend on "
                                     "conservative or primitive variables" % name)
        if name in self._linear_sources:
            raise ValueError("linear_source('%s'): already declared" % name)
        if name in self._source_terms:
            raise ValueError("linear_source('%s'): name collides with a source_term" % name)
        self._linear_sources[name] = wrapped
        return OperatorHandle(name, kind="local_linear_operator", owner=self.owner_path)

    def rate_operator(self, name: Any, *, flux: bool = True, sources: Any = ("default",),
                      fluxes: Any = None) -> Any:
        """Declare a NAMED composite rate operator ``R_name = -div F + sum(sources)`` (Spec 2,
        operator-first). It is a Program-side ALIAS for ``ctx.rhs(flux=, sources=, fluxes=)``: a typed
        calling the returned handle lowers to the same internal rhs IR,
        so a model-free Program can address the RHS by one operator name instead of spelling out
        flux/sources. The alias carries no new numerics (its flux/sources are already in the model and
        the hash) -- it never enters the model hash nor the codegen. ``flux`` / ``sources`` / ``fluxes``
        have the same meaning as :meth:`Program.rhs`. ``name`` must be a valid identifier, unique among
        rate operators, and must not collide with a source_term / linear_source.

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3): an
        inert typed reference (``.name`` / ``.kind == "local_rate"``) callable with Program values;
        string operator selection is absent."""
        if self.n_vars == 0:
            raise ValueError("rate_operator(%r): declare conservative_vars(...) first" % (name,))
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("rate_operator(%r): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % (name,))
        if name in self._rate_operators:
            raise ValueError("rate_operator('%s'): already declared" % name)
        if name in self._source_terms or name in self._linear_sources:
            raise ValueError("rate_operator('%s'): name collides with a source_term/linear_source"
                             % name)
        flx = list(fluxes) if fluxes else None
        if not flux and flx:
            raise ValueError("rate_operator('%s'): named fluxes require flux=True "
                             "(a source-only rate has no flux to divide)" % name)
        srcs = list(sources) if sources is not None else None
        self._rate_operators[name] = {"flux": bool(flux), "sources": srcs, "fluxes": flx}
        return OperatorHandle(name, kind="local_rate", owner=self.owner_path)

    def stability_speed(self, expr: Any) -> None:
        """STABILITY speed lambda* (expression of cons / prims / aux): drives the block CFL
        instead of ``max(|eigenvalues|)``. Emitted as ``stability_speed(U, aux, dir)`` (C++ trait
        ``HasStabilitySpeed``): System::step_cfl then uses it for the transport bound
        dt <= cfl*h/lambda*, while the Riemann solvers keep reading max_wave_speed
        (stability != accuracy). WITHOUT a call, the FALLBACK is strictly historical:
        max(abs(eigenvalues)) via max_wave_speed. Compiled like flux/source (no per-cell Python
        callback: compatible with GPU/MPI production). Wired into System AND AmrSystem (mono and
        multi-block; on the AMR side the reduction is evaluated on the COARSE level, where the CFL lives)."""
        self._stab_speed = _wrap(expr)

    def stability_dt(self, expr_dt: Any) -> None:
        """Direct ADMISSIBLE step dt(U, aux) (expression > 0, in time units): local step
        bound, emitted as ``stability_dt(U, aux)`` (C++ trait ``HasStabilityDt``). System::step_cfl
        imposes dt <= min_cells(stability_dt) * substeps / stride (the cfl is NOT applied: the
        model already declares an admissible step). The most general form (stiff source, local coupling,
        non-reducible transport+source formula). WITHOUT a call, no additional bound (historical
        step policy). Compiled like flux/source (GPU/MPI production). Wired into System AND
        AmrSystem (mono and multi-block; on the AMR side evaluated on the COARSE level)."""
        self._stab_dt = _wrap(expr_dt)

    def source_frequency(self, expr_mu: Any) -> None:
        """Local FREQUENCY mu(U, aux) [1/time] of the SOURCE (relaxation, collision, reaction):
        the 'second CFL' of the meeting -- bound dt <= cfl * substeps / (stride * max_cells(mu)),
        WITHOUT a space step (a source bounded in 1/time). Emitted as ``frequency(U, aux)`` on the
        generated SOURCE BRICK (C++ contract of source bricks, cf. physics/source.hpp);
        CompositeModel forwards it (HasSourceFrequency trait) and System/AmrSystem::step_cfl
        aggregate it. REQUIRES set_source/m.source (the frequency is a property of the source).
        WITHOUT a call, the source does not constrain the step (historical). Compiled (GPU/MPI production)."""
        self._src_freq = _wrap(expr_mu)

    def projection(self, exprs: Any) -> None:
        """PROJECTION PONCTUELLE post-pas (ADC-177) : U <- P(U, aux), une expression par composante
        conservative (en fonction des cons / prims / aux). Emise comme ``project(U, aux)`` sur la
        brique hyperbolique generee (trait C++ ``HasPointwiseProjection``) ; le System l'applique sur
        les cellules VALIDES de chaque bloc a la FIN de chaque macro-pas ENTIER (apres transport +
        etage source + couplages ; jamais par etage RK -- semantique POST-PAS). CONTRAT : P doit etre
        une PROJECTION (idempotente : P(P(U)) == P(U)) et PONCTUELLE (aucune lecture de voisin). Les
        formules de realisabilite restent cote cas ; les clamps s'ecrivent SANS branche, en max/min
        via abs_ / sign : p.ex. positivite q >= 0 : (q + abs_(q)) / 2. Compilee comme flux/source
        (CSE comprise, production GPU/MPI -- remplace le callback Python par cellule). Le package
        natif transporte ce hook sur System et l'applique par niveau apres reflux sur AMR. SANS
        appel : aucun hook emis."""
        exprs = [_wrap(e) for e in exprs]
        if len(exprs) != self.n_vars:
            raise ValueError("projection : %d expressions attendues (une par composante "
                             "conservative), recu %d" % (self.n_vars, len(exprs)))
        self._proj = exprs

    def projection_value(self, U: Any, aux: Any = None) -> Any:
        """EVALUATEUR numpy de la projection ponctuelle emise (miroir exact du project(U, aux) C++) :
        U (n_vars, ...) -> U projete. Reference de test / prototypage hote. ValueError si
        projection([...]) n'a pas ete appelee."""
        if self._proj is None:
            raise ValueError("projection_value : appeler projection([...]) d'abord")
        env = self._env(U, aux)
        shape = np.asarray(U[0]).shape
        return np.stack([np.broadcast_to(e.eval(env), shape) for e in self._proj], axis=0)

    def source_jacobian(self, rows: Any) -> None:
        """ANALYTICAL JACOBIAN of the source: dS/dU, n_vars x n_vars matrix of expressions
        (rows[r][c] = dS_r/dU_c, as a function of cons / prims / aux). Emitted as
        ``jacobian(U, aux, J)`` on the generated SOURCE brick, forwarded by CompositeModel
        (C++ trait ``HasSourceJacobian``): the Newton of the implicit source (IMEX /
        SourceImplicitBE) uses it INSTEAD of finite differences -- exactness (no more
        fd_eps noise) and saved source evaluations. REQUIRES m.source. WITHOUT a call: historical
        finite differences, bit-identical."""
        self._src_jac = [[_wrap(e) for e in row] for row in rows]
