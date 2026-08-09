"""Authoring mixin: variable / primitive / aux declarations.

Split out of the monolithic :class:`~pops.physics._model.HyperbolicModel` so no
single file exceeds the Spec-4 500-line bound. The mixin holds only methods; the
instance attributes they touch (``cons_names`` / ``prim_defs`` / ``_provider_components``
/ ``prim_state`` / ``cons_from`` / ``prim_roles``) are
created by ``HyperbolicModel.__init__`` (see :mod:`pops.physics._model`).

Imports only :mod:`pops._ir` and the package-level aux constants: this layer is
codegen-free and ``_pops``-free (Spec-4 import-graph rule).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._ir import Var, _wrap

if TYPE_CHECKING:
    from ._model_contract import _HyperbolicModel
else:
    _HyperbolicModel = object


def _require_provider_component_name(value: Any, *, where: str = "provider") -> str:
    """Validate one source-level component spelling used as a generated C++ local.

    Identity and storage are resolved exclusively by the ProviderPack.  This
    small authoring check only prevents an invalid formula local from reaching
    the C++ compiler.
    """
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(
            "%s name must be a valid identifier (letters/digits/_, without a leading digit)"
            % where
        )
    return value


class _VariablesMixin(_HyperbolicModel):
    """Conservative / primitive / auxiliary variable declarations."""

    def cons(self, name: Any) -> Any:
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names: Any, roles: Any = None) -> Any:
        """Declare the conservative variables. @p roles (optional): list of the same length explicitly
        setting the physical role of each component (string 'Density'/'MomentumX'... or None
        to fall back on the canonical mapping of the name); useful for a non-standard layout where the names
        do not suffice to deduce the meaning. Without roles, the canonical name -> role mapping applies."""
        if roles is not None and len(roles) != len(names):
            raise ValueError("conservative_vars: %d roles for %d variables" % (len(roles), len(names)))
        self.cons_roles = list(roles) if roles is not None else None
        return tuple(self.cons(n) for n in names)

    def primitive(self, name: Any, expr: Any) -> Any:
        """Define a primitive by its formula (in terms of the cons / previous primitives)."""
        self.prim_defs[name] = _wrap(expr)
        return Var(name, "prim")

    def aux(self, name: Any) -> Any:
        """Declare one ordinary auxiliary input/output by local identifier.

        The compiler resolves its owner, space, contract and compact native
        slot from the canonical ``ProviderPack``; no spelling has a reserved
        physical meaning.
        """
        name = _require_provider_component_name(name, where="provider")
        if name not in self._provider_components:
            self._provider_components.append(name)
            self._invalidate_authoring_views()
        return Var(name, "aux")

    def _aux_locals_lines(self) -> Any:
        """C++ locals read through exact compact provider-pack slots."""
        from pops._ir.visitors import _dependencies

        used = _dependencies(self._source or ())
        return [
            "    const pops::Real %s = a.template flux_provider<%d>();"
            % (name, self._consumer_provider_slot("source_default", name))
            for name in self._provider_components if name in used
        ]

    def _flux_provider_locals_lines(self) -> Any:
        """C++ locals read from the exact physical-flux provider protocol.

        This emits no global auxiliary storage access. ``BoundFluxProviders<Model>`` implements
        ``flux_provider<ConsumerSlot>()`` over the exact physical-flux consumer plan, so generated
        physical laws keep one formula while the finite-volume route consumes only its resolved pack.
        """
        from pops._ir.visitors import _dependencies

        expressions = [
            *[expr for values in self._flux.values() for expr in values],
            *[expr for values in self._eig.values() for expr in values],
        ]
        if self._wave_speeds is not None:
            expressions.extend(
                expr for values in self._wave_speeds.values() for expr in values
            )
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            expressions.extend(
                expr
                for matrix in self._ws_jacobian["rows"].values()
                for row in matrix
                for expr in row
            )
        used = _dependencies(expressions)
        return [
            "    const pops::Real %s = a.template flux_provider<%d>();"
            % (name, self._physical_flux_consumer_slot(name))
            for name in self._provider_components if name in used
        ]

    def _reads_aux(self) -> bool:
        """True if a formula reads an auxiliary provider."""
        return bool(self._provider_components)

    def _total_n_aux(self) -> Any:
        """Total width of the resolved compact provider-pack channel."""
        pack = getattr(self, "_auxiliary_provider_pack", None)
        if pack is None:
            raise ValueError(
                "auxiliary ProviderPack is absent; compile through the canonical Module authority"
            )
        return len(pack)

    def _physical_flux_consumer_slot(self, name: Any) -> int:
        """Return a physical-flux *consumer* slot, never a carrier slot."""
        plan = getattr(self, "_component_flux_consumer_plan", None)
        if plan is None:
            raise ValueError(
                "physical-flux consumer plan is absent; compile through Module"
            )
        checked = _require_provider_component_name(name, where="provider")
        matches = [
            row for row in plan if row["key"]["component"] == checked
        ]
        if len(matches) != 1:
            detail = "absent" if not matches else "ambiguous"
            raise ValueError(
                "auxiliary component %r is %s from the physical-flux consumer plan"
                % (checked, detail)
            )
        return matches[0]["consumer_slot"]

    def _consumer_provider_slot(self, consumer: Any, name: Any) -> int:
        """Return a named operator's local provider slot."""
        plans = getattr(self, "_component_operator_consumer_plans", None)
        if plans is None:
            raise ValueError("auxiliary consumer plans are absent; compile through Module")
        plan = plans.get(consumer)
        if plan is None:
            raise ValueError("auxiliary consumer plan %r is absent" % consumer)
        checked = _require_provider_component_name(name, where="provider")
        matches = [row for row in plan if row["key"]["component"] == checked]
        if len(matches) != 1:
            detail = "absent" if not matches else "ambiguous"
            raise ValueError(
                "auxiliary component %r is %s from consumer plan %r"
                % (checked, detail, consumer)
            )
        return matches[0]["consumer_slot"]

    def set_primitive_state(self, *vars_or_names: Any, roles: Any = None) -> None:
        """Declares the ORDERED layout of the primitive state (Prim): component names, in order.
        Necessary for the brick codegen (to_primitive fills Prim in this order). Each name must
        be a conservative variable or an already-defined primitive. @p roles (optional): same
        convention as conservative_vars (explicit per-component override, None = canonical mapping)."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]
        if roles is not None and len(roles) != len(self.prim_state):
            raise ValueError("set_primitive_state : %d roles for %d variables"
                             % (len(roles), len(self.prim_state)))
        self.prim_roles = list(roles) if roles is not None else None

    def set_conservative_from(self, exprs: Any) -> None:
        """Formulas of the conservative state as a function of the primitives (one per conservative
        variable, in conservative_vars order). Used to generate to_conservative: the DSL cannot invert
        the primitives symbolically, so the user provides the inverse explicitly."""
        self.cons_from = [_wrap(e) for e in exprs]

    @property
    def n_vars(self) -> int: return len(self.cons_names)
