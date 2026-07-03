"""Type-only contracts for the authoring mixins (ADC-628, static analysis only).

The concrete authoring classes (:class:`pops.physics.model.HyperbolicModel`,
:class:`pops.physics.board.Model`, :class:`pops.physics.facade.Model`) assemble their
behaviour from topical mixins (``_authoring_*`` / ``_board_*`` / ``_facade_*``). Each mixin
reads instance attributes and sibling methods that live on the composed class, not on the
mixin itself, so a static checker cannot resolve them from the mixin in isolation.

This module declares those shared surfaces ONCE. Each mixin lists the relevant contract as a
``TYPE_CHECKING``-only base; at runtime the guard is false and the base collapses to ``object``,
so the class layout, the MRO and every runtime behaviour are unchanged (the contract is never
imported, instantiated or introspected outside a type checker). The bodies are ``...`` stubs:
they describe shapes, never behaviour.
"""
from __future__ import annotations

from typing import Any


class _HyperbolicModel:
    """The :class:`HyperbolicModel` instance surface the ``_authoring_*`` mixins share.

    Every attribute is created by ``HyperbolicModel.__init__``; the methods/properties are
    defined across the sibling authoring mixins. Declared here so each mixin type-checks against
    the composed instance without a runtime base change.
    """

    name: str
    cons_names: Any
    cons_roles: Any
    cons_from: Any
    prim_defs: Any
    prim_roles: Any
    prim_state: Any
    aux_names: Any
    aux_extra_names: Any
    gamma: Any
    _flux: Any
    _flux_terms: Any
    _fluxes: Any
    _eig: Any
    _wave_speeds: Any
    _ws_jacobian: Any
    _source: Any
    _source_terms: Any
    _linear_sources: Any
    _rate_operators: Any
    _elliptic: Any
    _elliptic_fields: Any
    _proj: Any
    _roe: Any
    _roe_rows: Any
    _roe_jacobian: Any
    _riemann_hook_forms: Any
    _hllc: Any
    _src_freq: Any
    _src_jac: Any
    _stab_dt: Any
    _stab_speed: Any
    _invariants: Any
    _aliases: Any
    _to_expr: Any
    params: Any
    module: Any

    @property
    def n_vars(self) -> int: ...
    def _env(self, U: Any, aux: Any) -> Any: ...
    def flux_jacobian(self, dir: Any) -> Any: ...


class _BoardModel:
    """The :class:`pops.physics.board.Model` instance surface the ``_board_*`` mixins share."""

    name: str
    _species: Any
    _states: Any
    _fields: Any
    _fluxes: Any
    _sources: Any
    _field_solvers: Any
    _multi_module: Any
    _aliases: Any
    _invariants: Any
    _m: Any
    _dsl: Any

    @property
    def module(self) -> Any: ...
    def _to_expr(self, node: Any) -> Any: ...


class _FacadeModel:
    """The :class:`pops.physics.facade.Model` instance surface the ``_facade_*`` mixins share."""

    name: str
    module: Any
    backend_auto_reason: Any
    params: Any
    _m: Any
    _dsl: Any
