"""Type-only contracts for the runtime coupler mixins (ADC-628, static analysis only).

The concrete runtime couplers (:class:`pops.runtime.system.System` and
:class:`pops.runtime.amr_system.AmrSystem`) assemble their behaviour from topical mixins
(``_system_*`` / ``_amr_system_*``). Each mixin reads instance attributes (chiefly the native
facade ``self._s``) and sibling methods that live on the composed class, not on the mixin itself,
so a static checker cannot resolve them from the mixin in isolation.

This module declares those shared surfaces ONCE. Each mixin lists the relevant contract as a
``TYPE_CHECKING``-only base; at runtime the guard is false and the base collapses to ``object``,
so the class layout, the MRO and every runtime behaviour are unchanged (the contract is never
imported, instantiated or introspected outside a type checker). The bodies are ``...`` stubs:
they describe shapes, never behaviour.
"""
from __future__ import annotations

from typing import Any


class _System:
    """The :class:`System` instance surface the ``_system_*`` mixins share.

    ``_s`` is the native ``_pops.System`` facade (its ``__getattr__`` resolves any delegated
    accessor to ``Any``); the sibling methods below are defined across the other install / aux /
    diagnostics / io / lifecycle mixins and the concrete ``System`` body. Declared here so each
    mixin type-checks against the composed instance without a runtime base change.
    """

    _s: Any
    _aux_field_index: dict
    _output_policies: list
    _program_cadence_cfl: Any
    _lifecycle: str
    _bound_snapshot: Any

    def add_equation(self, *args: Any, **kwargs: Any) -> Any: ...
    def add_block(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_poisson(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_state(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_density(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_magnetic_field(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_aux_field(self, *args: Any, **kwargs: Any) -> Any: ...
    def install_program(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_program_cadence(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_program_params(self, *args: Any, **kwargs: Any) -> Any: ...
    def nx(self) -> int: ...
    def _finalize_bind(self, *args: Any, **kwargs: Any) -> Any: ...
