"""Type-only contracts for the runtime coupler mixins (ADC-628, static analysis only).

The concrete runtime couplers (:class:`pops.runtime._system.System` and
:class:`pops.runtime._amr_system.AmrSystem`) assemble their behaviour from topical mixins
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
    _step_strategy: Any
    _step_transaction_plan: Any
    _lifecycle: str
    _bound_snapshot: Any
    _last_run_manifest: Any
    _last_run_identity: Any
    _last_restart_identity: Any
    _temporal_restart_state: Any
    _execution_context: Any

    def add_equation(self, *args: Any, **kwargs: Any) -> Any: ...
    def add_block(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_poisson(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_poisson_native(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_state(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_density(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_magnetic_field(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_aux_field(self, *args: Any, **kwargs: Any) -> Any: ...
    def install_program(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_program_params(self, *args: Any, **kwargs: Any) -> Any: ...
    def nx(self) -> int: ...
    def time(self) -> float: ...
    def macro_step(self) -> int: ...
    @property
    def bound_snapshot(self) -> Any: ...
    @property
    def last_run_identity(self) -> Any: ...
    def _finalize_bind(self, *args: Any, **kwargs: Any) -> Any: ...
    def _checkpoint_identities(self) -> tuple[Any, Any, Any]: ...
