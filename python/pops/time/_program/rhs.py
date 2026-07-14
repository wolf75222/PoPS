"""Typed public RHS composition and its private lowering seam."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time._program.value_validation import rate_space_for
from pops.time.values import ProgramValue, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramRhs(_ProgramBase):
    """Compose a rate from typed terms, then lower it to primitive flux/source selectors."""

    def rhs(self, name: Any = None, state: Any = None, fields: Any = None, *,
            terms: Any) -> Any:
        """Build ``R`` from typed RHS terms.

        ``Flux()`` selects the default conservative divergence, while ``Flux(handle)`` selects an
        exact named grid operator. ``DefaultSource()`` explicitly selects the block model's
        default/composite source. Named sources are exact ``OperatorHandle`` values returned by
        ``m.source_term``, directly or wrapped in ``SourceTerm(handle)`` / ``LocalTerm(handle)``.
        Every selected operator retains its owner identity in the IR until lowering.
        """
        state = _resolve_handle(state)
        from pops.time._rhs_terms import lower_rhs_terms
        flux, sources, source_handles, fluxes, flux_handles = lower_rhs_terms(
            self, terms, state=state)
        result = self._rhs_primitive(
            name=name, state=state, fields=fields, flux=flux, sources=sources, fluxes=fluxes)
        if not source_handles and not flux_handles:
            return result
        attrs = dict(result.attrs)
        if source_handles:
            attrs["source_handles"] = source_handles
        if flux_handles:
            attrs["flux_handles"] = flux_handles
        return self._replace_value(result, attrs=attrs)

    def _rhs_primitive(self, name: Any = None, state: Any = None, fields: Any = None,
                       flux: Any = True, sources: Any = None, fluxes: Any = None) -> Any:
        """Private projection of typed terms to runtime-local flux/source selector tokens."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if isinstance(name, ProgramValue):
            raise ValueError("rhs: pass state=/fields= by keyword (first arg is the debug name)")
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("rhs: a State value is required (state=...)")
        if fields is not None and not (
                isinstance(fields, ProgramValue) and fields.vtype == "fields"):
            raise ValueError("rhs: fields must be a FieldContext from solve_fields")
        field_context = None
        if fields is not None:
            from pops.time.field_context import require_field_read
            field_context = require_field_read(fields, state, "rhs")
        src = list(sources) if sources is not None else None
        attrs = {
            "flux": bool(flux),
            "sources": src,
            "fluxes": list(fluxes) if fluxes else None,
        }
        inputs = (state, fields) if fields is not None else (state,)
        return self._new(
            "rhs", "rhs", inputs, attrs, name, state.block,
            space=rate_space_for(state.space), field_context=field_context)


__all__ = ["_ProgramRhs"]
