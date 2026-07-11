"""Authoring mixin: runtime parameter collection and hook-form validation.

Collects ``RuntimeParamRef`` nodes across the model formulas, assigns their flat
indices, and emits the generated ``pops::RuntimeParams`` member; also validates
arbitrary Riemann hook formulas. Methods only; touched attributes come from
``HyperbolicModel.__init__``. Codegen-free and ``_pops``-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.ir import Expr, _wrap  # noqa: F401  -- _validate_hook_form isinstance checks
from pops.ir.values import RuntimeParamRef, set_runtime_param_indices  # noqa: F401
from pops.ir.visitors import _children  # noqa: F401

from .aux import _K_MAX_RUNTIME_PARAMS, max_runtime_params  # noqa: F401 -- literal + _pops-preferring

if TYPE_CHECKING:
    from ._model_contract import _HyperbolicModel
else:
    _HyperbolicModel = object


class _RuntimeParamsMixin(_HyperbolicModel):
    """Runtime parameter discovery, index assignment, and hook validation."""

    def _all_exprs(self) -> Any:
        """All the Expr of the model (primitives, flux, eigenvalues, source, elliptic,
        cons_from). Used to discover the RuntimeParamRef nodes hidden in the tree."""
        out = list(self.prim_defs.values())
        for d in ("x", "y"):
            out += self._flux.get(d, [])
            out += self._eig.get(d, [])
        if self._wave_speeds is not None:  # explicit signed speeds: runtime params included
            for d in ("x", "y"):
                out += list(self._wave_speeds[d])
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):  # jacobian entries: runtime params included
                out += [e for row in self._ws_jacobian["rows"][d] for e in row]
        if self._source is not None:
            out += [_wrap(e) for e in self._source]
        # NAMED sources / linear sources / fluxes (ADC-510): a runtime param read ONLY by a named
        # m.source_term / m.linear_source / m.flux_term (which a compiled time Program lowers directly,
        # Spec 5 C5) must also get a stable index, so params.get(idx) is emitted (not an index -1 raise).
        for exprs in (getattr(self, "_source_terms", {}) or {}).values():
            out += [_wrap(e) for e in exprs]
        for rows in (getattr(self, "_linear_sources", {}) or {}).values():
            for row in rows:
                out += [_wrap(e) for e in row]
        for term in (getattr(self, "_flux_terms", {}) or {}).values():
            for d in ("x", "y"):
                out += [_wrap(e) for e in term.get(d, [])]
        if self.cons_from is not None:
            out += list(self.cons_from)
        if self._elliptic is not None:
            out.append(self._elliptic)
        if self._roe_rows is not None:  # Roe rows provided: discover their runtime params (via StateRef)
            out += self._roe_rows["x"] + self._roe_rows["y"]
        return out

    def runtime_param_nodes(self) -> Any:
        """RuntimeParamRef nodes PRESENT in the formulas, deduplicated by name (the same param may
        appear several times but shares the SAME node object). Order SORTED by name (stable index
        = position in this list, mirror of RuntimeParams on the C++ side)."""
        seen = {}

        def walk(e: Any) -> None:
            if isinstance(e, RuntimeParamRef):
                seen.setdefault(e.name, e)
                return
            for c in _children(e):
                walk(c)

        for e in self._all_exprs():
            walk(e)
        return [seen[k] for k in sorted(seen)]

    def assign_runtime_indices(self) -> Any:
        """Assigns to each RuntimeParamRef its STABLE index (sorted order of names) and returns the
        ordered list of nodes. CALLED before any brick codegen: without this call, to_cpp() would raise
        (index -1). Idempotent (reassigns the same indices). Rejects a model exceeding the C++ bound
        kMaxRuntimeParams EARLY, with a user-facing error naming the limit, the count, and the offending
        params -- the fixed-size device array RuntimeParams::values[kMaxRuntimeParams] would otherwise be
        read out of bounds on device (no bound check on the hot path get())."""
        nodes = self.runtime_param_nodes()
        limit = max_runtime_params()  # _pops.__max_runtime_params__ when present, else literal 32
        if len(nodes) > limit:
            names = ", ".join(repr(node.name) for node in nodes)
            raise ValueError(
                "model '%s': %d runtime parameters exceed kMaxRuntimeParams=%d "
                "(include/pops/runtime/config/runtime_params.hpp); the fixed-size device array would "
                "overflow. Reduce the number of runtime params or promote some to kind='const'. "
                "Declared runtime params: %s"
                % (self.name, len(nodes), limit, names))
        set_runtime_param_indices({node.name: k for k, node in enumerate(nodes)})
        return nodes

    def _runtime_params_member(self) -> str:
        """C++ line declaring the RuntimeParams member of a generated brick, initialized to the
        neutral carrier values.  Runtime defaults belong to the resolved BindSchema and are installed
        before execution; baking them into generated C++ would make a bind-time value change alter the
        artifact identity. Empty string if the model has no runtime param."""
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        vals = ", ".join("pops::Real(0)" for _ in nodes)
        return ("  pops::RuntimeParams params{%d, {%s}};  // defaults installed from BindSchema\n"
                % (len(nodes), vals))

    def has_runtime_params(self) -> bool:
        """True if at least one formula reads a runtime parameter (kind='runtime')."""
        return bool(self.runtime_param_nodes())

    def _validate_hook_form(self, hook: Any, form: Any, allow_aux: bool = True) -> None:
        """Reject an arbitrary-formula Riemann hook (ADC-456) that references a quantity the model
        cannot provide -- the same dependency rule as :meth:`check`, surfaced as a clear capability
        error. @p allow_aux: a single-state hook (e.g. pressure(U)) takes no Aux parameter, so an
        aux dependency is also a missing capability there."""
        known = set(self.cons_names) | set(self.prim_defs)
        if allow_aux:
            known |= set(self.aux_names) | set(self.aux_extra_names)
        missing = sorted(form.deps() - known)
        if missing:
            raise ValueError(
                "riemann hook %r references undeclared quantity %s: the formula needs model "
                "capabilities %s that are not provided (declare them, or use the role-derived "
                "default)" % (hook, missing, missing))
