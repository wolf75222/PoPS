"""Internal helpers for :mod:`pops.physics.board`.

These methods are split out of ``board.py`` so the public authoring class stays under the
per-file size budget while still building a real ``pops.model.Module`` directly.
"""
from .. import math as _bm
from .board_handles import FieldHandle


class _BoardInternalsMixin:
    """Validation and expression-normalisation helpers for the Module-native board facade."""

    def _validate_riemann_capabilities(self, kind, pressure, wave_speeds):
        """Reject a model lacking the capabilities required by the selected Riemann solver."""
        state = self._require_state("riemann")
        roles = set(state.roles.values())
        has_pressure = ("p" in self._primitive_defs) or (pressure is not None)
        fluid = {"Density", "MomentumX", "MomentumY"}
        if kind in ("hllc", "roe"):
            if not has_pressure:
                raise ValueError(
                    "riemann %s requires model capability 'pressure' for state %r: declare a "
                    "primitive m.primitive('p', ...) or pass m.riemann(..., pressure=...)"
                    % (kind.upper(), self._state_name()))
            missing = fluid - roles
            if missing:
                raise ValueError(
                    "riemann %s requires model capability 'hllc_star_state' for state %r: the "
                    "fluid roles %s are needed (declare m.state(..., roles={...})); missing %s"
                    % (kind.upper(), self._state_name(), sorted(fluid), sorted(missing)))
        elif kind == "hll":
            if wave_speeds is None and self._module._eigenvalues is None:
                raise ValueError(
                    "riemann HLL requires model capability 'wave_speeds': declare m.flux(..., "
                    "waves=...) or pass m.riemann('hll', wave_speeds=...)")

    def _state_name(self):
        return next(iter(self._states), "U")

    def _destructure_rate(self, rhs):
        """Split a rate right-hand side into ``(flux, [source names])``."""
        terms = _bm._as_rate(rhs)._rate_terms()
        flux = False
        sources = []
        for kind, payload, sign in terms:
            if kind == "flux":
                if sign >= 0:
                    raise ValueError(
                        "a rate equation flux term must be -div(F) (negative); "
                        "write 'ddt(U) == -div(F) + ...'")
                flux = True
            elif kind == "source":
                if sign <= 0:
                    raise ValueError(
                        "a rate equation source term %r must be added (positive sign)"
                        % (payload.name,))
                sources.append(payload.reg_name)
            else:  # pragma: no cover - defensive
                raise ValueError("unknown rate term kind %r" % (kind,))
        return flux, sources

    def _to_expr(self, node):
        """Resolve a board node to a ``pops.ir`` expression in this model's context."""
        if isinstance(node, _bm.Partial):
            field = node.field
            fname = field.name if isinstance(field, FieldHandle) else str(field)
            aux_name = self._gradient_aux(fname, node.axis)
            expr = self.aux(aux_name)
            if node.scale != 1.0:
                expr = node.scale * expr
            return expr
        if isinstance(node, _bm.Gradient):
            raise TypeError("a gradient is a vector; use grad(field).x / .y")
        if isinstance(node, _bm.Laplacian):
            raise TypeError("a laplacian only appears as a field-solve operator")
        return node

    def _require_state(self, who):
        if self._state_space is None:
            raise ValueError("%s requires a declared state; call m.state(...) first" % who)
        return self._state_space

    def _ensure_default_fields(self):
        if self._default_field_space is None:
            self._default_field_space = self._module.field_space(
                "fields", ("phi", "grad_x", "grad_y"))
        return self._default_field_space

    @staticmethod
    def _gradient_aux(field_name, axis):
        """Canonical gradient aux name of ``field_name`` along ``axis``."""
        if field_name == "phi":
            return "grad_x" if axis == 0 else "grad_y"
        return "%s_grad_%s" % (field_name, "x" if axis == 0 else "y")
