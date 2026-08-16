"""System auxiliary-data, disc-domain, and primitive-state methods.

Auxiliary data crosses the runtime only through a complete
``ComponentKey``.  Python neither assigns a carrier component nor attaches a
physical-name special case: the sealed native ProviderPack/registry owns the
address and accepts only declared ``InputAux`` routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object


class _SystemAuxState(_System):
    """Exact auxiliary component + disc domain + primitive state methods."""

    @staticmethod
    def _auxiliary_key(value: Any) -> Any:
        """Normalize public ComponentKey data without a bare-name fallback."""
        from pops.model.provider_pack import ComponentKey

        if type(value) is ComponentKey:
            return value
        if isinstance(value, dict):
            try:
                return ComponentKey(**value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "auxiliary component mapping must carry owner_qid, space_kind, space_name, component"
                ) from exc
        raise TypeError("auxiliary input requires an exact pops.model.ComponentKey")

    def stage_auxiliary_input(self, key: Any, values: Any) -> None:
        """Stage values for one declared owner-qualified ``InputAux`` component.

        The native registry validates that ``key`` is an input producer, owns
        its group/component storage address, and publishes it transactionally
        at the next auxiliary refresh.  ``DerivedAux`` and field outputs are
        intentionally rejected by native code rather than overwritten here.
        """
        import numpy as np

        exact = self._auxiliary_key(key)
        values_array = np.asarray(values, dtype=float)
        self._s.stage_auxiliary_input(
            exact.owner_qid, exact.space_kind, exact.space_name, exact.component,
            values_array.reshape(-1),
        )

    def auxiliary_component(self, key: Any) -> Any:
        """Read a declared owner-qualified auxiliary component after publication."""
        exact = self._auxiliary_key(key)
        return self._s.auxiliary_component(
            exact.owner_qid, exact.space_kind, exact.space_name, exact.component,
        )

    def set_geometry_mode(self, mode: Any) -> Any:
        """Switch ONLY the embedded-boundary transport mode ('none'|'staircase'|'cutcell') without
        redefining the level set. A mode != 'none' requires an analytic level set already set. Setting
        back to 'none' restores the full Cartesian transport (bit-identical).

        ``mode`` must be a typed :class:`pops.mesh.masks.TransportMask`; strings are rejected."""
        from pops.runtime._lifecycle import guard_assembling

        guard_assembling(self, "set_geometry_mode")  # frozen once pops.bind completes (ADC-592)
        from pops.mesh.masks import lower_transport_mask

        self._s.set_geometry_mode(lower_transport_mask(mode))

    def embedded_boundary_mask(self) -> Any:
        """Return the active-cell mask for any installed embedded LevelSet geometry."""
        return self._s.embedded_boundary_mask()

    def set_primitive_state(self, name: Any, **prims: Any) -> Any:
        """Initialize a block from its PRIMITIVE variables, named (rho/u/v/p ...):

            sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)

        Each primitive uses the exact ranked NumPy shape (native axes reversed). The expected names
        are those of variable_names(name, "primitive") (the order of the block model). The
        component-major array is
        assembled in that order, then CONVERTED to conservative variables by the block model (on the
        C++ side: compressible E = p/(g-1) + 1/2 rho|v|^2; isothermal rho u; scalar identity) and written
        to the state. Ergonomic counterpart of set_density (which only sets the density, leaving it at rest).

        Raises a clear error if a primitive name is unknown for the block, or if one is missing."""
        import numpy as np  # local: numpy is only required for this host assembly

        names = list(self._s.variable_names(name, "primitive"))
        shape = tuple(reversed(self.spatial_shape()))
        unknown = [k for k in prims if k not in names]
        if unknown:
            raise ValueError(
                "set_primitive_state: unknown primitive(s) %r for block '%s'; "
                "expected primitives: %r" % (unknown, name, names)
            )
        missing = [k for k in names if k not in prims]
        if missing:
            raise ValueError(
                "set_primitive_state: missing primitive(s) %r for block '%s'; "
                "provide all the primitives: %r" % (missing, name, names)
            )
        # Assemble component-major state in model order, not kwargs order.
        prim = np.empty((len(names), *shape), dtype=np.float64)
        for c, nm in enumerate(names):
            arr = np.asarray(prims[nm], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(
                    "set_primitive_state: primitive '%s' of shape %r, expected %r"
                    % (nm, tuple(arr.shape), shape)
                )
            prim[c] = arr
        self._s.set_primitive_state(name, prim)

    def get_primitive_state(self, name: Any) -> Any:
        """Read the conservative state of a block and return it in PRIMITIVE variables (diagnostic):

            P = sim.get_primitive_state("electrons")   # {"rho": ..., "u": ..., "v": ..., "p": ...}

        Returns a dict {primitive_name: ranked array} in the order of variable_names(name,
        "primitive"). Inverse of set_primitive_state (exact round-trip to machine precision, the
        model cons <-> prim conversion being consistent)."""
        names = list(self._s.variable_names(name, "primitive"))
        prim = self._s.get_primitive_state(name)  # component-major ranked state
        return {nm: prim[c] for c, nm in enumerate(names)}
