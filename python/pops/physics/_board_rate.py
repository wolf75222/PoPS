"""Rate-equation authoring for the physics blackboard facade."""
from __future__ import annotations

from typing import Any

from .. import math as _bm
from ._board_contract import atomic_attrs, normalize_sequence, require_name
from .board_handles import FluxHandle, SourceHandle, StateHandle, _safe_name


class _RateAuthoringMixin:
    """Build and validate explicit finite-volume/rate equations."""

    def rate(self, name: Any, *, equation: Any) -> Any:
        reg = _safe_name(name)
        if not isinstance(equation, _bm.Equation):
            raise TypeError("rate expects an equation 'ddt(U) == -div(F) + sources'")
        if not isinstance(equation.lhs, _bm.TimeDerivative):
            raise ValueError("rate left-hand side must be ddt(U) / rate(U)")
        state = equation.lhs.state
        if (not isinstance(state, StateHandle)
                or state.owner_path != self.owner_path
                or self._states.get(state.name) != state):
            raise ValueError(
                "rate left-hand side must reference a StateHandle declared by this physics model; "
                "got %r" % (state,))
        if self._multi_module is not None:
            state = self._species_handle("rate", name, state)
        flux, sources = self._destructure_rate(equation.rhs)
        if self._multi_module is not None:
            fluxes = () if flux is None else (
                self._multi_module.operator_handle(flux.name),)
            source_refs = tuple(
                self._multi_module.operator_handle(source.reg_name) for source in sources)
            result = self._multi_module.rate_operator(
                reg,
                state_space=state.space,
                flux=flux is not None,
                fluxes=fluxes,
                sources=source_refs,
            )
            self._rate_contracts[result] = {
                "state": state,
                "flux": flux,
                "sources": tuple(sources),
            }
            self._invalidate_authoring_views()
            return result
        hyp = self._dsl._m
        with atomic_attrs((hyp, "_rate_operators"), (self, "_rate_contracts")):
            self._dsl.rate_operator(
                reg, flux=flux is not None, sources=[source.reg_name for source in sources])
            result = self._registered_operator_handle(reg)
            self._rate_contracts[result] = {
                "state": state,
                "flux": flux,
                "sources": tuple(sources),
            }
        return result

    def rate_contract(self, rate: Any) -> dict[str, Any]:
        """Return the exact physical dependencies of a registered rate handle."""
        try:
            contract = self._rate_contracts[rate]
        except (KeyError, TypeError):
            raise ValueError("rate handle is not registered by this Model") from None
        return {"state": contract["state"], "flux": contract["flux"],
                "sources": tuple(contract["sources"])}

    def finite_volume_rate(self, name: Any, flux: Any = None, riemann: Any = None,
                           reconstruction: Any = None, sources: Any = ()) -> Any:
        """Declare a native finite-volume rate with owner-checked physical terms."""
        reg = _safe_name(name)
        if flux is not None and (not isinstance(flux, FluxHandle)
                or flux.owner_path != self.owner_path
                or self._fluxes.get(flux.name) != flux):
            raise ValueError(
                "finite_volume_rate flux must be a FluxHandle declared by this physics model; "
                "got %r" % (flux,))
        source_handles = []
        for source in normalize_sequence(sources, "finite_volume_rate sources"):
            if not isinstance(source, SourceHandle):
                raise TypeError("finite_volume_rate sources must contain SourceHandle objects")
            if (source.owner_path != self.owner_path
                    or self._sources.get(source.reg_name) != source):
                raise ValueError(
                    "finite_volume_rate source handle %r belongs to another physics model"
                    % (source.name,))
            source_handles.append(source)
        if riemann is not None:
            scheme = getattr(riemann, "scheme", riemann)
            self._validate_riemann_capabilities(
                require_name(scheme, "Riemann scheme").lower(), pressure=None, wave_speeds=None)

        hyp = self._dsl._m
        with atomic_attrs(
                (hyp, "aux_names"), (hyp, "aux_extra_names"), (hyp, "_hllc"), (hyp, "_roe"),
                (hyp, "_riemann_hook_forms"), (hyp, "_rate_operators"),
                (self, "_riemann"), (self, "_reconstruction"),
                (self, "_rate_contracts")):
            if riemann is not None:
                self.riemann(riemann)
            self._reconstruction = reconstruction
            self._dsl.rate_operator(
                reg, flux=flux is not None,
                sources=[source.reg_name for source in source_handles])
            result = self._registered_operator_handle(reg)
            state = next(iter(self._states.values()))
            self._rate_contracts[result] = {
                "state": state,
                "flux": flux,
                "sources": tuple(source_handles),
            }
        return result

    def _destructure_rate(self, rhs: Any) -> Any:
        terms = _bm._as_rate(rhs)._rate_terms()
        flux = None
        sources = []
        for kind, payload, sign in terms:
            if kind == "flux":
                if (not isinstance(payload, FluxHandle)
                        or payload.owner_path != self.owner_path
                        or self._fluxes.get(payload.name) != payload):
                    raise ValueError(
                        "a rate equation flux must be a FluxHandle declared by this physics model; "
                        "got %r" % (payload,))
                if sign != -1:
                    raise ValueError(
                        "a rate equation flux term must be -div(F) with exact unit coefficient -1; "
                        "the current finite-volume rate lowering cannot silently discard a scale "
                        "%r (absorb it into the declared flux or author a distinct operator)" % sign)
                if flux is not None:
                    raise ValueError(
                        "a rate equation may contain one -div(F) term in the current finite-volume "
                        "lowering; multiple divergence terms need an explicitly combined flux")
                flux = payload
            elif kind == "source":
                if (not isinstance(payload, SourceHandle)
                        or payload.owner_path != self.owner_path
                        or self._sources.get(payload.reg_name) != payload):
                    raise ValueError(
                        "a rate equation source must be a SourceHandle declared by this physics "
                        "model; got %r" % (payload,))
                if sign != 1:
                    raise ValueError(
                        "a rate equation source term %r must have the exact unit coefficient +1; "
                        "the current source lowering cannot silently discard scale %r"
                        % (payload.name, sign))
                sources.append(payload)
            else:  # pragma: no cover
                raise ValueError("unknown rate term kind %r" % (kind,))
        return flux, sources


__all__ = ["_RateAuthoringMixin"]
