"""Rate-equation authoring for the physics blackboard facade."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import math as _bm
from .._ir.balance import Balance, BalanceView
from ._board_contract import atomic_attrs, normalize_sequence, require_name
from .board_handles import FluxHandle, RateHandle, SourceHandle, StateHandle, _safe_name

if TYPE_CHECKING:
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


class _RateAuthoringMixin(_BoardModel):
    """Retain physical equations and derive checked finite-volume adapters."""

    def diffusive_flux(self, name: Any, *, state: Any, value: Any, boundaries: Any = None) -> Any:
        """Declare a constitutive A*grad(W) flux without choosing its discrete gradient."""
        from .diffusion import declare_diffusive_flux
        return declare_diffusive_flux(self, name, state=state, value=value, boundaries=boundaries)

    def drift_flux(self,name: Any,*,state: Any,mobility: Any,potential: Any,boundaries: Any=None) -> Any:
        """Declare the physical drift -mobility*n*grad(potential), without fitting a stencil."""
        from .drift_diffusion import declare_drift_flux
        return declare_drift_flux(self,name,state=state,mobility=mobility,potential=potential,boundaries=boundaries)

    def rate(self, name: Any, *, equation: Any) -> Any:
        reg = _safe_name(name)
        if not isinstance(equation, _bm.Equation):
            raise TypeError("rate expects an equation 'ddt(U) == signed physical terms'")
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
        terms = self._destructure_rate(equation.rhs, target=state)
        from pops.model import OperatorHandle
        handle = OperatorHandle(reg, kind="local_rate", owner=self.owner_path,
                                signature=self._balance_signature(state, terms))
        balance = Balance.capture(handle, state, terms, equation.lhs.accumulation)
        return self._register_balance_view(handle, balance.full_view())

    def _ensure_balance_registry(self) -> None:
        if not hasattr(self, "_retained_rates"):
            self._retained_rates = {}
            self._selected_balances = {}

    def _balance_signature(self, state: Any, terms: Any) -> Any:
        from pops.model import Signature, Rate
        registry = (self._multi_module.operator_registry() if self._multi_module is not None
                    else self._dsl._m.operator_registry())
        inputs = [state.space]
        for kind, payload, _coefficient in terms:
            if kind in {"diffusion", "drift"}:
                for space in payload.law.inputs:
                    if space not in inputs:
                        inputs.append(space)
                continue
            if kind == "projection":
                for space in payload.application.operator.signature.inputs:
                    if space not in inputs:
                        inputs.append(space)
                continue
            if kind == "source":
                operator = registry.get(payload.reg_name)
            elif kind == "flux" and self._multi_module is not None:
                binding = self._multi_module.operator_binding(payload)
                operator = registry.get(binding.registered_operator_name)
            elif kind == "flux":
                operator = registry.get(payload.reg_name)
            else:
                continue
            for space in operator.signature.inputs:
                if getattr(space, "kind", None) == "state":
                    if space != state.space:
                        # The sole board state is an explicit alias of the DSL's
                        # U storage. Authenticate every physical metadata field;
                        # neither a reused name nor an equal array shape suffices.
                        actual = {key: value for key, value in space.to_data().items() if key != "name"}
                        expected = {key: value for key, value in state.space.to_data().items()
                                    if key != "name"}
                        if self._multi_module is not None or actual != expected:
                            raise ValueError("balance %s targets an incompatible state quantity" % kind)
                elif space not in inputs:
                    inputs.append(space)
        return Signature(tuple(inputs), Rate(state.space))

    def _register_balance_view(self, handle: Any, view: BalanceView) -> RateHandle:
        self._guard_mutable("declare a balance or partition")
        self._ensure_balance_registry()
        reg = handle.local_id
        if any(previous.local_id == reg for previous in self._retained_rates):
            raise ValueError("balance operator %r is already declared" % reg)
        reason = view.legacy_incompatibility()
        state = view.target
        fluxes = tuple(item.payload for item in view.occurrences if item.kind == "flux")
        sources = tuple(item.payload for item in view.occurrences if item.kind == "source")
        flux = fluxes[0] if len(fluxes) == 1 else (fluxes or None)
        # An unsupported equation remains scientific IR. In particular it is never
        # encoded as flux=False or an empty source list to make old codegen accept it.
        result = RateHandle(handle, view, model=self)
        if reason is None:
            result = self._register_legacy_rate(reg, state, flux, sources, view)
        else:
            registry = (self._multi_module.operator_registry() if self._multi_module is not None
                        else self._dsl._m.operator_registry())
            if reg in registry.names():
                raise ValueError("balance operator %r collides with an existing operator" % reg)
            self._rate_contracts[result] = {
                "state": state, "flux": flux, "sources": sources,
            }
        self._retained_rates[result] = view
        self._invalidate_authoring_views()
        return result

    def _register_legacy_rate(self, reg: str, state: Any, flux: Any, sources: Any,
                              view: BalanceView) -> RateHandle:
        """The supported scalar adapter is derived solely from retained occurrences."""
        if self._multi_module is not None:
            if flux is None:
                fluxes = ()
                default_flux = None
            else:
                try:
                    flux_target = self._multi_module.operator_binding(flux)
                except KeyError:
                    raise RuntimeError(
                        "multi-state physical flux %r has no typed executable binding"
                        % flux.name
                    ) from None
                fluxes = (flux_target,)
                default_flux = fluxes[0] if flux.is_default else None
            source_refs = tuple(
                self._multi_module.operator_handle(source.reg_name) for source in sources)
            result = self._multi_module.rate_operator(
                reg,
                state_space=state.space,
                flux=flux is not None,
                fluxes=fluxes,
                default_flux=default_flux,
                sources=source_refs,
            )
            result = RateHandle(result, view, model=self)
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
                reg, flux=flux is not None, sources=[source.reg_name for source in sources],
                fluxes=([flux.reg_name] if flux is not None and not flux.is_default else None))
            result = RateHandle(self._registered_operator_handle(reg), view, model=self)
            self._rate_contracts[result] = {
                "state": state,
                "flux": flux,
                "sources": tuple(sources),
            }
        return result

    def _install_retained_rates(self, module: Any) -> None:
        """Populate the typed Module with authoritative equations and derived adapters."""
        from pops.model.operators import Operator
        from .interactions import joint_balance_supported
        from pops.provenance import ProvenanceRecord, source_span
        registry = module.operator_registry()
        from .diffusion import install_diffusive_fluxes
        install_diffusive_fluxes(self, module)
        from .drift_diffusion import install_drift_fluxes
        install_drift_fluxes(self,module)
        for handle, view in getattr(self, "_retained_rates", {}).items():
            reason = view.legacy_incompatibility()
            storage = {}
            if (self.frame is not None and view.accumulation.is_identity and view.occurrences
                    and all(row.kind == "source" for row in view.occurrences)):
                storage = {"storage_axes": tuple(axis.name for axis in self.frame.axes),
                           "storage_frame": self.frame.canonical_id}
            if handle.registered_operator_name in registry.names():
                operator = registry.get(handle.registered_operator_name)
                if storage:
                    operator.capabilities = {**operator.capabilities, **storage}
                if operator.lowering.get("physical_balance") is view:
                    continue
                lowering = dict(operator.lowering)
                lowering["physical_balance"] = view
                operator.lowering = lowering
            else:
                lowering = {"physical_balance": view}
                if joint_balance_supported(view):
                    lowering["joint_balance"] = True
                elif reason is not None:
                    lowering["native_unsupported"] = {
                        "code": "unsupported_balance_realization", "phase": "resolve",
                        "reason": reason,
                    }
                registry.register(Operator(handle.registered_operator_name, "local_rate",
                    handle.signature, lowering=lowering,
                    capabilities={"produces_rate": True, "local": False, **storage},
                    source=ProvenanceRecord(primary=source_span(), owner=self.owner_path,
                        authoring_api="pops.physics.Model.rate")))
            # Raw Module flux contracts use operator packs, while board contracts
            # retain the scientific FluxHandle. Keep each existing adapter's typed
            # public shape; physical_balance is the common scientific authority.
            if handle not in module._rate_contracts:
                module._rate_contracts[handle] = dict(self._rate_contracts[handle])

    def _select_rate_view(self, rate: RateHandle, view: BalanceView) -> RateHandle:
        self._guard_mutable("select a physical balance partition")
        if self.balance_contract(rate).balance is not view.balance:
            raise ValueError("a partition must belong to the registered physical balance")
        for existing, retained in self._retained_rates.items():
            if retained.balance is view.balance and retained.ordinals == view.ordinals:
                return existing
        from pops.model import OperatorHandle
        name = "__pops_balance_view_%s_%s" % (
            view.balance.handle.local_id.encode("utf-8").hex(),
            "_".join(map(str, view.ordinals)))
        terms = [(item.kind, item.payload, item.coefficient) for item in view.occurrences]
        handle = OperatorHandle(name, kind="local_rate", owner=self.owner_path,
                                signature=self._balance_signature(view.target, terms))
        return self._register_balance_view(handle, view)

    def balance_contract(self, rate: Any) -> BalanceView:
        """Authoritative scientific equation and exact partition for a rate handle."""
        try:
            return self._retained_rates[rate]
        except (AttributeError, KeyError, TypeError):
            raise ValueError("rate handle has no physical balance registered by this Model") from None

    def select_balance(self, rate: Any) -> Any:
        """Explicitly choose one of several alternative equations for a quantity."""
        self._guard_mutable("select an alternative physical balance")
        view = self.balance_contract(rate)
        if view.ordinals != tuple(range(len(view.balance.occurrences))):
            raise ValueError("select_balance requires a complete physical balance, not a partition")
        self._selected_balances[view.target] = view.balance.handle
        self._invalidate_authoring_views()
        return rate

    def selected_rate_contracts(self, *, states: Any = None) -> dict[Any, Any]:
        """Return the selected complete balances; numerical partitions remain views."""
        state_filter = None if states is None else set(states)
        retained = getattr(self, "_retained_rates", {})
        roots: dict[Any, list[Any]] = {}
        for handle, view in retained.items():
            if handle == view.balance.handle and (state_filter is None or view.target in state_filter):
                roots.setdefault(view.target, []).append(handle)
        result = {handle: contract for handle, contract in self._rate_contracts.items()
                  if handle not in retained and (state_filter is None or contract["state"] in state_filter)}
        for state, choices in roots.items():
            selected = getattr(self, "_selected_balances", {}).get(state)
            if selected is None and len(choices) != 1:
                raise ValueError(
                    "quantity %r has alternative physical balances %s; select_balance(rate) is required"
                    % (state.local_id, [handle.local_id for handle in choices]))
            chosen = choices[0] if selected is None else next(
                handle for handle in choices if handle == selected)
            result[chosen] = self._rate_contracts[chosen]
        return result

    def validate_balance_selection(self, *, states: Any = None) -> bool:
        self.selected_rate_contracts(states=states)
        return True

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
                (hyp, "_provider_components"), (hyp, "_hllc"), (hyp, "_roe"),
                (hyp, "_riemann_hook_forms"), (hyp, "_rate_operators"),
                (self, "_riemann"), (self, "_reconstruction"),
                (self, "_rate_contracts")):
            if riemann is not None:
                self.riemann(riemann)
            self._reconstruction = reconstruction
            state = next(iter(self._states.values()))
            terms = ([] if flux is None else [("flux", flux, -1)])
            terms.extend(("source", source, 1) for source in source_handles)
            result = self.rate(reg, equation=_bm.ddt(state) == _bm.RateExpr(terms))
        return result

    def _authenticate_rate_projection(self, projection: Any, *, target: Any) -> None:
        """Check a joint projection against the exact captured registry body, without calling it."""
        from pops._ir import Var, _children
        from pops._ir.application import RateApplicationProjection, substitute_quantities
        from pops._ir.quantity import QuantityRef, local_expression_identity
        from pops.model.hash_data import canonical_hash_data
        from pops.model.spaces import RateSpace
        if (not isinstance(projection, RateApplicationProjection) or projection.target != target
                or projection.rate_space.base_space != target.space):
            raise ValueError("a joint rate projection must target the differentiated quantity exactly")
        application = projection.application
        module = self._multi_module if self._multi_module is not None else self.module
        try:
            issued = module.operator_handle(application.operator.registered_operator_name)
        except (KeyError, ValueError):
            raise ValueError("joint application operator is not declared by this physics model") from None
        if application.operator != issued or application.operator.signature != issued.signature:
            raise ValueError("joint application operator identity or signature is not authenticated")
        declaration = module.operator_registry().get(issued.registered_operator_name)
        if declaration.body is None or callable(declaration.body):
            raise TypeError("a balance projection requires a captured expression body, not a callback")
        if len(application.inputs) != len(issued.signature.inputs):
            raise ValueError("joint application input count differs from its declared signature")
        bindings = {}
        declaration_index = self.declaration_index()
        for handle in application.declaration_references():
            try:
                declaration_index.authenticate(handle)
            except (KeyError, ValueError, TypeError):
                raise ValueError("joint application contains a foreign or undeclared dependency") from None
        for space, values in zip(issued.signature.inputs, application.inputs, strict=True):
            if len(values) != len(space.components):
                raise ValueError("joint application input shape differs from its declared signature")
            stack = list(values)
            seen = set()
            while stack:
                value = stack.pop()
                if id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, Var):
                    raise ValueError("joint application requires qualified input quantities")
                if isinstance(value, QuantityRef):
                    try:
                        expected = (module.state_handle(value.space) if value.handle.kind == "state"
                                    else module.field_handle(value.space))
                    except (KeyError, ValueError, TypeError):
                        raise ValueError("joint application reads an undeclared quantity type") from None
                    if value.handle != expected:
                        raise ValueError("joint application reads a foreign quantity owner")
                stack.extend(_children(value))
            symbols = (module.state_symbols(space) if space.kind == "state"
                       else module.field_symbols(space))
            bindings.update({(symbol.handle, symbol.index): value
                             for symbol, value in zip(symbols, values, strict=True)})
        body = substitute_quantities(declaration.body, bindings)
        expected_outputs = {"value": body} if isinstance(issued.signature.output, RateSpace) else body
        with local_expression_identity(module.owner_path):
            if canonical_hash_data(expected_outputs) != canonical_hash_data(application.outputs):
                raise ValueError("joint application outputs differ from its captured registered body")
        if application.effects != tuple(sorted(set(declaration.capabilities.get("effects", ())))):
            raise ValueError("joint application effects differ from its registered obligations")

    def _destructure_rate(self, rhs: Any, *, target: Any = None) -> Any:
        """Authenticate each occurrence without destructuring away its scientific meaning."""
        terms = _bm._as_rate(rhs)._rate_terms()
        for kind, payload, _coefficient in terms:
            if kind == "drift":
                from .drift_diffusion import DriftFluxHandle
                if (not isinstance(payload,DriftFluxHandle) or payload.owner_path!=self.owner_path
                        or getattr(self,"_drift_fluxes",{}).get(payload.name)!=payload or payload.state!=target):
                    raise ValueError("drift term must name this state's exact physical drift flux")
            elif kind == "diffusion":
                from .diffusion import DiffusiveFluxHandle
                if (not isinstance(payload, DiffusiveFluxHandle)
                        or payload.owner_path != self.owner_path
                        or getattr(self, "_diffusive_fluxes", {}).get(payload.name) != payload
                        or payload.state != target):
                    raise ValueError("a diffusive rate term must name this state's exact constitutive flux")
            elif kind == "flux":
                if (not isinstance(payload, FluxHandle)
                        or payload.owner_path != self.owner_path
                        or self._fluxes.get(payload.name) != payload):
                    raise ValueError(
                        "a rate equation flux must be a FluxHandle declared by this physics model; "
                        "got %r" % (payload,))
            elif kind == "source":
                if (not isinstance(payload, SourceHandle)
                        or payload.owner_path != self.owner_path
                        or self._sources.get(payload.reg_name) != payload):
                    raise ValueError(
                        "a rate equation source must be a SourceHandle declared by this physics "
                        "model; got %r" % (payload,))
            elif kind == "projection":
                self._authenticate_rate_projection(payload, target=target)
            else:  # pragma: no cover
                raise ValueError("unknown rate term kind %r" % (kind,))
        return tuple(terms)


__all__ = ["_RateAuthoringMixin"]
