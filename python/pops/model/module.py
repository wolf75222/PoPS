"""The :class:`Module` front-end of the operator-first type system (Spec 2, S2-3).

A Module owns the RULES -- state/field spaces, parameters, aux declarations and a
registry of typed operators a Program composes by signature. The Simulation owns
the DATA. ``Module.to_dsl()`` lowers a pure Module to a
:class:`pops.physics.facade.Model`.

Imports only the standard library (plus the sibling operator-first types) so it
can be exercised without the compiled ``_pops`` extension; the codegen engine is
imported lazily inside :meth:`Module.to_dsl` to avoid an import cycle.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .hash_data import body_identity as _body_identity, canonical_hash_data as _canonical_hash_data
from .operators import Operator, validate_operator_signature
from .handles import OperatorHandle, OwnerPath
from .registry import OperatorRegistry
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    ParameterSpace,
    RateSpace,
    StateSpace,
)


class Module:
    """A model as typed spaces + a registry of typed operators (Spec 2, operator-first).

    A Module owns the RULES -- state/field spaces, parameters, aux declarations and the
    typed operators a Program composes by signature. The Simulation owns the DATA
    (grid, arrays, solvers, clock). :class:`pops.physics.facade.Model` is the PDE convenience facade
    that populates a Module's registry (``source_term`` / ``linear_source`` /
    ``elliptic_field`` / ``flux`` register typed operators); a Module can also be built
    directly with ``state_space`` / ``field_space`` / ``parameters`` / ``aux_fields`` /
    ``operator``. A generic Program bound to ``module.operator_registry()`` runs against
    any Module that provides operators with the expected signatures.
    """

    def __init__(self, name: Any, *, owner: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Module name must be a non-empty string")
        self.name = name
        self._owner_path = (OwnerPath.coerce(owner) if owner is not None
                            else OwnerPath.fresh("module", self.name))
        self._state_spaces = {}
        self._field_spaces = {}
        self._params = {}
        self._aux = {}
        self._registry = OperatorRegistry(owner=self.owner_path)
        # Wave speeds for the Riemann solver of a compilable Module: {"x": [Expr], "y": [Expr]}
        # eigenvalues, or None (set via eigenvalues()). Carried so a pure Module is self-contained;
        # lowered to dsl.Model.eigenvalues by compile_problem.
        self._eigenvalues = None

    @property
    def owner_path(self) -> OwnerPath:
        """Immutable declaration owner anchoring all handles and registries."""
        return self._owner_path

    # --- spaces ---
    def state_space(self, name: Any = "U", components: Any = (), roles: Any = None, layout: str = "cell",
                    storage: str = "multifab") -> Any:
        """Declare and return a :class:`StateSpace`."""
        space = StateSpace(name, components, roles, layout, storage)
        return self._declare_descriptor(self._state_spaces, space, "StateSpace")

    def field_space(self, name: Any, components: Any = (), layout: str = "cell") -> Any:
        """Declare and return a :class:`FieldSpace`."""
        space = FieldSpace(name, components, layout)
        return self._declare_descriptor(self._field_spaces, space, "FieldSpace")

    # --- parameters / aux ---
    def param(self, name: Any, default: Any = 0.0, dtype: str = "real") -> Any:
        """Declare and return one :class:`ParameterSpace`."""
        p = ParameterSpace(name, default, dtype)
        return self._declare_descriptor(self._params, p, "parameter")

    def parameters(self, **defaults: Any) -> Any:
        """Declare several parameters by keyword; return ``{name: ParameterSpace}``."""
        return {k: self.param(k, v) for k, v in defaults.items()}

    def aux_field(self, name: Any, kind: str = "cell_scalar") -> Any:
        """Declare and return one :class:`AuxSpace`."""
        a = AuxSpace(name, kind)
        return self._declare_descriptor(self._aux, a, "aux field")

    def aux_fields(self, **kinds: Any) -> Any:
        """Declare several aux fields by keyword; return ``{name: AuxSpace}``."""
        return {k: self.aux_field(k, v) for k, v in kinds.items()}

    # --- operators ---
    def operator(self, name: Any = None, signature: Any = None, kind: Any = None,
                 capabilities: Any = None, requirements: Any = None, lowering: Any = None,
                 expr: Any = None) -> Any:
        """Register a typed operator.

        Builder mode (``expr`` given) registers the operator immediately and returns its
        public :class:`OperatorHandle`. Decorator mode records the decorated body internally
        and replaces the decorated name with the same public handle::

            @module.operator(name="explicit_rhs",
                             signature=(U, Fields) >> Rate(U), kind="local_rate")
            def explicit_rhs(U, fields):
                ...
        """
        if name is None or signature is None or kind is None:
            raise ValueError("module.operator requires name, signature and kind")
        if not isinstance(signature, Signature):
            raise TypeError(
                "module.operator(%r): signature must be a Signature (use the >> sugar or "
                "Signature(inputs, output)); got %r" % (name, signature))
        validate_operator_signature(kind, signature, operator_name=name)

        def _register(body: Any) -> Any:
            op = Operator(name, kind, signature, capabilities=capabilities,
                          requirements=requirements, lowering=lowering, source="module",
                          body=body)
            self._registry.register(op)
            return self.operator_handle(op.name)

        if expr is not None:
            return _register(expr)

        def decorator(func: Any) -> Any:
            return _register(func)

        return decorator

    def rate_operator(self, name: Any, state_space: Any = "U", flux: bool = True,
                      sources: Any = ("default",), fluxes: Any = None) -> Any:
        """Register a composite ``local_rate`` operator ``R = -div F + sum(sources)`` from named
        sub-operators (the flux and the listed source operators). Mirrors ``dsl.rate_operator``; the
        ``lowering`` carries the flux/sources/fluxes so ``P.call`` and the codegen compose it."""
        if isinstance(state_space, StateSpace):
            u = state_space
        else:
            u = self._state_spaces.get(state_space)
        if u is None:
            raise ValueError(
                "rate_operator(%r): state_space must name a declared StateSpace" % name)
        if not isinstance(flux, bool):
            raise TypeError("rate_operator(%r): flux must be a Python bool" % name)
        srcs = list(sources) if sources is not None else None
        flxs = list(fluxes) if fluxes else None
        if not flux and flxs:
            raise ValueError("rate_operator(%r): named fluxes require flux=True" % name)
        inputs = [u]
        selected = []
        for source_name in srcs or ():
            if not isinstance(source_name, str) or not source_name:
                raise TypeError("rate_operator(%r): source names must be non-empty strings" % name)
            if source_name == "default" and source_name not in self._registry:
                continue
            target = self._registry.target_for_handle(source_name)
            selected.append(self._registry.get(target))
        for flux_name in flxs or ():
            if not isinstance(flux_name, str) or not flux_name:
                raise TypeError("rate_operator(%r): flux names must be non-empty strings" % name)
            target = self._registry.target_for_handle(flux_name)
            selected.append(self._registry.get(target))
        field_inputs = []
        for selected_op in selected:
            for input_space in selected_op.signature.inputs:
                input_kind = getattr(input_space, "kind", None)
                if input_kind == "state" and input_space != u:
                    raise ValueError(
                        "rate_operator(%r): operator %r consumes incompatible StateSpace %r"
                        % (name, selected_op.name, getattr(input_space, "name", input_space)))
                if input_kind == "field" and input_space not in field_inputs:
                    field_inputs.append(input_space)
        if len(field_inputs) > 1:
            raise ValueError(
                "rate_operator(%r): composed sources require multiple incompatible FieldSpaces; "
                "declare one compatible field context or separate the rates" % name)
        inputs.extend(field_inputs)
        op = Operator(name, "local_rate", Signature(tuple(inputs), RateSpace(u)),
                      capabilities={"local": False, "requires_fields": bool(field_inputs),
                                    "produces_rate": True, "supports_device": True},
                      lowering={"flux": bool(flux), "sources": srcs,
                                "fluxes": flxs},
                      source="module")
        self._registry.register(op)
        return self.operator_handle(op.name)

    def eigenvalues(self, x: Any, y: Any) -> Any:
        """Declare the per-direction wave speeds (eigenvalues) the Riemann solver needs, as lists of
        IR expressions over the state. Carried so a pure Module is a self-contained, compilable model
        (lowered to ``dsl.Model.eigenvalues``)."""
        self._eigenvalues = {"x": list(x), "y": list(y)}
        return self._eigenvalues

    def adopt_registry(self, registry: Any) -> Any:
        """Use ``registry`` as this Module's operator registry (the dsl.Model facade adopts
        the derived registry of its HyperbolicModel). Returns ``self``."""
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("adopt_registry expects an OperatorRegistry")
        if registry.owner_path is None:
            raise ValueError("adopt_registry requires an owner-qualified OperatorRegistry")
        if registry.owner_path != self.owner_path:
            raise ValueError("adopt_registry cannot adopt a registry owned by another Module")
        self._registry = registry
        return self

    def operator_registry(self) -> Any:
        """The Module's :class:`OperatorRegistry` (bind it to a Program with P.bind_operators)."""
        return self._registry

    def operator_handle(self, name: Any) -> OperatorHandle:
        """Return the canonical public handle for one registered operator.

        ``Operator`` is the registry/codegen record.  User programs retain this
        immutable owner-qualified reference instead, so pure ``Module`` authoring
        has the same clean path as the physics facade.
        """
        target = self._registry.target_for_handle(name)
        operator = self._registry.get(target)
        return OperatorHandle(
            name, kind=operator.kind, owner=self._registry.owner_path,
            signature=operator.signature, registered_operator_name=target)

    def manifest(self) -> Any:
        """The self-describing :class:`pops.model.manifest.ModuleManifest` of this Module (ADC-585).

        The central, JSON-ready representation of the model -- spaces, params, aux, eigenvalue
        presence, the typed operator registry (each operator by its stable id) and the native
        route-registry components -- that supersedes the legacy flat ModelSpec POD. Read-only:
        it does not mutate the Module."""
        from pops.model.manifest import build_module_manifest
        return build_module_manifest(self)

    def to_dsl(self) -> Any:
        """Lower this Module to a :class:`pops.physics.facade.Model` -- the physical/codegen engine -- by mapping
        each typed operator (with its IR body) to the dsl method of its kind. Reuses the dsl backend
        (a translation, not a second codegen). ``pops.codegen.compile_problem(model=module, ...)``
        does this implicitly; call it directly to build the block model for ``sim.add_equation``."""
        # Lazy: codegen.compile imports this module, so import only when compiling.
        from pops.codegen.compile import _module_to_model
        return _module_to_model(self)

    # --- introspection (Spec 2, S2-5) ---
    def state_spaces(self) -> Any:
        return dict(self._state_spaces)

    def field_spaces(self) -> Any:
        return dict(self._field_spaces)

    def params(self) -> Any:
        return dict(self._params)

    def aux(self) -> Any:
        return dict(self._aux)

    def list_state_spaces(self) -> Any:
        """Names of the declared state spaces."""
        return list(self._state_spaces)

    def list_field_spaces(self) -> Any:
        """Names of the declared field spaces."""
        return list(self._field_spaces)

    def list_operators(self) -> Any:
        """Operator names in registration (id) order."""
        return self._registry.names()

    def operator_signature(self, name: Any) -> Any:
        """The :class:`Signature` of operator ``name``."""
        return self._registry.get(name).signature

    def operator_requirements(self, name: Any) -> Any:
        """The requirements dict of operator ``name`` (aux / solver / params / ...)."""
        return dict(self._registry.get(name).requirements)

    def unknown_requirement_keys(self, name: Any) -> Any:
        """Requirement keys of operator ``name`` outside the documented vocabulary (ADC-528).

        Returns the sorted list of keys in the operator's ``requirements`` dict that are NOT one of
        :data:`pops.model.operators.OPERATOR_REQUIREMENT_KEYS` (ghosts / fields / params / aux /
        solvers / layout / backend). Empty for a clean operator. This is a diagnostic, not a hard
        gate: a foreign key is allowed (the vocabulary is documented, not enforced), but surfacing it
        lets a typo (``"ghost"`` for ``"ghosts"``) be caught. Requirements are always declared by the
        operator's author; they are NEVER inferred from the operator name."""
        from .operators import OPERATOR_REQUIREMENT_KEYS
        reqs = self._registry.get(name).requirements
        return sorted(k for k in reqs if k not in OPERATOR_REQUIREMENT_KEYS)

    def validate_requirements(self) -> Any:
        """Warn on every operator requirement key outside the documented vocabulary (ADC-528).

        Emits one :class:`UserWarning` per operator that carries an unrecognized requirement key and
        returns ``{operator_name: [unknown_key, ...]}`` for the offending operators (empty when the
        whole registry is clean). Additive and non-fatal: it never rejects a Module, so a family that
        has not yet adopted the vocabulary keeps working; it only flags a likely typo."""
        import warnings
        from .operators import OPERATOR_REQUIREMENT_KEYS
        known = sorted(OPERATOR_REQUIREMENT_KEYS)
        offenders = {}
        for op in self._registry:
            unknown = self.unknown_requirement_keys(op.name)
            if unknown:
                offenders[op.name] = unknown
                warnings.warn(
                    "operator %r declares requirement key(s) %s outside the documented vocabulary %s"
                    % (op.name, unknown, known), UserWarning, stacklevel=2)
        return offenders

    def operator_capabilities(self, name: Any, **caps: Any) -> Any:
        """Get or set the capabilities of operator ``name``.

        Called with only a name it is a getter (returns a copy of the dict). Called with
        keyword capabilities (e.g. ``cacheable=True``, ``stale_allowed=True``,
        ``requires_fresh_inputs=True``) it UPDATES them in place and returns the new dict.
        ``cacheable`` is consumed by the Program scheduler to validate a ``hold`` schedule.
        """
        op = self._registry.get(name)
        if caps:
            op.capabilities.update(caps)
        return dict(op.capabilities)

    def module_hash(self) -> str:
        """Stable hash of the ModuleSpec for the compiled-artifact cache (Spec 2, S2-7).

        Folds the spaces, parameters, aux declarations and -- for every operator -- the name,
        kind, signature, capabilities, requirements and a body identity (the source of a callable
        body, else its repr). Sensitive to an operator body, signature, capability or space change;
        deterministic for an identical module. A spec2 tag namespaces it away from any spec1 key.
        """
        payload = {
            "schema": "spec2-module",
            "name": self.name,
            "state_spaces": [
                self._state_spaces[name].to_data() for name in sorted(self._state_spaces)
            ],
            "field_spaces": [
                self._field_spaces[name].to_data() for name in sorted(self._field_spaces)
            ],
            "parameters": [self._params[name].to_data() for name in sorted(self._params)],
            "aux": [self._aux[name].to_data() for name in sorted(self._aux)],
            "eigenvalues": None if self._eigenvalues is None else {
                direction: [_canonical_hash_data(value) for value in self._eigenvalues[direction]]
                for direction in ("x", "y")
            },
            # Registry order is semantic: it determines stable OperatorId values.
            "operators": [{
                "name": op.name,
                "kind": op.kind,
                "signature": op.signature.to_data(),
                "capabilities": op.capabilities,
                "requirements": op.requirements,
                "lowering": op.lowering,
                "body": _body_identity(op.body),
            } for op in self._registry],
        }
        canonical = json.dumps(
            _canonical_hash_data(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "Module(%r, operators=[%s])" % (self.name, ", ".join(self._registry.names()))

    @staticmethod
    def _declare_descriptor(registry: Any, descriptor: Any, label: str) -> Any:
        """Install one immutable descriptor, making compatible repeats idempotent."""
        existing = registry.get(descriptor.name)
        if existing is None:
            registry[descriptor.name] = descriptor
            return descriptor
        old_data = existing.to_data()
        new_data = descriptor.to_data()
        if old_data == new_data:
            return existing
        raise ValueError(
            "%s %r is already declared incompatibly: existing %r, requested %r"
            % (label, descriptor.name, old_data, new_data))
