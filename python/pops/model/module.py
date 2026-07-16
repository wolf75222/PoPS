"""Operator-first :class:`Module`: typed spaces, parameters, and operators.

The Module owns rules while the Simulation owns data. Codegen is imported lazily
inside :meth:`Module.to_dsl`, keeping this authoring layer extension-free.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any
from weakref import ref

from ._module_freeze import ModuleFreezable
from .operators import Operator, validate_operator_signature
from .handles import Handle, OperatorHandle, ParamHandle, StateHandle
from .ownership import OwnerKind, OwnerPath
from .param_registry import ParamRegistry
from .registry import DeclarationIndex, OperatorRegistry
from .signatures import Signature
from .spaces import AuxSpace, FieldSpace, RateSpace, StateSpace


class Module(ModuleFreezable):
    """Typed spaces plus the operator registry consumed by a generic Program.

    The physics facade populates this registry; direct authoring through spaces,
    explicit parameter declarations, auxiliary fields, and operators is equivalent.
    """
    def __init__(self, name: Any, *, owner: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Module name must be a non-empty string")
        self.name = name
        candidate_owner = (OwnerPath.coerce(owner) if owner is not None
                           else OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name))
        self._owner_path = candidate_owner.require_authoring_root(
            OwnerKind.MODEL_DEFINITION, name=self.name, where="Module owner")
        self._state_spaces = {}
        self._field_spaces = {}
        self._aux = {}
        self._state_handles = {}
        self._field_handles = {}
        self._aux_handles = {}
        self._param_registry = ParamRegistry(owner=self.owner_path, mutation_guard=self._guard_mutable)
        self._registry = OperatorRegistry(
            owner=self.owner_path, mutation_guard=self._guard_mutable)
        # Wave speeds for the Riemann solver of a compilable Module: {"x": [Expr], "y": [Expr]}
        # eigenvalues, or None (set via eigenvalues()). Carried so a pure Module is self-contained;
        # lowered to dsl.Model.eigenvalues by compile_problem.
        self._eigenvalues = None
        # Canonical detached source of the signed pair consumed by HLL.  This is metadata, not a
        # numerics selection: Einfeldt/Davis remain Riemann-provider strategies and are therefore
        # deliberately absent from this model-source vocabulary.
        self._wave_speed_provider = None
        # RateHandle -> exact physical dependencies consumed by DiscretizationPlan.  This is
        # derived authoring metadata: the operator registry remains the executable authority,
        # while the contract proves that a numerical method discretizes precisely the ordered
        # grid-operator pack selected by one rate.
        self._rate_contracts = {}
        # Scientific handles and executable operators are distinct declaration families. This
        # small typed registry records their explicit projections without injecting presentation
        # names into OperatorRegistry's flat callable namespace. Keys and values are handles, so a
        # projection can only be selected through an authenticated scientific declaration.
        self._operator_bindings = {}
        # Process-local membership authorities admitted explicitly by the model builder. They are
        # not semantic data: only successful typed bindings enter the hash/manifest. Object identity
        # prevents an arbitrary same-owner DeclarationIndex from authenticating a forged subject
        # through the public Module API; admission and writes are private facade-adapter seams.
        self._operator_binding_authorities = []
        # The canonical model owner is content-addressed by the complete definition. module_hash()
        # deliberately excludes OwnerPath/Handle identities, so this provider cannot recurse into
        # the identity it stabilizes. It supersedes OperatorRegistry's standalone fallback.
        module_ref = ref(self)

        def module_fingerprint() -> str | None:
            module = module_ref()
            return (None if module is None
                    else "pops.module:sha256:%s" % module.module_hash())

        self._owner_path._bind_definition_fingerprint_provider(
            module_fingerprint, priority=100)

    @property
    def owner_path(self) -> OwnerPath:
        """Immutable declaration owner anchoring all handles and registries."""
        return self._owner_path
    # --- spaces ---
    def state_space(self, name: Any = "U", components: Any = (), roles: Any = None, layout: str = "cell",
                    storage: str = "multifab", *, representation: Any = "conservative",
                    centering: Any = None, units: Any = None, frame: Any = "model",
                    clock: Any = "simulation") -> Any:
        """Declare and return a :class:`StateSpace`."""
        space = StateSpace(
            name, components, roles, layout, storage, representation=representation,
            centering=centering, units=units, frame=frame, clock=clock)
        return self._declare_descriptor(
            self._state_spaces, self._state_handles, space, "StateSpace", "state")

    def field_space(self, name: Any, components: Any = (), layout: str = "cell", *,
                    representation: Any = "field", centering: Any = None, units: Any = None,
                    frame: Any = "model", clock: Any = "simulation") -> Any:
        """Declare and return a :class:`FieldSpace`."""
        space = FieldSpace(
            name, components, layout, representation=representation, centering=centering,
            units=units, frame=frame, clock=clock)
        return self._declare_descriptor(
            self._field_spaces, self._field_handles, space, "FieldSpace", "field")

    # --- parameters / aux ---
    def param(self, declaration: Any) -> ParamHandle:
        """Register one canonical parameter declaration and return its typed handle.

        The kind is carried by ``RuntimeParam`` / ``ConstParam`` / ``DerivedParam``.  A
        ``(name, value)`` shorthand is deliberately not accepted because it would silently choose
        compile-time storage for a value whose intended lifetime is unknown.
        """
        self._guard_mutable("declare a parameter")
        return self._param_registry.register(declaration)

    def parameters(self, *declarations: Any) -> tuple[ParamHandle, ...]:
        """Register several explicit typed declarations in order."""
        return tuple(self.param(declaration) for declaration in declarations)

    def value(self, parameter: Any) -> Any:
        """Build the symbolic value read of one registered parameter.

        This explicit conversion keeps :class:`ParamHandle` identity separate
        from :class:`pops._ir.Expr` algebra: handles retain Boolean equality and
        dictionary-key semantics, while the returned node carries the declared
        compile/bind storage behavior.
        """
        from pops._ir import parameter_value

        return parameter_value(self._param_registry, parameter)

    def aux_field(self, name: Any, kind: str = "cell_scalar", *,
                  representation: Any = "auxiliary", centering: Any = "cell",
                  unit: Any = None, frame: Any = "model", clock: Any = "simulation") -> Any:
        """Declare and return one :class:`AuxSpace`."""
        a = AuxSpace(
            name, kind, representation=representation, centering=centering, unit=unit,
            frame=frame, clock=clock)
        return self._declare_descriptor(
            self._aux, self._aux_handles, a, "aux field", "aux")

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
        if kind == "field_operator":
            if lowering is None:
                lowering = {"field_provider": {"key": name}}
            elif not isinstance(lowering, dict) or "field_provider" not in lowering:
                raise ValueError(
                    "field_operator lowering must declare its field_provider route")

        def _register(body: Any) -> Any:
            self._guard_mutable("register an operator")
            from pops.provenance import ProvenanceRecord, callable_span, source_span
            primary = callable_span(body) if callable(body) else source_span()
            provenance = ProvenanceRecord(
                primary=primary,
                owner=self.owner_path,
                authoring_api="pops.model.Module.operator",
            )
            op = Operator(name, kind, signature, capabilities=capabilities,
                          requirements=requirements, lowering=lowering, source=provenance,
                          body=body)
            self._registry.register(op)
            return self.operator_handle(op.name)

        if expr is not None:
            return _register(expr)

        def decorator(func: Any) -> Any:
            return _register(func)

        return decorator

    def rate_operator(self, name: Any, state_space: Any, flux: bool = True,
                      sources: Any = (), fluxes: Any = None,
                      default_flux: Any = None) -> Any:
        """Register ``R = -div F + sum(sources)`` from typed declaration references.

        ``state_space`` is the declared :class:`StateSpace` (or its registry-issued state
        :class:`Handle`). ``sources`` and ``fluxes`` contain :class:`OperatorHandle` values issued by
        this Module. ``default_flux`` may identify the sole selected grid operator that lowering
        installs as the native model's default finite-volume flux.  Its physical identity remains in
        ``fluxes`` while Program calls route through the configured FV/Riemann ``rhs_into`` path.
        Names remain only in the private lowering payload; public semantic references are never
        looked up from strings.
        """
        self._guard_mutable("register a rate operator")
        state_ref = self.state_handle(state_space)
        u = self._state_spaces[state_ref.local_id]
        if not isinstance(flux, bool):
            raise TypeError("rate_operator(%r): flux must be a Python bool" % name)
        source_refs = list(sources) if sources is not None else []
        flux_refs = list(fluxes) if fluxes is not None else []
        if not flux and flux_refs:
            raise ValueError("rate_operator(%r): named fluxes require flux=True" % name)
        inputs = [u]
        selected = []
        source_names = []
        source_handles = []
        for source_ref in source_refs:
            target, operator = self._operator_reference(
                source_ref, expected_kind="local_source", label="source")
            source_names.append(target)
            source_handles.append(self.operator_handle(target))
            selected.append(operator)
        flux_names = []
        flux_handles = []
        for flux_ref in flux_refs:
            target, operator = self._operator_reference(
                flux_ref, expected_kind="grid_operator", label="flux")
            flux_names.append(target)
            flux_handles.append(self.operator_handle(target))
            selected.append(operator)
        default_flux_name = None
        if default_flux is not None:
            default_flux_name, _ = self._operator_reference(
                default_flux, expected_kind="grid_operator", label="default flux")
            if not flux:
                raise ValueError(
                    "rate_operator(%r): a default flux route requires flux=True" % name)
            if flux_names != [default_flux_name]:
                raise ValueError(
                    "rate_operator(%r): default_flux must be the sole exact operator selected "
                    "in fluxes" % name)
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
        from pops.provenance import ProvenanceRecord, source_span
        provenance = ProvenanceRecord(
            primary=source_span(), owner=self.owner_path,
            authoring_api="pops.model.Module.rate_operator")
        lowering = {"flux": flux, "sources": source_names, "fluxes": flux_names or None}
        if default_flux_name is not None:
            lowering["default_flux"] = default_flux_name
        op = Operator(
            name,
            "local_rate",
            Signature(tuple(inputs), RateSpace(u)),
            capabilities={
                "local": False,
                "requires_fields": bool(field_inputs),
                "produces_rate": True,
                "supports_device": True,
            },
            lowering=lowering,
            source=provenance,
        )
        self._registry.register(op)
        handle = self.operator_handle(op.name)
        self._rate_contracts[handle] = {
            "state": state_ref,
            "flux": tuple(flux_handles) if flux else None,
            "sources": tuple(source_handles),
        }
        return handle

    def rate_contract(self, rate: Any) -> dict[str, Any]:
        """Return the exact physical dependencies of a registered rate operator.

        A raw operator-first Module names a physical flux as an ordered pack of
        ``grid_operator`` handles.  Keeping the pack typed and ordered lets a finite-volume
        method authenticate both a single flux and an explicitly decomposed sum without
        collapsing either form to strings.
        """
        if not isinstance(rate, OperatorHandle) or rate.kind != "local_rate":
            raise TypeError("rate_contract requires a local_rate OperatorHandle")
        try:
            contract = self._rate_contracts[rate]
        except KeyError:
            raise ValueError("rate handle is not registered by this Module") from None
        return {
            "state": contract["state"],
            "flux": contract["flux"],
            "sources": tuple(contract["sources"]),
        }

    def eigenvalues(self, x: Any, y: Any) -> Any:
        """Declare the per-direction wave speeds (eigenvalues) the Riemann solver needs, as lists of
        IR expressions over the state. Carried so a pure Module is a self-contained, compilable model
        (lowered to ``dsl.Model.eigenvalues``)."""
        self._guard_mutable("declare eigenvalues")
        x_values, y_values = tuple(x), tuple(y)
        self._eigenvalues = {"x": x_values, "y": y_values}
        # An inspection result is deliberately detached.  Mutating a value returned during
        # authoring must never rewrite the Module behind its public setter.
        return {"x": list(x_values), "y": list(y_values)}

    def set_wave_speed_provider(self, kind: Any) -> str:
        """Record the one detached source kind that emits signed wave speeds.

        The operator-first manifest must retain this provenance after the physics facade and its
        live authoring graph have been discarded.  Only model sources belong here; Riemann-side
        estimates such as Einfeldt and Davis are numerical descriptors.
        """
        self._guard_mutable("declare the wave-speed provider")
        allowed = ("explicit_pair", "jacobian", "pressure_derived")
        if kind not in allowed:
            raise ValueError(
                "wave-speed provider kind %r must be one of %s"
                % (kind, ", ".join(allowed))
            )
        current = self._wave_speed_provider
        if current is not None and current != kind:
            raise ValueError(
                "Module %r already declares wave-speed provider %r; a model has one source"
                % (self.name, current)
            )
        self._wave_speed_provider = kind
        return kind

    @property
    def wave_speed_provider_kind(self) -> str | None:
        """Detached signed-wave source recorded by this module, if any."""
        return self._wave_speed_provider

    def adopt_registry(self, registry: Any) -> Any:
        """Use ``registry`` as this Module's operator registry (the dsl.Model facade adopts
        the derived registry of its HyperbolicModel). Returns ``self``."""
        self._guard_mutable("adopt an operator registry")
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("adopt_registry expects an OperatorRegistry")
        if registry.owner_path != self.owner_path:
            raise ValueError("adopt_registry cannot adopt a registry owned by another Module")
        registry._bind_mutation_guard(self._guard_mutable)
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

    def _bind_operator(
        self,
        subject: Any,
        operator: Any,
        *,
        declarations: Any,
    ) -> OperatorHandle:
        """Private facade-adapter seam binding a scientific handle to an executable operator.

        This is a projection registry, not an operator-alias registry. The subject and target keep
        their own typed namespaces, so (for example) a physical flux and a local rate may both be
        named ``transport``. Lookup accepts only the exact owner-qualified subject handle; strings
        never gain declaration authority. ``declarations`` is the originating family's read-only
        membership authority and must first be admitted explicitly by this Module; sharing an owner
        path alone cannot manufacture a binding subject.
        """
        self._guard_mutable("bind a scientific handle to an operator")
        if (
            not isinstance(subject, Handle)
            or isinstance(subject, OperatorHandle)
        ):
            raise TypeError(
                "operator binding subject must be a non-operator Handle"
            )
        if subject.owner_path != self.owner_path:
            raise ValueError("operator binding subject belongs to another Module authority")
        if not isinstance(declarations, DeclarationIndex):
            raise TypeError(
                "operator binding declarations must be a DeclarationIndex authority"
            )
        if declarations.owner_path != self.owner_path:
            raise ValueError("operator binding declaration index belongs to another authority")
        if not any(authority is declarations for authority in self._operator_binding_authorities):
            raise ValueError(
                "operator binding declaration index is not registered by this Module"
            )
        registered_subject = declarations.authenticate(subject)
        if isinstance(registered_subject, OperatorHandle):  # pragma: no cover - guarded above
            raise TypeError("operator binding subject must be a non-operator Handle")
        if not isinstance(operator, OperatorHandle):
            raise TypeError("operator binding target must be an OperatorHandle")
        registered = self._registry.declaration_index().authenticate(operator)
        if not isinstance(registered, OperatorHandle):  # pragma: no cover - registry invariant
            raise RuntimeError("operator registry returned a non-OperatorHandle declaration")
        if registered_subject in self._operator_bindings:
            raise ValueError(
                "operator binding subject %s is already registered"
                % registered_subject.qualified_id
            )
        target = registered.registered_operator_name
        canonical_target = self.operator_handle(target)
        self._operator_bindings[registered_subject] = canonical_target
        return canonical_target

    def _register_operator_binding_authority(
        self, declarations: Any,
    ) -> DeclarationIndex:
        """Private facade-adapter admission for one originating declaration index.

        The public Module surface cannot nominate a same-owner index as trusted. Sanctioned model
        facades call this private seam with their exact family projection; equivalent repeated
        projections are coalesced to keep persistent Modules bounded.
        """
        self._guard_mutable("register an operator binding declaration authority")
        if not isinstance(declarations, DeclarationIndex):
            raise TypeError(
                "operator binding authority must be a DeclarationIndex"
            )
        if declarations.owner_path != self.owner_path:
            raise ValueError("operator binding declaration index belongs to another authority")
        for authority in self._operator_binding_authorities:
            if authority is declarations or authority.records() == declarations.records():
                return authority
        self._operator_binding_authorities.append(declarations)
        return declarations

    def operator_binding(self, subject: Any) -> OperatorHandle:
        """Return the target for an exact registered subject handle; strings are refused."""
        if not isinstance(subject, Handle) or isinstance(subject, OperatorHandle):
            raise TypeError("operator_binding requires a non-operator Handle subject")
        if subject.owner_path != self.owner_path:
            raise ValueError("operator binding subject belongs to another Module authority")
        try:
            return self._operator_bindings[subject]
        except KeyError:
            raise KeyError("no operator binding is registered for %s" % subject.qualified_id) \
                from None

    def operator_bindings(self) -> MappingProxyType:
        """Immutable snapshot of typed scientific-handle to operator projections."""
        return MappingProxyType(dict(self._operator_bindings))

    def state_handle(self, state: Any) -> Handle:
        """Return the registry-issued handle of a declared :class:`StateSpace`."""
        return self._descriptor_handle(
            state, self._state_spaces, self._state_handles, "StateSpace")

    def state_symbols(self, state: Any) -> tuple[Any, ...]:
        """Return the owner-qualified conservative coordinates of ``state``.

        These expressions are the canonical authoring path for operators that read
        several state spaces.  Unlike bare component names, they remain unambiguous
        when, for example, both an electron and ion state contain ``"density"``.
        They are ordinary immutable :class:`pops._ir.Var` nodes and introduce no
        separate multi-species runtime or lowering path.
        """
        from pops._ir.expr import Var
        from pops.model.state_symbols import state_component_symbol

        handle = self.state_handle(state)
        space = self._state_spaces[handle.local_id]
        return tuple(
            Var(state_component_symbol(space, component), "cons")
            for component in space.components
        )

    def field_handle(self, field: Any) -> Handle:
        """Return the registry-issued handle of a declared :class:`FieldSpace`."""
        return self._descriptor_handle(
            field, self._field_spaces, self._field_handles, "FieldSpace")

    def param_handle(self, parameter: Any) -> ParamHandle:
        """Return the registry-issued handle of a declared parameter."""
        return self._param_registry.handle(parameter)

    def param_declaration(self, parameter: Any) -> Any:
        """Return the immutable declaration authenticated by ``parameter``."""
        return self._param_registry.declaration(parameter)

    def param_registry(self) -> ParamRegistry:
        """The unique parameter authority owned by this Module."""
        return self._param_registry

    def aux_handle(self, aux: Any) -> Handle:
        """Return the registry-issued handle of a declared auxiliary field."""
        return self._descriptor_handle(aux, self._aux, self._aux_handles, "aux field")

    def declaration_index(self) -> DeclarationIndex:
        """Read-only union of the Module's authoritative family registries."""
        candidates = [
            *self._state_handles.values(),
            *self._field_handles.values(),
            *self._param_registry.handles(),
            *self._aux_handles.values(),
            *self._operator_bindings,
            *self._registry.declaration_index().records(),
        ]
        handles = []
        by_key = {}
        for handle in candidates:
            key = (handle.kind, handle.local_id)
            previous = by_key.get(key)
            if previous is not None:
                if previous != handle:
                    raise ValueError(
                        "Module %r exposes conflicting %s declaration %r"
                        % (self.name, handle.kind, handle.local_id)
                    )
                continue
            by_key[key] = handle
            handles.append(handle)
        return DeclarationIndex(owner=self.owner_path, handles=handles)

    def manifest(self) -> Any:
        """The self-describing :class:`pops.model.manifest.ModuleManifest` of this Module (ADC-585).

        The central, JSON-ready representation of the model -- spaces, params, aux, eigenvalue
        presence, the typed operator registry (each operator by its stable id), scientific-handle
        projections and the native route-registry components -- that supersedes the legacy flat
        ModelSpec POD. Read-only: it does not mutate the Module."""
        from pops.model.manifest import build_module_manifest
        return build_module_manifest(self)

    def to_dsl(self) -> Any:
        """Lower this Module to a :class:`pops.physics._facade.Model` -- the physical/codegen engine -- by mapping
        each typed operator (with its IR body) to the dsl method of its kind. Reuses the dsl backend
        (a translation, not a second codegen). ``pops.codegen.compile_problem(model=module, ...)``
        does this implicitly; call it directly to build the block model for ``sim.add_equation``."""
        # Lazy: codegen.compile imports this module, so import only when compiling.
        from pops.codegen.module_lowering import _module_to_model
        return _module_to_model(self)

    def __pops_compiler_lowering__(self) -> Any:
        """Provide the explicit compiler-provider boundary for this canonical IR."""
        from pops.codegen import CompilerLowering
        from pops.codegen.module_lowering import _module_to_model

        return CompilerLowering(
            emit_model=_module_to_model(self),
            source_module=self,
            facade=self,
        )

    # --- introspection (Spec 2, S2-5) ---
    def state_spaces(self) -> Any:
        return dict(self._state_spaces)

    def field_spaces(self) -> Any:
        return dict(self._field_spaces)

    def params(self) -> Any:
        declarations = self._param_registry.declarations()
        return MappingProxyType(declarations) if getattr(self, "frozen", False) else declarations

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
            self._guard_mutable("change operator capabilities")
            op.capabilities.update(caps)
        return dict(op.capabilities)

    def module_hash(self) -> str:
        """Stable hash of the ModuleSpec for the compiled-artifact cache (Spec 2, S2-7).

        Folds the spaces, parameters, aux declarations and -- for every operator -- the name,
        kind, signature, capabilities, requirements and a body identity (the source of a callable
        body, else its repr), plus every authenticated public operator alias and typed scientific
        operator binding. Sensitive to an operator body, signature, capability, alias, binding or
        space change; deterministic for an identical module. A spec2 tag namespaces it away from
        any spec1 key.
        """
        from ._module_hash import module_content_hash
        return module_content_hash(self)

    def __repr__(self) -> str:
        return "Module(%r, operators=[%s])" % (self.name, ", ".join(self._registry.names()))

    def _declare_descriptor(
        self,
        registry: Any,
        handles: Any,
        descriptor: Any,
        label: str,
        kind: str,
    ) -> Any:
        """Install one descriptor exactly once under its semantic name."""
        self._guard_mutable("declare %s" % label)
        existing = registry.get(descriptor.name)
        if existing is None:
            registry[descriptor.name] = descriptor
            handles[descriptor.name] = (
                StateHandle(descriptor.name, owner=self.owner_path, space=descriptor)
                if kind == "state"
                else Handle(descriptor.name, kind=kind, owner=self.owner_path)
            )
            return descriptor
        old_data = existing.to_data()
        new_data = descriptor.to_data()
        raise ValueError(
            "%s %r is already declared; declarations are register-once: "
            "existing %r, requested %r"
            % (label, descriptor.name, old_data, new_data))

    @staticmethod
    def _descriptor_handle(
        descriptor_or_name: Any,
        descriptors: Any,
        handles: Any,
        label: str,
    ) -> Handle:
        if isinstance(descriptor_or_name, str):
            raise TypeError(
                "%s handle lookup requires the declared descriptor object or its Handle; "
                "a name string is not a semantic reference" % label)
        if isinstance(descriptor_or_name, Handle):
            registered = handles.get(descriptor_or_name.local_id)
            if registered is None:
                raise KeyError("unknown %s %r" % (label, descriptor_or_name.local_id))
            if registered != descriptor_or_name:
                raise ValueError(
                    "%s handle %s belongs to another Module registry"
                    % (label, descriptor_or_name.qualified_id))
            return registered
        name = getattr(descriptor_or_name, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError("%s handle lookup requires the declared descriptor object" % label)
        descriptor = descriptors.get(name)
        if descriptor is None:
            raise KeyError("unknown %s %r" % (label, name))
        if descriptor is not descriptor_or_name:
            raise ValueError("%s %r belongs to another Module registry" % (label, name))
        return handles[name]

    def _operator_reference(
        self,
        reference: Any,
        *,
        expected_kind: str,
        label: str,
    ) -> tuple[str, Operator]:
        """Authenticate one typed operator reference and expose its private registry target."""
        if not isinstance(reference, OperatorHandle):
            raise TypeError(
                "rate_operator %s references must be OperatorHandle values, not %r"
                % (label, type(reference).__name__))
        registered = self._registry.declaration_index().authenticate(reference)
        if not isinstance(registered, OperatorHandle):
            raise RuntimeError("operator registry returned a non-OperatorHandle declaration")
        if registered.kind != expected_kind:
            raise TypeError(
                "rate_operator %s reference %s has kind %r; expected %r"
                % (label, reference.qualified_id, registered.kind, expected_kind))
        target = registered.registered_operator_name
        return target, self._registry.get(target)
