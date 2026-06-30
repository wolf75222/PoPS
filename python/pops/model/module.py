"""The :class:`Module` front-end of the operator-first type system (Spec 2, S2-3).

A Module owns the RULES -- state/field spaces, parameters, aux declarations and a
registry of typed operators a Program composes by signature. The Simulation owns
the DATA. Modules are compiled by ``pops.codegen.compile_problem``; they do not
expose a public lowering escape hatch to the old DSL.

Imports only the standard library (plus the sibling operator-first types) so it
can be exercised without the compiled ``_pops`` extension.
"""
import hashlib

from .operators import Operator
from .registry import OperatorRegistry
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    ParameterSpace,
    RateSpace,
    StateSpace,
)
from ._module_helpers import (
    body_identity as _body_identity,
    metadata_record as _metadata_record,
    normalize_source_selectors as _normalize_source_selectors,
    operator_record as _operator_record,
    param_record as _param_record,
    space_record as _space_record,
    symbolic_args as _symbolic_args,
)


class Module:
    """A model as typed spaces + a registry of typed operators (Spec 2, operator-first).

    A Module owns the RULES -- state/field spaces, parameters, aux declarations and the
    typed operators a Program composes by signature. The Simulation owns the DATA
    (grid, arrays, solvers, clock). :class:`pops.physics.Model` is the board-like
    authoring facade that populates a Module's registry; a Module can also be built
    directly with ``state_space`` / ``field_space`` / ``parameters`` / ``aux_fields`` /
    ``operator``. A generic Program bound to ``module.operator_registry()`` runs against
    any Module that provides operators with the expected signatures.
    """

    def __init__(self, name):
        self.name = str(name)
        self._state_spaces = {}
        self._field_spaces = {}
        self._params = {}
        self._aux = {}
        self._registry = OperatorRegistry()
        self._requirements = {}
        self._capabilities = {}
        self._invariants = {}
        self._diagnostics = {}
        self._primitive_defs = {}
        self._riemann_metadata = {
            "hllc": False,
            "roe": False,
            "hooks": {},
            "wave_speeds": None,
        }
        # Wave speeds for the Riemann solver of a compilable Module: {"x": [Expr], "y": [Expr]}
        # eigenvalues, or None (set via eigenvalues()). Carried so a pure Module is
        # self-contained for compiled problem codegen.
        self._eigenvalues = None

    # --- spaces ---
    def state_space(self, name="U", components=(), roles=None, layout="cell",
                    storage="multifab"):
        """Declare and return a :class:`StateSpace`."""
        space = StateSpace(name, components, roles, layout, storage)
        self._state_spaces[space.name] = space
        return space

    def field_space(self, name, components=(), layout="cell"):
        """Declare and return a :class:`FieldSpace`."""
        space = FieldSpace(name, components, layout)
        self._field_spaces[space.name] = space
        return space

    # --- parameters / aux ---
    def param(self, name, default=0.0, dtype="real", *, _kind="const"):
        """Declare and return one :class:`ParameterSpace`."""
        p = ParameterSpace(name, default, dtype, kind=_kind)
        self._params[p.name] = p
        return p

    def parameters(self, **defaults):
        """Declare several parameters by keyword; return ``{name: ParameterSpace}``."""
        return {k: self.param(k, v) for k, v in defaults.items()}

    def aux_field(self, name, kind="cell_scalar"):
        """Declare and return one :class:`AuxSpace`."""
        a = AuxSpace(name, kind)
        self._aux[a.name] = a
        return a

    def aux_fields(self, **kinds):
        """Declare several aux fields by keyword; return ``{name: AuxSpace}``."""
        return {k: self.aux_field(k, v) for k, v in kinds.items()}

    def primitive(self, name, expr):
        """Declare a primitive expression used by generated model capabilities.

        The board facade writes primitives here so Module-native codegen can emit pressure,
        wave-speed and Riemann hooks without reconstructing a legacy HyperbolicModel.
        """
        self._primitive_defs[str(name)] = expr
        return expr

    def primitive_defs(self):
        return dict(self._primitive_defs)

    def riemann_metadata(self, *, hllc=None, roe=None, hooks=None, wave_speeds=None):
        """Get or update Riemann capability metadata for Module-native codegen."""
        if hllc is not None:
            self._riemann_metadata["hllc"] = bool(hllc)
        if roe is not None:
            self._riemann_metadata["roe"] = bool(roe)
        if hooks is not None:
            self._riemann_metadata["hooks"] = dict(hooks)
        if wave_speeds is not None:
            self._riemann_metadata["wave_speeds"] = wave_speeds
        return dict(self._riemann_metadata)

    # --- operators ---
    def operator(self, name=None, signature=None, kind=None, capabilities=None,
                 requirements=None, lowering=None, expr=None):
        """Register a typed operator.

        Builder mode (``expr`` given) registers the operator immediately and returns the
        :class:`Operator`. Decorator mode (no ``expr``) executes the decorated function
        once with symbolic typed arguments, records the returned IR body, and returns the
        :class:`Operator`::

            @module.operator(name="explicit_rhs",
                             signature=(U, Fields) >> Rate(U), kind="local_rate")
            def explicit_rhs(U, fields):
                ...

        The decorated Python function is never stored as a runtime callback.
        """
        if name is None or signature is None or kind is None:
            raise ValueError("module.operator requires name, signature and kind")
        if not isinstance(signature, Signature):
            raise TypeError(
                "module.operator(%r): signature must be a Signature (use the >> sugar or "
                "Signature(inputs, output)); got %r" % (name, signature))

        def _register(body):
            op = Operator(name, kind, signature, capabilities=capabilities,
                          requirements=requirements, lowering=lowering, source="module",
                          body=body)
            self._registry.register(op)
            return op

        if expr is not None:
            return _register(expr)

        def decorator(func):
            body = func(*_symbolic_args(signature.inputs))
            return _register(body)

        return decorator

    def rate_operator(self, name, state_space="U", flux=True, sources=("default",), fluxes=None):
        """Register a composite ``local_rate`` operator ``R = -div F + sum(sources)`` from named
        sub-operators (the flux and the listed source operators). The ``lowering`` metadata carries
        the flux/sources/fluxes so ``P.call`` and the generated module compose it."""
        u = self._state_spaces.get(state_space) or StateSpace(state_space)
        srcs = _normalize_source_selectors(sources, who="Module.rate_operator(%r)" % name)
        fields = self._rate_operator_field_input(srcs)
        inputs = (u, fields) if fields is not None else (u,)
        op = Operator(name, "local_rate", Signature(inputs, RateSpace(u)),
                      capabilities={"local": False, "produces_rate": True, "supports_device": True},
                      lowering={"flux": bool(flux), "sources": srcs,
                                "fluxes": list(fluxes) if fluxes else None},
                      source="module")
        self._registry.register(op)
        return op

    def _rate_operator_field_input(self, source_names):
        """Return the field input required by selected local sources, if any."""
        if not source_names:
            return None
        for source_name in source_names:
            candidates = ["source_default", "default"] if source_name == "default" else [source_name]
            for candidate in candidates:
                if candidate not in self._registry:
                    continue
                op = self._registry.get(candidate)
                if op.kind != "local_source":
                    continue
                if len(op.signature.inputs) > 1:
                    return op.signature.inputs[1]
        return None

    def eigenvalues(self, x=None, y=None):
        """Get or declare per-direction Riemann eigenvalues.

        Called as ``module.eigenvalues(x=[...], y=[...])`` it declares the per-direction
        eigenvalue expressions carried by a pure Module. Called with no arguments it returns a
        shallow copy of the current declaration, or ``None`` when none was declared.
        """
        if x is None and y is None:
            return None if self._eigenvalues is None else {
                "x": list(self._eigenvalues["x"]),
                "y": list(self._eigenvalues["y"]),
            }
        if x is None or y is None:
            raise ValueError("Module.eigenvalues requires both x= and y=, or neither for a getter")
        self._eigenvalues = {"x": list(x), "y": list(y)}
        return self.eigenvalues()

    def adopt_registry(self, registry):
        """Use ``registry`` as this Module's operator registry. Returns ``self``."""
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("adopt_registry expects an OperatorRegistry")
        self._registry = registry
        return self

    def operator_registry(self):
        """The Module's :class:`OperatorRegistry` (bind it to a Program with P.bind_operators)."""
        return self._registry

    def operator_handle(self, name):
        """Return an inert :class:`OperatorHandle` for a registered operator."""
        return self._registry.get(name).handle()

    # --- module metadata ---
    def requirements(self, **items):
        """Get or update module-level compile/runtime requirements."""
        if items:
            self._requirements.update(items)
        return dict(self._requirements)

    def capabilities(self, **items):
        """Get or update module-level declared capabilities."""
        if items:
            self._capabilities.update(items)
        return dict(self._capabilities)

    def invariant(self, name, expression=None, **metadata):
        """Declare an inspectable model invariant."""
        record = {"name": str(name), "expression": expression, **metadata}
        self._invariants[str(name)] = record
        return record

    def diagnostic(self, name, expression=None, **metadata):
        """Declare an inspectable model diagnostic."""
        record = {"name": str(name), "expression": expression, **metadata}
        self._diagnostics[str(name)] = record
        return record

    def invariants(self):
        return dict(self._invariants)

    def diagnostics(self):
        return dict(self._diagnostics)

    # --- introspection (Spec 2, S2-5) ---
    def state_spaces(self):
        return dict(self._state_spaces)

    def field_spaces(self):
        return dict(self._field_spaces)

    def params(self):
        return dict(self._params)

    def aux(self):
        return dict(self._aux)

    def list_state_spaces(self):
        """Names of the declared state spaces."""
        return list(self._state_spaces)

    def list_field_spaces(self):
        """Names of the declared field spaces."""
        return list(self._field_spaces)

    def list_operators(self):
        """Operator names in registration (id) order."""
        return self._registry.names()

    def operator_signature(self, name):
        """The :class:`Signature` of operator ``name``."""
        return self._registry.get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name`` (aux / solver / params / ...)."""
        return dict(self._registry.get(name).requirements)

    def operator_capabilities(self, name, **caps):
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

    def validate(self):
        """Validate the inert Module authoring graph without runtime or codegen imports."""
        for op in self._registry:
            if callable(op.body):
                raise ValueError(
                    "operator %r stores a Python callable body; Module operators must capture "
                    "IR at declaration time" % (op.name,))
        return self

    check = validate

    def inspect(self):
        """Plain-dict, runtime-free view of this Module."""
        return {
            "name": self.name,
            "state_spaces": {k: _space_record(v) for k, v in self._state_spaces.items()},
            "field_spaces": {k: _space_record(v) for k, v in self._field_spaces.items()},
            "params": {k: _param_record(v) for k, v in self._params.items()},
            "aux": {k: {"name": v.name, "kind": v.kind} for k, v in self._aux.items()},
            "requirements": dict(self._requirements),
            "capabilities": dict(self._capabilities),
            "invariants": {k: _metadata_record(v) for k, v in self._invariants.items()},
            "diagnostics": {k: _metadata_record(v) for k, v in self._diagnostics.items()},
            "operators": {op.name: _operator_record(self._registry, op) for op in self._registry},
        }

    def module_hash(self):
        """Stable hash of the ModuleSpec for the compiled-artifact cache (Spec 2, S2-7).

        Folds the spaces, parameters, aux declarations and -- for every operator -- the name,
        kind, signature, capabilities, requirements and a canonical body identity. Sensitive to an
        operator body, signature, capability or space change; deterministic for an identical module.
        A spec2 tag namespaces it away from any spec1 key.
        """
        parts = ["spec2-module", self.name]
        for nm in sorted(self._state_spaces):
            s = self._state_spaces[nm]
            parts.append("state:%s:%s:%s" % (
                s.name, ",".join(s.components), sorted(s.roles.items())))
        for nm in sorted(self._field_spaces):
            f = self._field_spaces[nm]
            parts.append("field:%s:%s" % (f.name, ",".join(f.components)))
        for nm in sorted(self._params):
            p = self._params[nm]
            parts.append("param:%s:%r:%s" % (p.name, p.default, p.dtype))
        for nm in sorted(self._aux):
            a = self._aux[nm]
            parts.append("aux:%s:%s" % (a.name, a.kind))
        parts.append("requirements:%s" % sorted(self._requirements.items()))
        parts.append("capabilities:%s" % sorted(self._capabilities.items()))
        parts.append("invariants:%s" % sorted((k, _metadata_record(v)) for k, v in self._invariants.items()))
        parts.append("diagnostics:%s" % sorted((k, _metadata_record(v)) for k, v in self._diagnostics.items()))
        if self._eigenvalues is not None:
            for direction in ("x", "y"):
                parts.append("eig_%s:%s" % (
                    direction, ";".join(_body_identity(e) for e in self._eigenvalues[direction])))
        for op in self._registry:  # registration (id) order
            parts.append("op:%s:%s:%s:caps=%s:reqs=%s:body=%s" % (
                op.name, op.kind, repr(op.signature),
                sorted(op.capabilities.items()), sorted(op.requirements.items()),
                _body_identity(op.body)))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def __repr__(self):
        return "Module(%r, operators=[%s])" % (self.name, ", ".join(self._registry.names()))
