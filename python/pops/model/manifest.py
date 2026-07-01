"""pops.model.manifest -- the Module / Operator manifests that replace ModelSpec (ADC-585).

ModelSpec (``pops._bootstrap.ModelSpec``) is a flat C++ POD the ``pops.Model(...)`` sugar builds
for the legacy native ``add_block`` bridge; it is NOT the target representation. The target
representation of a model is the operator-first :class:`pops.model.Module` (typed spaces + a
registry of typed operators) compiled to a Problem. This module serialises a Module into a
self-describing, JSON-ready MANIFEST -- the Python-side counterpart of the per-operator metadata
the generated ``.so`` carries (``include/pops/runtime/program/module_metadata.hpp``): each operator
by its stable integer :class:`~pops.model.registry.OperatorId`, its kind, its typed signature, its
inputs / output space names, its capabilities / requirements and its lowering route.

Three inert value objects and a read-only builder:

  - :class:`OperatorManifestEntry` -- one operator's manifest row (built from a registry Operator).
  - :class:`OperatorRegistryManifest` -- the ordered entries plus a stable :attr:`hash` and a
    :meth:`describe` lookup that cites the OPERATOR and the registry contents (never a historical
    tag) when a name is unknown.
  - :class:`ModuleManifest` -- the whole Module: spaces / params / aux / eigenvalues / operators /
    capabilities, plus the native route-registry components (from :mod:`pops.runtime.routes`) and
    an ABI-requirements slot the compile seam binds at compile time.

The :func:`build_module_manifest` builder reads the Module through its PUBLIC accessors only
(``state_spaces`` / ``field_spaces`` / ``params`` / ``aux`` / ``operator_registry``); it never
mutates the Module. Imports only the standard library plus the sibling model types and
:mod:`pops.runtime.routes` (itself import-free), so a manifest can be built without the compiled
``_pops`` extension.
"""
import hashlib
import json

SCHEMA_VERSION = 1


def _space_name(item):
    """The manifest name of a signature input / output item.

    A :class:`~pops.model.spaces.Space` reports its ``name``; an operator-valued type
    (``LocalLinearOperator`` / ``MatrixFreeOperator``) its ``domain -> range`` shape; a
    :class:`~pops.model.bundles.RateBundle` its per-block names; anything else its ``repr``.
    """
    name = getattr(item, "name", None)
    if name is not None:
        return name
    domain = getattr(item, "domain_name", None)
    range_ = getattr(item, "range_name", None)
    if domain is not None and range_ is not None:
        return "%s->%s" % (domain, range_)
    keys = getattr(item, "keys", None)
    if callable(keys):  # a RateBundle (multi-output): name it by its participating blocks
        return "RateBundle{%s}" % ", ".join(keys())
    return repr(item)


class OperatorManifestEntry:
    """One operator's manifest row (ADC-585), built from a registry :class:`Operator`.

    A frozen, JSON-ready value object. ``id`` is the stable integer OperatorId (the registration
    index); ``inputs`` / ``output`` are the space (or bundle) NAMES of the typed signature -- a
    multi-output ``coupled_rate`` names its :class:`~pops.model.bundles.RateBundle` output without
    any new flat field. ``capabilities`` / ``requirements`` are copies of the operator dicts;
    ``lowering_route`` is the operator's lowering hint (a dict for a composite rate, else a string).
    """

    __slots__ = ("id", "name", "kind", "signature", "inputs", "output",
                 "capabilities", "requirements", "lowering_route")

    def __init__(self, operator, operator_id):
        signature = operator.signature
        object.__setattr__(self, "id", int(operator_id))
        object.__setattr__(self, "name", operator.name)
        object.__setattr__(self, "kind", operator.kind)
        object.__setattr__(self, "signature", repr(signature))
        object.__setattr__(self, "inputs", [_space_name(i) for i in signature.inputs])
        object.__setattr__(self, "output", _space_name(signature.output))
        object.__setattr__(self, "capabilities", dict(operator.capabilities))
        object.__setattr__(self, "requirements", dict(operator.requirements))
        lowering = operator.lowering
        object.__setattr__(self, "lowering_route", dict(lowering) if lowering else {})

    def __setattr__(self, name, value):  # frozen
        raise AttributeError("OperatorManifestEntry is immutable")

    def to_dict(self):
        """A plain-dict view of this row (JSON-ready)."""
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "signature": self.signature, "inputs": list(self.inputs),
                "output": self.output, "capabilities": dict(self.capabilities),
                "requirements": dict(self.requirements),
                "lowering_route": dict(self.lowering_route)}

    def __repr__(self):
        return "OperatorManifestEntry(id=%d, name=%r, kind=%r)" % (self.id, self.name, self.kind)


class OperatorRegistryManifest:
    """The ordered operator manifest of a Module's registry (ADC-585).

    Carries the :class:`OperatorManifestEntry` rows in registration (id) order plus a stable
    :attr:`hash` over their serialisation; :meth:`describe` looks a row up by name and, on an
    unknown name, raises an error naming the OPERATOR and the registry contents (never a historical
    tag). :meth:`to_dict` / iteration expose the rows.
    """

    def __init__(self, entries):
        self._entries = list(entries)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def names(self):
        """The operator names in registration (id) order."""
        return [entry.name for entry in self._entries]

    def describe(self, name):
        """The :class:`OperatorManifestEntry` named @p name, or raise citing operator + registry.

        The refusal names the requested OPERATOR and the operators this registry manifest DOES
        carry (never a historical tag / route table), so a typo is diagnosable from the manifest
        alone.
        """
        for entry in self._entries:
            if entry.name == name:
                return entry
        known = ", ".join(self.names()) or "<none>"
        raise KeyError(
            "operator %r is not in this module's operator registry (registered: %s)"
            % (name, known))

    def to_dict(self):
        """A plain list-of-dicts view of the rows in id order (JSON-ready)."""
        return [entry.to_dict() for entry in self._entries]

    @property
    def hash(self):
        """A stable sha256 over the ordered, canonically serialised entries."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self):
        return "OperatorRegistryManifest(%s)" % ", ".join(self.names())


class ModuleManifest:
    """The self-describing manifest of a :class:`pops.model.Module` (ADC-585).

    Replaces ModelSpec as the central, inspectable representation of a model: the state / field
    spaces, the parameters, the aux declarations, the per-direction eigenvalue presence, the typed
    :class:`OperatorRegistryManifest`, the Module capabilities, plus the NATIVE route-registry
    components (version / hash / signature from :mod:`pops.runtime.routes`), the NATIVE brick catalog
    component (version + canonical ids from :mod:`pops.runtime.brick_catalog`, ADC-586, so a generated
    artifact can reference the native bricks by id) and an ABI-requirements slot.
    ``abi_requirements["abi_key"]`` is left ``None`` here (the ABI key is bound at compile time); the
    compile seam fills it from the CompiledProblem handle. Every field is JSON-ready.
    """

    def __init__(self, *, name, state_spaces, field_spaces, params, aux, has_eigenvalues,
                 operators, capabilities, native_routes, native_catalog, abi_requirements):
        self.schema_version = SCHEMA_VERSION
        self.name = name
        self.state_spaces = dict(state_spaces)
        self.field_spaces = dict(field_spaces)
        self.params = dict(params)
        self.aux = dict(aux)
        self.has_eigenvalues = dict(has_eigenvalues)
        self.operators = operators              # OperatorRegistryManifest
        self.capabilities = dict(capabilities)
        self.native_routes = dict(native_routes)
        self.native_catalog = dict(native_catalog)
        self.abi_requirements = dict(abi_requirements)

    def to_dict(self):
        """A plain-dict view of the whole manifest (JSON-ready)."""
        return {"schema_version": self.schema_version, "name": self.name,
                "state_spaces": {n: dict(s) for n, s in self.state_spaces.items()},
                "field_spaces": {n: dict(s) for n, s in self.field_spaces.items()},
                "params": {n: dict(p) for n, p in self.params.items()},
                "aux": dict(self.aux), "has_eigenvalues": dict(self.has_eigenvalues),
                "operators": self.operators.to_dict(),
                "capabilities": dict(self.capabilities),
                "native_routes": dict(self.native_routes),
                "native_catalog": dict(self.native_catalog),
                "abi_requirements": dict(self.abi_requirements)}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to @p path if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def hash(self):
        """A stable sha256 over the canonically serialised manifest (adding an operator changes it)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self):
        return ("ModuleManifest(name=%r, operators=[%s])"
                % (self.name, ", ".join(self.operators.names())))


def _state_space_row(space):
    """The manifest row of a state space: components / roles / layout / storage."""
    return {"components": list(space.components), "roles": dict(getattr(space, "roles", {}) or {}),
            "layout": getattr(space, "layout", "cell"),
            "storage": getattr(space, "storage", "multifab")}


def _field_space_row(space):
    """The manifest row of a field space: components / layout."""
    return {"components": list(space.components), "layout": getattr(space, "layout", "cell")}


def _param_row(param):
    """The manifest row of a parameter: default / dtype / kind (kind if the param carries one)."""
    row = {"default": getattr(param, "default", getattr(param, "value", None)),
           "dtype": getattr(param, "dtype", "real")}
    kind = getattr(param, "kind", None)
    if kind is not None:
        row["kind"] = kind
    return row


def _native_routes():
    """The native route-registry components (version / hash / signature) from pops.runtime.routes."""
    from pops.runtime.routes import (ROUTE_REGISTRY_VERSION, route_registry_hash,
                                      route_registry_signature)
    return {"version": ROUTE_REGISTRY_VERSION, "hash": route_registry_hash(),
            "signature": route_registry_signature()}


def _native_catalog():
    """The builtin native brick catalog component (version + ids) from pops.runtime.brick_catalog.

    The codegen-facing native-catalog manifest (ADC-586): the route-registry version pins the
    catalog vocabulary, and the ids are the canonical native bricks a generated artifact can
    reference. Imported in-builder (like the routes import) so the manifest stays buildable without
    the compiled ``_pops`` extension.
    """
    from pops.runtime.brick_catalog import brick_catalog
    from pops.runtime.routes import ROUTE_REGISTRY_VERSION
    return {"version": ROUTE_REGISTRY_VERSION,
            "bricks": [entry["id"] for entry in brick_catalog()]}


def build_module_manifest(module):
    """Build the :class:`ModuleManifest` of @p module WITHOUT mutating it (ADC-585).

    Reads the Module through its public accessors only -- ``state_spaces`` / ``field_spaces`` /
    ``params`` / ``aux`` / ``operator_registry`` -- and folds each typed operator into an
    :class:`OperatorManifestEntry` keyed by its stable integer OperatorId (registration order). The
    native route-registry components come from :mod:`pops.runtime.routes`; ``abi_requirements``
    carries the route-registry signature and an ``abi_key`` slot the compile seam binds later
    (``None`` here -- the ABI key is a compile-time fact, not a Module property).
    """
    registry = module.operator_registry()
    entries = [OperatorManifestEntry(op, registry.id_of(op.name)) for op in registry]
    operators = OperatorRegistryManifest(entries)

    state_spaces = {n: _state_space_row(s) for n, s in module.state_spaces().items()}
    field_spaces = {n: _field_space_row(s) for n, s in module.field_spaces().items()}
    params = {n: _param_row(p) for n, p in module.params().items()}
    aux = {n: getattr(a, "kind", "cell_scalar") for n, a in module.aux().items()}

    eigenvalues = getattr(module, "_eigenvalues", None)
    has_eigenvalues = {"x": bool(eigenvalues and eigenvalues.get("x")),
                       "y": bool(eigenvalues and eigenvalues.get("y"))}

    # Whatever capability surface the Module exposes; today a Module carries per-operator
    # capabilities (in the entries above) rather than a module-level dict, so a Module accessor is
    # used when present and an empty dict otherwise (never a fabricated flag).
    caps_fn = getattr(module, "capabilities", None)
    capabilities = dict(caps_fn()) if callable(caps_fn) else {}

    routes = _native_routes()
    catalog = _native_catalog()
    abi_requirements = {"route_registry_signature": routes["signature"], "abi_key": None}

    return ModuleManifest(
        name=module.name, state_spaces=state_spaces, field_spaces=field_spaces, params=params,
        aux=aux, has_eigenvalues=has_eigenvalues, operators=operators,
        capabilities=capabilities, native_routes=routes, native_catalog=catalog,
        abi_requirements=abi_requirements)


def _is_manifestable_module(obj):
    """True when @p obj exposes the full Module accessor surface the builder reads.

    A dsl.Model DELEGATES ``operator_registry`` to its backing Module but does NOT expose the space
    accessors, so the manifest builder needs the ``state_spaces`` accessor to distinguish a real
    :class:`Module` from a Model facade.
    """
    return obj is not None and hasattr(obj, "operator_registry") and hasattr(obj, "state_spaces")


def module_manifest_of(model_or_module):
    """The :class:`ModuleManifest` of a model-or-Module, or ``None`` (the compile-seam helper).

    Accepts a :class:`pops.model.Module` directly, or a dsl.Model that carries its backing Module on
    ``.module``; a bare dsl.Model with no Module (or ``None``) yields ``None`` -- the manifest is
    honestly absent, never fabricated. Read-only: it delegates to :func:`build_module_manifest`.
    """
    module = model_or_module
    if not _is_manifestable_module(module):
        module = getattr(module, "module", None)
    if not _is_manifestable_module(module):
        return None
    return build_module_manifest(module)


__all__ = ["OperatorManifestEntry", "OperatorRegistryManifest", "ModuleManifest",
           "build_module_manifest", "module_manifest_of", "SCHEMA_VERSION"]
