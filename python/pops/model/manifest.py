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
from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = 1


def _space_name(item: Any) -> Any:
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
    keys: Any = getattr(item, "keys", None)
    if callable(keys):  # a RateBundle (multi-output): name it by its participating blocks
        block_names: Any = keys()
        return "RateBundle{%s}" % ", ".join(block_names)
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

    def __init__(self, operator: Any, operator_id: Any) -> None:
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

    def __setattr__(self, name: Any, value: Any) -> None:  # frozen
        raise AttributeError("OperatorManifestEntry is immutable")

    def to_dict(self) -> Any:
        """A plain-dict view of this row (JSON-ready)."""
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "signature": self.signature, "inputs": list(self.inputs),
                "output": self.output, "capabilities": dict(self.capabilities),
                "requirements": dict(self.requirements),
                "lowering_route": dict(self.lowering_route)}

    def __repr__(self) -> str:
        return "OperatorManifestEntry(id=%d, name=%r, kind=%r)" % (self.id, self.name, self.kind)


class OperatorRegistryManifest:
    """The ordered operator manifest of a Module's registry (ADC-585).

    Carries the :class:`OperatorManifestEntry` rows in registration (id) order plus a stable
    :attr:`hash` over their serialisation; :meth:`describe` looks a row up by name and, on an
    unknown name, raises an error naming the OPERATOR and the registry contents (never a historical
    tag). :meth:`to_dict` / iteration expose the rows.
    """

    def __init__(self, entries: Any) -> None:
        self._entries = list(entries)

    def __iter__(self) -> Any:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> Any:
        """The operator names in registration (id) order."""
        return [entry.name for entry in self._entries]

    def describe(self, name: Any) -> Any:
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

    def to_dict(self) -> Any:
        """A plain list-of-dicts view of the rows in id order (JSON-ready)."""
        return [entry.to_dict() for entry in self._entries]

    @property
    def hash(self) -> str:
        """A stable sha256 over the ordered, canonically serialised entries."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
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

    def __init__(self, *, name: Any, state_spaces: Any, field_spaces: Any, params: Any, aux: Any,
                 has_eigenvalues: Any, operators: Any, capabilities: Any, native_routes: Any,
                 native_catalog: Any, abi_requirements: Any, params_utilization: Any = None) -> None:
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
        # Runtime-param capacity utilization (ADC-610): {count, limit, status}. Additive; a reader that
        # ignores it is unaffected (SCHEMA_VERSION unchanged for a purely-additive field).
        self.params_utilization = dict(params_utilization or _params_utilization(self.params))

    def to_dict(self) -> Any:
        """A plain-dict view of the whole manifest (JSON-ready)."""
        return {"schema_version": self.schema_version, "name": self.name,
                "state_spaces": {n: dict(s) for n, s in self.state_spaces.items()},
                "field_spaces": {n: dict(s) for n, s in self.field_spaces.items()},
                "params": {n: dict(p) for n, p in self.params.items()},
                "params_utilization": dict(self.params_utilization),
                "aux": dict(self.aux), "has_eigenvalues": dict(self.has_eigenvalues),
                "operators": self.operators.to_dict(),
                "capabilities": dict(self.capabilities),
                "native_routes": dict(self.native_routes),
                "native_catalog": dict(self.native_catalog),
                "abi_requirements": dict(self.abi_requirements)}

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        """Serialise :meth:`to_dict` to JSON; write to @p path if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def hash(self) -> str:
        """A stable sha256 over the canonically serialised manifest (adding an operator changes it)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return ("ModuleManifest(name=%r, operators=[%s])"
                % (self.name, ", ".join(self.operators.names())))


def _state_space_row(space: Any) -> Any:
    """The manifest row of a state space: components / roles / layout / storage."""
    return {"components": list(space.components), "roles": dict(getattr(space, "roles", {}) or {}),
            "layout": getattr(space, "layout", "cell"),
            "storage": getattr(space, "storage", "multifab")}


def _field_space_row(space: Any) -> Any:
    """The manifest row of a field space: components / layout."""
    return {"components": list(space.components), "layout": getattr(space, "layout", "cell")}


def _param_row(param: Any) -> Any:
    """The manifest row of a parameter: default / dtype / kind (kind if the param carries one)."""
    row = {"default": getattr(param, "default", getattr(param, "value", None)),
           "dtype": getattr(param, "dtype", "real")}
    kind = getattr(param, "kind", None)
    if kind is not None:
        row["kind"] = kind
    return row


def _params_utilization(params: Any) -> Any:
    """Runtime-param capacity utilization row (ADC-610): {count, limit, status}. Surfaces the
    previously-hidden kMaxRuntimeParams bound so an artifact's headroom is introspectable. @p params is
    the already-built {name: row} map; count = number of kind='runtime' params. status is 'ok' below the
    limit, 'at_limit' exactly at it, 'exceeded' above (the codegen would already have refused it -- the
    row records the fact rather than fabricating a pass)."""
    from pops.physics.aux import max_runtime_params  # lazy: keep manifest import-light
    limit = max_runtime_params()
    count = sum(1 for row in params.values() if row.get("kind") == "runtime")
    status = "ok" if count < limit else ("at_limit" if count == limit else "exceeded")
    return {"count": count, "limit": limit, "status": status}


def _native_routes() -> Any:
    """The native route-registry components (version / hash / signature) from pops.runtime.routes."""
    from pops.runtime.routes import (ROUTE_REGISTRY_VERSION, route_registry_hash,
                                      route_registry_signature)
    return {"version": ROUTE_REGISTRY_VERSION, "hash": route_registry_hash(),
            "signature": route_registry_signature()}


def _native_catalog() -> Any:
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


def build_module_manifest(module: Any) -> Any:
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
    caps_fn: Any = getattr(module, "capabilities", None)
    caps_value: Any = caps_fn() if callable(caps_fn) else {}
    capabilities = dict(caps_value)

    routes = _native_routes()
    catalog = _native_catalog()
    abi_requirements = {"route_registry_signature": routes["signature"], "abi_key": None}

    return ModuleManifest(
        name=module.name, state_spaces=state_spaces, field_spaces=field_spaces, params=params,
        aux=aux, has_eigenvalues=has_eigenvalues, operators=operators,
        capabilities=capabilities, native_routes=routes, native_catalog=catalog,
        abi_requirements=abi_requirements, params_utilization=_params_utilization(params))


# Program IR ops that lower to the GENERIC condensed-implicit solve (ADC-637): the inline block_inverse
# emitters (block_inverse.hpp), NO coupling/schur call. Kept in lock-step with
# pops.codegen.program_emit_kernels._CONDENSED_OPS (the codegen block_inverse include gate); duplicated
# here (not imported) so the manifest stays buildable without pulling the codegen package.
_CONDENSED_ROUTE_OPS = ("condensed_coeffs", "condensed_rhs", "condensed_reconstruct")


def condensed_route_manifest(program: Any) -> Any:
    """The STRUCTURED condensed-implicit route descriptor of a compiled time @p program, or ``None``
    when its IR carries no condensed op (ADC-637).

    A compiled Program that lowers the ``pops.lib.time.condensed_schur`` macro drives the generic
    condensed-implicit electrostatic-Lorentz push, authored entirely in the DSL and emitted inline via
    ``pops::detail::block_inverse`` -- no coupling/schur call. This descriptor makes that route -- and its
    documented LIMITATION -- machine-visible in the manifest instead of buried in the macro docstring: a
    reader can tell from the manifest alone that the .so pulls ``numerics/linalg/block_inverse.hpp`` and
    that the route is a documented near-match to the native ``pops.CondensedSchur`` stepper (bit-exact
    only at ``theta == 1`` for the FIRST step; the cross-step ``phi^n`` warm-start carry is deferred).
    Purely-additive, JSON-ready; a condensed-free Program yields ``None`` (the route is honestly absent).

    ``ops`` lists the condensed op names actually present (id-stable IR op tags); ``operator_header`` is
    the intrinsic the .so includes; ``limitations`` is a structured record of the theta constraint so a
    tool can gate on it without parsing prose.
    """
    values = getattr(program, "_values", None)
    if values is None:
        return None
    present = [v.op for v in values if v.op in _CONDENSED_ROUTE_OPS]
    if not present:
        return None
    # De-duplicate while preserving first-seen order (a Program may carry several condensed ops).
    seen = []
    for op in present:
        if op not in seen:
            seen.append(op)
    return {
        "route": "condensed_implicit",
        "ops": seen,
        "operator_module": "pops::detail (block_inverse)",
        "operator_header": "pops/numerics/linalg/block_inverse.hpp",
        "native_reference": "pops.CondensedSchur",
        "limitations": {
            "bit_exact_theta": 1.0,
            "bit_exact_scope": "first_step",
            "theta_range": "(0, 1]",
            "note": ("near-match to the native CondensedSchur stepper; matrix-free BiCGStab without "
                     "the native GeometricMG preconditioner, and the cross-step phi^n warm-start "
                     "carry is deferred, so bit-exactness holds only for the first step at theta == 1"),
        },
    }


def coupling_operator_manifest(compiled: Any, conserved: Any = (), created: Any = (),
                               frequency: Any = None) -> Any:
    """The STRUCTURED manifest row of a compiled inter-species coupling operator (ADC-595).

    A named coupling preset (Ionization / Collision / ThermalExchange) or a
    :class:`~pops.physics.multispecies.CompiledCoupledSource` lowers to the ONE generic coupled-source
    representation. This row makes that operator -- its declared CONSERVATION contract, its FREQUENCY
    bound and its capacity UTILIZATION against the frozen ``kCsMax*`` bounds -- machine-visible in the
    ModuleManifest / report, exactly as :func:`condensed_route_manifest` surfaces the condensed-implicit route.
    Purely additive, JSON-ready; reads @p compiled through its public attributes only (no mutation, no
    ``_pops`` import).

    ``conservation`` records the DECLARED conserved / created roles (empty both -> "unchecked", a raw
    user CoupledSource); ``frequency`` records the constant mu bound and whether a per-cell mu(U) program
    is carried; ``utilization`` is the compiled source's ``utilization()`` (registers / terms / program
    against the C++ fixed-array capacities).
    """
    conserved, created = list(conserved), list(created)
    freq = getattr(compiled, "frequency", 0.0) if frequency is None else float(frequency)
    per_cell = bool(getattr(compiled, "freq_prog_ops", []) or getattr(compiled, "freq_prog_args", []))
    return {
        "route": "coupled_source",
        "name": getattr(compiled, "name", "coupled_source"),
        "operator_module": "pops::CouplingOperator",
        "operator_header": "pops/coupling/source/coupling_operator.hpp",
        "in_fields": list(zip(getattr(compiled, "in_blocks", []), getattr(compiled, "in_roles", []),
                              strict=True)),
        "out_terms": list(zip(getattr(compiled, "out_blocks", []), getattr(compiled, "out_roles", []),
                              strict=True)),
        "conservation": {"conserved_roles": conserved, "created_roles": created,
                         "unchecked": not (conserved or created)},
        "frequency": {"constant_mu": freq, "per_cell": per_cell},
        "utilization": compiled.utilization() if hasattr(compiled, "utilization") else None,
    }


def _is_manifestable_module(obj: Any) -> bool:
    """True when @p obj exposes the full Module accessor surface the builder reads.

    A dsl.Model DELEGATES ``operator_registry`` to its backing Module but does NOT expose the space
    accessors, so the manifest builder needs the ``state_spaces`` accessor to distinguish a real
    :class:`Module` from a Model facade.
    """
    return obj is not None and hasattr(obj, "operator_registry") and hasattr(obj, "state_spaces")


def module_manifest_of(model_or_module: Any) -> Any:
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
           "build_module_manifest", "module_manifest_of", "condensed_route_manifest",
           "coupling_operator_manifest", "SCHEMA_VERSION"]
