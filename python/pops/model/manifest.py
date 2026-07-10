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

from .manifest_data import freeze_json as _freeze_json, require_manifest_id, require_manifest_name, thaw_json as _thaw_json
from .manifest_support import (
    field_space_row as _field_space_row,
    native_catalog as _native_catalog,
    native_routes as _native_routes,
    param_row as _param_row,
    params_utilization as _params_utilization,
    space_name as _space_name,
    state_space_row as _state_space_row,
)
SCHEMA_VERSION = 3


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
        require_manifest_id(operator_id)
        object.__setattr__(self, "id", operator_id)
        object.__setattr__(self, "name", operator.name)
        object.__setattr__(self, "kind", operator.kind)
        object.__setattr__(self, "signature", _freeze_json(
            signature.to_data(), where="operator %s signature" % operator.name))
        object.__setattr__(self, "inputs", tuple(_space_name(i) for i in signature.inputs))
        object.__setattr__(self, "output", _space_name(signature.output))
        object.__setattr__(self, "capabilities", _freeze_json(
            operator.capabilities, where="operator %s capabilities" % operator.name))
        object.__setattr__(self, "requirements", _freeze_json(
            operator.requirements, where="operator %s requirements" % operator.name))
        lowering = operator.lowering
        object.__setattr__(self, "lowering_route", _freeze_json(
            lowering or {}, where="operator %s lowering" % operator.name))

    def __setattr__(self, name: Any, value: Any) -> None:  # frozen
        raise AttributeError("OperatorManifestEntry is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("OperatorManifestEntry is immutable")

    def to_dict(self) -> Any:
        """A plain-dict view of this row (JSON-ready)."""
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "signature": _thaw_json(self.signature), "inputs": list(self.inputs),
                "output": self.output, "capabilities": _thaw_json(self.capabilities),
                "requirements": _thaw_json(self.requirements),
                "lowering_route": _thaw_json(self.lowering_route)}

    def __repr__(self) -> str:
        return "OperatorManifestEntry(id=%d, name=%r, kind=%r)" % (self.id, self.name, self.kind)


class OperatorRegistryManifest:
    """The ordered operator manifest of a Module's registry (ADC-585).

    Carries the :class:`OperatorManifestEntry` rows in registration (id) order plus a stable
    :attr:`hash` over their serialisation; :meth:`describe` looks a row up by name and, on an
    unknown name, raises an error naming the OPERATOR and the registry contents (never a historical
    tag). :meth:`to_dict` / iteration expose the rows.
    """

    __slots__ = ("_entries", "_aliases")

    def __init__(self, entries: Any, aliases: Any = None) -> None:
        frozen = tuple(entries)
        if any(not isinstance(entry, OperatorManifestEntry) for entry in frozen):
            raise TypeError("OperatorRegistryManifest entries must be OperatorManifestEntry values")
        object.__setattr__(self, "_entries", frozen)
        object.__setattr__(self, "_aliases", _freeze_json(
            aliases or {}, where="operator registry aliases"))

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError("OperatorRegistryManifest is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("OperatorRegistryManifest is immutable")

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

    def aliases(self) -> Any:
        """Detached public-alias table included in module identity."""
        return _thaw_json(self._aliases)

    @property
    def hash(self) -> str:
        """A stable sha256 over the ordered, canonically serialised entries."""
        blob = json.dumps(
            {"entries": self.to_dict(), "aliases": self.aliases()},
            sort_keys=True, separators=(",", ":"), allow_nan=False)
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

    __slots__ = (
        "schema_version", "name", "state_spaces", "field_spaces", "params", "aux",
        "has_eigenvalues", "operators", "capabilities", "native_routes", "native_catalog",
        "abi_requirements", "params_utilization",
    )

    def __init__(self, *, name: Any, state_spaces: Any, field_spaces: Any, params: Any, aux: Any,
                 has_eigenvalues: Any, operators: Any, capabilities: Any, native_routes: Any,
                 native_catalog: Any, abi_requirements: Any, params_utilization: Any = None) -> None:
        if not isinstance(operators, OperatorRegistryManifest):
            raise TypeError("ModuleManifest operators must be an OperatorRegistryManifest")
        require_manifest_name(name)
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "state_spaces", _freeze_json(
            state_spaces, where="module state_spaces"))
        object.__setattr__(self, "field_spaces", _freeze_json(
            field_spaces, where="module field_spaces"))
        object.__setattr__(self, "params", _freeze_json(params, where="module params"))
        object.__setattr__(self, "aux", _freeze_json(aux, where="module aux"))
        object.__setattr__(self, "has_eigenvalues", _freeze_json(
            has_eigenvalues, where="module has_eigenvalues"))
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "capabilities", _freeze_json(
            capabilities, where="module capabilities"))
        object.__setattr__(self, "native_routes", _freeze_json(
            native_routes, where="module native_routes"))
        object.__setattr__(self, "native_catalog", _freeze_json(
            native_catalog, where="module native_catalog"))
        object.__setattr__(self, "abi_requirements", _freeze_json(
            abi_requirements, where="module abi_requirements"))
        # Runtime-param capacity utilization (ADC-610): {count, limit, status}. Additive; a reader that
        # ignores it is unaffected.
        object.__setattr__(self, "params_utilization", _freeze_json(
            params_utilization or _params_utilization(self.params),
            where="module params_utilization"))

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError("ModuleManifest is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("ModuleManifest is immutable")

    def with_abi_key(self, abi_key: Any) -> "ModuleManifest":
        """Return a new manifest with the compile-time ABI key bound.

        The Module manifest remains an immutable build result; CompiledProblem
        functionally derives its artifact-specific copy instead of mutating a
        shared model manifest after hashing/introspection.
        """
        requirements = _thaw_json(self.abi_requirements)
        requirements["abi_key"] = abi_key
        return ModuleManifest(
            name=self.name,
            state_spaces=_thaw_json(self.state_spaces),
            field_spaces=_thaw_json(self.field_spaces),
            params=_thaw_json(self.params),
            aux=_thaw_json(self.aux),
            has_eigenvalues=_thaw_json(self.has_eigenvalues),
            operators=self.operators,
            capabilities=_thaw_json(self.capabilities),
            native_routes=_thaw_json(self.native_routes),
            native_catalog=_thaw_json(self.native_catalog),
            abi_requirements=requirements,
            params_utilization=_thaw_json(self.params_utilization),
        )

    def to_dict(self) -> Any:
        """A plain-dict view of the whole manifest (JSON-ready)."""
        return {"schema_version": self.schema_version, "name": self.name,
                "state_spaces": _thaw_json(self.state_spaces),
                "field_spaces": _thaw_json(self.field_spaces),
                "params": _thaw_json(self.params),
                "params_utilization": _thaw_json(self.params_utilization),
                "aux": _thaw_json(self.aux),
                "has_eigenvalues": _thaw_json(self.has_eigenvalues),
                "operators": self.operators.to_dict(),
                "operator_aliases": self.operators.aliases(),
                "capabilities": _thaw_json(self.capabilities),
                "native_routes": _thaw_json(self.native_routes),
                "native_catalog": _thaw_json(self.native_catalog),
                "abi_requirements": _thaw_json(self.abi_requirements)}

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        """Serialise :meth:`to_dict` to JSON; write to @p path if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def hash(self) -> str:
        """A stable sha256 over the canonically serialised manifest (adding an operator changes it)."""
        blob = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return ("ModuleManifest(name=%r, operators=[%s])"
                % (self.name, ", ".join(self.operators.names())))


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
    operators = OperatorRegistryManifest(entries, aliases=registry.aliases())

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
    from pops.ir.literals import scalar_data
    conserved, created = list(conserved), list(created)
    freq = getattr(compiled, "frequency", 0.0) if frequency is None else frequency
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
        "frequency": {"constant_mu": scalar_data(freq), "per_cell": per_cell},
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
