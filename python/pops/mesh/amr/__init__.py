"""pops.mesh.amr -- typed AMR policy descriptors (Spec 5 sec.5.11 / sec.8).

AMR is mesh / runtime infrastructure, not physics. These descriptors declare the
refinement criteria, patch clustering, regrid cadence, proper nesting, and the
checkpoint / output policy that :class:`pops.mesh.layouts.AMR` consumes and the C++
runtime executes after validation. Spec 5 (sec.8.6) replaces the string forms
(``set_refinement(0.05, variable="rho")`` / ``set_phi_refinement(0.5)``) with typed
criteria: ``Refine.on(block.role(Density)).above(0.05)`` and ``TagUnion(...)``.

Everything here is an inert descriptor; nothing tags a cell in Python.
"""
from __future__ import annotations

from typing import Any

from .._descriptor import Availability, MeshDescriptor

# Current native AMR capability envelope (Spec 5 sec.8.7): the production AMR route
# supports 2 levels at refinement ratio 2. A request beyond this is refused BEFORE the
# runtime, with a clear message, rather than silently clamped.
NATIVE_MAX_LEVELS = 2
NATIVE_RATIOS = (2,)


def _require_handle(reference: Any, where: str) -> Any:
    """Require one typed declaration reference without adding a mesh -> model import edge."""
    from pops.model import Handle

    if not isinstance(reference, Handle):
        raise TypeError(
            "%s requires a pops.model.Handle declaration reference, got %r; names and strings "
            "are not declaration identities" % (where, type(reference).__name__))
    return reference


def _require_value_handle(reference: Any, where: str) -> Any:
    """Require a Handle that represents a readable scientific value."""
    reference = _require_handle(reference, where)
    if getattr(reference, "expression_readable", None) is not True:
        raise TypeError(
            "%s requires a value-readable Handle, got %s(kind=%r); block, operator, endpoint, "
            "and other control identities are not scientific values"
            % (where, type(reference).__name__, reference.kind))
    return reference


def _resolve_handle_reference(reference: Any, resolver: Any, where: str) -> Any:
    """Resolve an authoring handle through the owning assembly's small resolver protocol."""
    reference = _require_value_handle(reference, where)
    resolve = resolver if callable(resolver) else getattr(resolver, "resolve", None)
    if not callable(resolve):
        raise TypeError(
            "%s reference resolution requires a callable resolver or an object exposing "
            "resolve(handle), got %r"
            % (where, type(resolver).__name__))
    resolved = _require_value_handle(resolve(reference), where)
    if not resolved.is_resolved:
        raise ValueError(
            "%s resolver returned an authoring-owned Handle; references must be canonical before "
            "compile" % where)
    return resolved


def _require_refine_subject(subject: Any) -> Any:
    """Require a typed Handle or a semantic expression with reference resolution."""
    from pops.ir import Expr
    from pops.model import Handle

    if isinstance(subject, Handle):
        return _require_value_handle(subject, "Refine.on(...)")
    if isinstance(subject, Expr):
        references = subject.declaration_references()
        if not references:
            raise TypeError(
                "Refine.on(...) expression has no typed declaration Handle leaves; build it from "
                "ValueExpr(handle), not Var/free names or strings")
        for reference in references:
            _require_value_handle(reference, "Refine.on(...) expression")
        return subject
    if isinstance(subject, str) or not callable(getattr(subject, "resolve_references", None)):
        raise TypeError(
            "Refine.on(...) requires a pops.model.Handle or a semantic expression exposing "
            "resolve_references(resolver), got %r; names and strings are not declaration "
            "identities" % type(subject).__name__)
    return subject


def _resolve_refine_subject(subject: Any, resolver: Any) -> Any:
    """Resolve every declaration leaf without lowering the scientific indicator."""
    from pops.model import Handle

    subject = _require_refine_subject(subject)
    if isinstance(subject, Handle):
        return _resolve_handle_reference(subject, resolver, "Refine.on(...)")
    resolved = subject.resolve_references(resolver)
    return _require_refine_subject(resolved)


def _refine_subject_options(subject: Any) -> dict:
    """Structured inspection that keeps a semantic expression distinct from a Handle."""
    from pops.model import Handle

    if isinstance(subject, Handle):
        return {"reference_type": "handle", "handle": _stable_handle_projection(subject)}
    return {
        "reference_type": "expression",
        "expression_type": "%s.%s" % (type(subject).__module__, type(subject).__qualname__),
        "expression": _semantic_projection(subject, set()),
    }


def _stable_handle_projection(handle: Any) -> dict:
    """Deterministic Handle inspection that never exposes an authoring capability token."""
    canonical_owner = handle.owner_path.canonical()
    return {
        "kind": handle.kind,
        "local_id": handle.local_id,
        "owner_path": str(canonical_owner),
    }


def _semantic_projection(value: Any, active: set[int]) -> Any:
    """Project an indicator graph structurally without repr() or local-authority text."""
    from collections.abc import Mapping
    from decimal import Decimal
    from fractions import Fraction
    from pops.model import Handle

    if isinstance(value, Handle):
        return {"handle": _stable_handle_projection(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (Decimal, Fraction)):
        return {"type": type(value).__name__, "value": str(value)}
    if isinstance(value, (tuple, list)):
        return [_semantic_projection(item, active) for item in value]
    if isinstance(value, Mapping):
        entries = [
            {"key": _semantic_projection(key, active),
             "value": _semantic_projection(item, active)}
            for key, item in value.items()
        ]
        return {"entries": sorted(entries, key=repr)}
    if isinstance(value, (set, frozenset)):
        projected = [_semantic_projection(item, active) for item in value]
        return sorted(projected, key=repr)

    object_id = id(value)
    if object_id in active:
        return {"type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                "cycle": True}
    active.add(object_id)
    try:
        to_data = getattr(value, "to_data", None)
        if callable(to_data):
            return {
                "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                "data": _semantic_projection(to_data(), active),
            }
        names = set(getattr(value, "__dict__", {}))
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            names.update(slots)
        fields = {}
        for name in sorted(names):
            if not isinstance(name, str) or name.startswith("_") \
                    or name in ("__dict__", "__weakref__") or not hasattr(value, name):
                continue
            fields[name] = _semantic_projection(getattr(value, name), active)
        return {
            "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
            "fields": fields,
        }
    finally:
        active.remove(object_id)


class Refine(MeshDescriptor):
    """A typed refinement criterion (Spec 5 sec.8.6).

    Build with the fluent form ``Refine.on(subject).above(threshold)``. ``subject`` is either a real
    declaration :class:`pops.model.Handle` or a semantic expression (for example a gradient-norm
    indicator) implementing ``resolve_references(resolver)``. Bare strings are deliberately
    refused. Every Handle leaf is authenticated and block-qualified by the Problem before compile;
    the expression itself stays symbolic until a backend explicitly lowers or refuses it.
    """

    category = "refinement_criterion"

    def __init__(self, subject: Any, predicate: Any = None, threshold: Any = None) -> None:
        self.subject = _require_refine_subject(subject)
        self.predicate = predicate
        self.threshold = threshold
        self._references_authenticated = False

    @classmethod
    def on(cls, subject: Any) -> Refine:
        return cls(subject)

    def _with(self, predicate: Any, threshold: Any) -> Refine:
        result = Refine(self.subject, predicate, float(threshold))
        result._references_authenticated = self._references_authenticated
        return result

    def above(self, threshold: Any) -> Refine:
        return self._with("above", threshold)

    def below(self, threshold: Any) -> Refine:
        return self._with("below", threshold)

    def gradient_above(self, threshold: Any) -> Refine:
        return self._with("gradient_above", threshold)

    def magnitude_above(self, threshold: Any) -> Refine:
        return self._with("magnitude_above", threshold)

    def options(self) -> dict:
        return {"subject": _refine_subject_options(self.subject), "predicate": self.predicate,
                "threshold": self.threshold}

    def resolve_references(self, resolver: Any) -> Refine:
        """Return a detached criterion with canonical, authenticated Handle leaves."""
        result = Refine(
            _resolve_refine_subject(self.subject, resolver),
            self.predicate,
            self.threshold,
        )
        result._references_authenticated = True
        return result

    @property
    def references_authenticated(self) -> bool:
        """Whether an authoritative assembly resolved every declaration leaf."""
        return self._references_authenticated is True

    def validate(self, context: Any = None) -> bool:
        """Validate the criterion shape; ownership is validated by ``resolve_references``."""
        if self.predicate is None or self.threshold is None:
            raise ValueError(
                "Refine criterion is incomplete: use Refine.on(subject).above(value) "
                "(or .below / .gradient_above / .magnitude_above)")
        _require_refine_subject(self.subject)
        return True


class TagUnion(MeshDescriptor):
    """The union of several refinement criteria (a cell is tagged if ANY fires)."""

    category = "refinement_criterion"

    def __init__(self, *criteria: Any) -> None:
        flat = []
        for c in criteria:
            if not isinstance(c, (Refine, TagUnion)):
                raise TypeError("TagUnion: every entry must be a Refine / TagUnion (got %r)" % (c,))
            flat.append(c)
        self.criteria = flat

    def options(self) -> dict:
        return {"n_criteria": len(self.criteria)}

    def validate(self, context: Any = None) -> bool:
        for c in self.criteria:
            c.validate(context)
        return True

    def resolve_references(self, resolver: Any) -> TagUnion:
        """Return a detached union whose indicator graphs carry canonical Handle leaves."""
        return TagUnion(*(criterion.resolve_references(resolver)
                          for criterion in self.criteria))

    @property
    def references_authenticated(self) -> bool:
        """Whether every leaf criterion crossed an authoritative resolver boundary."""
        return all(getattr(criterion, "references_authenticated", False)
                   for criterion in self.criteria)


class RegridEvery(MeshDescriptor):
    """Regrid the hierarchy every N macro-steps."""

    category = "regrid_policy"

    def __init__(self, steps: Any) -> None:
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("RegridEvery: steps must be > 0 (use FrozenRegrid for no regrid)")

    def options(self) -> dict:
        return {"steps": self.steps}


class FrozenRegrid(MeshDescriptor):
    """No dynamic regrid: the hierarchy is built once and frozen."""

    category = "regrid_policy"

    def options(self) -> dict:
        return {"frozen": True}


class PatchLayout(MeshDescriptor):
    """Patch-clustering policy: coarse distribution + max grid size."""

    category = "patch_layout"

    def __init__(self, distribute_coarse: Any = False, coarse_max_grid: Any = 32) -> None:
        self.distribute_coarse = bool(distribute_coarse)
        self.coarse_max_grid = int(coarse_max_grid)

    def options(self) -> dict:
        return {"distribute_coarse": self.distribute_coarse,
                "coarse_max_grid": self.coarse_max_grid}


class PatchClustering(MeshDescriptor):
    """Berger-Rigoutsos clustering policy of the regrid layout (ADC-615/616).

    Tunes how tagged coarse cells are grouped into fine patches:

    * ``min_efficiency`` in (0, 1] -- the tagged fraction a candidate box must reach to be accepted
      (higher = tighter, more patches; default 0.7);
    * ``min_box_size`` -- the smallest admissible patch side (default 1);
    * ``max_box_size`` -- the largest patch side; accepted boxes are chopped to it (default 32).

    Defaults reproduce the historical native ``ClusterParams{0.7, 1, 32}`` bit-for-bit. Out-of-domain
    values (efficiency outside (0, 1], sizes < 1, min > max) are refused STRUCTURALLY at construction.
    """

    category = "clustering_policy"

    def __init__(self, min_efficiency: Any = 0.7, min_box_size: Any = 1,
                 max_box_size: Any = 32) -> None:
        if isinstance(min_efficiency, bool) or not isinstance(min_efficiency, (int, float)):
            raise TypeError("PatchClustering: min_efficiency must be a number (got %r)"
                            % (min_efficiency,))
        self.min_efficiency = float(min_efficiency)
        if not (0.0 < self.min_efficiency <= 1.0):
            raise ValueError("PatchClustering: min_efficiency must be in (0, 1] (got %r)"
                             % (min_efficiency,))
        if isinstance(min_box_size, bool) or not isinstance(min_box_size, int) or min_box_size < 1:
            raise ValueError("PatchClustering: min_box_size must be an int >= 1 (got %r)"
                             % (min_box_size,))
        if isinstance(max_box_size, bool) or not isinstance(max_box_size, int) or max_box_size < 1:
            raise ValueError("PatchClustering: max_box_size must be an int >= 1 (got %r)"
                             % (max_box_size,))
        if min_box_size > max_box_size:
            raise ValueError("PatchClustering: min_box_size <= max_box_size required (got %d > %d)"
                             % (min_box_size, max_box_size))
        self.min_box_size = int(min_box_size)
        self.max_box_size = int(max_box_size)

    def options(self) -> dict:
        return {"min_efficiency": self.min_efficiency, "min_box_size": self.min_box_size,
                "max_box_size": self.max_box_size}


class ProperNesting(MeshDescriptor):
    """Proper-nesting policy with a buffer (cells of guaranteed coarse padding)."""

    category = "nesting_policy"

    def __init__(self, buffer: Any = 1) -> None:
        self.buffer = int(buffer)
        if self.buffer < 0:
            raise ValueError("ProperNesting: buffer must be >= 0")

    def options(self) -> dict:
        return {"buffer": self.buffer}


class BufferCells(MeshDescriptor):
    """Tag-buffer width: extra cells tagged around each flagged cell."""

    category = "tag_policy"

    def __init__(self, cells: Any = 1) -> None:
        self.cells = int(cells)

    def options(self) -> dict:
        return {"cells": self.cells}


# --- level selection policies (shared semantics with pops.output, Spec 5 sec.5.14) -----
class AllLevels(MeshDescriptor):
    category = "level_policy"

    def options(self) -> dict:
        return {"levels": "all"}


class CoarseOnly(MeshDescriptor):
    category = "level_policy"

    def options(self) -> dict:
        return {"levels": "coarse"}


class SelectedLevels(MeshDescriptor):
    category = "level_policy"

    def __init__(self, *levels: Any) -> None:
        self.levels = tuple(int(l) for l in levels)

    def options(self) -> dict:
        return {"levels": self.levels}


class CheckpointPolicy(MeshDescriptor):
    """AMR checkpoint / restart policy (Spec 5 sec.8.11).

    Spec 5 keeps a single checkpoint semantics: :class:`pops.output.CheckpointPolicy` is
    the general policy and this one is the AMR-compatible specialisation. ``restartable``
    requests a bit-identical restart; the route validates whether the current native AMR
    supports it (single block / single rank / frozen regrid) before runtime.
    """

    category = "checkpoint_policy"

    def __init__(self, restartable: Any = False, require_bit_identical: Any = False) -> None:
        self.restartable = bool(restartable)
        self.require_bit_identical = bool(require_bit_identical)

    def options(self) -> dict:
        return {"restartable": self.restartable,
                "require_bit_identical": self.require_bit_identical}


class AMROutput(MeshDescriptor):
    """AMR output policy: which fields on which levels, with patch metadata (sec.8.11)."""

    category = "amr_output"

    def __init__(self, fields: Any = (), levels: Any = None,
                 include_patch_boxes: Any = False) -> None:
        self.fields = tuple(
            _require_value_handle(field, "AMROutput(fields=...)") for field in fields)
        self.levels = levels if levels is not None else AllLevels()
        self.include_patch_boxes = bool(include_patch_boxes)

    def options(self) -> dict:
        return {"n_fields": len(self.fields),
                "fields": [_stable_handle_projection(field) for field in self.fields],
                "levels": self.levels.options().get("levels"),
                "include_patch_boxes": self.include_patch_boxes}

    def resolve_references(self, resolver: Any) -> AMROutput:
        """Return a detached output policy with canonical, authenticated field Handles."""
        return AMROutput(
            fields=tuple(_resolve_handle_reference(field, resolver, "AMROutput(fields=...)")
                         for field in self.fields),
            levels=self.levels,
            include_patch_boxes=self.include_patch_boxes,
        )


class IgnoreAMRCriteria(MeshDescriptor):
    """The explicit escape hatch for a ``Uniform(...)`` layout carrying AMR criteria.

    Spec 5 sec.8.6 / sec.5.14 (ADC-589 / ADC-555): a :class:`~pops.mesh.layouts.Uniform` layout
    with an active refinement criterion attached is refused by default (a criterion silently
    ignored on a single-level mesh is a correctness trap, not a convenience). Passing
    ``Uniform(mesh, refine=..., ignore_amr=IgnoreAMRCriteria())`` is the one explicit,
    self-documenting way to opt out: it is a marker descriptor (it carries no behaviour of its
    own) that :meth:`pops.Problem.validate` looks for before it raises.
    """

    category = "amr_override"

    def options(self) -> dict:
        return {"ignore_amr_criteria": True}


__all__ = [
    "Refine", "TagUnion", "RegridEvery", "FrozenRegrid", "PatchLayout", "PatchClustering",
    "ProperNesting", "BufferCells", "AllLevels", "CoarseOnly", "SelectedLevels",
    "CheckpointPolicy", "AMROutput", "IgnoreAMRCriteria", "NATIVE_MAX_LEVELS",
    "NATIVE_RATIOS", "Availability",
]
