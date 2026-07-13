"""Typed, inert AMR policies consumed by layouts and the native runtime.

Refinement uses typed criteria such as ``Refine.on(subject).above(0.05)``; Python never tags cells.
"""
from __future__ import annotations

from typing import Any

from pops.params.use_sites import ParamUse, resolve_param_use

from .._descriptor import Availability, MeshDescriptor
from ._param_threshold import resolve_refine_threshold
from . import bootstrap as _bootstrap, hierarchy as _hierarchy
from . import hierarchy_regrid as _hierarchy_regrid, transfer as _transfer
from . import hierarchy_resolution as _hierarchy_resolution
from .bootstrap import *  # noqa: F403
from .hierarchy import *  # noqa: F403
from .hierarchy_regrid import *  # noqa: F403
from .hierarchy_resolution import *  # noqa: F403
from .transfer import *  # noqa: F403
from . import tagging_graph as _tagging_graph, tagging_resolution as _tagging_resolution
from .tagging_graph import *  # noqa: F403
from .tagging_resolution import *  # noqa: F403

# ResolvedHierarchy transports an arbitrary positive level count.  Only the refinement ratio is a
# native kernel capability; the old layout-only adapter remains a separate two-level compatibility
# seam and must not masquerade as a backend maximum.
LEGACY_CONFIG_LEVELS = 2
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
    """Typed ``Refine.on(subject).above(threshold)`` criterion.

    Subjects are declaration handles or semantic expressions. Handles are authenticated and
    block-qualified before compile; the indicator stays symbolic until backend lowering.
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
        # Tagging thresholds are genuine runtime values.  Preserve RuntimeParam /
        # DerivedParam descriptors for the resolved plan instead of silently
        # erasing their storage class through float(...); ConstParam is unwrapped.
        threshold = resolve_param_use(
            threshold, ParamUse.RUNTIME_VALUE, where="Refine.%s(threshold=)" % predicate)
        result = Refine(self.subject, predicate, threshold)
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
            resolve_refine_threshold(
                self.threshold, resolver, _resolve_handle_reference
            ),
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
        self.steps = int(resolve_param_use(
            steps, ParamUse.REGRID_SCHEDULE, where="RegridEvery(steps=)"))
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
        self.distribute_coarse = bool(resolve_param_use(
            distribute_coarse, ParamUse.MESH_TOPOLOGY,
            where="PatchLayout(distribute_coarse=)"))
        self.coarse_max_grid = int(resolve_param_use(
            coarse_max_grid, ParamUse.SHAPE, where="PatchLayout(coarse_max_grid=)"))

    def options(self) -> dict:
        return {"distribute_coarse": self.distribute_coarse,
                "coarse_max_grid": self.coarse_max_grid}


class PatchClustering(MeshDescriptor):
    """Berger-Rigoutsos clustering controls (ADC-615/616).

    Efficiency is in ``(0, 1]`` and box sizes are positive with ``min <= max``. Defaults preserve
    the native ``ClusterParams{0.7, 1, 32}`` policy.
    """

    category = "clustering_policy"

    def __init__(self, min_efficiency: Any = 0.7, min_box_size: Any = 1,
                 max_box_size: Any = 32) -> None:
        min_efficiency = resolve_param_use(
            min_efficiency, ParamUse.AMR_HIERARCHY,
            where="PatchClustering(min_efficiency=)")
        min_box_size = resolve_param_use(
            min_box_size, ParamUse.SHAPE, where="PatchClustering(min_box_size=)")
        max_box_size = resolve_param_use(
            max_box_size, ParamUse.SHAPE, where="PatchClustering(max_box_size=)")
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
        self.buffer = int(resolve_param_use(
            buffer, ParamUse.AMR_HIERARCHY, where="ProperNesting(buffer=)"))
        if self.buffer < 0:
            raise ValueError("ProperNesting: buffer must be >= 0")

    def options(self) -> dict:
        return {"buffer": self.buffer}


class BufferCells(MeshDescriptor):
    """Tag-buffer width: extra cells tagged around each flagged cell."""

    category = "tag_policy"

    def __init__(self, cells: Any = 1) -> None:
        self.cells = int(resolve_param_use(
            cells, ParamUse.AMR_HIERARCHY, where="BufferCells(cells=)"))

    def options(self) -> dict:
        return {"cells": self.cells}


class IgnoreAMRCriteria(MeshDescriptor):
    """The explicit escape hatch for a ``Uniform(...)`` layout carrying AMR criteria.

    Spec 5 sec.8.6 / sec.5.14 (ADC-589 / ADC-555): a :class:`~pops.layouts.Uniform` layout
    with an active refinement criterion attached is refused by default (a criterion silently
    ignored on a single-level mesh is a correctness trap, not a convenience). Passing
    ``Uniform(mesh, refine=..., ignore_amr=IgnoreAMRCriteria())`` is the one explicit,
    self-documenting way to opt out: it is a marker descriptor (it carries no behaviour of its
    own) that :meth:`pops.Case.validate` looks for before it raises.
    """

    category = "amr_override"

    def options(self) -> dict:
        return {"ignore_amr_criteria": True}


__all__ = [
    "Refine", "TagUnion", "RegridEvery", "FrozenRegrid", "PatchLayout", "PatchClustering",
    "ProperNesting", "BufferCells", "IgnoreAMRCriteria",
    "NATIVE_RATIOS", "Availability",
] + _tagging_graph.__all__ + _tagging_resolution.__all__ \
    + _hierarchy.__all__ + _hierarchy_regrid.__all__ + _hierarchy_resolution.__all__ \
    + _transfer.__all__ + _bootstrap.__all__
