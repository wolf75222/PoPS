"""pops.diagnostics.measures -- typed diagnostic-measure descriptors (Spec 5 sec.5.13 / 14.2.7).

Spec 5 names a diagnostic with a TYPED object, not the string form
``diagnostics.norm(kind="l2")``. :class:`Norm` / :class:`Integral` / :class:`MinMax` /
:class:`ConservationCheck` are those objects -- inert descriptors that DESCRIBE a scalar
reduction over a block (and an optional model role): the reduction kind, whether it needs an
MPI reduction, its cadence slot and its AMR / multi-level compatibility, all carried as
METADATA. They compute nothing; the C++ / Kokkos / MPI runtime evaluates the reduction.

Each typed measure lowers to an inert native reduction scheme label (``norm`` / ``integral`` /
``min_max`` / ``conservation_check``). ``norm`` and ``integral`` reuse the same-name
:mod:`pops.diagnostics` factory scheme; ``conservation_check`` matches the
:mod:`pops.diagnostics.invariants` reduction; ``min_max`` is a new inert label. No native symbol
is fabricated -- the labels name the reduction the C++ runtime evaluates. The :class:`Norm`
measure takes a typed norm kind from :mod:`pops.linalg.norms` (``L1`` / ``L2`` / ``LInf``); a
bare string is rejected.
"""
from __future__ import annotations

import math
from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.linalg.norms import _Norm


def _ref_name(value: Any) -> Any:
    """The stable display name for a block / role reference (its ``name`` or its repr).

    A block is named by a string and a role by a typed role object (carrying a ``name``); the
    measure references it WITHOUT interpreting it. ``None`` stays ``None`` so an unscoped
    measure (whole-domain, default role) reads cleanly.
    """
    if value is None:
        return None
    from pops.physics.roles import ComponentRole, native_role_token
    if isinstance(value, ComponentRole):
        return native_role_token(value)
    return getattr(value, "name", None) or (value if isinstance(value, str) else repr(value))


def _role_name(value: Any) -> str | None:
    """Return the canonical physical-role selector carried into native execution."""
    if value is None:
        return None
    from pops.physics.roles import native_role_token
    try:
        return native_role_token(value)
    except TypeError as exc:
        raise TypeError(
            "diagnostic role must be a typed pops.physics.roles.ComponentRole"
        ) from exc


def _operation(name: str, reduction: str, *, transform: str = "identity",
               metric_weighted: bool = False) -> dict[str, Any]:
    """Build one callback-free native scalar-reduction instruction."""
    return {
        "name": name,
        "reduction": reduction,
        "transform": transform,
        "metric_weighted": metric_weighted,
    }


class _Measure(Descriptor):
    """Base of the typed diagnostic measures: a scalar reduction over a block / role.

    A measure stores its (opaque) ``block`` and ``role`` references and surfaces them by name;
    it never reads a cell. Subclasses set :attr:`category` + :attr:`scheme` (the native
    reduction identity shared with the legacy factory) and declare their reduction metadata via
    :meth:`capabilities` / :meth:`requirements`. ``cadence`` is any inert schedule object (e.g.
    ``pops.time._schedule.api.every(20)``) or an int step interval; it is stored, not interpreted.
    """

    #: The native diagnostic reduction scheme this measure lowers to (an inert authoring label;
    #: ``norm`` / ``integral`` reuse the same-name legacy factory scheme, ``min_max`` is new).
    #: Subclasses set it.
    scheme = None
    #: The reduction kind this measure performs ("sum" / "norm" / "min_max" / "check").
    reduction = None

    def __init__(self, block: Any = None, role: Any = None, cadence: Any = None) -> None:
        if block is not None:
            from pops.problem.handles import BlockHandle
            if not isinstance(block, BlockHandle):
                raise TypeError(
                    "diagnostic block must be a BlockHandle; names/strings are not references "
                    "(got %r)" % type(block).__name__)
        self.block = block
        _role_name(role)
        self.role = role
        self.cadence = cadence

    def options(self) -> dict:
        return {"scheme": self.scheme, "block": _ref_name(self.block),
                "role": _ref_name(self.role), "cadence": _ref_name(self.cadence)}

    def resolve_references(self, resolver: Any) -> Any:
        """Return a detached measure whose optional block reference is canonical.

        The authoring descriptor is never mutated. External diagnostic consumers can rely on this
        small protocol instead of branching on every concrete measure subclass.
        """
        if not callable(resolver):
            raise TypeError("diagnostic reference resolver must be callable")
        from copy import copy
        from pops.problem.handles import BlockHandle
        resolved = copy(self)
        if self.block is not None:
            block = resolver(self.block)
            if not isinstance(block, BlockHandle):
                raise TypeError("diagnostic block resolver must return a BlockHandle")
            resolved.block = block
        return resolved

    def declaration_references(self) -> tuple[Any, ...]:
        """Return the complete typed ownership surface consumed by this measure.

        Consumer authoring deliberately depends on this small protocol instead of inspecting
        concrete diagnostic classes or guessing from ``options()`` display strings.
        """
        return () if self.block is None else (self.block,)

    def consumer_data(self) -> dict[str, Any]:
        """Return callback-free diagnostic semantics for ``ConsumerGraph`` identity."""
        return {
            "category": self.category,
            "scheme": self.scheme,
            "reduction": self.reduction,
            "options": self.options(),
            "requirements": self.requirements().to_dict(),
            "capabilities": self.capabilities().to_dict(),
        }

    def diagnostic_execution(self) -> dict[str, Any]:
        """Return the closed native-reduction plan consumed by ``ConsumerGraph``.

        This is deliberately a tiny protocol rather than runtime dispatch on concrete descriptor
        classes.  The plan contains no callback and no array operation; the runtime only invokes
        the named native collective and applies the declared scalar transform.
        """
        raise NotImplementedError(
            "%s must provide an exact diagnostic_execution() plan" % type(self).__name__)

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        # A scalar reduction over a distributed mesh needs an MPI all-reduce to be correct.
        return RequirementSet({"mpi_reduction": True})

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet
        # Reduction metadata, declared not computed: a single scalar reduction is AMR /
        # multi-level safe (it sums / folds across levels) and runs on the diagnostic cadence.
        return CapabilitySet({"reduction": self.reduction, "mpi_reduction": True,
                              "amr_compatible": True, "multi_level": True,
                              "cadence_slot": "diagnostic"})

    def lower(self, context: Any = None) -> Any:
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self.name, category=self.category,
                                 native_id=self.native_id, options=self.options(),
                                 scheme=self.scheme)

    def inspect(self) -> Any:
        info = super().inspect()
        info["scheme"] = self.scheme
        return info


class Norm(_Measure):
    """A typed norm reduction over a block: ``Norm(L2(), block=..., role=...)``.

    The norm kind is a typed :mod:`pops.linalg.norms` object (``L1`` / ``L2`` / ``LInf``), NOT
    the string ``kind="l2"`` (Spec 5 sec.7 rejects a free-string selector). The measure lowers
    to the native ``norm`` reduction the legacy ``diagnostics.norm`` factory already names.
    """

    category = "diagnostic_norm"
    scheme = "norm"
    reduction = "norm"

    def __init__(self, norm: Any, block: Any = None, role: Any = None,
                 cadence: Any = None) -> None:
        if not isinstance(norm, _Norm):
            from pops.descriptors import reject_string_selector
            if isinstance(norm, str):
                reject_string_selector(norm, "norm", "pops.linalg.norms.L2()")
            raise TypeError(
                "Norm(norm=...) takes a typed pops.linalg.norms object (L1 / L2 / LInf), "
                "got %r. Spec 5 forbids a string norm selector." % (norm,))
        super().__init__(block=block, role=role, cadence=cadence)
        self.norm = norm

    def options(self) -> dict:
        opts = super().options()
        opts["norm"] = self.norm.kind
        return opts

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet
        caps = super().capabilities().to_dict()
        caps["norm_kind"] = self.norm.kind
        return CapabilitySet(caps)

    def diagnostic_execution(self) -> dict[str, Any]:
        operations = {
            "l1": _operation("l1", "abs_sum", metric_weighted=True),
            "l2": _operation(
                "l2", "sum_sq", transform="sqrt", metric_weighted=True),
            "linf": _operation("linf", "abs_max"),
        }
        kind = self.norm.kind
        if kind is None:
            raise ValueError("typed norm descriptor has no canonical kind")
        return {
            "schema_version": 1,
            "role": _role_name(self.role),
            "operations": [operations[kind]],
            "conservation": None,
        }


class StepChangeNorm(_Measure):
    """Norm of the accepted macro-step change ``U[n+1] - U[n]``.

    The previous state is the runtime transaction snapshot, so the reduction is evaluated
    natively on the execution backend and collectively under MPI.  It is intentionally a
    whole-state diagnostic: selecting one component would no longer describe the solution
    change reported by the time integrator.
    """

    category = "diagnostic_step_change_norm"
    scheme = "step_change_norm"
    reduction = "step_change_norm"

    def __init__(self, norm: Any, block: Any = None, cadence: Any = None) -> None:
        if not isinstance(norm, _Norm):
            raise TypeError(
                "StepChangeNorm(norm=...) takes a typed pops.linalg.norms object")
        if norm.kind != "l2":
            raise ValueError("StepChangeNorm currently supports exactly pops.linalg.norms.L2()")
        super().__init__(block=block, role=None, cadence=cadence)
        self.norm = norm

    def options(self) -> dict:
        opts = super().options()
        opts["norm"] = self.norm.kind
        return opts

    def diagnostic_execution(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "role": None,
            "operations": [
                _operation("step_change_l2", "step_change_l2"),
            ],
            "conservation": None,
        }


class Integral(_Measure):
    """A typed domain-integral reduction over a block: ``Integral(role=Density())``.

    Sums the (role-selected) quantity over the block volume; ``mass`` is
    ``Integral(role=Density())``. Lowers to the native ``integral`` reduction.
    """

    category = "diagnostic_integral"
    scheme = "integral"
    reduction = "sum"

    def diagnostic_execution(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "role": _role_name(self.role),
            "operations": [
                _operation("integral", "sum", metric_weighted=True),
            ],
            "conservation": None,
        }


class MinMax(_Measure):
    """A typed min / max reduction over a block: ``MinMax(block=..., role=...)``.

    Reports the (role-selected) extrema over the block. Lowers to the native ``min_max``
    reduction. Unlike a sum / norm it is a fold, but it is equally MPI- and AMR-safe.
    """

    category = "diagnostic_minmax"
    scheme = "min_max"
    reduction = "min_max"

    def diagnostic_execution(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "role": _role_name(self.role),
            "operations": [
                _operation("min", "min"),
                _operation("max", "max"),
            ],
            "conservation": None,
        }


class ConservationCheck(Descriptor):
    """A typed conservation check on a diagnostic quantity: ``ConservationCheck(Integral(...))``.

    Names a tolerance check that the runtime applies to a measured ``quantity`` (a diagnostic
    measure descriptor, e.g. an :class:`Integral`) -- the drift of that quantity must stay
    within ``tolerance``. ``quantity`` MUST be a diagnostic descriptor (Spec 5 sec.5.13); a
    string or anything else is rejected. The check itself computes nothing; the runtime
    measures the quantity and compares the drift.
    """

    category = "conservation_check"
    scheme = "conservation_check"

    def __init__(self, quantity: Any, tolerance: float = 1e-12) -> None:
        self.quantity = quantity
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("ConservationCheck tolerance must be a finite number >= 0")
        self.cadence = getattr(quantity, "cadence", None)

    def options(self) -> dict:
        return {"scheme": self.scheme, "quantity": _ref_name(self.quantity),
                "tolerance": self.tolerance}

    def resolve_references(self, resolver: Any) -> Any:
        """Return a detached check containing a reference-resolved diagnostic quantity."""
        if not callable(resolver):
            raise TypeError("diagnostic reference resolver must be callable")
        resolve_quantity = getattr(self.quantity, "resolve_references", None)
        if not callable(resolve_quantity):
            raise TypeError(
                "ConservationCheck quantity must implement resolve_references(resolver)")
        from copy import copy
        resolved = copy(self)
        resolved.quantity = resolve_quantity(resolver)
        return resolved

    def declaration_references(self) -> tuple[Any, ...]:
        references = getattr(self.quantity, "declaration_references", None)
        if not callable(references):
            raise TypeError(
                "ConservationCheck quantity must implement declaration_references()")
        values = references()
        if not isinstance(values, tuple):
            raise TypeError("diagnostic declaration_references() must return a tuple")
        return values

    def consumer_data(self) -> dict[str, Any]:
        projection = getattr(self.quantity, "consumer_data", None)
        if not callable(projection):
            raise TypeError("ConservationCheck quantity must implement consumer_data()")
        tolerance = self.tolerance.hex()
        capabilities = self.capabilities().to_dict()
        capabilities["tolerance"] = tolerance
        return {
            "category": self.category,
            "scheme": self.scheme,
            "quantity": projection(),
            "tolerance": tolerance,
            "requirements": self.requirements().to_dict(),
            "capabilities": capabilities,
        }

    def diagnostic_execution(self) -> dict[str, Any]:
        provider = getattr(self.quantity, "diagnostic_execution", None)
        if not callable(provider):
            raise TypeError(
                "ConservationCheck quantity must implement diagnostic_execution()")
        plan = provider()
        if type(plan) is not dict or plan.get("schema_version") != 1:
            raise TypeError("ConservationCheck quantity returned an invalid execution plan")
        operations = plan.get("operations")
        if not isinstance(operations, list) or len(operations) != 1:
            raise ValueError(
                "ConservationCheck requires one scalar diagnostic quantity; "
                "a multi-valued MinMax check is ambiguous")
        return {
            "schema_version": 1,
            "role": plan.get("role"),
            "operations": [dict(operations[0])],
            "conservation": {"tolerance": self.tolerance.hex()},
        }

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        # The checked quantity is itself a reduction, so the check inherits its MPI need.
        return RequirementSet({"mpi_reduction": True, "quantity": True})

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet
        return CapabilitySet({"reduction": "check", "mpi_reduction": True, "amr_compatible": True,
                              "multi_level": True, "cadence_slot": "diagnostic",
                              "tolerance": self.tolerance})

    def available(self, context: Any = None) -> Any:
        if not isinstance(self.quantity, Descriptor):
            return Availability.no(
                "ConservationCheck(quantity=...) needs a diagnostic descriptor "
                "(e.g. Integral(role=Density())), got %r" % (self.quantity,),
                missing=["quantity"],
                alternatives=["Integral(...)", "Norm(L2(), ...)", "MinMax(...)"])
        return Availability.yes()

    def validate(self, context: Any = None) -> bool:
        status = self.available(context)
        if not status.ok:
            raise TypeError("%s is not valid:\n%s" % (self.name, status))
        return True

    def lower(self, context: Any = None) -> Any:
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self.name, category=self.category,
                                 native_id=self.native_id, options=self.options(),
                                 scheme=self.scheme)

    def inspect(self) -> Any:
        info = super().inspect()
        info["scheme"] = self.scheme
        return info


__all__ = ["Norm", "Integral", "MinMax", "ConservationCheck"]
