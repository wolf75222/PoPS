"""pops.output.policies -- output / checkpoint / level descriptors (Spec 5 sec.5.14 / 8.11).

Typed replacements for ``output(format="hdf5", every=20)`` / ``checkpoint(mode=...)``. An
output or checkpoint policy declares its format, cadence, fields, diagnostics and level
selection; the runtime performs the I/O. Inert descriptors.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor


_WRITABLE_HANDLE_KINDS = frozenset({"aux", "field", "state"})


def _require_writable_handle(reference: Any, *, where: str) -> None:
    """Reject control/configuration handles at the output declaration boundary."""
    from pops.model import Handle

    if not isinstance(reference, Handle):
        raise TypeError(
            "%s must be declaration Handle values; names/strings are not references "
            "(got %r)" % (where, type(reference).__name__))
    if reference.kind not in _WRITABLE_HANDLE_KINDS:
        raise TypeError(
            "%s accepts only writable state/field/aux handles; got kind %r"
            % (where, reference.kind))


class _LevelPolicy(Descriptor):
    category = "level_policy"


class AllLevels(_LevelPolicy):
    def options(self) -> dict:
        return {"levels": "all"}


class CoarseOnly(_LevelPolicy):
    def options(self) -> dict:
        return {"levels": "coarse"}


class SelectedLevels(_LevelPolicy):
    def __init__(self, *levels: Any) -> None:
        self.levels = tuple(int(l) for l in levels)

    def options(self) -> dict:
        return {"levels": self.levels}


class OutputPolicy(Descriptor):
    """An output policy: a format, a cadence, the fields/diagnostics, and the level selection.

    ``OutputPolicy(format=HDF5(), cadence=every(20), fields=[phi, E], levels=AllLevels())``.
    ``cadence`` is any inert schedule object (e.g. ``pops.time.schedule.every(20)``) or an int
    step interval; it is stored, not interpreted, here.
    """

    category = "output_policy"

    def __init__(self, format: Any = None, cadence: Any = None, fields: Any = (),
                 diagnostics: Any = (), levels: Any = None, require_parallel: bool = False,
                 prefix: Any = None) -> None:
        field_refs = list(fields)
        for reference in field_refs:
            _require_writable_handle(reference, where="OutputPolicy fields")
        diagnostic_refs = list(diagnostics)
        invalid_diagnostics = [
            value for value in diagnostic_refs
            if not _is_diagnostic_category(getattr(value, "category", None))]
        if invalid_diagnostics:
            raise TypeError(
                "OutputPolicy diagnostics must be typed pops.diagnostics measures; names/strings "
                "are not references (got %r)" % type(invalid_diagnostics[0]).__name__)
        self.format = format
        self.cadence = cadence
        self.fields = field_refs
        self.diagnostics = diagnostic_refs
        self.levels = levels if levels is not None else AllLevels()
        self.require_parallel = bool(require_parallel)
        #: Optional file-name prefix the run-loop driver writes under output_dir (default "output").
        self.prefix = prefix

    def options(self) -> dict:
        return {"format": getattr(self.format, "name", self.format),
                "cadence": getattr(self.cadence, "name", self.cadence),
                "n_fields": len(self.fields), "n_diagnostics": len(self.diagnostics),
                "levels": self.levels.options().get("levels"),
                "require_parallel": self.require_parallel, "prefix": self.prefix}

    def resolve_references(self, resolver: Any) -> Any:
        """Return a detached policy with canonical field and diagnostic references."""
        if not callable(resolver):
            raise TypeError("OutputPolicy reference resolver must be callable")
        from copy import copy
        resolved = copy(self)
        resolved.fields = [resolver(reference) for reference in self.fields]
        for reference in resolved.fields:
            _require_writable_handle(
                reference, where="OutputPolicy resolved fields")
        resolved.diagnostics = []
        for measure in self.diagnostics:
            resolve_measure = getattr(measure, "resolve_references", None)
            if not callable(resolve_measure):
                raise TypeError(
                    "%s must implement resolve_references(resolver) to be used by OutputPolicy"
                    % type(measure).__name__)
            resolved.diagnostics.append(resolve_measure(resolver))
        return resolved

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        req = {}
        if self.require_parallel:
            req["parallel_io"] = True
        # Union the chosen format's own requirements (e.g. HDF5(parallel=True) -> parallel_io).
        if self.format is not None and hasattr(self.format, "requirements"):
            req.update(self.format.requirements().to_dict())
        return RequirementSet(req)


def _is_diagnostic_category(category: Any) -> bool:
    """Whether a descriptor category participates in the diagnostic-measure protocol."""
    return isinstance(category, str) and (
        category.startswith("diagnostic_") or category == "conservation_check")


class CheckpointPolicy(Descriptor):
    """The general checkpoint / restart policy (Spec 5 sec.8.4 / 8.11).

    ``CheckpointPolicy(cadence=every(100), restartable=True)``. This is the single general
    policy; :class:`pops.mesh.amr.CheckpointPolicy` is the AMR-compatible specialisation
    (Spec 5 Phase D reconciles them so there is one semantics, not two divergent APIs).
    """

    category = "checkpoint_policy"

    def __init__(self, cadence: Any = None, restartable: bool = False,
                 require_bit_identical: bool = False, prefix: Any = None) -> None:
        self.cadence = cadence
        self.restartable = bool(restartable)
        self.require_bit_identical = bool(require_bit_identical)
        #: Optional file-name prefix the run-loop driver writes under output_dir (default "checkpoint").
        self.prefix = prefix

    def options(self) -> dict:
        return {"cadence": getattr(self.cadence, "name", self.cadence),
                "restartable": self.restartable,
                "require_bit_identical": self.require_bit_identical,
                "prefix": self.prefix}

    def resolve_references(self, resolver: Any) -> Any:
        """Return a detached snapshot declaration (checkpoint policies retain no references)."""
        if not callable(resolver):
            raise TypeError("CheckpointPolicy reference resolver must be callable")
        from copy import copy
        return copy(self)

    def capabilities(self):
        """The checkpoint route's declared capabilities (parity with every other typed route).

        ``amr_compatible`` is True: the AMR checkpoint restarts a hierarchy under ACTIVE regridding
        (format v3 rebuilds the hierarchy from the manifest), so a restartable checkpoint is honest on
        an AMR route now (ADC-542). ``device_host_sync`` is True: the checkpoint gathers to the host
        (rank-0) writer.
        """
        from pops.descriptors_report import CapabilitySet
        return CapabilitySet({"restartable": self.restartable,
                              "require_bit_identical": self.require_bit_identical,
                              "cadence_slot": "checkpoint",
                              "amr_compatible": True,
                              "device_host_sync": True})

    def requirements(self):
        """What the checkpoint route needs of the resolved runtime (a restartable / bit-identical route)."""
        from pops.descriptors_report import RequirementSet
        req = {}
        if self.restartable:
            req["restartable_route"] = True
        if self.require_bit_identical:
            req["bit_identical_route"] = True
        return RequirementSet(req)

    def validate(self, context=None):
        """Refuse ONLY the physically / correctness-impossible residue (ADC-542 addendum B.8).

        Restartable checkpoints WORK under active regridding (v3 rebuilds the hierarchy), so
        ``restartable=True`` is never refused for being AMR. The honest residue the gate still refuses:

        (r1) ``require_bit_identical=True`` when the resolved @p context declares a restart that
             CHANGES the rank count (``restart_ranks`` != the compiled ``ranks``): per-rank partial
             sums re-associate (IEEE754 non-associativity), so a BIT guarantee across a rank-count
             change is physically impossible. The restart itself is still CORRECT (values restored
             exactly); only the bit guarantee is refused. Absent that explicit context, never refused.

        A mismatched-identity restore (r2) is a RESTART-time guard (program-hash / abi_key / grid),
        not a compile-time policy concern; it is enforced by the v3 reader, not here.
        """
        ctx = context or {}
        if self.require_bit_identical and isinstance(ctx, dict):
            ranks = ctx.get("ranks")
            restart_ranks = ctx.get("restart_ranks")
            if (ranks is not None and restart_ranks is not None and int(ranks) != int(restart_ranks)):
                raise ValueError(
                    "CheckpointPolicy(require_bit_identical=True) cannot be honored across a rank-count "
                    "change (compiled ranks=%d, restart ranks=%d): per-rank partial sums re-associate "
                    "(IEEE754 non-associativity), so a bit-identical guarantee is physically impossible "
                    "across a different rank count. The restart is still correct (values restored "
                    "exactly); declare require_bit_identical=False for a cross-rank restart, or restart "
                    "on the same rank count." % (int(ranks), int(restart_ranks)))
        return True


__all__ = ["OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels"]
