"""pops.output.policies -- output / checkpoint / level descriptors (Spec 5 sec.5.14 / 8.11).

Typed output/checkpoint policies declare their format, cadence, fields, diagnostics and level
selection; the runtime performs the I/O. Inert descriptors.
"""
from pops.descriptors import reject_string_selector
from pops.descriptors import Descriptor
from pops.output.formats import NPZ


_OUTPUT_CADENCE_KINDS = ("always", "every", "on_start", "on_end")


def _validate_output_format(fmt):
    if fmt is None:
        return NPZ()
    if isinstance(fmt, str):
        reject_string_selector(fmt, "format", "pops.output.NPZ() / VTK() / HDF5()")
    if getattr(fmt, "category", None) != "output_format":
        raise TypeError(
            "OutputPolicy: format must be a pops.output format descriptor "
            "(NPZ() / VTK() / HDF5()), got %r" % type(fmt).__name__)
    return fmt


def _validate_output_cadence(cadence):
    if cadence is None or (isinstance(cadence, int) and not isinstance(cadence, bool)):
        return cadence
    if isinstance(cadence, bool):
        raise TypeError("OutputPolicy: cadence must be an int interval or typed schedule, got bool")
    kind = getattr(cadence, "kind", None)
    if kind not in _OUTPUT_CADENCE_KINDS:
        raise NotImplementedError(
            "OutputPolicy: cadence %r is not implemented by the output run loop; use "
            "always(), every(N), on_start(), on_end(), an int interval, or implement the "
            "runtime hook before exposing this policy." % (kind,))
    return cadence


class _LevelPolicy(Descriptor):
    category = "level_policy"


class AllLevels(_LevelPolicy):
    def options(self):
        return {"levels": "all"}


class CoarseOnly(_LevelPolicy):
    def options(self):
        return {"levels": "coarse"}


class SelectedLevels(_LevelPolicy):
    def __init__(self, *levels):
        self.levels = tuple(int(l) for l in levels)

    def options(self):
        return {"levels": self.levels}


class OutputPolicy(Descriptor):
    """An output policy: a format, a cadence, the fields/diagnostics, and the level selection.

    ``OutputPolicy(format=HDF5(), cadence=every(20), fields=[phi, E], levels=AllLevels())``.
    ``cadence`` is a schedule the run-loop actually honors (``always`` / ``every`` /
    ``on_start`` / ``on_end``) or an int step interval.
    """

    category = "output_policy"

    def __init__(self, format=None, cadence=None, fields=(), diagnostics=(),
                 levels=None, require_parallel=False, prefix=None):
        self.format = _validate_output_format(format)
        self.cadence = _validate_output_cadence(cadence)
        self.fields = list(fields)
        self.diagnostics = list(diagnostics)
        self.levels = levels if levels is not None else AllLevels()
        self.require_parallel = bool(require_parallel)
        #: Optional file-name prefix the run-loop driver writes under output_dir (default "output").
        self.prefix = prefix

    def options(self):
        return {"format": getattr(self.format, "name", self.format),
                "cadence": getattr(self.cadence, "name", self.cadence),
                "n_fields": len(self.fields), "n_diagnostics": len(self.diagnostics),
                "levels": self.levels.options().get("levels"),
                "require_parallel": self.require_parallel, "prefix": self.prefix}

    def requirements(self):
        req = {}
        if self.require_parallel:
            req["parallel_io"] = True
        # Union the chosen format's own requirements (e.g. HDF5(parallel=True) -> parallel_io).
        if self.format is not None and hasattr(self.format, "requirements"):
            req.update(self.format.requirements())
        return req


class CheckpointPolicy(Descriptor):
    """The general checkpoint / restart policy (Spec 5 sec.8.4 / 8.11).

    ``CheckpointPolicy(cadence=every(100), restartable=True)``. This is the single general
    policy; :class:`pops.mesh.amr.CheckpointPolicy` is the AMR-compatible specialisation
    (Spec 5 Phase D reconciles them so there is one semantics, not two divergent APIs).
    """

    category = "checkpoint_policy"

    def __init__(self, cadence=None, restartable=False, require_bit_identical=False,
                 prefix=None):
        self.cadence = cadence
        self.restartable = bool(restartable)
        self.require_bit_identical = bool(require_bit_identical)
        #: Optional file-name prefix the run-loop driver writes under output_dir (default "checkpoint").
        self.prefix = prefix

    def options(self):
        return {"cadence": getattr(self.cadence, "name", self.cadence),
                "restartable": self.restartable,
                "require_bit_identical": self.require_bit_identical,
                "prefix": self.prefix}


__all__ = ["OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels"]
