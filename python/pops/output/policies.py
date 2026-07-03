"""pops.output.policies -- output / checkpoint / level descriptors (Spec 5 sec.5.14 / 8.11).

Typed replacements for ``output(format="hdf5", every=20)`` / ``checkpoint(mode=...)``. An
output or checkpoint policy declares its format, cadence, fields, diagnostics and level
selection; the runtime performs the I/O. Inert descriptors.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor


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
        self.format = format
        self.cadence = cadence
        self.fields = list(fields)
        self.diagnostics = list(diagnostics)
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

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        req = {}
        if self.require_parallel:
            req["parallel_io"] = True
        # Union the chosen format's own requirements (e.g. HDF5(parallel=True) -> parallel_io).
        if self.format is not None and hasattr(self.format, "requirements"):
            req.update(self.format.requirements().to_dict())
        return RequirementSet(req)


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


__all__ = ["OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels"]
