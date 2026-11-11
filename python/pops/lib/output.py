"""The writers of the outputs of a run (spec 0.3, 2.6.9, table of ``pops.lib.output``; CC14).

A writer is given as ``write=`` of ``pops.Output`` and keeps the whole series of its Output in one
file, one record per publication with its achieved time (2.6.2). It writes after each run of the
simulation, from the publications that the compiled run returned, outside every step (CF9); its
path is relative to the ``directory=`` of ``pops.bind``.

``npz(path, metadata=None, overwrite=False)`` writes a NumPy archive. Each field ``f`` of the record
is the array ``f`` of shape ``(n,) + cells``, the n publications in order, each in the order of the
support's axes; ``time`` and ``step`` are the achieved time and the accepted macro step of each
publication; each field of ``metadata``, a ``pops.record`` of numbers, is a scalar array of its
name. A file that this simulation did not write is refused with ``OutputExists`` unless
``overwrite=True``, before the first run of the simulation starts. ``hdf5`` is a later lot.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

_RESERVED = ("time", "step")


class OutputExists(FileExistsError):
    """A writer met a file that another run wrote, and ``overwrite=`` is not true (2.6.9)."""


class Npz:
    """The writer ``npz(path, metadata=None, overwrite=False)``."""

    __slots__ = ("path", "metadata", "overwrite")

    def __init__(self, path: Any, metadata: Any, overwrite: bool) -> None:
        self.path = os.fspath(path)
        self.metadata = metadata
        self.overwrite = overwrite

    def __repr__(self) -> str:
        return "npz(%r%s%s)" % (self.path, "" if self.metadata is None else ", metadata=...",
                                ", overwrite=True" if self.overwrite else "")

    def check(self, directory: str) -> None:
        """Refuse, before a simulation runs, a file that it would replace and did not write."""
        target = os.path.join(directory, self.path)
        if os.path.exists(target) and not self.overwrite:
            raise OutputExists("npz(%r): %s exists, written by another run; give "
                               "overwrite=True or another path" % (self.path, target))

    def write_series(self, directory: str, record: Any, series: tuple, *, first: bool) -> str:
        """Write ``series`` (the publications of one Output of a simulation) under
        ``directory``; ``first`` is true when this simulation has not written it yet."""
        target = os.path.join(directory, self.path)
        if first:
            self.check(directory)
        arrays: dict = {}
        names = [name for name, _ in record.items()]
        for name in names:
            if name in _RESERVED:
                raise ValueError("npz(%r): the record field %s is the series of the publications; "
                                 "name the field otherwise" % (self.path, name))
        for name in names:
            arrays[name] = np.stack([publication[name] for publication in series]) if series \
                else np.zeros((0,))
        arrays["time"] = np.array([publication.time for publication in series], dtype=np.float64)
        arrays["step"] = np.array([publication.step for publication in series], dtype=np.int64)
        if self.metadata is not None:
            for name, value in self.metadata.items():
                arrays[name] = np.asarray(value)
        parent = os.path.dirname(target) or "."
        os.makedirs(parent, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".pops-npz-", suffix=".npz", dir=parent)
        os.close(handle)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        return target


def _is_record(value: Any) -> bool:
    """Whether ``value`` is a ``pops.record`` of the surface (this module is not written with the
    surface, so it reads the type by name rather than importing the surface)."""
    kind = type(value)
    return kind.__name__ == "Record" and kind.__module__ == "pops.surface._run"


def npz(path: Any, metadata: Any = None, overwrite: bool = False) -> Npz:
    """``npz(path, metadata=None, overwrite=False)``: the series of an Output in a NumPy archive at
    ``path``, relative to the ``directory=`` of ``pops.bind``; ``metadata`` is a ``pops.record`` of
    numbers."""
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise TypeError("npz(path) takes a file path, got %r" % (path,))
    if os.path.isabs(os.fspath(path)):
        raise ValueError("npz(%r): the path is relative to the directory= of pops.bind" % (path,))
    if metadata is not None:
        if not _is_record(metadata):
            raise TypeError("npz(metadata=...) takes a pops.record of numbers, got %r"
                            % (metadata,))
        for name, value in metadata.items():
            if name in _RESERVED:
                raise ValueError("npz(metadata=pops.record(%s=...)): %s is the series of the "
                                 "publications; name the metadata otherwise" % (name, name))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("npz(metadata=pops.record(%s=%r)): metadata are numbers"
                                % (name, value))
    if type(overwrite) is not bool:
        raise TypeError("npz(overwrite=...) is True or False, got %r" % (overwrite,))
    return Npz(path, metadata, overwrite)


def hdf5(path: Any, overwrite: bool = False) -> Any:
    """``hdf5(path, overwrite=False)``: refused until a later lot of N15."""
    # This module is infrastructure, not a word written with the surface (SU-T1 lints those), so it
    # reaches the refusal class of the surface through importlib.
    import importlib

    surface = importlib.import_module("pops.surface")
    raise surface.NotYetElaborated("hdf5(path), the HDF5 writer", "a later lot of N15 with the writers of "
                           "pops.lib.output").with_traceback(None)


__all__ = ["Npz", "OutputExists", "hdf5", "npz"]
