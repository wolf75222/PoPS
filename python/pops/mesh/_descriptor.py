"""Mesh / layout / AMR descriptor base, aligned on the root DescriptorProtocol (Spec 5).

Spec 5 (sec.6) requires every object that *chooses a route* -- a layout, an AMR policy, a
geometry, a boundary, a mask -- to be a typed descriptor that can declare its ``requirements``
/ ``capabilities`` / ``options`` and answer ``available(context)`` with an *explainable*
status (not just a bool). These mesh descriptors are inert: they describe a choice the C++
runtime will materialise after validation; nothing here computes a cell, a face or a patch.

Spec 5 Phase D unifies the two descriptor families: :class:`MeshDescriptor` now subclasses the
shared :class:`pops.descriptors.Descriptor` (so the mesh objects honour the same documented
``DescriptorProtocol`` as the native :class:`pops.descriptors.BrickDescriptor`), and
:class:`Availability` is consumed from its unique owner :mod:`pops.descriptors`; it is not
re-exported by :mod:`pops.mesh`. The ``mesh -> descriptors`` import is on a flat root module
(``pops.descriptors`` is not a tracked layer), so it does not add a cross-layer edge.
"""
from __future__ import annotations

from pops.descriptors import Descriptor


class MeshDescriptor(Descriptor):
    """Base of the inert mesh / layout / AMR descriptors (Spec 5 sec.6).

    A thin specialisation of :class:`pops.descriptors.Descriptor` that defaults
    :attr:`category` to ``"mesh"``; subclasses override :meth:`options` (and, where a route can
    be refused, :meth:`available` / :meth:`validate`). The full inert contract -- empty
    requirements / capabilities, an unconditionally-available status, the plain-dict
    :meth:`inspect`, the inert :meth:`lower`, and the short deterministic ``str`` -- is inherited
    unchanged from the shared base. A mesh descriptor computes nothing.
    """

    category = "mesh"


__all__ = ["MeshDescriptor"]
