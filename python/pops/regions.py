"""The region level of spec 0.3 (SU15): the typed index maps of IR9 and the entities they act on.

``pops.regions`` is the one public home of the typed maps: ``Shift``, ``Incidence``, ``Trace``,
``Parent``, ``Child``, ``Project``, ``Embed``, ``Compose`` and ``Inverse``, and of the entity kinds
``cell``, ``face(a)`` and ``boundary(a, side)`` that they read and give. A map applied to an index,
``m(i)``, gives the index to which m maps i, and two maps are equal when their normal forms are
(IR9).

Until the index sets of IR2 exist, ``Incidence`` takes entity kinds, ``Incidence(face(a), cell,
side="left")``, and ``Project`` and ``Embed`` take the base rank that a ``ProductSet`` will give
them. The index sets and the region builder of IR4 join this module with the issue that builds
them; the spellings stay provisional until N15 settles the user surface.
"""
from pops._ir.index_maps import (
    Child, Compose, Embed, Incidence, Inverse, Parent, Project, Shift, Trace, boundary, cell, face)

__all__ = ["Child", "Compose", "Embed", "Incidence", "Inverse", "Parent", "Project", "Shift",
           "Trace", "boundary", "cell", "face"]
