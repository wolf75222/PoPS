"""pops.lib.descriptors -- transitional re-export of :mod:`pops.descriptors`.

Spec 5 (sec.4 / sec.6) moves the canonical :class:`BrickDescriptor` and the shared
descriptor factories out of ``pops.lib`` (which becomes presets-only) into the
top-level :mod:`pops.descriptors`. This module is a thin alias kept ONLY so the
catalogs still parked under ``pops.lib`` (``lib.spatial`` / ``lib.fields`` /
``lib.solvers``) keep resolving ``from ..descriptors import ...`` until their Spec 5
Phase A2 relocation lands. It is removed once those catalogs move; new code imports
from :mod:`pops.descriptors`.
"""
from pops.descriptors import (  # noqa: F401
    BRICK_TYPES,
    BrickDescriptor,
    _clear_external_catalog,
    _external_descriptor,
    _native,
    _planned,
    _register_manifest,
    _split_csv,
    external,
    load_cpp_library,
)

__all__ = ["BrickDescriptor", "load_cpp_library", "external",
           "_register_manifest", "_clear_external_catalog", "_native", "_planned",
           "_external_descriptor", "_split_csv", "BRICK_TYPES"]
