"""pops.lib.reconstruction.limiters -- the slope-limiter brick catalog (Spec 3).

Minmod / VanLeer have native types; MC / Superbee are catalogued but have no
native type yet (available=False).
"""
from types import SimpleNamespace

from ..descriptors import _native, _planned

limiters = SimpleNamespace(
    Minmod=lambda: _native("minmod", "pops::Minmod", "minmod", category="limiter"),
    VanLeer=lambda: _native("vanleer", "pops::VanLeer", "vanleer", category="limiter"),
    # MC / Superbee are catalogued but have no native type yet (available=False).
    MC=lambda: _planned("mc", "mc", category="limiter"),
    Superbee=lambda: _planned("superbee", "superbee", category="limiter"),
)

__all__ = ["limiters"]
