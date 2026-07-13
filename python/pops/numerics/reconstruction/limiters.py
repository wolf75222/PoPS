"""Executable slope-limiter descriptors exposed by the final catalog."""
from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _native

limiters = SimpleNamespace(
    Minmod=lambda: _native("minmod", "pops::Minmod", "minmod", category="limiter"),
    VanLeer=lambda: _native("vanleer", "pops::VanLeer", "vanleer", category="limiter"),
)

# Spec 5: expose the limiters at module scope.
Minmod = limiters.Minmod
VanLeer = limiters.VanLeer

__all__ = ["limiters", "Minmod", "VanLeer"]
