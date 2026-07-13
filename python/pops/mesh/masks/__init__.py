"""Typed transport-mask descriptors for embedded boundaries.

The public selector is always a descriptor: :class:`NoMask`, :class:`Staircase`,
or :class:`CutCell`. Native string tokens exist only as lowering output and are never accepted as
authoring input. This keeps validation, capabilities, and cut-cell thresholds attached to the
selected policy instead of allowing untyped values to bypass the descriptor contract.
"""
from __future__ import annotations

import math
from typing import Any

from .._descriptor import MeshDescriptor
from ...descriptors_report import RequirementSet, CapabilitySet

# Native-only tokens consumed by the C++ runtime. They are deliberately not exported.
_DISC_MODE_TOKENS = ("none", "staircase", "cutcell")


class TransportMask(MeshDescriptor):
    """Extensible typed interface for a disc-transport policy.

    Implementations provide a native token through :meth:`lower`; callers select the policy with
    the descriptor itself, never with that token.
    """

    category = "transport_mask"
    #: The native ``set_disc_domain`` / ``set_geometry_mode`` token this mask selects.
    mode_token = ""

    def options(self) -> dict:
        return {"mode": self.mode_token}

    def lower(self, context: Any = None) -> Any:
        """Return the private native disc-transport token."""
        return self.mode_token


class NoMask(TransportMask):
    """No masking: the embedded geometry is ignored by transport (mode='none')."""

    mode_token = "none"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": False})


class Staircase(TransportMask):
    """Staircase masking: cells fully inside the wall are excluded (mode='staircase')."""

    mode_token = "staircase"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": True, "conservative": False})


class CutCell(TransportMask):
    """Cut-cell masking: conservative masked transport on cut cells (mode='cutcell').

    ADC-615 exposes the cut-cell numeric thresholds, previously hardcoded native constants:

    * ``kappa_min`` -- small-cell volume-fraction floor (default 1e-2): bounds the 1/kappa
      amplification so an arbitrarily cut cell keeps a finite, stable explicit step;
    * ``face_open_eps`` -- aperture in ``(0, 1]`` below which a face is CLOSED (default 1e-6);
    * ``cut_theta_min`` -- the cut-fraction clamp (default 1e-3) SHARED with the elliptic
      Shortley-Weller wall, so the finite-volume aperture stays bit-consistent with the wall.

    ``None`` for any keyword keeps the native default (bit-identical). Out-of-domain values are
    refused STRUCTURALLY (never silently clamped)."""

    mode_token = "cutcell"

    def __init__(self, kappa_min: Any = None, face_open_eps: Any = None,
                 cut_theta_min: Any = None) -> None:
        self.kappa_min = self._check(kappa_min, "kappa_min", unit_interval=True)
        self.face_open_eps = self._check(face_open_eps, "face_open_eps", unit_interval=True)
        self.cut_theta_min = self._check(cut_theta_min, "cut_theta_min", unit_interval=True)

    @staticmethod
    def _check(value: Any, name: str, unit_interval: bool) -> float:
        """Validate a threshold: None -> 0.0 (native default); else positive, and in (0, 1] when
        ``unit_interval``. Rejects out-of-domain values (a degenerate clamp is a structural error)."""
        if value is None:
            return 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("CutCell(%s=) must be a real number or None; got %r" % (name, value))
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("CutCell(%s=) must be finite; got %r" % (name, value))
        if v <= 0.0:
            raise ValueError("CutCell(%s=) must be > 0 (or None for the default); got %r"
                             % (name, value))
        if unit_interval and v > 1.0:
            raise ValueError("CutCell(%s=) must be in (0, 1]; got %r" % (name, value))
        return v

    def options(self) -> dict:
        opt = {"mode": self.mode_token}
        opt.update(self.thresholds())
        return opt

    def thresholds(self) -> dict:
        """The resolved native cut-cell thresholds (0.0 = keep the native default). ADC-615."""
        return {"kappa_min": self.kappa_min, "face_open_eps": self.face_open_eps,
                "cut_theta_min": self.cut_theta_min}

    def requirements(self) -> Any:
        return RequirementSet({"embedded_boundary_support": True})

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": True, "conservative": True})


def lower_disc_mode(mode: Any) -> str:
    """Lower a typed transport mask to its native token.

    This function is an implementation seam, not a coercion layer: strings are rejected even when
    they spell a valid native token. Subclassing :class:`TransportMask` is the extension protocol.
    """
    if not isinstance(mode, TransportMask):
        raise TypeError(
            "transport mode must be a pops.mesh.masks.TransportMask descriptor "
            "(NoMask / Staircase / CutCell), got %s" % type(mode).__name__)
    token = mode.lower()
    if not isinstance(token, str) or token not in _DISC_MODE_TOKENS:
        raise ValueError(
            "%s.lower() returned unsupported native transport token %r"
            % (type(mode).__name__, token))
    return token


def disc_mode_thresholds(mode: Any) -> dict:
    """Return numeric thresholds carried by a validated typed transport mask."""
    lower_disc_mode(mode)
    thresholds = getattr(mode, "thresholds", None)
    if callable(thresholds):
        resolved = thresholds()
        if isinstance(resolved, dict):
            return resolved
    return {}


__all__ = ["TransportMask", "NoMask", "Staircase", "CutCell", "lower_disc_mode",
           "disc_mode_thresholds"]
