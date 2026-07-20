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
_TRANSPORT_MODE_TOKENS = ("none", "staircase", "cutcell")


class TransportMask(MeshDescriptor):
    """Extensible typed interface for an embedded-domain transport policy.

    Implementations provide a native token through :meth:`lower`; callers select the policy with
    the descriptor itself, never with that token.
    """

    category = "transport_mask"
    #: The native embedded-domain transport token selected by this policy.
    mode_token = ""

    def options(self) -> dict:
        return {"mode": self.mode_token}

    def lower(self, context: Any = None) -> Any:
        """Return the private native embedded-boundary transport token."""
        return self.mode_token


class NoMask(TransportMask):
    """No masking: the embedded geometry is ignored by transport (mode='none')."""

    mode_token = "none"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": False})


class Staircase(TransportMask):
    """Conservative active-cell masking with binary closed faces (mode='staircase')."""

    mode_token = "staircase"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": True, "conservative": True})


class CutCell(TransportMask):
    """Disc-specific center-sampled cut-distance transport (mode='cutcell').

    The present native route is retained for the historical :class:`~pops.mesh.geometry.Disc`
    problem.  It is not accepted for arbitrary ``LevelSet`` or CSG geometry: those require true
    face apertures, intersection volumes and a typed wall-flux policy.  Use :class:`Staircase` for
    generic implicit geometry until that complete route exists; PoPS never silently substitutes it.

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


def lower_transport_mask(mode: Any) -> str:
    """Lower a typed transport mask to its native token.

    This function is an implementation seam, not a coercion layer: strings are rejected even when
    they spell a valid native token. Subclassing :class:`TransportMask` is the extension protocol.
    """
    if not isinstance(mode, TransportMask):
        raise TypeError(
            "transport mode must be a pops.mesh.masks.TransportMask descriptor "
            "(NoMask / Staircase / CutCell), got %s" % type(mode).__name__)
    token = mode.lower()
    if not isinstance(token, str) or token not in _TRANSPORT_MODE_TOKENS:
        raise ValueError(
            "%s.lower() returned unsupported native transport token %r"
            % (type(mode).__name__, token))
    return token


def transport_mask_thresholds(mode: Any) -> dict:
    """Return finite normalized thresholds carried by a typed transport mask.

    Extension policies may implement ``thresholds()`` but cannot weaken the native safety domain:
    every published value is an exact real scalar in ``[0, 1]`` (zero selects the native default).
    """
    lower_transport_mask(mode)
    thresholds = getattr(mode, "thresholds", None)
    if not callable(thresholds):
        return {}
    resolved = thresholds()
    if type(resolved) is not dict:
        raise TypeError("TransportMask.thresholds() must return an exact dict")
    supported = {"kappa_min", "face_open_eps", "cut_theta_min"}
    if not set(resolved).issubset(supported):
        raise ValueError(
            "TransportMask.thresholds() uses unsupported keys %s"
            % sorted(set(resolved) - supported)
        )
    normalized = {}
    for name, value in resolved.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("TransportMask threshold %s must be a real scalar" % name)
        checked = float(value)
        if not math.isfinite(checked) or checked < 0.0 or checked > 1.0:
            raise ValueError("TransportMask threshold %s must be finite and in [0, 1]" % name)
        normalized[name] = checked
    return normalized


__all__ = [
    "TransportMask",
    "NoMask",
    "Staircase",
    "CutCell",
    "lower_transport_mask",
    "transport_mask_thresholds",
]
