"""pops.mesh.masks -- transport-mask descriptors for embedded boundaries (Spec 5 sec.8.16.1).

The typed replacement for the ``set_disc_domain(..., mode="none"|"staircase"|"cutcell")``
string. A mask says HOW transport is masked at an embedded boundary; the runtime applies
it. Inert descriptors.

Each mask carries the native disc-transport token (``none`` / ``staircase`` / ``cutcell``) that
the C++ ``set_disc_domain`` / ``set_geometry_mode`` consume, exposed via :meth:`lower` (and the
shared :func:`lower_disc_mode`, which also passes a legacy string through unchanged). The lowered
token is byte-identical to what a user passes today in the string form.
"""
from __future__ import annotations

from typing import Any

from .._descriptor import MeshDescriptor
from ...descriptors_report import RequirementSet, CapabilitySet

#: The native disc-transport tokens (single source). A typed mask lowers to one of these.
DISC_MODE_TOKENS = ("none", "staircase", "cutcell")


class _TransportMask(MeshDescriptor):
    """Base of the disc-transport masks: carries the native token via :attr:`mode_token`."""

    category = "transport_mask"
    #: The native ``set_disc_domain`` / ``set_geometry_mode`` token this mask selects.
    mode_token = "none"

    def options(self) -> dict:
        return {"mode": self.mode_token}

    def lower(self, context: Any = None) -> Any:
        """The native disc-transport token (byte-identical to the legacy ``mode=`` string)."""
        return self.mode_token


class NoMask(_TransportMask):
    """No masking: the embedded geometry is ignored by transport (mode='none')."""

    mode_token = "none"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": False})


class Staircase(_TransportMask):
    """Staircase masking: cells fully inside the wall are excluded (mode='staircase')."""

    mode_token = "staircase"

    def capabilities(self) -> Any:
        return CapabilitySet({"masked_transport": True, "conservative": False})


class CutCell(_TransportMask):
    """Cut-cell masking: conservative masked transport on cut cells (mode='cutcell').

    ADC-615 exposes the cut-cell numeric thresholds, previously hardcoded native constants:

    * ``kappa_min`` -- small-cell volume-fraction floor (default 1e-2): bounds the 1/kappa
      amplification so an arbitrarily cut cell keeps a finite, stable explicit step;
    * ``face_open_eps`` -- aperture below which a face is treated as CLOSED (default 1e-6);
    * ``cut_theta_min`` -- the cut-fraction clamp (default 1e-3) SHARED with the elliptic
      Shortley-Weller wall, so the finite-volume aperture stays bit-consistent with the wall.

    ``None`` for any keyword keeps the native default (bit-identical). Out-of-domain values are
    refused STRUCTURALLY (never silently clamped)."""

    mode_token = "cutcell"

    def __init__(self, kappa_min: Any = None, face_open_eps: Any = None,
                 cut_theta_min: Any = None) -> None:
        self.kappa_min = self._check(kappa_min, "kappa_min", unit_interval=True)
        self.face_open_eps = self._check(face_open_eps, "face_open_eps", unit_interval=False)
        self.cut_theta_min = self._check(cut_theta_min, "cut_theta_min", unit_interval=True)

    @staticmethod
    def _check(value: Any, name: str, unit_interval: bool) -> float:
        """Validate a threshold: None -> 0.0 (native default); else positive, and in (0, 1] when
        ``unit_interval``. Rejects out-of-domain values (a degenerate clamp is a structural error)."""
        if value is None:
            return 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("CutCell(%s=) must be a number or None; got %r" % (name, value))
        v = float(value)
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


def lower_disc_mode(mode: Any) -> Any:
    """Lower a disc-transport ``mode`` to its native token (Spec 5 sec.8.16).

    Accepts a typed :class:`_TransportMask` (``NoMask`` / ``Staircase`` / ``CutCell``) -> its
    ``mode_token``, OR the legacy string (``"none"`` / ``"staircase"`` / ``"cutcell"``), which is
    validated and passed through unchanged. Any other type is a clear :class:`TypeError`; an
    unknown string is a :class:`ValueError` naming the accepted tokens. Mirrors the
    string-or-typed coercion used elsewhere (the string path stays byte-identical).

    Args:
        mode: A ``pops.mesh.masks`` descriptor or a legacy disc-mode string.

    Returns:
        The native disc-transport token (``"none"`` / ``"staircase"`` / ``"cutcell"``).
    """
    if isinstance(mode, str):
        if mode not in DISC_MODE_TOKENS:
            raise ValueError(
                "set_disc_domain: unknown mode %r (expected one of %s, or a typed "
                "pops.mesh.masks.NoMask() / Staircase() / CutCell())"
                % (mode, ", ".join(DISC_MODE_TOKENS)))
        return mode
    token = getattr(mode, "mode_token", None)
    if token is None or getattr(mode, "category", None) != "transport_mask":
        raise TypeError(
            "set_disc_domain: mode must be a pops.mesh.masks transport mask "
            "(NoMask / Staircase / CutCell) or a disc-mode string (none / staircase / "
            "cutcell), got %r" % (type(mode).__name__,))
    return token


def disc_mode_thresholds(mode: Any) -> dict:
    """The resolved cut-cell numeric thresholds carried by a disc-transport ``mode`` (ADC-615).

    A typed :class:`CutCell` returns its ``kappa_min`` / ``face_open_eps`` / ``cut_theta_min`` (0.0 =
    keep the native default). Any other mask, a legacy string, or a non-mask returns ``{}`` (the
    native ``set_disc_domain`` then keeps every kEb* default, bit-identical).

    Args:
        mode: A ``pops.mesh.masks`` descriptor or a legacy disc-mode string.

    Returns:
        A dict with ``kappa_min`` / ``face_open_eps`` / ``cut_theta_min``, or ``{}``.
    """
    thresholds = getattr(mode, "thresholds", None)
    if callable(thresholds):
        resolved = thresholds()
        if isinstance(resolved, dict):
            return resolved
    return {}


__all__ = ["NoMask", "Staircase", "CutCell", "DISC_MODE_TOKENS", "lower_disc_mode",
           "disc_mode_thresholds"]
