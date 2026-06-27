"""pops.mesh.masks -- transport-mask descriptors for embedded boundaries (Spec 5 sec.8.16.1).

The typed replacement for the ``set_disc_domain(..., mode="none"|"staircase"|"cutcell")``
string. A mask says HOW transport is masked at an embedded boundary; the runtime applies
it. Inert descriptors.

Each mask carries the native disc-transport token (``none`` / ``staircase`` / ``cutcell``) that
the C++ ``set_disc_domain`` / ``set_geometry_mode`` consume, exposed via :meth:`lower` (and the
shared :func:`lower_disc_mode`, which also passes a legacy string through unchanged). The lowered
token is byte-identical to what a user passes today in the string form.
"""
from .._descriptor import MeshDescriptor

#: The native disc-transport tokens (single source). A typed mask lowers to one of these.
DISC_MODE_TOKENS = ("none", "staircase", "cutcell")


class _TransportMask(MeshDescriptor):
    """Base of the disc-transport masks: carries the native token via :attr:`mode_token`."""

    category = "transport_mask"
    #: The native ``set_disc_domain`` / ``set_geometry_mode`` token this mask selects.
    mode_token = "none"

    def options(self):
        return {"mode": self.mode_token}

    def lower(self, context=None):
        """The native disc-transport token (byte-identical to the legacy ``mode=`` string)."""
        return self.mode_token


class NoMask(_TransportMask):
    """No masking: the embedded geometry is ignored by transport (mode='none')."""

    mode_token = "none"

    def capabilities(self):
        return {"masked_transport": False}


class Staircase(_TransportMask):
    """Staircase masking: cells fully inside the wall are excluded (mode='staircase')."""

    mode_token = "staircase"

    def capabilities(self):
        return {"masked_transport": True, "conservative": False}


class CutCell(_TransportMask):
    """Cut-cell masking: conservative masked transport on cut cells (mode='cutcell')."""

    mode_token = "cutcell"

    def requirements(self):
        return {"embedded_boundary_support": True}

    def capabilities(self):
        return {"masked_transport": True, "conservative": True}


def lower_disc_mode(mode):
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


__all__ = ["NoMask", "Staircase", "CutCell", "DISC_MODE_TOKENS", "lower_disc_mode"]
