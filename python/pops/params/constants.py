"""Dimensioned constants use the canonical :class:`ConstParam` declaration."""

from pops.params.runtime import ConstParam


# An alias, not a fourth parameter kind or a parallel declaration hierarchy.
Constant = ConstParam


__all__ = ["Constant"]
