"""pops.fields.rhs -- typed right-hand-side sources for a field solve (Spec 5 sec.5.5).

The right-hand side of an elliptic field solve (the source of a Poisson problem) is a
typed descriptor rather than a raw array. :class:`ChargeDensity` describes the charge
density assembled from the contributing physics blocks; :meth:`ChargeDensity.from_blocks`
builds it from the named blocks that deposit charge. :class:`FixedSource` names a static
aux field the runtime adds, and :class:`SumRHS` composes several RHS descriptors into a
single multi-block / multi-source right-hand side (``ChargeDensity(...) + FixedSource(...)``).

Inert descriptors; the runtime assembles the actual density field.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet


class _RHS(Descriptor):
    """Base of the RHS descriptors: composable with ``+`` into a :class:`SumRHS`."""

    category = "rhs"

    def __add__(self, other: Any) -> Any:
        if not isinstance(other, _RHS):
            return NotImplemented
        return SumRHS(self, other)


class ChargeDensity(_RHS):
    """The charge-density right-hand side of a Poisson-type field solve.

    Built with :meth:`from_blocks` from the names of the physics blocks that deposit
    charge; the runtime assembles the density from those blocks' conserved state.
    """

    def __init__(self, blocks: Any = ()) -> None:
        self.blocks = tuple(str(b) for b in blocks)

    @classmethod
    def from_blocks(cls, *blocks: Any) -> ChargeDensity:
        """A :class:`ChargeDensity` summed over the named contributing :paramref:`blocks`.

        Accepts the block names either as varargs (``from_blocks("ions", "electrons")``)
        or as a single iterable (``from_blocks(["ions", "electrons"])``).
        """
        if len(blocks) == 1 and not isinstance(blocks[0], str) and hasattr(blocks[0], "__iter__"):
            blocks = tuple(blocks[0])
        return cls(blocks=blocks)

    def options(self) -> dict:
        return {"rhs": "charge_density", "blocks": self.blocks}

    def requirements(self) -> Any:
        return RequirementSet({"blocks": list(self.blocks)})


class FixedSource(_RHS):
    """A static source term named :paramref:`aux_field` the runtime adds to the RHS.

    Names an aux field (a background charge, a fixed forcing) that contributes to the
    right-hand side alongside the block-deposited charge; the runtime reads its values.
    """

    def __init__(self, aux_field: Any) -> None:
        self.aux_field = str(aux_field)

    @property
    def name(self) -> str:
        return self.aux_field

    def options(self) -> dict:
        return {"rhs": "fixed_source", "aux_field": self.aux_field}

    def requirements(self) -> Any:
        return RequirementSet({"aux_field": self.aux_field})


class SumRHS(_RHS):
    """A composed right-hand side: the sum of several typed RHS descriptors (Spec 5 sec.9).

    Built by ``+`` on the RHS descriptors (``ChargeDensity.from_blocks("ions", "electrons") +
    FixedSource("rho_background")``) or directly (``SumRHS(a, b, c)``). It flattens nested sums
    so the composition stays a single flat list of terms, and unions each term's requirements.
    """

    def __init__(self, *terms: Any) -> None:
        flat = []
        for term in terms:
            if isinstance(term, SumRHS):
                flat.extend(term.terms)
            elif isinstance(term, _RHS):
                flat.append(term)
            else:
                raise TypeError(
                    "SumRHS: every term must be a typed pops.fields.rhs descriptor "
                    "(ChargeDensity / FixedSource / SumRHS); got %r" % (type(term).__name__,))
        if not flat:
            raise ValueError("SumRHS: needs at least one RHS term")
        self.terms = tuple(flat)

    def options(self) -> dict:
        return {"rhs": "sum", "n_terms": len(self.terms),
                "terms": [t.options().get("rhs") for t in self.terms]}

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        blocks, aux = [], []
        for term in self.terms:
            req = term.requirements().to_dict()
            blocks.extend(req.get("blocks", []))
            if "aux_field" in req:
                aux.append(req["aux_field"])
        out = {}
        if blocks:
            out["blocks"] = blocks
        if aux:
            out["aux_fields"] = aux
        return RequirementSet(out)


__all__ = ["ChargeDensity", "FixedSource", "SumRHS"]
