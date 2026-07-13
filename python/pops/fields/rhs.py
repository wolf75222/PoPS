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

from ._references import reference_label, resolve_handle


class _RHS(Descriptor):
    """Base of the RHS descriptors: composable with ``+`` into a :class:`SumRHS`."""

    __pops_ir_immutable__ = True
    category = "rhs"

    def __add__(self, other: Any) -> Any:
        if not isinstance(other, _RHS):
            return NotImplemented
        return SumRHS(self, other)

    def freeze(self) -> Any:
        """RHS declarations are immutable value nodes and need no freeze-time rewrite."""
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)


class ChargeDensity(_RHS):
    """The charge-density right-hand side of a Poisson-type field solve.

    Built with :meth:`from_blocks` from the names of the physics blocks that deposit
    charge; the runtime assembles the density from those blocks' conserved state.
    """

    def __init__(self, blocks: Any = ()) -> None:
        from pops.problem.handles import BlockHandle

        refs = tuple(blocks)
        invalid = [block for block in refs if not isinstance(block, BlockHandle)]
        if invalid:
            raise TypeError(
                "ChargeDensity blocks must be BlockHandle values; names/strings are not "
                "references (got %r)" % type(invalid[0]).__name__
            )
        object.__setattr__(self, "blocks", refs)

    @classmethod
    def from_blocks(cls, *blocks: Any) -> ChargeDensity:
        """A :class:`ChargeDensity` summed over the named contributing :paramref:`blocks`.

        Accepts block handles either as varargs or as a single iterable.
        """
        if (
            len(blocks) == 1
            and not hasattr(blocks[0], "qualified_id")
            and hasattr(blocks[0], "__iter__")
        ):
            blocks = tuple(blocks[0])
        return cls(blocks=blocks)

    def options(self) -> dict:
        return {
            "rhs": "charge_density",
            "blocks": [
                reference_label(block, where="ChargeDensity block") for block in self.blocks
            ],
        }

    def requirements(self) -> Any:
        return RequirementSet(
            {
                "blocks": [
                    reference_label(block, where="ChargeDensity block") for block in self.blocks
                ]
            }
        )

    def resolve_references(self, resolver: Any) -> ChargeDensity:
        return type(self)(
            blocks=tuple(
                resolve_handle(block, resolver, where="ChargeDensity block")
                for block in self.blocks
            )
        )

    def declaration_references(self) -> tuple[Any, ...]:
        return self.blocks


class FixedSource(_RHS):
    """A static source term named :paramref:`aux_field` the runtime adds to the RHS.

    Names an aux field (a background charge, a fixed forcing) that contributes to the
    right-hand side alongside the block-deposited charge; the runtime reads its values.
    """

    def __init__(self, aux_field: Any) -> None:
        from pops.model import Handle

        if not isinstance(aux_field, Handle):
            raise TypeError(
                "FixedSource aux_field must be a declaration Handle; names/strings are not "
                "references (got %r)" % type(aux_field).__name__
            )
        object.__setattr__(self, "aux_field", aux_field)

    @property
    def name(self) -> str:
        return self.aux_field.local_id

    def options(self) -> dict:
        return {
            "rhs": "fixed_source",
            "aux_field": reference_label(self.aux_field, where="FixedSource aux_field"),
        }

    def requirements(self) -> Any:
        return RequirementSet(
            {"aux_field": reference_label(self.aux_field, where="FixedSource aux_field")}
        )

    def resolve_references(self, resolver: Any) -> FixedSource:
        return type(self)(resolve_handle(self.aux_field, resolver, where="FixedSource aux_field"))

    def declaration_references(self) -> tuple[Any, ...]:
        return (self.aux_field,)


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
                    "(ChargeDensity / FixedSource / SumRHS); got %r" % (type(term).__name__,)
                )
        if not flat:
            raise ValueError("SumRHS: needs at least one RHS term")
        object.__setattr__(self, "terms", tuple(flat))

    def options(self) -> dict:
        return {
            "rhs": "sum",
            "n_terms": len(self.terms),
            "terms": [t.options().get("rhs") for t in self.terms],
        }

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

    def resolve_references(self, resolver: Any) -> SumRHS:
        return type(self)(*(term.resolve_references(resolver) for term in self.terms))

    def declaration_references(self) -> tuple[Any, ...]:
        return tuple(
            reference for term in self.terms for reference in term.declaration_references()
        )


__all__ = ["ChargeDensity", "FixedSource", "SumRHS"]
