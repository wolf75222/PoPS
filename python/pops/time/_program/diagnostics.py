"""Program stage grouping and scalar diagnostic authoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.value_collections import StageStateSet
from pops.time.value_metadata import positive_scalar_literal
from pops.time._authoring import atomic_authoring
from pops.time.values import ProgramValue

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramDiagnostics(_ProgramBase):
    def state_set(self, name: Any, mapping: Any) -> StageStateSet:
        return StageStateSet(name, mapping)

    def record(self, name: Any, value: Any) -> ProgramValue:
        """Record an already-authored scalar reduction as a named diagnostic."""
        if not (isinstance(value, ProgramValue) and value.vtype == "scalar"):
            raise ValueError(
                "record(%r): value must be a Program scalar (e.g. P.sum / P.norm2); got %r"
                % (name, value))
        return self.record_scalar(name, value)

    @atomic_authoring
    def record_balance(
        self,
        ledger: Any,
        *,
        storage_change: Any,
        outward_boundary_flux: Any,
        sources: Any,
        reflux: Any = None,
        projection: Any = None,
    ) -> tuple[ProgramValue, ...]:
        """Publish one exact five-term balance into the current native attempt.

        Every explicitly authored term is a signed, time-integrated increment for this Program
        invocation and must be an additive global Program reduction (sum/dot), or scalar arithmetic
        composed exclusively from such reductions and exact literals. A ledger that explicitly
        delegates ``reflux`` or ``projection`` to its native producer requires the corresponding
        argument to remain ``None``. The native mailbox accumulates all increments across cadence
        substeps in the same public macro-step. Raw Python values, extrema/norm reductions, and
        rank-local runtime scalars are rejected. The records are attempt-local: a rejected step or
        consumer rollback cannot leave evidence for a later sample.
        """
        from pops._balance_contract import (
            BALANCE_TERM_NAMES,
            BalanceLedger,
            balance_record_name,
        )

        if type(ledger) is not BalanceLedger:
            raise TypeError(
                "record_balance ledger must be an exact pops.diagnostics.BalanceLedger"
            )
        supplied = {
            "storage_change": storage_change,
            "outward_boundary_flux": outward_boundary_flux,
            "sources": sources,
            "reflux": reflux,
            "projection": projection,
        }

        def require_reduced(value: Any, term: str, seen: set[int]) -> ProgramValue:
            value = self._canonical_value(value)
            if not isinstance(value, ProgramValue) or value.prog is not self \
                    or value.vtype != "scalar":
                raise TypeError(
                    "record_balance %s must be a scalar from this Program" % term
                )
            if value.id in seen:
                return value
            seen.add(value.id)
            if value.op == "reduce":
                if value.attrs.get("kind") not in {"sum", "dot"}:
                    raise ValueError(
                        "record_balance %s requires additive sum/dot reductions; got %r"
                        % (term, value.attrs.get("kind"))
                    )
                return value
            if value.op == "scalar_op" and value.inputs:
                for item in value.inputs:
                    require_reduced(item, term, seen)
                return value
            raise ValueError(
                "record_balance %s must be a global reduction or arithmetic composed "
                "only from global reductions; got scalar op %r" % (term, value.op)
            )

        automatic = set(ledger.automatic_terms)
        for name in automatic:
            if supplied[name] is not None:
                raise ValueError(
                    "record_balance %s is owned by the ledger's native automatic producer; "
                    "leave it as None" % name
                )
        terms = {
            name: require_reduced(supplied[name], name, set())
            for name in BALANCE_TERM_NAMES
            if name not in automatic
        }
        if automatic:
            expected_component = ledger.component

            def reduced_components(
                value: ProgramValue, term: str, seen: set[int]
            ) -> set[int]:
                if value.id in seen:
                    return set()
                seen.add(value.id)
                if value.op == "reduce":
                    component = value.attrs.get("comp")
                    if value.attrs.get("kind") != "sum" or type(component) is not int:
                        raise ValueError(
                            "record_balance %s must use component-qualified sum reductions "
                            "when native terms are selected" % term
                        )
                    return {component}
                components: set[int] = set()
                for item in value.inputs:
                    components.update(reduced_components(item, term, seen))
                return components

            for name, value in terms.items():
                components = reduced_components(value, name, set())
                if components != {expected_component}:
                    raise ValueError(
                        "record_balance %s selects components %s but the native ledger owns "
                        "component %d" % (name, sorted(components), expected_component)
                    )
        blocks = {value.block for value in terms.values()}
        if None in blocks or len(blocks) != 1:
            raise ValueError(
                "record_balance terms must reduce one exact common physics block"
            )
        route = ledger.route_identity(next(iter(blocks)))
        return tuple(
            self._new(
                "scalar",
                "record_balance_term",
                (terms[name],),
                {
                    "diagnostic": balance_record_name(route, name),
                    "route": route.token,
                    "term": name,
                },
                balance_record_name(route, name),
                terms[name].block,
            )
            for name in BALANCE_TERM_NAMES
            if name in terms
        )

    @atomic_authoring
    def check_invariant(self, name: Any, before: Any = None, after: Any = None,
                        tolerance: Any = 1e-10) -> ProgramValue:
        """Record invariant drift with immutable tolerance metadata."""
        if not (isinstance(before, ProgramValue) and before.vtype == "scalar"
                and isinstance(after, ProgramValue) and after.vtype == "scalar"):
            raise ValueError(
                "check_invariant(%r): before/after must be Program scalars" % (name,))
        tolerance_literal = positive_scalar_literal(
            tolerance, where="check_invariant: tolerance")
        out = self.record_scalar(name + "_drift", after - before)
        attrs = dict(out.attrs)
        attrs["tolerance"] = tolerance_literal
        return self._replace_value(out, attrs=attrs)


__all__ = ["_ProgramDiagnostics"]
