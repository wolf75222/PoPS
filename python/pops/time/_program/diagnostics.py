"""Program stage grouping and scalar diagnostic authoring."""
from __future__ import annotations

from typing import Any

from pops.time.value_collections import StageStateSet
from pops.time.value_metadata import positive_scalar_literal
from pops.time._authoring import atomic_authoring
from pops.time.values import ProgramValue


class _ProgramDiagnostics:
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
