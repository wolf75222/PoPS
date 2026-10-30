"""The compiled control of an elaborated Advance (spec 0.3 sections 5.2 and 11.1, T28, T38).

An ``Advance`` elaborates into a Program whose candidate passes through its assessments before
the commit, and into an ``AdvanceControl``: the record of those assessments, of the retry budget
and of the first proposal. The record enters the Program identity, the generated artifact exports
its attempt budget, and the artifact manifest publishes it, the default acceptance included with
its origin (SU9). ``System::run_advance`` runs the attempt loop of the record in C++.
"""
from __future__ import annotations

import math
from typing import Any

ADVANCE_CONTROL_SCHEMA = "pops.advance-control@1"
"""Schema of ``AdvanceControl.to_data``."""

ADVANCE_ASSESSMENT_CONTRACT = "pops.advance-assessment@1"
"""Attribute of an acceptance guard that is an assessment of an Advance: its rejection is raised
in the phase that the compiled run repeats (T28), every other rejection is a trial failure."""

_MAX_ATTEMPTS = 2**31 - 1


class AdvanceControl:
    """The assessments, the retry budget and the first proposal of one elaborated Advance."""

    __slots__ = ("assessments", "max_attempts", "max_transforms", "first_dt")

    def __init__(self, *, assessments: tuple, max_attempts: int, max_transforms: int,
                 first_dt: float) -> None:
        if type(assessments) is not tuple or not assessments or any(
                type(row) is not tuple or len(row) != 2
                or any(type(item) is not str or not item for item in row)
                for row in assessments):
            raise TypeError("AdvanceControl.assessments holds (name, origin) pairs of text")
        if type(max_attempts) is not int or not 1 <= max_attempts <= _MAX_ATTEMPTS:
            # The artifact exports the budget as a C int (pops_program_advance_max_attempts).
            raise ValueError("AdvanceControl.max_attempts is an integer in [1, 2**31 - 1]")
        if type(max_transforms) is not int or max_transforms != 0:
            raise ValueError("AdvanceControl.max_transforms is 0 until transforms are lowered")
        if type(first_dt) is not float or not math.isfinite(first_dt) or first_dt <= 0.0:
            raise ValueError("AdvanceControl.first_dt is a finite positive binary64 duration")
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "max_transforms", max_transforms)
        object.__setattr__(self, "first_dt", first_dt)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AdvanceControl is immutable")

    def __eq__(self, other: object) -> bool:
        return type(other) is AdvanceControl and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash((self.assessments, self.max_attempts, self.max_transforms, self.first_dt))

    def __repr__(self) -> str:
        return "AdvanceControl(%r)" % (self.to_data(),)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ADVANCE_CONTROL_SCHEMA,
            "accept": [{"assess": name, "origin": origin} for name, origin in self.assessments],
            "retry": {"max_attempts": self.max_attempts, "max_transforms": self.max_transforms},
            "proposal": {"dt": self.first_dt.hex()},
        }

    @classmethod
    def from_data(cls, data: Any) -> AdvanceControl:
        if not isinstance(data, dict) or set(data) != {"schema", "accept", "retry", "proposal"} \
                or data["schema"] != ADVANCE_CONTROL_SCHEMA:
            raise ValueError("Advance control record has another schema")
        accept, retry, proposal = data["accept"], data["retry"], data["proposal"]
        if type(accept) is not list or any(
                not isinstance(row, dict) or set(row) != {"assess", "origin"} for row in accept):
            raise ValueError("Advance control record has malformed assessments")
        if not isinstance(retry, dict) or set(retry) != {"max_attempts", "max_transforms"}:
            raise ValueError("Advance control record has a malformed retry budget")
        if not isinstance(proposal, dict) or set(proposal) != {"dt"} \
                or type(proposal["dt"]) is not str:
            raise ValueError("Advance control record has a malformed proposal")
        control = cls(assessments=tuple((row["assess"], row["origin"]) for row in accept),
                      max_attempts=retry["max_attempts"], max_transforms=retry["max_transforms"],
                      first_dt=float.fromhex(proposal["dt"]))
        if control.to_data() != data:
            raise ValueError("Advance control record is not canonical")
        return control


__all__ = ["ADVANCE_ASSESSMENT_CONTRACT", "ADVANCE_CONTROL_SCHEMA", "AdvanceControl"]
