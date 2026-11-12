"""The compiled control of an elaborated Advance (spec 0.3 sections 5.2 and 11.1, T28, T38).

An ``Advance`` elaborates into a Program whose candidate passes through its assessments before
the commit, and into an ``AdvanceControl``: the record of those assessments, of the retry budget
and of the first proposal. The record enters the Program identity, the generated artifact exports
its attempt budget, and the artifact manifest publishes it, the default acceptance included with
its origin (SU9). ``System::run_advance`` runs the attempt loop of the record in C++. The record
also lists the Outputs of the Advance (CC14): the schedule of each, ``when`` and ``at``, and the
shape of its record, which arrays and which scalars it publishes.
"""
from __future__ import annotations

import json
import math
from typing import Any

ADVANCE_CONTROL_SCHEMA = "pops.advance-control@1"
"""Schema of ``AdvanceControl.to_data`` for an Advance without Outputs."""

ADVANCE_CONTROL_OUTPUTS_SCHEMA = "pops.advance-control@2"
"""Schema of ``AdvanceControl.to_data`` for an Advance with Outputs (CC14): schema 1 and the
schedule and record shape of each Output."""

ADVANCE_CONTROL_MEMORY_SCHEMA = "pops.advance-control@3"
"""Schema of ``AdvanceControl.to_data`` for an Advance whose method has memory or field requests
(N8 lot 5): a superset of schema 2, whose keys ``outputs``, ``memory`` and ``trial_failures`` are
always present. ``memory`` lists each member (T27, CN1): its path, role, type identity, writer,
readers and ring; ``trial_failures`` lists each failure site of a field request and whether a
retry can change it (T38), the sites that no lot classifies yet with the lot that will, and each
admissibility guard of the method body (IR4, N8 lot 6), whose variant is ``NotAdmissible``."""

ADVANCE_ASSESSMENT_CONTRACT = "pops.advance-assessment@1"
"""Attribute of an acceptance guard that is an assessment of an Advance: its rejection is raised
in the phase that the compiled run repeats (T28), every other rejection is a trial failure."""

_MAX_ATTEMPTS = 2**31 - 1


class AdvanceControl:
    """The assessments, the retry budget and the first proposal of one elaborated Advance."""

    __slots__ = ("assessments", "max_attempts", "max_transforms", "first_dt", "outputs",
                 "memory", "trial_failures")

    def __init__(self, *, assessments: tuple, max_attempts: int, max_transforms: int,
                 first_dt: float, outputs: tuple = (), memory: tuple = (),
                 trial_failures: tuple = ()) -> None:
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
        object.__setattr__(self, "outputs", _outputs(outputs))
        object.__setattr__(self, "memory", _rows(memory, _MEMORY_KEYS, "memory"))
        object.__setattr__(self, "trial_failures",
                           _rows(trial_failures, _TRIAL_KEYS, "trial failure"))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AdvanceControl is immutable")

    def __eq__(self, other: object) -> bool:
        return type(other) is AdvanceControl and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash((self.assessments, self.max_attempts, self.max_transforms, self.first_dt,
                     len(self.outputs), len(self.memory), len(self.trial_failures)))

    def __repr__(self) -> str:
        return "AdvanceControl(%r)" % (self.to_data(),)

    @property
    def schema(self) -> str:
        if self.memory or self.trial_failures:
            return ADVANCE_CONTROL_MEMORY_SCHEMA
        return ADVANCE_CONTROL_OUTPUTS_SCHEMA if self.outputs else ADVANCE_CONTROL_SCHEMA

    def to_data(self) -> dict[str, Any]:
        schema = self.schema
        data = {
            "schema": schema,
            "accept": [{"assess": name, "origin": origin} for name, origin in self.assessments],
            "retry": {"max_attempts": self.max_attempts, "max_transforms": self.max_transforms},
            "proposal": {"dt": self.first_dt.hex()},
        }
        if self.outputs or schema == ADVANCE_CONTROL_MEMORY_SCHEMA:
            data["outputs"] = _thaw(self.outputs)
        if schema == ADVANCE_CONTROL_MEMORY_SCHEMA:
            data["memory"] = _thaw(self.memory)
            data["trial_failures"] = _thaw(self.trial_failures)
        return data

    @classmethod
    def from_data(cls, data: Any) -> AdvanceControl:
        keys = {"schema", "accept", "retry", "proposal"}
        if not isinstance(data, dict) or not (
                data.get("schema") == ADVANCE_CONTROL_SCHEMA and set(data) == keys
                or data.get("schema") == ADVANCE_CONTROL_OUTPUTS_SCHEMA
                and set(data) == keys | {"outputs"}
                or data.get("schema") == ADVANCE_CONTROL_MEMORY_SCHEMA
                and set(data) == keys | {"outputs", "memory", "trial_failures"}):
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
                      first_dt=float.fromhex(proposal["dt"]),
                      outputs=tuple(data.get("outputs", ())),
                      memory=tuple(data.get("memory", ())),
                      trial_failures=tuple(data.get("trial_failures", ())))
        if control.to_data() != data:
            raise ValueError("Advance control record is not canonical")
        return control


_MEMORY_KEYS = frozenset(("path", "role", "type", "writer", "readers", "ring"))
_TRIAL_KEYS = frozenset(("site", "guard", "variant", "retry_can_change", "classified", "lot"))


def _rows(rows: Any, keys: frozenset, what: str) -> tuple:
    """Validate the %s rows of schema 3 and freeze them."""
    if type(rows) is not tuple:
        raise TypeError("AdvanceControl %s rows are a tuple" % what)
    for row in rows:
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("an AdvanceControl %s row holds %s" % (what, ", ".join(sorted(keys))))
    return tuple(_freeze(row) for row in rows)


def _freeze(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def memory_rings(control: Any) -> dict:
    """``{member: ring}`` of the method memory of ``control`` (T27), in declaration order."""
    return {row["path"].rsplit("/", 1)[1]: row["ring"] for row in _thaw(control.memory)}


def _thaw(rows: tuple) -> list:
    return [json.loads(row) for row in rows]


def _hex_time(text: Any, where: str) -> float:
    if type(text) is not str:
        raise ValueError("%s is canonical float.hex() text" % where)
    value = float.fromhex(text)
    if not math.isfinite(value) or value.hex() != text:
        raise ValueError("%s is canonical finite float.hex() text" % where)
    return value


def _outputs(outputs: Any) -> tuple:
    """Validate the schedule rows of the Outputs (``OutputRegion.schedule_data``) and freeze them."""
    if type(outputs) is not tuple:
        raise TypeError("AdvanceControl.outputs is a tuple of Output schedule rows")
    for row in outputs:
        if not isinstance(row, dict) or set(row) != {"when", "at", "record"}:
            raise ValueError("an Output schedule row holds when, at and record")
        when, at, record = row["when"], row["at"], row["record"]
        if when is not None and (not isinstance(when, dict) or set(when) != {"every"} or
                                 type(when["every"]) is not int or when["every"] < 1):
            raise ValueError("an Output schedule row has a malformed when")
        if at is not None:
            if isinstance(at, dict) and set(at) == {"every_dt"}:
                if not _hex_time(at["every_dt"], "Output every_dt") > 0.0:
                    raise ValueError("an Output every_dt is a positive duration")
            elif isinstance(at, dict) and set(at) == {"times"} and type(at["times"]) is list \
                    and at["times"]:
                times = [_hex_time(text, "Output time") for text in at["times"]]
                if any(not left < right for left, right in zip(times, times[1:])):
                    raise ValueError("Output times are strictly increasing")
            else:
                raise ValueError("an Output schedule row has a malformed at")
        if type(record) is not list or not record:
            raise ValueError("an Output schedule row has a non-empty record")
        scalars, names = 0, set()
        for item in record:
            if not isinstance(item, dict) or len(item) != 2 or type(item.get("name")) is not str \
                    or item["name"] in names:
                raise ValueError("an Output record row has a unique name and one value")
            names.add(item["name"])
            if "array" in item:
                if type(item["array"]) is not str or not item["array"]:
                    raise ValueError("an Output record array names its block")
            elif item.get("scalar") != scalars:
                raise ValueError("an Output record numbers its scalars in order")
            else:
                scalars += 1
    return tuple(_freeze(row) for row in outputs)


__all__ = ["ADVANCE_ASSESSMENT_CONTRACT", "ADVANCE_CONTROL_MEMORY_SCHEMA",
           "ADVANCE_CONTROL_OUTPUTS_SCHEMA",
           "ADVANCE_CONTROL_SCHEMA", "AdvanceControl", "memory_rings"]
