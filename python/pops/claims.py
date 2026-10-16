"""Scientific claims with explicit, non-interchangeable evidence categories.

A receipt authenticates a reported execution, not a mathematical theorem. A
structural graph identity proves only equality of the retained finite graphs.
Neither category silently promotes an analytical assumption to a proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pops.representations import _freeze_data, _data_to_wire, _data_identity


def _text(value, where):
    if type(value) is not str or not value or value.strip() != value:
        raise TypeError(where + " requires canonical nonempty text")
    return value


def _digest(data):
    return hashlib.sha256(bytes.fromhex(_data_identity(data))).hexdigest()


def _path(data, path):
    for key in path:
        if type(key) is str and type(data) is dict:
            data = data[key]
        elif type(key) is int and type(data) is list and 0 <= key < len(data):
            data = data[key]
        else:
            raise ValueError("claim receipt path does not identify an explicit value")
    return data


@dataclass(frozen=True, slots=True, init=False)
class Claim:
    """An immutable claim created only through an evidence-specific constructor."""

    statement: str
    category: str
    evidence: object
    __pops_ir_immutable__ = True

    @classmethod
    def _make(cls, statement, category, evidence):
        claim = object.__new__(cls)
        object.__setattr__(claim, "statement", _text(statement, "Claim.statement"))
        object.__setattr__(claim, "category", category)
        object.__setattr__(claim, "evidence", _freeze_data(evidence))
        return claim

    @classmethod
    def assumption(cls, statement, *, scope, justification):
        """Declare an analytical assumption, without marking it verified."""
        return cls._make(statement, "analytical_assumption", {
            "scope": _text(scope, "assumption scope"),
            "justification": _text(justification, "assumption justification"),
        })

    @classmethod
    def unestablished(cls, statement, *, reason):
        return cls._make(statement, "unestablished", {
            "reason": _text(reason, "unestablished reason"),
        })

    @classmethod
    def graph_identity(cls, statement, *, left, right):
        """Verify exact identity of two closed finite GraphFunction definitions.

        This deliberately does not use floating-point samples or algebraic
        rewrites such as reassociation, and proves nothing about a continuous PDE.
        """
        from pops.graph_functions import GraphFunction
        if type(left) is not GraphFunction or type(right) is not GraphFunction:
            raise TypeError("graph identity requires two closed GraphFunction objects")
        first, second = left.to_data(), right.to_data()
        if _data_identity(first) != _data_identity(second):
            raise ValueError("the retained finite graphs are not structurally identical")
        return cls._make(statement, "verified_graph_identity", {
            "verification": "exact-canonical-finite-graph-identity@1",
            "graph_sha256": _digest(first), "graph": first,
        })

    @classmethod
    def numerical_receipt(cls, statement, *, path, sha256, check_path, artifact_path):
        """Authenticate one successful check from an existing JSON run receipt.

        ``check_path`` selects a literal boolean success; ``artifact_path``
        selects the exact artifact identifier recorded by that run. File bytes
        are read now and verified against the supplied digest. This is reported
        numerical evidence with provenance, never universal proof or permission
        to substitute a different graph. The receipt's producer remains trusted.
        """
        if type(sha256) is not str or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("a numerical claim requires an exact lowercase SHA-256")
        receipt_path = Path(path).resolve()
        payload = receipt_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("numerical receipt bytes differ from the declared SHA-256")
        receipt = json.loads(payload)
        check_path, artifact_path = tuple(check_path), tuple(artifact_path)
        if not check_path or not artifact_path:
            raise ValueError("numerical receipt requires explicit nonempty evidence paths")
        if _path(receipt, check_path) is not True:
            raise ValueError("numerical receipt does not report the selected check as passed")
        artifact = _path(receipt, artifact_path)
        _text(artifact, "numerical artifact identity")
        return cls._make(statement, "reported_numerical_execution", {
            "receipt": str(receipt_path), "receipt_sha256": sha256,
            "check_path": check_path, "artifact_path": artifact_path,
            "artifact_identity": artifact, "passed": True,
            "scope": "selected check in authenticated receipt; producer provenance is trusted",
        })

    def to_data(self):
        return {"contract": "pops.scientific-claim@1", "statement": self.statement,
                "category": self.category, "evidence": _data_to_wire(self.evidence)}


def claims_manifest(claims):
    """Export obligations separately from evidence; preserve their exact nature."""
    claims = tuple(claims)
    if any(type(claim) is not Claim for claim in claims):
        raise TypeError("claims require Claim objects, never bare proof flags")
    categories = ("verified_graph_identity", "reported_numerical_execution",
                  "analytical_assumption", "unestablished")
    return {category: tuple(claim.to_data() for claim in claims if claim.category == category)
            for category in categories}


__all__ = ["Claim", "claims_manifest"]
