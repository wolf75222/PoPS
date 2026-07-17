"""Typed, domain-separated identities over canonical PoPS bytes."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .encoding import canonical_bytes


_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ALGORITHM = "sha256"
_PROTOCOL = "pops.identity"
_TOKEN_RE = re.compile(
    r"^pops\.([a-z][a-z0-9_.-]*)\.v([1-9][0-9]*):sha256:([0-9a-f]{64})$")


def _domain(value: Any) -> str:
    if not isinstance(value, str) or _DOMAIN_RE.fullmatch(value) is None:
        raise ValueError(
            "identity domain must match [a-z][a-z0-9_.-]*, got %r" % (value,)
        )
    return value


def _schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("identity schema_version must be an integer >= 1")
    if value > (1 << 63) - 1:
        raise OverflowError("identity schema_version is outside signed int64")
    return value


@dataclass(frozen=True, slots=True)
class Identity:
    """Immutable reference to one versioned, domain-separated identity."""

    domain: str
    schema_version: int
    algorithm: str
    digest: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _domain(self.domain))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.algorithm != _ALGORITHM:
            raise ValueError("identity algorithm must be exactly 'sha256'")
        if not isinstance(self.digest, bytes) or len(self.digest) != 32:
            raise ValueError("identity digest must be exactly 32 bytes")

    @property
    def hexdigest(self) -> str:
        return self.digest.hex()

    @property
    def token(self) -> str:
        return "pops.%s.v%d:%s:%s" % (
            self.domain, self.schema_version, self.algorithm, self.hexdigest)

    def to_data(self) -> dict[str, Any]:
        """Return the strict CBOR-ready identity reference."""
        return {
            "domain": self.domain,
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "digest": self.digest,
        }

    @classmethod
    def from_data(cls, data: Any) -> Identity:
        """Decode the exact current identity-reference schema; no legacy shape is accepted."""
        required = {"domain", "schema_version", "algorithm", "digest"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("Identity data must contain exactly %s" % sorted(required))
        return cls(
            domain=data["domain"],
            schema_version=data["schema_version"],
            algorithm=data["algorithm"],
            digest=data["digest"],
        )

    @classmethod
    def from_token(cls, token: Any) -> Identity:
        """Decode the exact printable identity schema; no aliases are accepted."""
        if not isinstance(token, str):
            raise TypeError("Identity token must be a string")
        match = _TOKEN_RE.fullmatch(token)
        if match is None:
            raise ValueError("invalid PoPS identity token")
        return cls(match.group(1), int(match.group(2)), _ALGORITHM,
                   bytes.fromhex(match.group(3)))

    def __str__(self) -> str:
        return self.token


def make_identity(domain: Any, payload: Any, *, schema_version: Any = 1) -> Identity:
    """Hash one payload inside the canonical versioned PoPS identity envelope."""
    checked_domain = _domain(domain)
    checked_version = _schema_version(schema_version)
    envelope = {
        "protocol": _PROTOCOL,
        "domain": checked_domain,
        "schema_version": checked_version,
        "payload": payload,
    }
    digest = hashlib.sha256(canonical_bytes(envelope)).digest()
    return Identity(checked_domain, checked_version, _ALGORITHM, digest)


__all__ = ["Identity", "make_identity"]
