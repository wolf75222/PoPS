"""Typed and domain-separated PoPS identity tests."""
from __future__ import annotations

import hashlib

import pytest

from pops.identity import Identity, canonical_bytes, make_identity


def test_make_identity_hashes_the_exact_versioned_envelope():
    payload = {"components": ["rho", "rho_u"], "roles": {"density": "rho"}}
    identity = make_identity("semantic", payload, schema_version=3)
    envelope = {
        "protocol": "pops.identity",
        "domain": "semantic",
        "schema_version": 3,
        "payload": payload,
    }
    assert identity.digest == hashlib.sha256(canonical_bytes(envelope)).digest()
    assert identity.algorithm == "sha256"
    assert identity.hexdigest == identity.digest.hex()
    assert identity.token == "pops.semantic.v3:sha256:%s" % identity.hexdigest
    assert str(identity) == identity.token


def test_domain_and_schema_versions_are_separate_identity_namespaces():
    payload = {"same": "payload"}
    semantic = make_identity("semantic", payload)
    artifact = make_identity("artifact", payload)
    semantic_v2 = make_identity("semantic", payload, schema_version=2)
    assert len({semantic.digest, artifact.digest, semantic_v2.digest}) == 3


def test_payload_map_insertion_order_is_irrelevant():
    left = make_identity("semantic", {"a": 1, "bb": 2})
    right = make_identity("semantic", {"bb": 2, "a": 1})
    assert left == right


def test_identity_data_round_trips_without_hex_or_string_fallbacks():
    identity = make_identity("restart.state", {"clock": "0x1.0p+0"}, schema_version=4)
    data = identity.to_data()
    assert data == {
        "domain": "restart.state",
        "schema_version": 4,
        "algorithm": "sha256",
        "digest": identity.digest,
    }
    assert Identity.from_data(data) == identity
    assert canonical_bytes(Identity.from_data(data).to_data()) == canonical_bytes(data)


def test_identity_is_immutable():
    identity = make_identity("run", {"steps": 10})
    with pytest.raises((AttributeError, TypeError)):
        identity.domain = "artifact"


@pytest.mark.parametrize("domain", ["", "Semantic", "two words", ".leading", "x/y"])
def test_invalid_domains_are_rejected(domain):
    with pytest.raises(ValueError, match="identity domain"):
        make_identity(domain, {})


@pytest.mark.parametrize("version", [True, False, 0, -1, 1.0, "1"])
def test_invalid_schema_versions_are_rejected(version):
    with pytest.raises(ValueError, match="schema_version"):
        make_identity("semantic", {}, schema_version=version)


def test_identity_rejects_non_sha256_and_non_binary_digest():
    digest = b"\x00" * 32
    with pytest.raises(ValueError, match="algorithm"):
        Identity("semantic", 1, "sha512", digest)
    with pytest.raises(ValueError, match="32 bytes"):
        Identity("semantic", 1, "sha256", b"short")
    with pytest.raises(ValueError, match="32 bytes"):
        Identity("semantic", 1, "sha256", digest.hex())


def test_from_data_requires_the_exact_current_shape():
    identity = make_identity("semantic", {})
    data = identity.to_data()
    with pytest.raises(TypeError, match="exactly"):
        Identity.from_data({**data, "token": identity.token})
    with pytest.raises(TypeError, match="exactly"):
        Identity.from_data({key: value for key, value in data.items() if key != "digest"})
    with pytest.raises(TypeError, match="exactly"):
        Identity.from_data(identity.token)
