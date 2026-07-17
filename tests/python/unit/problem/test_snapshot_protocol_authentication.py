"""AuthoringSnapshot special encodings accept authenticated PoPS value types only."""
from enum import Enum

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops._ir.literals import ScalarLiteral as PopsScalarLiteral  # noqa: E402
from pops.model import DeclarationIndex, OwnerKind  # noqa: E402
from pops.model.handles import Handle, OwnerPath  # noqa: E402
from pops.problem._snapshot import AuthoringSnapshot  # noqa: E402


def test_enum_members_have_a_qualified_cycle_free_snapshot_encoding():
    class Equality(Enum):
        HOLD = "hold"

    class Conflict(Enum):
        HOLD = "hold"

    equality = AuthoringSnapshot({"policy": Equality.HOLD})
    conflict = AuthoringSnapshot({"policy": Conflict.HOLD})

    data = equality.to_dict()["policy"]["$enum"]
    assert data["type"].endswith(".Equality")
    assert data["member"] == "HOLD"
    assert data["value"] == "hold"
    assert equality.hash != conflict.hash


def test_same_named_scalar_literal_cannot_collide_with_real_literal():
    class ScalarLiteral:
        def to_data(self):
            return {"kind": "integer", "value": "7"}

    real = AuthoringSnapshot({"value": PopsScalarLiteral.from_value(7)})
    fake = AuthoringSnapshot({"value": ScalarLiteral()})

    assert real.hash != fake.hash
    assert "$scalar" in real.to_dict()["value"]
    assert "$object" in fake.to_dict()["value"]


def test_duck_typed_handle_cannot_collide_with_authenticated_handle():
    real_handle = Handle("rho", kind="state", owner=OwnerPath.model("fluid"))

    class FakeHandle:
        def __init__(self):
            self.qualified_id = real_handle.qualified_id
            self.schema_version = "1"
            self.kind = "state"
            self.owner_path = real_handle.owner_path
            self.local_id = "rho"

        def canonical_identity(self):
            return real_handle.canonical_identity()

    real = AuthoringSnapshot({"handle": real_handle})
    fake = AuthoringSnapshot({"handle": FakeHandle()})

    assert real.hash != fake.hash
    assert "$handle" in real.to_dict()["handle"]
    assert "$object" in fake.to_dict()["handle"]


def test_authenticated_handle_canonical_identity_is_strictly_validated():
    class CoercibleHandle(Handle):
        def canonical_identity(self):
            identity = super().canonical_identity()
            identity["schema_version"] = "1"
            identity["kind"] = 42
            return identity

    handle = CoercibleHandle("rho", kind="state", owner=OwnerPath.model("fluid"))

    with pytest.raises(TypeError, match="schema_version"):
        AuthoringSnapshot({"handle": handle})


def test_authoring_handle_requires_an_authoritative_snapshot_resolver():
    owner = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "fluid")
    handle = Handle("rho", kind="state", owner=owner)

    with pytest.raises(TypeError, match="authoritative resolver"):
        AuthoringSnapshot({"handle": handle})

    index = DeclarationIndex(owner=owner, handles=(handle,))
    resolved = AuthoringSnapshot(
        {"handle": handle},
        handle_resolver=lambda value: index.authenticate(value)._resolved(),
    )
    identity = resolved.to_dict()["handle"]["$handle"]
    assert Handle.from_canonical_identity(identity).canonical_identity() == identity


def test_snapshot_resolver_must_reauthenticate_canonical_identity_without_rewriting_it():
    authored = Handle("rho", kind="state", owner=OwnerPath.model("fluid"))
    rewritten = Handle("rho", kind="state", owner=OwnerPath.model("other"))

    with pytest.raises(ValueError, match="changed an already canonical identity"):
        AuthoringSnapshot({"handle": authored}, handle_resolver=lambda value: rewritten)
