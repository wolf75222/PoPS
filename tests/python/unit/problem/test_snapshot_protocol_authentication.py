"""ProblemSnapshot special encodings accept authenticated PoPS value types only."""
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.ir.literals import ScalarLiteral as PopsScalarLiteral  # noqa: E402
from pops.model.handles import Handle, OwnerPath  # noqa: E402
from pops.problem._snapshot import ProblemSnapshot  # noqa: E402


def test_same_named_scalar_literal_cannot_collide_with_real_literal():
    class ScalarLiteral:
        def to_data(self):
            return {"kind": "integer", "value": "7"}

    real = ProblemSnapshot({"value": PopsScalarLiteral.from_value(7)})
    fake = ProblemSnapshot({"value": ScalarLiteral()})

    assert real.hash != fake.hash
    assert "$scalar" in real.to_dict()["value"]
    assert "$object" in fake.to_dict()["value"]


def test_duck_typed_handle_cannot_collide_with_authenticated_handle():
    real_handle = Handle("rho", kind="state", owner=OwnerPath("model", "fluid"))

    class FakeHandle:
        def __init__(self):
            self.qualified_id = real_handle.qualified_id
            self.schema_version = "1"
            self.kind = "state"
            self.owner_path = real_handle.owner_path
            self.local_id = "rho"

        def canonical_identity(self):
            return real_handle.canonical_identity()

    real = ProblemSnapshot({"handle": real_handle})
    fake = ProblemSnapshot({"handle": FakeHandle()})

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

    handle = CoercibleHandle("rho", kind="state", owner=OwnerPath("model", "fluid"))

    with pytest.raises(TypeError, match="schema_version"):
        ProblemSnapshot({"handle": handle})

