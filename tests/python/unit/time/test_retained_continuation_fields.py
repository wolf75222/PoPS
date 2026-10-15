"""Explicit persistent Fields are data; only declared State paths are projections."""

import pytest

from pops.continuations import ContinuationSchema
from pops.time._program.versioned_values import validate_schema_data
from pops.time.variant_types import Product, Variant, type_of
from examples.api_v02.methods_T01_T12 import _storage


def fixture(*, container="array"):
    parts = _storage("retained_field_schema", ("u",), (1.0,))
    p, q = parts[5:7]
    current = q.n
    if container == "array":
        domain = p.range_set(2, capacity=2, ownership="replicated")
        history = p.map(domain, lambda _p, _i: current)
        path = ("history", "[]")
    elif container == "variant":
        kind = Variant(empty=Product(), saved=type_of(current))
        history = p.variant(kind, "saved", current)
        path = ("history", "saved")
    else:
        history, path = current, ("history",)
    initial = p.product(state=current, history=history, time=p.float64(0), step=p.int64(0))
    kwargs = dict(fields=((q.next, ("state",)),), time=("time",), step=("step",))
    return p, q, initial, path, kwargs


@pytest.mark.parametrize("container", ["field", "array", "variant"])
def test_retained_snapshots_are_explicit_and_do_not_project_or_relabel(container):
    _, q, initial, path, kwargs = fixture(container=container)
    schema = ContinuationSchema(initial, retained_fields=(path,), **kwargs)
    assert schema.fields == ((q.next, ("state",)),)
    assert schema.retained_fields == (path,)
    before = dict(schema.read_type.signature["members"])
    after = dict(schema.write_type.signature["members"])
    assert after["state"]["point"] == q.next.point
    assert before["history"] == after["history"]
    validate_schema_data(schema.to_data())


def test_implicit_unbound_history_still_rejected():
    _, _, initial, _, kwargs = fixture()
    with pytest.raises(NotImplementedError, match="explicit retained_fields"):
        ContinuationSchema(initial, **kwargs)


@pytest.mark.parametrize("paths", [(("state",),), (("history", "[]"), ("history", "[]")),
                                   (("missing",),), (("history",),)])
def test_retention_is_an_exact_nonoverlapping_field_path_cover(paths):
    _, _, initial, _, kwargs = fixture()
    with pytest.raises(ValueError, match="retained fields|cover every persistent Field"):
        ContinuationSchema(initial, retained_fields=paths, **kwargs)


def test_retention_declaration_is_part_of_compiled_identity():
    _, _, initial, path, kwargs = fixture()
    schema = ContinuationSchema(initial, retained_fields=(path,), **kwargs)
    forged = dict(schema.to_data(), retained_fields=())
    with pytest.raises(ValueError, match="compiled identity"):
        validate_schema_data(forged)


def test_empty_retention_preserves_legacy_schema_identity():
    parts = _storage("legacy_field_schema", ("u",), (1.0,))
    p, q = parts[5:7]
    initial = p.product(state=q.n, time=p.float64(0), step=p.int64(0))
    kwargs = dict(fields=((q.next, ("state",)),), time=("time",), step=("step",))
    implicit = ContinuationSchema(initial, **kwargs)
    explicit = ContinuationSchema(initial, retained_fields=(), **kwargs)
    assert implicit.identity == explicit.identity
    assert "retained_fields" not in implicit.to_data()
