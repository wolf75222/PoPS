"""Source-only preparation of the future genuine MPI2 fixture."""
import io
import numpy as np
import pytest
from tests.python.integration.runtime.test_mapped_field_binding_collective_runtime import mutate_dto, exact_payload

@pytest.mark.parametrize("kind", ("bool", "overflow", "execution"))
def test_rank_one_argument_mutation_preserves_peer_and_original(kind):
    spec = {"mapped_field_components": 1}
    context = {"context_version": 1}
    peer = mutate_dto(spec, context, kind, 0)
    bad = mutate_dto(spec, context, kind, 1)
    assert peer == (spec, context)
    assert spec == {"mapped_field_components": 1} and context == {"context_version": 1}
    if kind != "bool": assert bad != peer
    if kind == "bool": assert type(bad[0]["mapped_field_components"]) is bool
    elif kind == "overflow": assert bad[0]["mapped_field_components"] == 2**63
    else: assert "context_version" not in bad[1]


def child(values):
    stream = io.BytesIO()
    np.savez_compressed(stream, state_carriers_checkpoint=np.asarray(values, dtype=np.uint8))
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def test_composite_comparison_decodes_child_and_refuses_grown_byte_change():
    before = {"layout_checkpoint_0": child([1, 0, 128]), "t": np.asarray(0.)}
    after = {"layout_checkpoint_0": child([1, 0, 128]), "t": np.asarray(0.)}
    exact_payload(before, after)
    after["layout_checkpoint_0"] = child([1, 0, 0])
    with pytest.raises(AssertionError, match="state_carriers_checkpoint"):
        exact_payload(before, after)


def test_valid_state_comparison_refuses_signed_zero_change():
    with pytest.raises(AssertionError, match="u"):
        exact_payload({"u": np.asarray([-0.])}, {"u": np.asarray([0.])})
