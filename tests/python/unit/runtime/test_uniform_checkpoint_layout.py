"""Check exact stored layouts for rectangular Uniform checkpoint field aliases."""
import numpy as np
import pytest

from pops.runtime._system_io import _DEFAULT_FIELD_SLOT, _validated_uniform_phi_alias


@pytest.mark.parametrize("logical_shape", [(6,), (2, 3), (2, 3, 4)])
def test_default_phi_requires_exact_native_ranked_storage(logical_shape):
    values = np.arange(np.prod(logical_shape), dtype=np.float64)
    values[1] = -0.0
    phi = values.reshape(logical_shape[::-1])
    payload = {"phi": phi.copy(), "field_potential_0": phi.copy()}
    before = {key: value.tobytes() for key, value in payload.items()}
    assert _validated_uniform_phi_alias(
        payload, spatial_shape=logical_shape, field_slots=[_DEFAULT_FIELD_SLOT]
    ) is True
    assert before == {key: value.tobytes() for key, value in payload.items()}
    if logical_shape != logical_shape[::-1]:
        malformed = {key: value.reshape(logical_shape) for key, value in payload.items()}
        with pytest.raises(ValueError, match="potential payload"):
            _validated_uniform_phi_alias(
                malformed, spatial_shape=logical_shape, field_slots=[_DEFAULT_FIELD_SLOT]
            )


@pytest.mark.parametrize("logical_shape", [(6,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("slots", [[], ["pops.test.named-field"]])
def test_field_free_legacy_logical_storage_stays_exact(logical_shape, slots):
    payload = {"phi": np.zeros(logical_shape, dtype=np.float64)}
    assert _validated_uniform_phi_alias(payload, spatial_shape=logical_shape, field_slots=slots) is False
    if logical_shape != logical_shape[::-1]:
        with pytest.raises(ValueError, match="potential payload"):
            _validated_uniform_phi_alias(
                {"phi": np.zeros(logical_shape[::-1], dtype=np.float64)},
                spatial_shape=logical_shape, field_slots=slots,
            )


@pytest.mark.parametrize("key", ["phi", "field_potential_0"])
@pytest.mark.parametrize("kind", ["logical_shape", "binary32", "fortran", "signed_zero_mismatch"])
def test_rectangular_default_retains_dtype_contiguity_and_bitwise_guards(key, kind):
    phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    phi[0, 1] = -0.0
    payload = {"phi": phi.copy(), "field_potential_0": phi.copy()}
    if kind == "logical_shape":
        payload[key] = payload[key].reshape(2, 3)
        error, match = (ValueError if key == "phi" else RuntimeError), "spatial shape"
    elif kind == "binary32":
        payload[key] = payload[key].astype(np.float32)
        error, match = (ValueError if key == "phi" else RuntimeError), "binary64"
    elif kind == "fortran":
        payload[key] = np.asfortranarray(payload[key])
        error, match = ValueError, "C-contiguous"
    else:
        payload[key][0, 1] = 0.0
        error, match = ValueError, "differs bitwise"
    before = {name: value.tobytes() for name, value in payload.items()}
    with pytest.raises(error, match=match):
        _validated_uniform_phi_alias(payload, spatial_shape=(2, 3), field_slots=[_DEFAULT_FIELD_SLOT])
    assert before == {name: value.tobytes() for name, value in payload.items()}


@pytest.mark.parametrize("kind", ["binary32", "fortran", "negative_zero", "nonzero", "nan"])
def test_rectangular_field_free_retains_exact_zero_representation(kind):
    phi = np.zeros((2, 3), dtype=np.float64)
    if kind == "binary32":
        phi = phi.astype(np.float32)
        match = "binary64"
    elif kind == "fortran":
        phi = np.asfortranarray(phi)
        match = "C-contiguous"
    else:
        phi[0, 1] = {"negative_zero": -0.0, "nonzero": 1., "nan": np.nan}[kind]
        match = r"canonical \+0.0"
    before = phi.tobytes()
    with pytest.raises(ValueError, match=match):
        _validated_uniform_phi_alias({"phi": phi}, spatial_shape=(2, 3), field_slots=[])
    assert phi.tobytes() == before


def test_rectangular_default_alias_cannot_borrow_named_field_bytes():
    phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    payload = {"phi": phi.copy(), "field_potential_0": phi.copy(), "field_potential_1": phi+1.}
    with pytest.raises(ValueError, match="differs bitwise"):
        _validated_uniform_phi_alias(payload, spatial_shape=(2, 3),
            field_slots=["pops.test.named-field", _DEFAULT_FIELD_SLOT])
