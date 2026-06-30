import pytest

from pops.codegen.module_emit_helpers import _roles_for


def test_codegen_roles_normalize_public_and_missing_names():
    assert _roles_for(
        ["rho", "mx", "my", "M20", "q", "custom_state"],
        ["density", "momentum_x", "MomentumY", None, "None", "non_native_user_role"],
    ) == [
        "Density",
        "MomentumX",
        "MomentumY",
        "Custom",
        "Custom",
        "Custom",
    ]


def test_codegen_roles_reject_mismatched_override_length():
    with pytest.raises(ValueError, match="roles: 1 roles for 2 variables"):
        _roles_for(["rho", "mx"], ["Density"])
