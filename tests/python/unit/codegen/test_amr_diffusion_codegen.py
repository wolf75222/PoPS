"""ADC-942: AMR diffusion lowers through stage ghosts and the sole reflux authority."""

from types import SimpleNamespace

import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges
from pops.mesh._amr.transfer import AMRTransfer
from pops.model import Handle, OwnerPath
from pops.numerics import Diffusion
from tests.python.integration.runtime.test_amr_transport_diffusion_qualification import (
    _author as _author_amr_diffusion,
    _stable_dt,
)
from tests.python.integration.runtime.test_public_diffusion_matrix import _build


def _combined_source():
    case, _, _ = _build("constant", 8, 1.0e-4, transport=(0.2, -0.1))
    model = dict(case._block_registry.items())["heat"]["model"]
    lowered, _ = lower_and_validate(model)
    return emit_cpp_program(list(case._time_registry)[0], model=lowered, target="amr_system")


def test_explicit_amr_diffusion_prepares_stage_and_registers_each_face_basis_once():
    source = _combined_source()
    prepared = source.index("ctx.prepare_generated_state(0,")
    applied = source.index(".apply(", prepared)
    diffusive_basis = source.index("ctx.attach_diffusive_flux_basis(0,", applied)
    transport_basis = source.index("ctx.neg_div_flux_default_with_faces_into(0,", applied)
    assert prepared < applied < diffusive_basis < transport_basis
    assert source.count("ctx.attach_diffusive_flux_basis(") == 1
    assert source.count("ctx.neg_div_flux_default_with_faces_into(") == 1
    assert "combined_transport_diffusion_stability" in source


def test_amr_accepted_exchange_inventory_uses_composite_active_cells():
    source = _combined_source()
    assert source.count("ctx.pointwise_active_mask(0,") == 1
    assert source.count(".stage_accepted_exchanges(ctx,0,") == 1
    assert "prepared_amr_ghosts" not in source


def test_repeated_same_face_emissions_have_independent_cpp_scopes():
    emitted = emit_transport_exchanges(
        "retained_faces",
        "operation",
        "occurrence",
        "evaluation",
        "dt",
        program_block=0,
        active_field="prototype",
    )
    assert emitted[0] == "{" and emitted[-1] == "}"
    repeated = "\n".join((*emitted, *emitted))
    assert repeated.count("* retained_faces_active = nullptr") == 2
    assert repeated.count("\n}\n{") == 1


@pytest.mark.parametrize("physical", (False, True))
def test_real_amr_resolution_recognizes_exact_combined_and_pure_diffusion(physical):
    assert (
        type(_author_amr_diffusion(16, _stable_dt(16), physical=physical)).__name__
        == "ResolvedSimulationPlan"
    )


def test_amr_diffusion_accuracy_rejects_lookalike_and_mismatched_transport_state():
    owner = OwnerPath.case("amr-diffusion-accuracy")
    state = Handle("u", kind="state", owner=owner)
    other = Handle("other", kind="state", owner=owner)
    lookalike = SimpleNamespace(
        category="diffusion", law=SimpleNamespace(state=state), formal_order=2, ghost_depth=1
    )
    plan = SimpleNamespace(rates=(SimpleNamespace(method=lookalike),))
    assert AMRTransfer._resolved_spatial_accuracy(state, (plan,), 2) is None

    mismatched = object.__new__(Diffusion)
    mismatched.law = SimpleNamespace(state=state)
    mismatched.transport = SimpleNamespace(variables=SimpleNamespace(options={"state": other}))
    plan = SimpleNamespace(rates=(SimpleNamespace(method=mismatched),))
    with pytest.raises(ValueError, match="exact constitutive state"):
        AMRTransfer._resolved_spatial_accuracy(state, (plan,), 2)
