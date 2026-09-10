"""ADC-942: AMR diffusion lowers through stage ghosts and the sole reflux authority."""

import re
from fractions import Fraction
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


@pytest.mark.parametrize("physical,expected", [(True, 1), (False, 2)])
def test_diffusion_flux_budget_counts_exact_native_face_providers(physical, expected):
    from pops.codegen.program_emit_amr import _flux_expression_budgets

    resolved = _author_amr_diffusion(16, _stable_dt(16), physical=physical)
    assert _flux_expression_budgets(resolved.time) == ((expected, 1),)
    assert "pops_program_flux_rhs_basis_bound" in _combined_source()


def test_ssprk2_combined_diffusion_budgets_both_face_providers_at_both_stages():
    from pops.codegen.program_emit_amr import _flux_expression_budgets
    from tests.python.unit.codegen.test_diffusion_program import resolved_heat

    resolved, _, _ = resolved_heat(method="ssprk2", transport=(0.2, -0.1))
    assert _flux_expression_budgets(resolved.time) == ((4, 1),)


def test_ssprk2_flux_families_are_operation_typed_and_installed_before_restore_hooks():
    from tests.python.unit.codegen.test_diffusion_program import resolved_heat

    resolved, _, model = resolved_heat(method="ssprk2", transport=(0.2, -0.1))
    source = emit_cpp_program(
        resolved.time, model=lower_and_validate(model)[0], target="amr_system"
    )
    declaration = next(
        line for line in source.splitlines() if "ctx.install_flux_temporal_families(" in line
    )
    rows = re.findall(r'\{0, (\d+), ([14]), "([^"]+)"\}', declaration)
    assert len(rows) == 4
    by_provider = {
        provider: {family for _, candidate, family in rows if candidate == provider}
        for provider in ("1", "4")
    }
    assert all(len(families) == 1 for families in by_provider.values())
    assert by_provider["1"] != by_provider["4"]
    assert source.index(declaration) < source.index("auto _make_level_program")
    assert source.index(declaration) < source.index("ctx.install([=](double dt)")
    for family in by_provider["1"]:
        assert source.count('neg_div_flux_default_with_faces_into(0,u') == 2
        assert source.count(f',"{family}");') >= 2
    for family in by_provider["4"]:
        assert source.count("ctx.attach_diffusive_flux_basis(") == 2
        assert source.count(f',"{family}");') >= 2


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
    # The producer can visit different active face counts on different execution ranks.
    source = "\n".join(emitted)
    assert source.count("ctx.stage_exchange_batch(") == 1
    assert source.index("ctx.stage_exchange_batch(") < source.index("for (std::size_t retained_faces_local")
    assert "ctx.stage_exchange(" not in source
    assert "stage_exchange(pops::runtime::program::ExchangeRecord{" in source


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


@pytest.mark.parametrize("n", [16, 32, 64])
def test_actual_subcycled_diffusion_rhs_keeps_spatial_sum_at_dt_power_zero(n):
    resolved = _author_amr_diffusion(n, _stable_dt(n))
    source = emit_cpp_program(
        resolved.time, model=lower_and_validate(resolved.blocks[0].model)[0], target="amr_system"
    )
    additions = re.findall(r"ctx\.axpy\(diffusive_rhs_\d+,1,diffusive_transport_\d+([^;]*);", source)
    assert additions == [",dt,{{0, 1, 1}})"]
    # The temporal update still owns exactly one dt power after spatial assembly.
    assert re.search(r"ctx\.axpy\([^;]*diffusive_rhs_\d+, dt, \{\{1, 1, 1\}\}\);", source)
    assert "ctx.advance_hierarchy(dt" in source


def test_ssprk2_diffusion_keeps_both_spatial_sums_constant_and_rational_time_weights_exact():
    from tests.python.unit.codegen.test_diffusion_program import resolved_heat

    resolved, _, model = resolved_heat(method="ssprk2", transport=(0.2, -0.1))
    source = emit_cpp_program(
        resolved.time, model=lower_and_validate(model)[0], target="amr_system"
    )
    additions = re.findall(r"ctx\.axpy\(diffusive_rhs_\d+,1,diffusive_transport_\d+([^;]*);", source)
    assert additions == [",dt,{{0, 1, 1}})"] * 2
    assert "dt, {{1, 1, 2}})" in source
    assert "dt, {{1, 1, 1}})" in source


@pytest.mark.parametrize("weight", [Fraction(-2, 3), Fraction(-1, 10**30)])
def test_source_only_sum_has_no_spurious_native_flux_coefficient_cap(weight):
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.identity.scalar import scalar_cpp
    from pops.lib.time import ForwardEuler
    from pops.math import ddt, div, grad
    from pops.numerics import DiscretizationPlan
    from pops.time import FixedDt

    frame = Rectangle("signed-source", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("signed-source", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux("diffusion", state=state, value=0.1 * grad(state[0]))
    source = model.source("reaction", on=state, value=(state[0],))
    rate = model.rate("balance", equation=ddt(state) == div(flux) + weight * source)
    case = pops.Case("signed-source")
    block = case.block("heat", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(_stable_dt(32)))
    case.program(program)
    emitted = emit_cpp_program(program, model=lower_and_validate(model)[0], target="amr_system")
    source_additions = [
        line for line in emitted.splitlines()
        if "ctx.axpy(diffusive_rhs_" in line and "diffusive_source_" in line
    ]
    assert len(source_additions) == 1
    # Source kernels carry no face basis. Keep their full scalar range, including
    # coefficients outside the int64 rational envelope required by conservative fluxes.
    assert ",%s,diffusive_source_" % scalar_cpp(weight) in source_additions[0]
    assert ",dt," not in source_additions[0]
