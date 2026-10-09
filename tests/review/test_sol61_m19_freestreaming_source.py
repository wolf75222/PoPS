"""Public source admission and independent equations, no native execution."""
import math

import numpy as np
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.module_codegen import emit_cpp_brick
from pops.codegen.module_lowering import lower_and_validate
from pops.model.provider_pack import build_operator_provider_pack, MissingInputProvider
from tests.python.support.m19_freestreaming import build, CASES, MODEL, MODULE, FRAME, RATE
from tests.review.sol61_m19_freestreaming_oracle import initial, discrete_fourier, receive, exact_cell_average


@pytest.mark.parametrize("nx,nv", CASES)
def test_actual_public_case_resolves_and_emits_signed_kinetic_transport(nx, nv):
    case, layout, quadrature, dt = build(nx, nv)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert tuple(layout.mesh.cells) == (nv, nx)
    assert len(case.initials) == 1
    assert quadrature.axis == 0 and float(dt) == 1 / (4 * nx)
    assert source.count("ctx.neg_div_flux_default_into(") == 2
    assert source.count("ctx.prepare_provider_values(") == 2
    assert "ctx.set_stage_time(0, 1);" in source and "ctx.set_stage_time(1, 1);" in source
    assert source.count('"aux", "velocity_coordinate", "velocity_coordinate"') == 2
    carrier, _ = lower_and_validate(MODEL)
    routes = _emit_auxiliary_route_registration(carrier._m, target="system")
    assert "geometry.cell_coordinate(0, index[0])" in routes
    assert "AuxiliaryFreshness::evaluation" in routes
    brick = emit_cpp_brick(carrier._m)
    assert "provider_value<0>" in brick
    assert "F[0] = (velocity_coordinate * population);" in brick
    assert "smin = velocity_coordinate; smax = velocity_coordinate;" in brick
    assert "F[0] = (pops::Real(0) * population);" in brick
    assert not any(name.startswith("pops._native.dim") for name in __import__("sys").modules)


def test_empty_fields_token_has_no_storage_but_physical_flux_has_exact_auxiliary():
    rate = MODULE.operator_registry().get(RATE.registered_operator_name)
    assert len(rate.signature.inputs[-1].components) == 0
    assert len(build_operator_provider_pack(MODULE, rate)) == 0
    flux = MODULE.operator_registry().get("flux_default")
    pack = build_operator_provider_pack(MODULE, flux)
    assert len(pack) == 1
    assert next(iter(pack)).space_kind == "aux"


@pytest.mark.parametrize("attack", ("missing", "nonempty", "frame", "layout"))
def test_empty_token_fix_does_not_admit_a_missing_or_changed_field_contract(attack):
    rate = MODULE.operator_registry().get(RATE.registered_operator_name)
    field = rate.signature.inputs[-1]
    data = field.to_data()
    if attack == "missing":
        data["name"] = "foreign_missing"
    elif attack == "nonempty":
        data["components"] = ["missing_component"]
        data["units"] = [None]
    elif attack == "frame":
        data["frame"] = "foreign-frame"
    else:
        data["layout"] = "foreign-layout"
    from pops.model import FieldSpace
    data.pop("kind")
    data.pop("value_shape")
    forged = FieldSpace(**data)
    from pops.model.operators import Operator
    rate = Operator(rate.name, rate.kind,
                    type(rate.signature)(inputs=(*rate.signature.inputs[:-1], forged),
                                         output=rate.signature.output),
                    requirements=rate.requirements)
    with pytest.raises(MissingInputProvider):
        build_operator_provider_pack(MODULE, rate)


@pytest.mark.parametrize("nx,nv", CASES)
def test_independent_discrete_equation_and_continuum_guards_are_nontrivial(nx, nv):
    dt = 1 / (4 * nx)
    seed = initial(nx, nv)
    for steps in (nx // 2, nx):
        state = discrete_fourier(nx, nv, dt, steps)
        report = receive(state, seed, nx, nv, dt, steps)
        assert report["discrete_error"] == 0
        assert report["continuum_l1"] > 1e-4
        assert abs(report["moments"][0] - 2) < 2e-14
        for bad in (seed, discrete_fourier(nx, nv, dt, steps, sign=-1),
                    discrete_fourier(nx, nv, dt, steps, temporal_order=1)):
            with pytest.raises(ValueError, match="original signed"):
                receive(bad, seed, nx, nv, dt, steps)


def test_continuum_original_shift_integral_matches_independent_gauss_integration():
    nx, nv, time = 16, 8, .125
    nodes, weights = np.polynomial.legendre.leggauss(32)
    expected = np.empty((1, nx, nv))
    for i in range(nx):
        for j in range(nv):
            total = math.fsum(float(wx * wv) * (1 + .1 * math.cos(2 * math.pi *
                (((i + .5) / nx + a / (2 * nx)) - (-1 + 2 * (j + .5) / nv + b / nv) * time)))
                * (1 + .25 * (-1 + 2 * (j + .5) / nv + b / nv))
                for a, wx in zip(nodes, weights, strict=True)
                for b, wv in zip(nodes, weights, strict=True)) / 4
            expected[0, i, j] = total
    np.testing.assert_allclose(exact_cell_average(nx, nv, time), expected, rtol=0, atol=8e-16)


def test_continuum_convergence_control_keeps_first_order_spatial_transport():
    errors = [receive(discrete_fourier(nx, nv, 1 / (4 * nx), nx // 2), initial(nx, nv),
                      nx, nv, 1 / (4 * nx), nx // 2)["continuum_l1"] for nx, nv in CASES]
    assert math.log2(errors[0] / errors[1]) > .8
    assert FRAME.domain.lower == (-1., 0.)


def test_empty_token_does_not_erase_an_actual_missing_auxiliary_requirement():
    from pops.model.operators import Operator
    original = MODULE.operator_registry().get("flux_default")
    missing = Operator(original.name, original.kind, original.signature,
                       requirements={"aux": ("foreign_missing_aux",)})
    with pytest.raises(MissingInputProvider):
        build_operator_provider_pack(MODULE, missing)


def test_native_fixture_uses_single_initial_authority_and_disjoint_checkpoint_observations():
    # Static protocol proof, not a native bind. Actual save/run assertions remain native-only.
    import ast
    from pathlib import Path
    path = Path(__file__).parents[1]/"python/integration/runtime/test_m19_freestreaming_runtime.py"
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    binds = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "bind"]
    assert len(binds) == 2
    assert all(tuple(kw.arg for kw in node.keywords) == ("resources",) for node in binds)
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "dump_ir" for node in calls)
    saves = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_snapshot"]
    assert [node.args[3].value for node in saves] == ["initial", "accepted", "continuous", "restored", "replay"]
    assert "phase+\"-checkpoint\"" in path.read_text()
    assert "phase+\"-state.npz\"" in path.read_text()


@pytest.mark.parametrize("mutation", ("history", "cache", "dtype", "shape", "missing"))
def test_restored_checkpoint_comparison_rejects_nonpopulation_payload_changes(tmp_path, mutation):
    from tests.python.integration.runtime.test_m19_freestreaming_runtime import _compare_restored_payload
    left = {"state": np.ones(3), "history": np.zeros(2), "cache": np.zeros(1)}
    right = {key: value.copy() for key, value in left.items()}
    if mutation in ("history", "cache"):
        right[mutation][0] = 1
    elif mutation == "dtype":
        right["history"] = right["history"].astype(np.float32)
    elif mutation == "shape":
        right["history"] = right["history"].reshape(1, 2)
    else:
        del right["cache"]
    accepted, restored = tmp_path/"accepted.npz", tmp_path/"restored.npz"
    np.savez(accepted, **left)
    np.savez(restored, **left)
    assert _compare_restored_payload(accepted, restored)["exact_payload"]
    np.savez(restored, **right)
    with pytest.raises(AssertionError):
        _compare_restored_payload(accepted, restored)



def test_empty_token_exact_type_guard_refuses_subclass_alias():
    from pops.model import FieldSpace
    from pops.model.operators import Operator
    class ForeignFieldSpace(FieldSpace):
        pass
    rate=MODULE.operator_registry().get(RATE.registered_operator_name)
    data=rate.signature.inputs[-1].to_data();data.pop("kind");data.pop("value_shape")
    forged=ForeignFieldSpace(**data)
    assert forged.to_data()==rate.signature.inputs[-1].to_data()
    operator=Operator(rate.name,rate.kind,type(rate.signature)(inputs=(*rate.signature.inputs[:-1],forged),output=rate.signature.output),requirements=rate.requirements)
    with pytest.raises(MissingInputProvider):build_operator_provider_pack(MODULE,operator)


@pytest.mark.parametrize("nx,nv", CASES)
def test_source_preparation_receipt_is_bounded_and_current_route_stays_public(nx, nv):
    import json, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
    assert not list((root / "python/pops").rglob("_pops*.so"))
    assert "pops._pops" not in sys.modules
    assert not any(name.startswith("pops._native.dim") for name in sys.modules)
    receipt = json.loads((root / "docs/development/api_040/m19_current_source_preparation_sol61.json").read_text())
    assert receipt["status"] == "SOURCE_ONLY"
    assert receipt["native_executed"] is False and receipt["pops_extension_present"] is False
    assert receipt["base"] == "05159edd06dddd05ed23105d50a12bb65217e95e"
    assert receipt["requires_remeasurement_before_native_qualification"] is True
    row = next(row for row in receipt["cases"] if (row["nx"], row["nv"]) == (nx, nv))
    assert row["native_cells"] == [nv, nx] and row["dt"] == [1, 4 * nx]
    assert row["ir_version"] == 5  # historical measured Source snapshot, not a future emitter promise
    assert type(row["emitted_cpp_bytes"]) is int and row["emitted_cpp_bytes"] > 0
    assert len(row["emitted_cpp_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in row["emitted_cpp_sha256"])
    case, layout, _, _ = build(nx, nv)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.neg_div_flux_default_into(") == 2
    assert type(resolved.time._serialize()["version"]) is int
    # Future emitter/header changes need a fresh measurement, not this old hash.
