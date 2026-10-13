"""Source guards for the authored restart fixtures; no native module is loaded."""
from __future__ import annotations

import ast
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pops
import pytest

ROOT = Path(__file__).resolve().parents[4]
MPI = ROOT / "tests/python/integration/mpi"


def _fixture_function(file, name, namespace):
    tree = ast.parse((MPI / file).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(MPI / file), "exec"), namespace)
    return namespace[name]


def test_shared_ab2_fixture_prepares_both_states_before_one_rhs_round():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.physics import Model

    frame = Rectangle("restart-fixture-source", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x, y = frame.axes
    model = Model("restart-fixture-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux("transport", frame=frame, state=state,
                      components={x: (rho,), y: (0.25*rho,)},
                      waves={x: (1.+0.*rho,), y: (0.25+0.*rho,)})
    rate = model.rate("transport-rate", equation=ddt(state) == -div(flux))
    case = pops.Case("restart-fixture-case")
    left, right = case.block("left", model), case.block("right", model)
    build = _fixture_function("test_amr_regrid_on_restart_mpi.py", "_shared_ab2_program",
                              {"pops": pops, "Fraction": Fraction, "_SHARED_DT": 1.e-3})
    program = build(left[state], right[state], rate)
    for target in ("system", "amr_system"):
        source = emit_cpp_program(program, target=target)
        assert source.count("ctx.rhs_group(") == 1
        assert "ctx.rhs_into(" not in source


def test_rank_change_fixture_reserves_the_complete_archive_via_public_api(tmp_path):
    configure = _fixture_function("probe_amr_rank_change_restart.py",
                                  "_configure_restart_diagnostic_capacity",
                                  {"np": np, "Any": object, "Path": Path, "_COMM": SimpleNamespace(size=1)})
    checkpoint = tmp_path / "two-ranks.npz"
    np.savez(checkpoint, program_diagnostics_state=np.arange(80, dtype=np.uint8),
             program_diagnostics_offsets=np.array([0, 40, 80], dtype=np.int64))
    requests = []
    runtime = SimpleNamespace(configure_checkpoint_diagnostics=lambda **kw: requests.append(kw))
    configure(runtime, checkpoint)
    assert requests == [{"capacity_per_rank": 88}]
    np.savez(checkpoint, program_diagnostics_state=np.zeros((2, 40), dtype=np.uint8),
             program_diagnostics_offsets=np.array([0, 40, 80], dtype=np.int64))
    with pytest.raises(AssertionError, match="malformed diagnostic bytes"):
        configure(runtime, checkpoint)
    assert requests == [{"capacity_per_rank": 88}]


def test_restart_fixture_authenticates_transformed_run_with_exact_receipt_scalars():
    from pops.identity import make_identity
    from pops.output import RegridOnRestart

    build = _fixture_function("test_amr_regrid_on_restart_mpi.py", "_transformed_restart_source_run",
                              {"json": json})
    source = make_identity("run", {"source": "accepted"})
    restart = make_identity("restart", {"restart": "transformed"})
    receipt = {"accepted_time": 0.125,
               "composite_integrals_before": [{"block": "left", "value": -0.0}],
               "composite_integrals_after": [{"block": "left", "value": 2.0}],
               "changed": True}
    expected = make_identity("run", {
        "continuation": "regrid_on_restart", "source_run_identity": source.to_data(),
        "restart_identity": restart.to_data(),
        "hierarchy_policy_identity": RegridOnRestart().identity.to_data(),
        "regrid_receipt": {"accepted_time": "0x1.0000000000000p-3",
                           "composite_integrals_before": [{"block": "left", "value": "-0x0.0p+0"}],
                           "composite_integrals_after": [{"block": "left", "value": "0x1.0000000000000p+1"}],
                           "changed": True},
    })
    assert build(source, restart, receipt) == expected != source
    assert receipt["accepted_time"] == 0.125
    assert receipt["composite_integrals_before"][0]["value"].hex() == "-0x0.0p+0"


def test_restart_receipt_projection_preserves_bytes_tuples_and_rejects_arrays():
    from pops.identity import make_identity
    from pops.output import RegridOnRestart

    build = _fixture_function("test_amr_regrid_on_restart_mpi.py", "_transformed_restart_source_run", {})
    source, restart = make_identity("run", {"source": 1}), make_identity("restart", {"restart": 1})
    receipt = {"accepted_time": 0.125, "composite_integrals_before": ({"value": -0.0},),
               "composite_integrals_after": [{"value": 2.0}], "opaque": b"\x00\xff",
               "opaque_sequence": (1, 2)}
    normalized = {"accepted_time": "0x1.0000000000000p-3",
                  "composite_integrals_before": [{"value": "-0x0.0p+0"}],
                  "composite_integrals_after": [{"value": "0x1.0000000000000p+1"}],
                  "opaque": b"\x00\xff", "opaque_sequence": (1, 2)}
    expected = make_identity("run", {
        "continuation": "regrid_on_restart", "source_run_identity": source.to_data(),
        "restart_identity": restart.to_data(), "hierarchy_policy_identity": RegridOnRestart().identity.to_data(),
        "regrid_receipt": normalized})
    assert build(source, restart, receipt) == expected
    assert isinstance(receipt["composite_integrals_before"], tuple)
    assert receipt["opaque"] == b"\x00\xff" and receipt["opaque_sequence"] == (1, 2)
    array = np.arange(3, dtype=np.int64)
    receipt["opaque_array"] = array
    with pytest.raises(TypeError):
        build(source, restart, receipt)
    assert np.array_equal(array, np.arange(3, dtype=np.int64))
    receipt.pop("opaque_array")
    receipt["composite_integrals_before"] = np.asarray([0.0])
    with pytest.raises(AssertionError, match="list or tuple"):
        build(source, restart, receipt)
