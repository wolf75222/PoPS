"""Counter-probes use only actual saved observations; no PoPS import or solve."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("sol61_t3_saved_state_oracle.py")
spec = importlib.util.spec_from_file_location("independent_t3", SCRIPT)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.fixture(scope="module")
def inventory():
    path = os.environ.get("SOL61_T3_INVENTORY")
    if not path:
        pytest.skip("requires an explicit inventory of actual root-run saved states")
    return oracle.json_read(path)


def datasets(inventory):
    return [dataset for run in inventory["runs"] for dataset in run["datasets"]]


def test_oracle_has_no_pops_or_author_function_import():
    import ast
    tree = ast.parse(SCRIPT.read_text())
    imported = [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
    imported += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(not name.startswith(("pops", "tests.python")) for name in imported)
    assert "np.linalg.solve" not in SCRIPT.read_text()
    assert "np.linalg.inv" not in SCRIPT.read_text()


def test_all_actual_saved_solutions_and_contrary_equations(inventory):
    rows = datasets(inventory)
    assert len(rows) == 6
    for dataset in rows:
        data = oracle.load_arrays(dataset["npz"]["path"])
        report = oracle.scientific_report(data)
        assert max(report["state_errors"]) < oracle.STATE_ATOL
        assert max(report["original_residuals"]) < oracle.RESIDUAL_ATOL
        assert all(value > oracle.RESIDUAL_ATOL for value in report["counter_equation_residuals"].values())
        assert report["second_seed_original_residual"] > 1e-3


@pytest.mark.parametrize("fault", ("route", "capture", "seed_rhs", "stale_forcing"))
def test_wrong_routes_captures_and_seed_substitution_refuse_actual_data(inventory, fault):
    for dataset in datasets(inventory):
        data = oracle.load_arrays(dataset["npz"]["path"])
        # Deliberate tainted in-memory counterprobes, never saved as native states.
        if fault == "route":
            order = [1, 0] + list(range(2, len(data["forcing"])))
            data["solved"] = data["solved"][:, order]
        elif fault == "capture":
            data["parameter"] = data["solved"][0, :1]
        elif fault == "seed_rhs":
            data["forcing"] = .8 * data["solved"][0]
        else:
            data["forcing"] = .8 * data["forcing"]
        with pytest.raises(oracle.ReceptionError):
            oracle.scientific_report(data)


def test_serial_mpi_saved_observations_are_byte_identical(inventory):
    serial, mpi = inventory["runs"]
    assert serial["name"] == "serial" and mpi["name"] == "mpi2"
    for left, right in zip(serial["datasets"], mpi["datasets"], strict=True):
        assert left["order"] == right["order"]
        assert oracle.sha(left["npz"]["path"]) == oracle.sha(right["npz"]["path"])
        a, b = oracle.load_arrays(left["npz"]["path"]), oracle.load_arrays(right["npz"]["path"])
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])


def test_self_declared_inventory_cannot_receive_positive_owner_authority(inventory, tmp_path):
    path = tmp_path / "unsealed.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(oracle.ReceptionError, match="owner must independently review"):
        oracle.verify(path, oracle.sha(path))
    with pytest.raises(oracle.ReceptionError, match="external owner pins SHA mismatch"):
        oracle.verify(path, "0" * 64)


def test_resealing_receipt_order_cannot_refresh_external_byte_pin(inventory, tmp_path):
    for index, dataset in enumerate(datasets(inventory)):
        source = Path(dataset["receipt"]["path"])
        copy = tmp_path / (str(index) + ".json")
        copy.write_bytes(source.read_bytes())
        pin = oracle.file_record(copy)
        receipt = oracle.json_read(copy)
        receipt["unknown_order"] = list(reversed(receipt["unknown_order"]))
        receipt["saved_state_sha256"] = oracle.sha(dataset["npz"]["path"])
        copy.write_text(json.dumps(receipt, indent=2) + "\n")
        with pytest.raises(oracle.ReceptionError, match="external file (hash|size) mismatch"):
            oracle.check_file(pin)


def test_source_contract_tolerances_cannot_be_read_from_untrusted_receipts(inventory):
    assert oracle.STATE_ATOL == oracle.RESIDUAL_ATOL == 2e-8
    for dataset in datasets(inventory):
        data = oracle.load_arrays(dataset["npz"]["path"])
        actual = oracle.scientific_report(data)
        assert max(actual["original_residuals"]) < 1e-12
        # Reporting recomputes the original residual and ignores receipt residuals.
        assert "state_errors" not in data and "original_residuals" not in data
