"""SOURCE_ONLY stdlib analytic/protocol checks; never Native evidence."""

import importlib.util
import inspect
import tempfile
import unittest
import json
from pathlib import Path
import sys
import struct
import numpy as np

spec = importlib.util.spec_from_file_location(
    "spatial_independent_saved_reader",
    Path(__file__).with_name("sol61_spatial_interaction_saved_reception.py"),
)
reader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reader
spec.loader.exec_module(reader)


def source_only_arrays():
    arrays = {}
    for side in reader.SIDES:
        arrays[side + "_level"] = np.array([0, 0], dtype="int64")
        arrays[side + "_index"] = np.array([[0], [1]], dtype="int64")
        arrays[side + "_position"] = np.array([[0.25], [0.75]])
        arrays[side + "_volume"] = np.array([0.5, 0.5])
        arrays[side + "_kappa"] = np.array([0.25, 0.5])
        arrays[side + "_active"] = np.ones(2)
        arrays[side + "_coverage"] = np.ones(2)
        arrays[side + "_owner"] = np.zeros(2, dtype="int64")
        arrays[side + "_resident"] = np.zeros(2, dtype="int64")
    arrays["source_values"] = np.array([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.0]])
    arrays["target_values"] = np.zeros((2, 2))
    levels = {
        0: {
            "level": 0,
            "distribution": "distributed",
            "domain_lo": [0],
            "domain_hi": [1],
            "lower": [(0.0).hex()],
            "upper": [(1.0).hex()],
            "boxes": [{"lo": [0], "hi": [1], "owner": 0}],
        }
    }
    return (arrays, levels)


TREE = [
    "sub",
    ["add", ["constant", (1.0).hex()], ["x", 0]],
    ["mul", ["constant", (2.0).hex()], ["y", 0]],
]


def check_SOURCE_ONLY_signed_nonsymmetric_kernel_and_permutation():
    arrays, levels = source_only_arrays()
    sources = reader.quotient(arrays, "source", levels, 1, 3, "<f8")
    targets = reader.quotient(arrays, "target", levels, 1, 2, "<f8")
    actual = reader.convolution(arrays, TREE, [2, 0], 1, sources, targets)
    np.testing.assert_array_equal(actual, [[0.375, 0.25], [0.625, 0.25]])
    arrays["target_kappa"][:] = 1
    np.testing.assert_array_equal(
        reader.convolution(arrays, TREE, [2, 0], 1, sources, targets), actual
    )


def check_SOURCE_ONLY_covered_and_zero_eb_nan_are_not_sources():
    arrays, levels = source_only_arrays()
    arrays["source_active"][1] = 0
    arrays["source_values"][1] = np.nan
    sources = reader.quotient(arrays, "source", levels, 1, 3, "<f8")
    targets = reader.quotient(arrays, "target", levels, 1, 2, "<f8")
    expected = [[0.375, 0.1875], [0.625, 0.3125]]
    np.testing.assert_array_equal(
        reader.convolution(arrays, TREE, [2, 0], 1, sources, targets), expected
    )
    arrays["source_active"][1] = 1
    arrays["source_kappa"][1] = 0
    np.testing.assert_array_equal(
        reader.convolution(arrays, TREE, [2, 0], 1, sources, targets), expected
    )


def check_SOURCE_ONLY_topology_geometry_countermodels(mutation):
    arrays, levels = source_only_arrays()
    if mutation == "missing":
        for key in list(arrays):
            if key.startswith("source_"):
                arrays[key] = arrays[key][:1]
    elif mutation == "duplicate":
        arrays["source_index"][1] = [0]
        arrays["source_position"][1] = [0.25]
    elif mutation == "foreign_owner":
        arrays["source_owner"][0] = 1
    elif mutation == "coordinate":
        arrays["source_position"][0] += 0.125
    else:
        arrays["source_kappa"][0] = np.nan
    with raises(ValueError):
        reader.quotient(arrays, "source", levels, 1, 3, "<f8")


def check_SOURCE_ONLY_replicas_owner_once_and_raw_zero_measure_masks():
    arrays, levels = source_only_arrays()
    for key, value in list(arrays.items()):
        if key.startswith("source_"):
            arrays[key] = np.concatenate((value, value))
    arrays["source_resident"] = np.array([0, 0, 1, 1], dtype="int64")
    levels[0]["distribution"] = "replicated"
    owners = reader.quotient(arrays, "source", levels, 2, 3, "<f8")
    assert owners == [0, 1]
    arrays["source_active"][[0, 2]] = 0
    arrays["source_kappa"][2] = 0.75
    with raises(ValueError, match="replica geometry"):
        reader.quotient(arrays, "source", levels, 2, 3, "<f8")


def check_SOURCE_ONLY_invalid_ast_refuses(tree):
    with raises(ValueError):
        reader.kernel(tree, 1, (reader.Fraction(1, 4),), (reader.Fraction(3, 4),))


def check_SOURCE_ONLY_self_pair_is_evaluated_without_regularization():
    singular = ["div", ["constant", (1.0).hex()], ["abs", ["sub", ["x", 0], ["y", 0]]]]
    with raises(ValueError, match="singular"):
        reader.kernel(singular, 1, (reader.Fraction(1, 4),), (reader.Fraction(1, 4),))


def check_SOURCE_ONLY_typed_dimension_countermodels(power):
    with raises(ValueError):
        reader.dimension({"kind": "physical_dimension", "powers": [power]})


def check_SOURCE_ONLY_pending_assembly_never_approves(tmp_path):
    leaf = tmp_path / "source-only.txt"
    leaf.write_text("SOURCE_ONLY protocol bytes; never native")
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "sol61.spatial-direct-capture-index@1",
                "capture_class": "SOURCE_ONLY",
                "mode": "serial",
                "ranks": 1,
                "cases": [],
                "junit": [],
                "origins": [{"role": "source_only", "path": str(leaf)}],
            }
        )
    )
    pending = reader.assemble(index)
    assert pending["status"] == "PENDING_ROOT_APPROVAL"
    assert pending["source_commit"] is None
    assert pending["native_build_source_commit"] is None
    with raises(ValueError, match="two external"):
        reader.receive(index, None, index, None)


def check_SOURCE_ONLY_alias_and_json_duplicates_refuse(tmp_path):
    actual = tmp_path / "actual"
    actual.write_text("SOURCE_ONLY")
    alias = tmp_path / "alias"
    alias.symlink_to(actual)
    with raises(ValueError, match="aliases"):
        reader.file_pin(alias)
    with raises(ValueError):
        reader.strict_json(b'{"mode":"serial","mode":"mpi"}')


def check_SOURCE_ONLY_amr_coverage_is_derived_from_fine_boxes():
    arrays, levels = source_only_arrays()
    levels[1] = {
        "level": 1,
        "distribution": "distributed",
        "domain_lo": [0],
        "domain_hi": [3],
        "lower": [(0.0).hex()],
        "upper": [(1.0).hex()],
        "boxes": [{"lo": [2], "hi": [3], "owner": 0}],
    }
    additions = {
        "level": [1, 1],
        "index": [[2], [3]],
        "position": [[0.625], [0.875]],
        "volume": [0.25, 0.25],
        "kappa": [1.0, 1.0],
        "active": [1.0, 1.0],
        "coverage": [1.0, 1.0],
        "owner": [0, 0],
        "resident": [0, 0],
        "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }
    for key, values in additions.items():
        old = arrays["source_" + key]
        arrays["source_" + key] = np.concatenate((old, np.array(values, dtype=old.dtype)))
    arrays["source_coverage"][1] = 0
    arrays["source_values"][1] = np.nan
    assert reader.quotient(arrays, "source", levels, 1, 3, "<f8") == [0, 1, 2, 3]
    arrays["source_coverage"][1] = 1
    arrays["source_values"][1] = 0
    with raises(ValueError, match="coverage differs"):
        reader.quotient(arrays, "source", levels, 1, 3, "<f8")


def check_SOURCE_ONLY_level_topology_types_fail_closed(bad):
    arrays, levels = source_only_arrays()
    if bad == "unknown_distribution":
        levels[0]["distribution"] = "anonymous"
    elif bad == "boolean_owner":
        levels[0]["boxes"][0]["owner"] = False
    elif bad == "box_escape":
        levels[0]["boxes"][0]["hi"] = [2]
    else:
        levels[0]["lower"] = []
    with raises(ValueError):
        reader.quotient(arrays, "source", levels, 1, 3, "<f8")


def check_SOURCE_ONLY_index_protocol_fails_before_pin_discovery(tmp_path, bad):
    leaf = tmp_path / "only-source-protocol"
    leaf.write_text("SOURCE_ONLY, never Native")
    value = {
        "schema": "sol61.spatial-direct-capture-index@1",
        "capture_class": "SOURCE_ONLY",
        "mode": "serial",
        "ranks": 1,
        "cases": [],
        "junit": [],
        "origins": [{"role": "source_only", "path": str(leaf)}],
    }
    if bad == "origin_duplicate":
        value["origins"] *= 2
    elif bad == "junit_duplicate":
        value["junit"] = [str(leaf), str(leaf)]
    elif bad == "mode_boolean":
        value["ranks"] = True
    else:
        value["origins"][0]["source_guess"] = "not allowed"
    index = tmp_path / "index.json"
    index.write_text(json.dumps(value))
    with raises(ValueError):
        reader.assemble(index)


def check_SOURCE_ONLY_unselected_nan_is_not_a_hidden_physical_constraint():
    arrays, levels = source_only_arrays()
    arrays["source_values"][:, 1] = np.nan
    rows = reader.quotient(arrays, "source", levels, 1, 3, "<f8", finite_components=[2, 0])
    assert rows == [0, 1]


def raises(error, *, match=None):
    case = unittest.TestCase()
    return case.assertRaises(error) if match is None else case.assertRaisesRegex(error, match)


def source_only_ir():
    # Handwritten protocol record, never emitted/native evidence or a ROOT seal.
    space = dict(
        components=["rho"],
        units=[None],
        frame="SOURCE_ONLY-frame",
        clock="physical",
        support=None,
        layout="cell",
        centering="cell",
    )
    point = dict(
        schema_version=1,
        clock={"name": "SOURCE_ONLY-clock"},
        step=0,
        offset={"kind": "integer", "value": "0"},
    )
    node = dict(id=0, op="state", inputs=[], point=point, block="SOURCE_ONLY-block", space=space)
    output = dict(
        id=1,
        op="spatial_interaction",
        vtype="scalar_field",
        inputs=[0],
        point=point,
        block=node["block"],
        space=dict(space, components=["I"]),
        attrs={
            "contract": "pops.spatial-interaction@1",
            "kernel": {
                "contract": "pops.spatial-interaction-kernel@1",
                "dimension": 1,
                "tree": TREE,
                "units": None,
            },
            "measure": {"contract": "pops.cell-volume-eb@1", "coordinate_units": []},
            "quadrature": "pops.cell-midpoint@1",
            "realization": "pops.direct-spatial-interaction@1",
            "max_workspace_bytes": {"uint64_hex": "0000000000010000"},
            "components": [0],
            "ncomp": 1,
            "source_scope": "accepted",
        },
    )
    ir = {"version": 17, "nodes": [node, output]}
    return ir


def check_SOURCE_ONLY_IR17_declared_operator_and_typed_ids():
    ir = source_only_ir()
    assert reader.interaction(ir, 1)[0] is ir["nodes"][1]
    ir["nodes"][0]["id"] = False
    with raises(ValueError, match="exact IR"):
        reader.interaction(ir, 1)


def check_SOURCE_ONLY_IR18_does_not_promote_storage_Q_to_seed_T():
    import copy

    ir = source_only_ir()
    state, output = ir["nodes"]
    state["state"] = {"qualified_id": "SOURCE_ONLY-State-T"}
    contract = dict(
        state=state["state"], space=state["space"], clock=state["point"]["clock"], depth=2
    )
    oldpoint = dict(state["point"], step=-1)
    retained = dict(
        id=2,
        op="history",
        inputs=[],
        point=oldpoint,
        block=state["block"],
        state=state["state"],
        space=state["space"],
        attrs=dict(history="T", lag=1, history_contract=contract),
    )
    store = dict(id=3, op="store_history", inputs=[0], attrs={"history": "T"})
    output.update(inputs=[2], point=oldpoint)
    output["attrs"].update(
        contract="pops.spatial-interaction@2",
        source_scope="issued",
        history_source=dict(
            contract="pops.spatial-interaction-history-source@1",
            history="T",
            lag=1,
            state={"handle": state["state"]},
            space=state["space"],
            clock=state["point"]["clock"],
            seed_id=0,
            seed_point=state["point"],
            cold_start="copy_current",
        ),
    )
    ir.update(version=18, history_contracts=[contract], nodes=[state, store, retained, output])
    assert reader.interaction(ir, 1)[0] is output
    # A rescellable IR with the same owner is still not an original State.n.
    forged = copy.deepcopy(ir)
    forged["nodes"][0]["op"] = "linear_combine"
    with raises(ValueError, match="seed/storage closure"):
        reader.interaction(forged, 1)
    forged = copy.deepcopy(ir)
    forged["nodes"][-1]["attrs"]["history_source"]["lag"] = True
    with raises(ValueError, match="source type"):
        reader.interaction(forged, 1)
    forged = copy.deepcopy(ir)
    forged["nodes"][0]["point"]["step"] = 1
    with raises(ValueError, match="seed/lag point"):
        reader.interaction(forged, 1)


def check_SOURCE_ONLY_large_power_is_scalar_not_unbounded_rational_storage():
    assert (
        reader.kernel(
            ["pow", ["constant", (0.5).hex()], ["constant", (1e9).hex()]],
            1,
            (reader.Fraction(0),),
            (reader.Fraction(0),),
        )
        == 0
    )


def check_SOURCE_ONLY_cross_axes_signed_convolution_two_Real_widths():
    tree = [
        "sub",
        [
            "add",
            ["sub", ["mul", ["x", 0], ["y", 2]], ["mul", ["constant", (2.0).hex()], ["y", 1]]],
            ["x", 2],
        ],
        ["y", 0],
    ]
    for real in ("float32", "float64"):
        arrays, _ = source_only_arrays()
        arrays["source_position"] = np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]], dtype=real)
        arrays["target_position"] = np.array([[0.5, 0.25, 0.75], [0.75, 0.75, 0.25]], dtype=real)
        for key in ("source_values", "source_kappa", "source_volume"):
            arrays[key] = arrays[key].astype(real)
        # Manual dyadic W rows=(-1/8,-7/8),(-7/16,-21/16).
        result = reader.convolution(arrays, tree, [2, 0], 3, [0, 1], [0, 1])
        np.testing.assert_array_equal(result, [[-0.0625, 0.1875], [-0.21875, 0.21875]])


PARAMETERS = {
    "check_SOURCE_ONLY_topology_geometry_countermodels": (
        "mutation",
        ["missing", "duplicate", "foreign_owner", "coordinate", "kappa"],
    ),
    "check_SOURCE_ONLY_invalid_ast_refuses": (
        "tree",
        [["x", True], ["y", 1], ["constant", "nan"], ["foreign"]],
    ),
    "check_SOURCE_ONLY_typed_dimension_countermodels": (
        "power",
        [["L", True, 1], ["L", 0, 1], ["L", 2, 4], ["L", 1, -1]],
    ),
    "check_SOURCE_ONLY_level_topology_types_fail_closed": (
        "bad",
        ["unknown_distribution", "boolean_owner", "box_escape", "axis_count"],
    ),
    "check_SOURCE_ONLY_index_protocol_fails_before_pin_discovery": (
        "bad",
        ["origin_duplicate", "junit_duplicate", "mode_boolean", "origin_extra"],
    ),
}


class TestSourceOnlyReception(unittest.TestCase):
    def test_33_source_protocol_checks(self):
        checks = 0
        for name, function in sorted(globals().copy().items()):
            if not name.startswith("check_SOURCE_ONLY_"):
                continue
            variants = [{}]
            if name in PARAMETERS:
                key, values = PARAMETERS[name]
                variants = [{key: value} for value in values]
            for arguments in variants:
                with self.subTest(check=name, arguments=arguments):
                    with tempfile.TemporaryDirectory(prefix="SOURCE_ONLY-nonlocal-") as directory:
                        if "tmp_path" in inspect.signature(function).parameters:
                            arguments = dict(arguments, tmp_path=Path(directory).resolve())
                        function(**arguments)
                        checks += 1
        self.assertEqual(checks, 33)


def source_only_native_piece_arrays(*, ranks=1, replicated=False):
    """In-memory protocol model ONLY. Never written as a Native NPZ or receipt."""
    shape = (2, 2)
    spacing = np.array([(2.3 - 0.3) / 2, (2.6 + 0.4) / 2])
    values = np.arange(12, dtype=float).reshape(3, *shape) - 4
    arrays = dict(
        time=np.array(0.0),
        step=np.array(0),
        topology_epoch=np.array(0),
        level_0_coverage=np.zeros(shape, dtype=bool),
        level_0_valid_cells=np.ones(shape, dtype=bool),
        level_0_cell_volumes=np.full(shape, spacing[0] * spacing[1]),
        level_0_boxes=np.array([[0, 0, 2, 2]], dtype=np.int64),
        level_0_native_cell_shape=np.array(shape, dtype=np.int64),
        level_0_origin=np.array([0.3, -0.4]),
        level_0_spacing=spacing,
    )
    manifest = []
    for rank in range(ranks):
        arrays["rank_%d_auxiliary_0" % rank] = np.frombuffer(b"POPSAUX2SOURCE_ONLY", dtype=np.uint8)
        arrays["level_0_rank_%d_history_metadata_json" % rank] = np.frombuffer(
            b"{}", dtype=np.uint8
        )
        if rank and not replicated:
            continue
        for field, value in (
            ("rho", values),
            ("active", np.ones((1, *shape))),
            ("kappa", np.array([[[0.25, 0.5], [0.75, 1.0]]])),
        ):
            name = "level_0_rank_%d_%s_piece_0" % (rank, field)
            arrays[name] = value.copy()
            manifest.append(
                dict(
                    array=name,
                    level=0,
                    resident_rank=rank,
                    lower=[0, 0],
                    upper=[2, 2],
                    field=field,
                    global_box_index=0,
                    reported_owner=rank if replicated else 0,
                    replicated=replicated,
                    physical_owner=0,
                )
            )
    arrays["piece_manifest_json"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    receipt = dict(
        width=3,
        components=[2, 0],
        dimension=2,
        adaptive=False,
        coordinate_authority="bound-normalized-native-Cartesian",
        raw_coverage_semantics="true_is_covered",
    )
    return arrays, receipt


class TestSourceOnlyRawCapture(unittest.TestCase):
    def test_SOURCE_ONLY_halfopen_native_axes_and_signed_component_quadrature(self):
        a, r = source_only_native_piece_arrays()
        rows, levels, _ = reader.fixture_snapshot(a, r, 1)
        owners = reader.quotient(rows, "source", levels, 1, 3, "<f8", finite_components=[2, 0])
        rows["target_values"] = np.zeros((len(rows["target_level"]), 2))
        targets = reader.quotient(rows, "target", levels, 1, 2, "<f8")
        tree = [
            "sub",
            ["add", ["constant", (1.0).hex()], ["mul", ["x", 0], ["y", 1]]],
            ["mul", ["constant", (2.0).hex()], ["y", 0]],
        ]
        actual = reader.convolution(rows, tree, [2, 0], 2, owners, targets)
        expected = []
        F = reader.Fraction.from_float
        for target in targets:
            x = rows["target_position"][target]
            expected.append(
                [
                    float(
                        sum(
                            (
                                1
                                + F(float(x[0])) * F(float(rows["source_position"][s, 1]))
                                - 2 * F(float(rows["source_position"][s, 0]))
                            )
                            * F(float(rows["source_volume"][s]))
                            * F(float(rows["source_kappa"][s]))
                            * F(float(rows["source_values"][s, c]))
                            for s in owners
                        )
                    )
                    for c in (2, 0)
                ]
            )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(rows["source_index"].tolist(), [[0, 0], [1, 0], [0, 1], [1, 1]])

    def test_SOURCE_ONLY_replicas_and_empty_rank(self):
        for replicated in (False, True):
            with self.subTest(replicated=replicated):
                a, r = source_only_native_piece_arrays(ranks=3, replicated=replicated)
                rows, levels, _ = reader.fixture_snapshot(a, r, 3)
                owners = reader.quotient(rows, "source", levels, 3, 3, "<f8")
                self.assertEqual(len(owners), 4)
                self.assertTrue(all(rows["source_resident"][i] == 0 for i in owners))
        a["level_0_rank_1_kappa_piece_0"][0, 0, 0] = 0.75
        rows, levels, _ = reader.fixture_snapshot(a, r, 3)
        with self.assertRaisesRegex(ValueError, "replica geometry bits"):
            reader.quotient(rows, "source", levels, 3, 3, "<f8")

    def test_SOURCE_ONLY_raw_capture_forgery_refusals(self):
        for mutation in (
            "unknown_array",
            "mask_omission",
            "box_alias",
            "false_valid",
            "double_kappa",
        ):
            with self.subTest(mutation=mutation):
                a, r = source_only_native_piece_arrays()
                if mutation == "unknown_array":
                    a["unclaimed_positive"] = np.array(1.0)
                elif mutation == "mask_omission":
                    a.pop("level_0_rank_0_kappa_piece_0")
                elif mutation == "box_alias":
                    a["level_0_boxes"][0, 3] = 1
                elif mutation == "false_valid":
                    a["level_0_valid_cells"][0, 0] = False
                else:
                    a["level_0_cell_volumes"] *= a["level_0_rank_0_kappa_piece_0"][0]
                with self.assertRaises((ValueError, KeyError)):
                    rows, levels, _ = reader.fixture_snapshot(a, r, 1)
                    reader.quotient(rows, "source", levels, 1, 3, "<f8")

    def test_SOURCE_ONLY_sample_codec_and_closure(self):
        def sample(name="keeper", level=-1, slot_count=2, kind=2, ordinal=1):
            raw = (
                b"POPSHID1"
                + struct.pack("<Q", len(name))
                + name.encode()
                + struct.pack("<qQ", level, slot_count)
            )
            return raw + b"".join(
                struct.pack("<QddQ", kind, 0.0, 0.01, ordinal) for _ in range(slot_count)
            )

        raw = sample()
        self.assertEqual(
            reader.history_samples(raw.hex(), "keeper", -1, 2), [(2, 0.0, 0.01, 1)] * 2
        )
        for forged in (
            sample(name="other"),
            sample(level=0),
            sample(slot_count=1),
            sample(kind=0),
            sample(ordinal=0),
            raw + b"X",
        ):
            with self.subTest(forged=forged.hex()):
                with self.assertRaises(ValueError):
                    reader.history_samples(forged.hex(), "keeper", -1, 2)

    def test_SOURCE_ONLY_junit_closed_native_refusals(self):
        from xml.etree.ElementTree import Element, SubElement, tostring

        index = dict(
            ranks=2,
            cases=[
                dict(id=k, receipt="/SOURCE_ONLY/" + k + "/receipt.json")
                for k in (*reader.SCIENCE_IDS, *reader.CONTROL_IDS)
            ],
            junit=["rank0", "rank1"],
        )

        def xml(rank, malice=None):
            suite = Element("testsuite", tests="6", failures="0", errors="0", skipped="0")
            for key in (*reader.SCIENCE_IDS, *reader.CONTROL_IDS):
                science = key in reader.SCIENCE_IDS
                prefix = (
                    "test_public_spatial_interaction_saved_history_and_exact_replay"
                    if science
                    else "test_public_spatial_interaction_refuses_without_publication"
                )
                node = SubElement(suite, "testcase", name=prefix + "[" + key + "]")
                props = SubElement(node, "properties")
                data = dict(
                    evidence_path="/SOURCE_ONLY/" + key,
                    rank=rank,
                    size=2,
                    fixture_schema=reader.FIXTURE,
                )
                if science:
                    data.update(
                        dimension=2,
                        width=1 if key == reader.SCIENCE_IDS[0] else 3,
                        adaptive=key == reader.SCIENCE_IDS[2],
                    )
                else:
                    data["failure"] = key
                if malice == "path":
                    data["evidence_path"] += "/other"
                for k, v in data.items():
                    SubElement(props, "property", name=k, value=str(v))
                if malice == "skip":
                    SubElement(node, "skipped")
            return tostring(suite)

        reader.closed_junit(index, lambda key: xml(int(key[-1])))
        for bad in ("skip", "path", "same_rank"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    reader.closed_junit(
                        index,
                        lambda key, bad=bad: xml(0 if bad == "same_rank" else int(key[-1]), bad),
                    )


class TestSourceOnlyFrozenFixtureAdmission(unittest.TestCase):
    def test_SOURCE_ONLY_exact_frozen_bind_uses_one_declared_authority(self):
        """Real frozen function bodies; transport/runtime are named host seams."""
        import ast
        import subprocess
        from types import SimpleNamespace

        frozen = "e7da8f75a6f142ef627b1e4dccd2ca073e27796b"

        def body(path, names):
            text = subprocess.check_output(
                ["git", "show", frozen + ":" + path], cwd=Path(__file__).parents[2], text=True
            )
            return ast.Module(
                body=[
                    node
                    for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name in names
                ],
                type_ignores=[],
            )

        env = dict(np=np, CELLS=(8, 6))
        exec(
            compile(
                body(
                    "tests/python/support/spatial_interaction_receipts.py",
                    {"initial_values", "bound_initial_values"},
                ),
                "FROZEN_SOURCE_ONLY_helper",
                "exec",
            ),
            env,
        )
        subject = object()
        plan = SimpleNamespace(
            bindings=[SimpleNamespace(subject=subject)], canonical_subject=lambda value: value
        )
        artifact = SimpleNamespace(plan=SimpleNamespace(initial_condition_plan=plan))
        calls = []
        context = object()

        def bound(actual, *, initial_values, resources):
            self.assertIs(actual, artifact)
            self.assertEqual(set(initial_values), {subject})
            self.assertIs(resources["execution_context"], context)
            self.assertTrue(initial_values[subject].flags.c_contiguous)
            calls.append(initial_values[subject].copy())
            return "SOURCE_ONLY_RUNTIME_SEAM"

        env.update(
            pops=SimpleNamespace(bind=bound),
            collective_call=lambda world, call: call(),
            artifact_execution_context=lambda actual: context,
        )
        exec(
            compile(
                body(
                    "tests/python/integration/runtime/test_public_spatial_interaction.py", {"bind"}
                ),
                "FROZEN_SOURCE_ONLY_bind",
                "exec",
            ),
            env,
        )
        self.assertEqual(
            env["bind"](object(), artifact, 1, failure="nonfinite"), "SOURCE_ONLY_RUNTIME_SEAM"
        )
        self.assertEqual(calls[0][0, 3, 4], 3.0)
        self.assertTrue(np.isfinite(calls[0]).all())
        plan.canonical_subject = lambda value: object()
        with self.assertRaisesRegex(ValueError, "canonical Handle"):
            env["bind"](object(), artifact, 1)
        self.assertEqual(len(calls), 1)


class TestSourceOnlyCrossAuthority(unittest.TestCase):
    def test_SOURCE_ONLY_distinct_original_observations_and_keeper(self):
        import copy

        ir = source_only_ir()
        state, accepted = ir["nodes"]
        state["state"] = {"qualified_id": "SOURCE_ONLY-State"}
        contract = dict(
            state=state["state"], space=state["space"], clock=state["point"]["clock"], depth=1
        )
        oldpoint = dict(state["point"], step=-1)
        history = dict(
            id=2,
            op="history",
            inputs=[],
            state=state["state"],
            space=state["space"],
            point=oldpoint,
            block=state["block"],
            attrs=dict(history="keeper", lag=1, history_contract=contract),
        )
        output = copy.deepcopy(accepted)
        output.update(id=4, inputs=[2], point=oldpoint)
        output["attrs"].update(
            contract="pops.spatial-interaction@2",
            source_scope="issued",
            history_source=dict(
                contract="pops.spatial-interaction-history-source@1",
                history="keeper",
                lag=1,
                state={"handle": state["state"]},
                space=state["space"],
                clock=state["point"]["clock"],
                seed_id=0,
                seed_point=state["point"],
                cold_start="copy_current",
            ),
        )
        ir.update(
            version=18,
            history_contracts=[contract],
            nodes=[
                state,
                accepted,
                history,
                output,
                dict(id=3, op="store_history", inputs=[0], attrs={"history": "keeper"}),
                dict(id=5, op="store_history", inputs=[1], attrs={"history": "I_accepted"}),
                dict(id=6, op="store_history", inputs=[4], attrs={"history": "I_history"}),
            ],
        )
        maps = reader.interaction_observations(ir)
        self.assertEqual([maps[k][0]["id"] for k in ("I_accepted", "I_history")], [1, 4])
        ir["nodes"][-1]["inputs"] = [1]
        with self.assertRaises(ValueError):
            reader.interaction_observations(ir)

    def test_SOURCE_ONLY_source_checkpoint_and_auxiliary_bits(self):
        arrays, receipt = source_only_native_piece_arrays(ranks=3, replicated=True)
        _, levels, meta = reader.fixture_snapshot(arrays, receipt, 3)
        cp = dict(
            state_density=arrays["level_0_rank_0_rho_piece_0"].ravel().copy(),
            auxiliary_checkpoint=arrays["rank_0_auxiliary_0"].copy(),
        )
        reader.history_checkpoint(arrays, cp, meta, levels, 3, False)
        cp["state_density"][0] += 1
        with self.assertRaisesRegex(ValueError, "source piece/checkpoint"):
            reader.history_checkpoint(arrays, cp, meta, levels, 3, False)
        cp["state_density"][0] -= 1
        arrays["rank_2_auxiliary_0"] = np.frombuffer(b"POPSAUX2SOURCE_ONLY_CHANGED", dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "auxiliary snapshot/checkpoint"):
            reader.history_checkpoint(arrays, cp, meta, levels, 3, False)


if __name__ == "__main__":
    unittest.main()
