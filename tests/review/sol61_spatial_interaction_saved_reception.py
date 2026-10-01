"""Independent finite-convolution evidence reader. No PoPS/native/author oracle.

assemble only pins existing files. receive requires TWO externally supplied ROOT
digests. Analytic unit tests are SOURCE_ONLY and are never native receipts.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys
import struct
import xml.etree.ElementTree as ET

import numpy as np


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Shared strict NPZ/JSON/CBOR/checkpoint integrity primitives only; its M19
# physical reference is never called. All convolution mathematics is below.
wire = _load("nonlocal_generic_checkpoint_integrity", "sol61_m19_saved_reception.py")
need, strict_json, digest = wire.need, wire.strict_json, wire.digest
SCHEMA = "sol61.spatial-direct-saved-reception@1"
QUALIFICATION = "finite-spatial-convolution-saved-state@1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
PHASES = ("initial", "accepted", "continuous", "reloaded", "replay")
SCIENCE_IDS = ("scalar-cutcell", "signed-vector-permuted", "partial-amr-cutcell")
CONTROL_IDS = ("budget", "pole", "nonfinite")
FIXTURE = "pops.spatial-interaction-native-fixture@1"
ANCHORS = {
    "accepted": "initial",
    "continuous": "accepted",
    "reloaded": "initial",
    "replay": "reloaded",
}
SIDES = ("source", "target")
VECTOR = ("index", "position", "values")
SCALAR = ("level", "volume", "kappa", "active", "coverage", "owner", "resident")
NPZ_KEYS = {side + "_" + key for side in SIDES for key in (*VECTOR, *SCALAR)}


def exact(value, keys, where):
    need(type(value) is dict and set(value) == set(keys), where + " exact keys differ")


def canonical(path):
    raw = str(path)
    path = Path(path)
    need(str(path) == raw and "\\" not in raw, "path spelling aliases origin")
    need(
        path.is_absolute() and str(path) == str(path.absolute()) and ".." not in path.parts,
        "path is not exact absolute spelling",
    )
    need(not any(p.is_symlink() for p in (path, *path.parents)), "path aliases an origin")
    need(path == path.resolve(strict=True), "path is not canonical")
    return path


def file_pin(path):
    path = canonical(path)
    need(path.is_file(), "missing actual file")
    budget = 64 * 1024**2 if path.suffix == ".npz" else 1024**3
    before = path.stat()
    need(before.st_size <= budget, "evidence reader byte budget exceeded")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            value.update(part)
    after = path.stat()
    need(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "origin changed during read",
    )
    return {"path": str(path), "sha256": value.hexdigest()}


def pinned(row):
    exact(row, ("path", "sha256"), "file pin")
    need(type(row["sha256"]) is str and HEX.fullmatch(row["sha256"]), "invalid digest")
    need(file_pin(row["path"]) == row, "file digest differs")
    raw = canonical(row["path"]).read_bytes()
    need(digest(raw) == row["sha256"], "origin changed after digest read")
    return raw


def fraction(value):
    need(
        isinstance(value, (float, np.floating)) and math.isfinite(float(value)),
        "nonfinite physical value",
    )
    return Fraction.from_float(float(value))


def kernel_shape(tree, dimension_count):
    """Authenticate syntax without demanding any manufactured evaluation."""
    need(type(tree) in (tuple, list) and tree, "invalid kernel tree")
    op = tree[0]
    need(type(op) is str, "invalid kernel operation")
    if op == "constant":
        need(len(tree) == 2 and type(tree[1]) is str, "invalid literal type")
        value = float.fromhex(tree[1])
        need(math.isfinite(value) and value.hex() == tree[1], "noncanonical kernel literal")
    elif op in ("x", "y"):
        need(
            len(tree) == 2 and type(tree[1]) is int and 0 <= tree[1] < dimension_count,
            "foreign kernel coordinate",
        )
    else:
        arity = 2 if op in ("neg", "abs", "sqrt", "exp") else 3
        need(
            op in ("neg", "abs", "sqrt", "exp", "add", "sub", "mul", "div", "pow")
            and len(tree) == arity,
            "foreign kernel operation/arity",
        )
        for part in tree[1:]:
            kernel_shape(part, dimension_count)


def kernel(tree, dimension, x, y):
    """Evaluate the inspectable AST independently; rational operations are exact.

    sqrt/exp/powers use stdlib scalar math, then exact representable
    binary64 values. Their rounding tolerance must be supplied by ROOT, not
    widened by this reader. No periodic images or diagonal exclusion is added.
    """
    need(type(tree) in (tuple, list) and tree, "invalid kernel tree")
    op = tree[0]
    if op == "constant" and len(tree) == 2:
        need(type(tree[1]) is str, "invalid literal type")
        value = float.fromhex(tree[1])
        need(math.isfinite(value) and value.hex() == tree[1], "noncanonical kernel literal")
        return Fraction.from_float(value)
    if op in ("x", "y") and len(tree) == 2:
        need(type(tree[1]) is int and 0 <= tree[1] < dimension, "foreign kernel coordinate")
        return (x if op == "x" else y)[tree[1]]
    if op in ("neg", "abs", "sqrt", "exp") and len(tree) == 2:
        a = kernel(tree[1], dimension, x, y)
        if op == "neg":
            return -a
        if op == "abs":
            return abs(a)
        result = math.sqrt(float(a)) if op == "sqrt" else math.exp(float(a))
        need(math.isfinite(result), "nonfinite evaluated kernel")
        return Fraction.from_float(result)
    need(
        op in ("add", "sub", "mul", "div", "pow") and len(tree) == 3,
        "foreign kernel operation/arity",
    )
    a, b = (kernel(part, dimension, x, y) for part in tree[1:])
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        need(b != 0, "singular evaluated kernel")
        return a / b
    need(a != 0 or b >= 0, "singular evaluated power")
    result = math.pow(float(a), float(b))
    need(math.isfinite(result), "nonfinite evaluated power")
    return Fraction.from_float(result)


def dimension(data, *, unknown=True):
    if data is None:
        need(unknown, "unknown coordinate measure dimension")
        return None
    exact(data, ("kind", "powers"), "dimension")
    need(
        data["kind"] == "physical_dimension" and type(data["powers"]) is list,
        "dimension kind differs",
    )
    result = {}
    for row in data["powers"]:
        need(
            type(row) is list
            and len(row) == 3
            and type(row[0]) is str
            and row[0]
            and type(row[1]) is int
            and type(row[2]) is int
            and row[1] != 0
            and row[2] > 0,
            "dimension is not typed canonical",
        )
        need(
            row[0] not in result and math.gcd(row[1], row[2]) == 1,
            "dimension is duplicate/unreduced",
        )
        result[row[0]] = Fraction(row[1], row[2])
    need(list(result) == sorted(result), "dimension order differs")
    return result


def interaction(ir, node_id):
    need(
        type(ir) is dict and type(ir.get("version")) is int and ir["version"] in (17, 18),
        "actual IR17 absent",
    )
    nodes = {}
    pending = list(ir["nodes"])
    while pending:
        node = pending.pop()
        need(
            type(node) is dict
            and type(node.get("id")) is int
            and node["id"] >= 0
            and type(node.get("inputs")) is list
            and all(type(i) is int and i >= 0 for i in node["inputs"]),
            "invalid exact IR node/input ids",
        )
        need(node["id"] not in nodes, "duplicate IR node")
        nodes[node["id"]] = node
        for key in (
            "true_block",
            "false_block",
            "body_block",
            "cond_block",
            "apply_block",
            "residual_block",
        ):
            pending.extend(node.get("attrs", {}).get(key, ()))
    need(type(node_id) is int and node_id in nodes, "interaction node absent")
    node = nodes[node_id]
    need(
        node["op"] == "spatial_interaction"
        and node["vtype"] == "scalar_field"
        and len(node["inputs"]) == 1,
        "interaction IR type differs",
    )
    attrs = node["attrs"]
    history_v2 = attrs.get("contract") == "pops.spatial-interaction@2"
    exact(
        attrs,
        (
            "contract",
            "kernel",
            "measure",
            "quadrature",
            "realization",
            "max_workspace_bytes",
            "components",
            "ncomp",
            "source_scope",
        )
        + (("history_source",) if history_v2 else ()),
        "interaction attrs",
    )
    need(
        attrs["contract"] in ("pops.spatial-interaction@1", "pops.spatial-interaction@2")
        and attrs["quadrature"] == "pops.cell-midpoint@1"
        and attrs["realization"] == "pops.direct-spatial-interaction@1",
        "interaction contracts differ",
    )
    physical = attrs["kernel"]
    exact(physical, ("contract", "dimension", "tree", "units"), "physical kernel")
    d = physical["dimension"]
    need(
        type(d) is int
        and 1 <= d <= 3
        and physical["contract"] == "pops.spatial-interaction-kernel@1",
        "kernel dimension/contract differs",
    )
    kernel_shape(physical["tree"], d)
    workspace = attrs["max_workspace_bytes"]
    exact(workspace, ("uint64_hex",), "workspace")
    need(
        type(workspace["uint64_hex"]) is str
        and re.fullmatch("[0-9a-f]{16}", workspace["uint64_hex"])
        and int(workspace["uint64_hex"], 16) > 0,
        "workspace authority differs",
    )
    measure = attrs["measure"]
    exact(measure, ("contract", "coordinate_units"), "measure")
    need(measure["contract"] == "pops.cell-volume-eb@1", "measure differs")
    coordinate = [dimension(unit, unknown=False) for unit in measure["coordinate_units"]]
    need(not coordinate or len(coordinate) == d, "coordinate measure dimension count differs")
    source = nodes[node["inputs"][0]]
    if history_v2:
        need(
            ir["version"] == 18 and source["op"] == "history" and attrs["source_scope"] == "issued",
            "HistoryComposite requires its distinct IR18 source contract",
        )
        history = attrs["history_source"]
        exact(
            history,
            (
                "contract",
                "history",
                "lag",
                "state",
                "space",
                "clock",
                "seed_id",
                "seed_point",
                "cold_start",
            ),
            "history source",
        )
        need(
            history["contract"] == "pops.spatial-interaction-history-source@1"
            and history["cold_start"] == "copy_current"
            and type(history["history"]) is str
            and type(history["lag"]) is int
            and history["lag"] >= 1
            and type(history["seed_id"]) is int
            and history["seed_id"] in nodes,
            "history source type/policy differs",
        )
        seed = nodes[history["seed_id"]]
        contracts = [
            c
            for c in ir.get("history_contracts", [])
            if c.get("state") == source["state"] and c.get("clock") == history["clock"]
        ]
        need(
            len(contracts) == 1
            and contracts[0] == source["attrs"].get("history_contract")
            and contracts[0]["space"] == history["space"]
            and type(contracts[0]["depth"]) is int
            and history["lag"] <= contracts[0]["depth"],
            "HistoryComposite declaration/retained lag differs",
        )
        stores = [
            v
            for v in nodes.values()
            if v["op"] == "store_history" and v["attrs"].get("history") == history["history"]
        ]
        need(
            len(stores) == 1
            and stores[0]["inputs"] == [seed["id"]]
            and seed["op"] == "state"
            and seed["state"] == source["state"]
            and history["state"] == {"handle": seed["state"]}
            and seed["space"] == source["space"] == history["space"]
            and seed["block"] == source["block"]
            and seed["point"] == history["seed_point"],
            "HistoryComposite seed/storage closure differs; cannot relabel Q as T",
        )
        need(
            seed["point"]
            == {
                "schema_version": 1,
                "clock": history["clock"],
                "step": 0,
                "offset": {"kind": "integer", "value": "0"},
            }
            and source["attrs"].get("history") == history["history"]
            and type(source["attrs"].get("lag")) is int
            and source["attrs"].get("lag") == history["lag"]
            and source["point"]
            == {
                "schema_version": 1,
                "clock": history["clock"],
                "step": -history["lag"],
                "offset": {"kind": "integer", "value": "0"},
            },
            "HistoryComposite native seed/lag point differs",
        )
    components = attrs["components"]
    need(
        type(components) is list
        and components
        and len(set(components)) == len(components)
        and all(type(i) is int and 0 <= i < len(source["space"]["components"]) for i in components)
        and type(attrs["ncomp"]) is int
        and attrs["ncomp"] == len(components),
        "component selection differs",
    )
    need(
        source["point"] == node["point"] and source["block"] == node["block"],
        "source point/owner differs",
    )
    for key in ("frame", "clock", "support", "layout", "centering"):
        need(source["space"][key] == node["space"][key], "physical space differs: " + key)
    w = dimension(physical["units"])
    for selected, output in zip(components, node["space"]["units"], strict=True):
        rho = dimension(source["space"]["units"][selected])
        actual = dimension(output)
        if rho is None or w is None or not coordinate:
            need(actual is None, "output invents units")
        else:
            total = {}
            for unit in (rho, w, *coordinate):
                for key, power in unit.items():
                    total[key] = total.get(key, 0) + power
            need(
                actual == {key: power for key, power in total.items() if power},
                "W*rho*dmu units differ",
            )
    need(attrs["source_scope"] in ("issued", "accepted"), "source scope differs")
    need(
        attrs["source_scope"] != "accepted" or source["op"] == "state",
        "accepted candidate invented",
    )
    need(
        len(node["space"]["components"]) == len(components)
        and len(node["space"]["units"]) == len(components),
        "output component width differs",
    )
    seen = set()
    pending = [source]
    while pending:
        value = pending.pop()
        if value["id"] in seen:
            continue
        seen.add(value["id"])
        if value["op"] == "state":
            point = value["point"]
            need(
                type(point.get("schema_version")) is int and type(point.get("step")) is int,
                "State.n point integer image differs",
            )
            need(
                value["point"]
                == {
                    "schema_version": 1,
                    "clock": value["point"]["clock"],
                    "step": 0,
                    "offset": {"kind": "integer", "value": "0"},
                },
                "native State.n leaf point differs",
            )
        pending.extend(nodes[i] for i in value["inputs"])
        for key in (
            "true_block",
            "false_block",
            "body_block",
            "cond_block",
            "apply_block",
            "residual_block",
        ):
            pending.extend(value.get("attrs", {}).get(key, ()))
    return node, source, attrs


def topology(levels, dimension_count, ranks):
    need(type(levels) is dict and levels, "missing actual level topology")
    keys = sorted(levels)
    need(
        keys == list(range(len(keys))) and all(type(k) is int for k in keys),
        "level closure differs",
    )
    for number, geom in levels.items():
        exact(
            geom,
            ("level", "distribution", "domain_lo", "domain_hi", "lower", "upper", "boxes"),
            "level topology",
        )
        need(
            type(geom["level"]) is int
            and geom["level"] == number
            and geom["distribution"] in ("distributed", "replicated"),
            "level identity differs",
        )
        for key in ("domain_lo", "domain_hi", "lower", "upper"):
            need(
                type(geom[key]) is list and len(geom[key]) == dimension_count,
                "geometry axis count differs",
            )
        need(
            all(type(i) is int for key in ("domain_lo", "domain_hi") for i in geom[key]),
            "domain integer image differs",
        )
        for lower, upper, lo, hi in zip(
            geom["lower"], geom["upper"], geom["domain_lo"], geom["domain_hi"], strict=True
        ):
            need(type(lower) is str and type(upper) is str, "physical bound type differs")
            a, b = float.fromhex(lower), float.fromhex(upper)
            need(
                math.isfinite(a)
                and math.isfinite(b)
                and a.hex() == lower
                and b.hex() == upper
                and a < b
                and lo <= hi,
                "physical bounds differ",
            )
        need(type(geom["boxes"]) is list, "actual boxes absent")
        for box in geom["boxes"]:
            exact(box, ("lo", "hi", "owner"), "valid box")
            need(type(box["owner"]) is int and 0 <= box["owner"] < ranks, "foreign box owner")
            need(
                type(box["lo"]) is list
                and type(box["hi"]) is list
                and len(box["lo"]) == len(box["hi"]) == dimension_count,
                "box dimension differs",
            )
            need(
                all(
                    type(a) is int and type(b) is int and lo <= a <= b <= hi
                    for a, b, lo, hi in zip(
                        box["lo"], box["hi"], geom["domain_lo"], geom["domain_hi"], strict=True
                    )
                ),
                "box escapes domain",
            )
    ratios = {}
    for coarse, fine in zip(keys, keys[1:], strict=False):
        a, b = levels[coarse], levels[fine]
        need(a["lower"] == b["lower"] and a["upper"] == b["upper"], "tower physical bounds differ")
        ratio = []
        for al, ah, bl, bh in zip(
            a["domain_lo"], a["domain_hi"], b["domain_lo"], b["domain_hi"], strict=True
        ):
            n, m = ah - al + 1, bh - bl + 1
            need(m % n == 0 and m // n >= 1, "invalid refinement ratio")
            ratio.append(m // n)
        need(any(r > 1 for r in ratio), "unrefined duplicate level")
        need(
            all(
                bl == al * r and bh == (ah + 1) * r - 1
                for al, ah, bl, bh, r in zip(
                    a["domain_lo"],
                    a["domain_hi"],
                    b["domain_lo"],
                    b["domain_hi"],
                    ratio,
                    strict=True,
                )
            ),
            "refined index domains differ",
        )
        ratios[coarse] = ratio
    return ratios


def expected_coverage(level, index, levels, ratios):
    if level not in ratios:
        return 1
    fine, ratio = levels[level + 1], ratios[level]
    # Native coverage excludes mathematical-floor coarsened fine footprints.
    # A split patch need not itself begin/end at a coarse boundary.
    return int(
        not any(
            all(
                lo // r <= i <= hi // r
                for i, lo, hi, r in zip(index, box["lo"], box["hi"], ratio, strict=True)
            )
            for box in fine["boxes"]
        )
    )


def quotient(arrays, side, levels, ranks, width, real, *, finite_components=None, hierarchy=None):
    """Validate every resident row, then take exactly the physical owner row.

    The level/box inventory is actual captured topology, not a reconstructed
    model grid. Covered source rows remain in the inventory but do not integrate.
    """
    need(
        type(ranks) is int and ranks > 0 and type(width) is int and width > 0,
        "rank/component authority differs",
    )
    values = arrays[side + "_values"]
    need(values.ndim == 2 and arrays[side + "_index"].ndim == 2, "physical row rank differs")
    if finite_components is None:
        finite_components = list(range(width))
    n = len(values)
    need(
        values.dtype.str == real and values.ndim == 2 and values.shape[1] == width,
        "physical values shape/dtype differs",
    )
    dimension_count = arrays[side + "_index"].shape[1]
    need(1 <= dimension_count <= 3, "physical dimension differs")
    ratios = topology(hierarchy or levels, dimension_count, ranks)
    need(
        all(
            level in (hierarchy or levels) and geom == (hierarchy or levels)[level]
            for level, geom in levels.items()
        ),
        "target topology differs from source hierarchy",
    )
    for key in SCALAR:
        need(arrays[side + "_" + key].shape == (n,), "row scalar shape differs")
    for key in ("level", "owner", "resident", "index"):
        need(arrays[side + "_" + key].dtype == np.dtype("int64"), "integer topology dtype differs")
    need(arrays[side + "_position"].shape == (n, dimension_count), "position shape differs")
    for key in ("position", "volume", "kappa", "active", "coverage"):
        need(
            arrays[side + "_" + key].dtype.str == real
            and np.isfinite(arrays[side + "_" + key]).all(),
            "geometry dtype/nonfinite differs",
        )
    groups = {}
    for row in range(n):
        level = int(arrays[side + "_level"][row])
        index = tuple(int(v) for v in arrays[side + "_index"][row])
        key = level, index
        owner = int(arrays[side + "_owner"][row])
        resident = int(arrays[side + "_resident"][row])
        need(0 <= owner < ranks and 0 <= resident < ranks, "foreign physical rank")
        need(level in levels, "foreign level")
        geom = levels[level]
        lo, hi = geom["domain_lo"], geom["domain_hi"]
        lower = [float.fromhex(v) for v in geom["lower"]]
        upper = [float.fromhex(v) for v in geom["upper"]]
        # Geometry::spacing is extent * rounded reciprocal, NOT extent/n.
        # Admit the two explicit IEEE evaluation graphs for cell_coordinate:
        # separate mul/add, or one compiler-contracted FMA. No epsilon window.
        scalar = np.dtype(real).type
        need(
            all(float(scalar(v)) == v for v in (*lower, *upper)),
            "physical bounds are not Native Real",
        )
        spacing = [
            scalar(
                scalar(scalar(upper[a]) - scalar(lower[a]))
                * scalar(scalar(1) / scalar(hi[a] - lo[a] + 1))
            )
            for a in range(dimension_count)
        ]
        need(all(math.isfinite(float(v)) and v > 0 for v in spacing), "invalid physical spacing")
        for a in range(dimension_count):
            offset = scalar(scalar(scalar(index[a]) - scalar(lo[a])) + scalar(0.5))
            separate = scalar(scalar(lower[a]) + scalar(offset * spacing[a]))
            fused = scalar(
                float(fraction(scalar(lower[a])) + fraction(offset) * fraction(spacing[a]))
            )
            actual = arrays[side + "_position"][row, a]
            need(
                actual.tobytes() in (separate.tobytes(), fused.tobytes()),
                "physical coordinate differs from explicit IEEE evaluation graphs",
            )
        volume = scalar(1)
        for dx in spacing:
            volume = scalar(volume * dx)
        need(
            volume.tobytes() == arrays[side + "_volume"][row].tobytes(),
            "physical cell volume differs",
        )
        active, coverage, kappa = (
            float(arrays[side + "_" + k][row]) for k in ("active", "coverage", "kappa")
        )
        need(active in (0, 1) and coverage in (0, 1) and 0 <= kappa <= 1, "invalid physical masks")
        need(
            coverage == expected_coverage(level, index, hierarchy or levels, ratios),
            "coverage differs from actual finer boxes",
        )
        groups.setdefault(key, []).append(row)
    need(
        sum(
            math.prod(b - a + 1 for a, b in zip(box["lo"], box["hi"], strict=True))
            for geom in levels.values()
            for box in geom["boxes"]
        )
        <= n,
        "valid-cell inventory exceeds resident rows",
    )
    expected = {}
    for level, geom in levels.items():
        for box in geom["boxes"]:
            for index in itertools.product(
                *(range(a, b + 1) for a, b in zip(box["lo"], box["hi"], strict=True))
            ):
                need((level, index) not in expected, "overlapping valid boxes")
                expected[level, index] = box["owner"]
    need(set(groups) == set(expected), "closed valid-cell inventory differs")
    owners = []
    for key, rows in groups.items():
        geom = levels[key[0]]
        residents = [int(arrays[side + "_resident"][i]) for i in rows]
        need(len(set(residents)) == len(rows), "duplicate resident cell")
        physical_owner = 0 if geom["distribution"] == "replicated" else expected[key]
        need(
            all(int(arrays[side + "_owner"][i]) == physical_owner for i in rows),
            "forged physical owner",
        )
        need(
            set(residents)
            == (set(range(ranks)) if geom["distribution"] == "replicated" else {physical_owner}),
            "missing/extra physical resident",
        )
        owner_row = rows[residents.index(physical_owner)]
        for row in rows:
            for key_name in ("position", "volume", "kappa", "active", "coverage"):
                need(
                    arrays[side + "_" + key_name][row].tobytes()
                    == arrays[side + "_" + key_name][owner_row].tobytes(),
                    "replica geometry bits differ",
                )
            included = (
                all(arrays[side + "_" + key_name][row] == 1 for key_name in ("active", "coverage"))
                and arrays[side + "_kappa"][row] > 0
            )
            if included:
                need(
                    np.isfinite(values[row, finite_components]).all()
                    and values[row, finite_components].tobytes()
                    == values[owner_row, finite_components].tobytes(),
                    "active source/output nonfinite or replica bits differ",
                )
        owners.append(owner_row)
    return owners


def convolution(arrays, tree, components, dimension_count, sources, targets):
    result = np.zeros((len(targets), len(components)), dtype=np.float64)
    source_rows = []
    for row in sources:
        if (
            arrays["source_active"][row] != 1
            or arrays["source_coverage"][row] != 1
            or arrays["source_kappa"][row] == 0
        ):
            continue
        y = tuple(fraction(v) for v in arrays["source_position"][row])
        measure = fraction(arrays["source_volume"][row]) * fraction(arrays["source_kappa"][row])
        values = tuple(
            fraction(arrays["source_values"][row, component]) for component in components
        )
        source_rows.append((y, measure, values))
    for output, row in enumerate(targets):
        if (
            arrays["target_active"][row] != 1
            or arrays["target_coverage"][row] != 1
            or arrays["target_kappa"][row] == 0
        ):
            continue
        x = tuple(fraction(v) for v in arrays["target_position"][row])
        sums = [Fraction(0) for _ in components]
        for y, measure, values in source_rows:
            weight = kernel(tree, dimension_count, x, y)
            for component, value in enumerate(values):
                sums[component] += weight * measure * value
        result[output] = [float(value) for value in sums]
    need(np.isfinite(result).all(), "nonfinite independent quadrature")
    return result


def json_array(array, where):
    need(array.dtype == np.uint8 and array.ndim == 1, where + " JSON bytes differ")
    return strict_json(array.tobytes())


def same_arrays(a, b, where):
    need(set(a) == set(b), where + " member inventory differs")
    need(
        all(
            a[k].dtype == b[k].dtype
            and a[k].shape == b[k].shape
            and a[k].tobytes() == b[k].tobytes()
            for k in a
        ),
        where + " bits differ",
    )


def history_samples(raw, name, level, depth):
    """Independent POPSHID1 decoder; no provenance reconstructed from values."""
    need(
        type(raw) is str and len(raw) % 2 == 0 and re.fullmatch("[0-9a-f]*", raw),
        "history sample hex differs",
    )
    data = bytes.fromhex(raw)
    need(data[:8] == b"POPSHID1" and len(data) >= 32, "history sample version differs")
    length = struct.unpack_from("<Q", data, 8)[0]
    need(length <= len(data) - 32, "history name size differs")
    end = 16 + length
    need(data[16:end].decode("utf-8") == name, "history sample name differs")
    actual_level, count = struct.unpack_from("<qQ", data, end)
    need(
        actual_level == level and count == depth and len(data) == end + 16 + 32 * depth,
        "history sample level/depth/closure differs",
    )
    result = []
    for slot in range(depth):
        kind, start, interval, ordinal = struct.unpack_from("<QQQQ", data, end + 16 + 32 * slot)
        need(kind in (1, 2), "unknown history provenance cannot be inferred")
        a, b = struct.unpack("<dd", struct.pack("<QQ", start, interval))
        if kind == 1:
            need(start == interval == ordinal == 0, "zero-start sample has a publication")
        else:
            need(
                math.isfinite(a) and math.isfinite(b) and b > 0 and ordinal > 0,
                "history publication window differs",
            )
        result.append((kind, a, b, ordinal))
    return result


def fixture_snapshot(arrays, receipt, ranks):
    """Adapt only actual native pieces. Native array y,x -> physical index x,y.

    Boxes are half-open. coverage=True is COVERED, while native inclusion is 1.
    The finite fixture declares these bounds; no limits are added to PoPS.
    """
    width = receipt["width"]
    need(type(width) is int and width > 0, "fixture width differs")
    need(
        receipt["coordinate_authority"] == "bound-normalized-native-Cartesian"
        and receipt["raw_coverage_semantics"] == "true_is_covered",
        "coordinate/mask authority differs",
    )
    need(
        receipt["dimension"] == 2 and type(receipt["dimension"]) is int,
        "this capture profile requires its declared two physical axes",
    )
    expected = {"time", "step", "topology_epoch", "piece_manifest_json"}
    for key in ("time", "step", "topology_epoch"):
        value = wire.protocol.scalar(arrays[key], "real" if key == "time" else "int")
        need(math.isfinite(value) and value >= 0, "native clock/epoch differs")
    manifests = json_array(arrays["piece_manifest_json"], "pieces")
    need(type(manifests) is list and manifests, "native pieces absent")
    levels = sorted({row.get("level") for row in manifests})
    need(
        all(type(level) is int for level in levels) and levels == list(range(len(levels))),
        "level closure differs",
    )
    if "cells" in receipt:
        need(
            levels == ([0, 1] if receipt["adaptive"] else [0]),
            "finite fixture level inventory differs",
        )
    topology_rows, rows, metadata = {}, [], {}
    for level in levels:
        prefix = "level_%d" % level
        expected.update(
            prefix + "_" + k
            for k in (
                "coverage",
                "valid_cells",
                "cell_volumes",
                "boxes",
                "native_cell_shape",
                "origin",
                "spacing",
            )
        )
        shape = arrays[prefix + "_native_cell_shape"]
        need(
            shape.dtype == np.int64 and shape.shape == (2,) and np.all(shape > 0),
            "native array shape differs",
        )
        shape = tuple(int(n) for n in shape)
        if "cells" in receipt:
            need(
                shape == tuple(n * 2**level for n in reversed(receipt["cells"])),
                "finite fixture cell inventory differs",
            )
        origin, spacing = arrays[prefix + "_origin"], arrays[prefix + "_spacing"]
        need(
            origin.dtype == spacing.dtype == np.float64
            and origin.shape == spacing.shape == (2,)
            and np.isfinite(origin).all()
            and np.isfinite(spacing).all()
            and np.all(spacing > 0),
            "native coordinate declaration differs",
        )
        # Literal bounds belong to this saved fixture profile, not production.
        lower, upper = (0.3, -0.4), (2.3, 2.6)
        need(origin.tobytes() == np.array(lower).tobytes(), "declared physical lower differs")
        need(
            spacing.tobytes()
            == np.array([(upper[a] - lower[a]) / shape[1 - a] for a in range(2)]).tobytes(),
            "declared bound spacing differs",
        )
        covered, valid, volumes = (
            arrays[prefix + "_" + k] for k in ("coverage", "valid_cells", "cell_volumes")
        )
        need(
            covered.dtype == valid.dtype == np.bool_
            and covered.shape == valid.shape == shape
            and volumes.dtype == np.float64
            and volumes.shape == shape
            and np.isfinite(volumes).all()
            and np.all(volumes > 0),
            "native geometry shape/dtype/measure differs",
        )
        boxes = arrays[prefix + "_boxes"]
        need(
            boxes.dtype == np.int64 and boxes.ndim == 2 and boxes.shape[1] == 4,
            "actual box inventory differs",
        )
        level_pieces = [row for row in manifests if row["level"] == level]
        field_maps = {}
        names = set()
        for row in level_pieces:
            exact(
                row,
                (
                    "array",
                    "level",
                    "resident_rank",
                    "lower",
                    "upper",
                    "field",
                    "global_box_index",
                    "reported_owner",
                    "replicated",
                    "physical_owner",
                ),
                "native piece",
            )
            for key in (
                "level",
                "resident_rank",
                "global_box_index",
                "reported_owner",
                "physical_owner",
            ):
                need(type(row[key]) is int, "piece integer authority differs")
            rank, boxid = row["resident_rank"], row["global_box_index"]
            need(
                0 <= rank < ranks
                and 0 <= boxid < len(boxes)
                and row["field"] in ("rho", "active", "kappa")
                and type(row["replicated"]) is bool
                and 0 <= row["reported_owner"] < ranks,
                "foreign native piece",
            )
            need(
                row["physical_owner"] == (0 if row["replicated"] else row["reported_owner"])
                and (row["replicated"] or row["reported_owner"] == rank),
                "piece ownership differs",
            )
            lo, hi = row["lower"], row["upper"]
            need(
                type(lo) is list
                and type(hi) is list
                and len(lo) == len(hi) == 2
                and all(type(v) is int for v in (*lo, *hi))
                and tuple((*lo, *hi)) == tuple(boxes[boxid]),
                "piece box linkage differs",
            )
            need(
                all(0 <= lo[a] < hi[a] <= shape[a] for a in range(2)), "piece escapes native domain"
            )
            name = "level_%d_rank_%d_%s_piece_%d" % (
                level,
                rank,
                row["field"],
                sum(
                    1
                    for r in level_pieces[: level_pieces.index(row)]
                    if r["resident_rank"] == rank and r["field"] == row["field"]
                ),
            )
            need(row["array"] == name and name not in names, "piece array identity differs")
            names.add(name)
            expected.add(name)
            value = arrays[name]
            ncomp = width if row["field"] == "rho" else 1
            need(
                value.dtype == np.float64 and value.shape == (ncomp, hi[0] - lo[0], hi[1] - lo[1]),
                "native piece width/shape differs",
            )
            key = rank, boxid
            entry = field_maps.setdefault(key, {})
            need(row["field"] not in entry, "duplicate native field piece")
            entry[row["field"]] = row, value
        box_rows = []
        all_valid = np.zeros(shape, dtype=np.bool_)
        replicated = {row["replicated"] for row in level_pieces}
        need(len(replicated) == 1, "mixed distribution within level")
        for boxid, box in enumerate(boxes):
            residents = {rank for rank, b in field_maps if b == boxid}
            need(residents, "missing native box")
            physical = {field_maps[rank, boxid]["rho"][0]["physical_owner"] for rank in residents}
            need(len(physical) == 1, "box owner differs between replicas")
            owner = physical.pop()
            lo, hi = box[:2], box[2:]
            need(
                not all_valid[lo[0] : hi[0], lo[1] : hi[1]].any(), "overlapping valid native boxes"
            )
            all_valid[lo[0] : hi[0], lo[1] : hi[1]] = True
            box_rows.append(
                {
                    "lo": [int(lo[1]), int(lo[0])],
                    "hi": [int(hi[1] - 1), int(hi[0] - 1)],
                    "owner": owner,
                }
            )
            for rank in residents:
                fields = field_maps[rank, boxid]
                need(set(fields) == {"rho", "active", "kappa"}, "missing native mask/source piece")
                authority = tuple(
                    fields["rho"][0][k]
                    for k in ("physical_owner", "reported_owner", "replicated", "lower", "upper")
                )
                need(
                    all(
                        tuple(
                            fields[k][0][a]
                            for a in (
                                "physical_owner",
                                "reported_owner",
                                "replicated",
                                "lower",
                                "upper",
                            )
                        )
                        == authority
                        for k in fields
                    ),
                    "mask/source authority differs",
                )
                for iy, ix in itertools.product(
                    range(int(lo[0]), int(hi[0])), range(int(lo[1]), int(hi[1]))
                ):
                    offset = iy - int(lo[0]), ix - int(lo[1])
                    x = np.array(
                        [origin[0] + (ix + 0.5) * spacing[0], origin[1] + (iy + 0.5) * spacing[1]]
                    )
                    rows.append(
                        (
                            level,
                            (ix, iy),
                            x,
                            fields["rho"][1][:, offset[0], offset[1]],
                            volumes[iy, ix],
                            fields["kappa"][1][(0,) + offset],
                            fields["active"][1][(0,) + offset],
                            float(not covered[iy, ix]),
                            owner,
                            rank,
                        )
                    )
        need(np.array_equal(valid, all_valid), "valid cells differ from actual boxes")
        topology_rows[level] = dict(
            level=level,
            distribution="replicated" if replicated.pop() else "distributed",
            domain_lo=[0, 0],
            domain_hi=[shape[1] - 1, shape[0] - 1],
            lower=[v.hex() for v in lower],
            upper=[v.hex() for v in upper],
            boxes=box_rows,
        )
        for rank in range(ranks):
            key = prefix + "_rank_%d_history_metadata_json" % rank
            expected.add(key)
            meta = json_array(arrays[key], "history metadata")
            metadata[level, rank] = meta
            if not meta:
                continue
            exact(meta, ("rho_name", "rho", "I_accepted", "I_history"), "history inventory")
            need(type(meta["rho_name"]) is str and meta["rho_name"], "actual keeper name absent")
            for name in ("rho", "I_accepted", "I_history"):
                record = meta[name]
                exact(
                    record,
                    ("native_name", "sample_hex", "depth", "fill_count", "initialized", "slot_dt"),
                    "history metadata",
                )
                need(
                    record["native_name"] == (meta["rho_name"] if name == "rho" else name),
                    "history alias differs",
                )
                need(
                    type(record["depth"]) is int
                    and record["depth"] == 2
                    and type(record["fill_count"]) is int
                    and 0 <= record["fill_count"] <= 2
                    and type(record["initialized"]) is bool
                    and type(record["slot_dt"]) is list
                    and len(record["slot_dt"]) == 2,
                    "physical two-slot history differs",
                )
                samples = history_samples(
                    record["sample_hex"],
                    record["native_name"],
                    level if receipt["adaptive"] else -1,
                    2,
                )
                for slot in range(2):
                    akey = prefix + "_rank_%d_%s_slot_%d" % (rank, name, slot)
                    expected.add(akey)
                    ncomp = width if name == "rho" else len(receipt["components"])
                    need(
                        arrays[akey].dtype == np.float64 and arrays[akey].shape == (ncomp, *shape),
                        "history dense shape differs",
                    )
                    dt = record["slot_dt"][slot]
                    need(
                        type(dt) is float
                        and math.isfinite(dt)
                        and dt >= 0
                        and (samples[slot][0] != 2 or dt == samples[slot][2]),
                        "sample/dt linkage differs",
                    )
                alias = prefix + "_rank_%d_%s" % (rank, "rho_retained" if name == "rho" else name)
                expected.add(alias)
                need(
                    arrays[alias].dtype
                    == arrays[prefix + "_rank_%d_%s_slot_1" % (rank, name)].dtype
                    and arrays[alias].shape
                    == arrays[prefix + "_rank_%d_%s_slot_1" % (rank, name)].shape
                    and arrays[alias].tobytes()
                    == arrays[prefix + "_rank_%d_%s_slot_1" % (rank, name)].tobytes(),
                    "latest history alias differs",
                )
    aux = {key for key in arrays if re.fullmatch(r"rank_[0-9]+_auxiliary_[0-9]+", key)}
    expected.update(aux)
    need(
        aux
        == {
            "rank_%d_auxiliary_%d" % (rank, level)
            for rank in range(ranks)
            for level in (levels if receipt["adaptive"] else (0,))
        },
        "auxiliary rank/level inventory differs",
    )
    for key in aux:
        value = arrays[key]
        need(
            value.dtype == np.uint8 and value.ndim == 1 and value.tobytes().startswith(b"POPSAUX2"),
            "accepted auxiliary bytes absent",
        )
    need(set(arrays) == expected, "closed scientific NPZ inventory differs")
    columns = tuple(zip(*rows, strict=True))
    normalized = {}
    for side in SIDES:
        for ordinal, key in enumerate(
            (
                "level",
                "index",
                "position",
                "values",
                "volume",
                "kappa",
                "active",
                "coverage",
                "owner",
                "resident",
            )
        ):
            normalized[side + "_" + key] = np.asarray(
                columns[ordinal],
                dtype=np.int64 if key in ("level", "index", "owner", "resident") else np.float64,
            )
    return normalized, topology_rows, metadata


def history_checkpoint(arrays, checkpoint, meta, levels, ranks, adaptive):
    """Bind every exported slot and sample to its actual checkpoint counterpart."""
    for level in levels:
        prefix = "level_%d" % level
        key = "state_density_%d" % level if adaptive else "state_density"
        state = checkpoint[key]
        shape = tuple(int(v) for v in arrays[prefix + "_native_cell_shape"])
        width = next(
            arrays[row["array"]].shape[0]
            for row in json_array(arrays["piece_manifest_json"], "pieces")
            if row["level"] == level and row["field"] == "rho"
        )
        need(
            state.dtype == np.float64 and state.size == width * math.prod(shape),
            "checkpoint density dimensions differ",
        )
        state = state.reshape(width, *shape)
        for row in json_array(arrays["piece_manifest_json"], "pieces"):
            if row["level"] != level or row["field"] != "rho":
                continue
            lo, hi = row["lower"], row["upper"]
            need(
                arrays[row["array"]].tobytes() == state[:, lo[0] : hi[0], lo[1] : hi[1]].tobytes(),
                "native source piece/checkpoint bits differ",
            )
        if adaptive:
            geom = levels[level]
            need(
                str(checkpoint["distribution_mode_%d" % level].item()) == geom["distribution"],
                "AMR checkpoint distribution differs",
            )
            owners = checkpoint["dmap_%d" % level]
            need(
                owners.dtype == np.int64
                and owners.ndim == 1
                and owners.tolist()
                == (
                    []
                    if geom["distribution"] == "replicated"
                    else [b["owner"] for b in geom["boxes"]]
                ),
                "AMR checkpoint owners differ",
            )
        aux = "auxiliary_checkpoint_%d" % level if adaptive else "auxiliary_checkpoint"
        for rank in range(ranks):
            need(
                arrays["rank_%d_auxiliary_%d" % (rank, level if adaptive else 0)].tobytes()
                == checkpoint[aux].tobytes(),
                "auxiliary snapshot/checkpoint differs",
            )
            record = meta[level, rank]
            if not record:
                continue
            for alias in ("rho", "I_accepted", "I_history"):
                row = record[alias]
                name = row["native_name"]
                suffix = name + ("_level_%d" % level if adaptive else "")
                need(
                    wire.protocol.scalar(checkpoint["history_depth_" + name], "int") == row["depth"]
                    and wire.protocol.scalar(checkpoint["history_fill_count_" + suffix], "int")
                    == row["fill_count"]
                    and checkpoint["history_init_" + suffix].shape == ()
                    and type(checkpoint["history_init_" + suffix].item()) is bool
                    and checkpoint["history_init_" + suffix].item() == row["initialized"],
                    "history checkpoint metadata differs",
                )
                need(
                    checkpoint["history_sample_identity_" + suffix].tobytes()
                    == bytes.fromhex(row["sample_hex"])
                    and checkpoint["history_slot_dt_" + suffix].tobytes()
                    == np.asarray(row["slot_dt"]).tobytes(),
                    "history checkpoint sample/duration differs",
                )
                for slot in range(2):
                    exported = arrays["level_%d_rank_%d_%s_slot_%d" % (level, rank, alias, slot)]
                    key = (
                        "history_%s_level_%d_%d" % (name, level, slot)
                        if adaptive
                        else "history_%s_%d" % (name, slot)
                    )
                    native = checkpoint[key]
                    need(
                        native.dtype == exported.dtype
                        and native.size == exported.size
                        and native.tobytes() == exported.tobytes(),
                        "history slot/checkpoint bits differ",
                    )


def fixture_receipt(raw, case_id, ranks):
    receipt = strict_json(raw)
    exact(
        receipt,
        (
            "schema",
            "dimension",
            "width",
            "components",
            "cells",
            "adaptive",
            "dt",
            "gamma",
            "workspace",
            "failure",
            "rank",
            "size",
            "phases",
            "checkpoints",
            "failure_errors",
            "fixture_sources",
            "artifact_identity",
            "platform",
            "components_origins",
            "native",
            "consumption_anchors",
            "history_source_contract",
            "raw_coverage_semantics",
            "coordinate_authority",
            "association",
            "aggregate_cryptographic_binding",
            "m26_pde_qualification",
        ),
        "fixture receipt",
    )
    science = case_id in SCIENCE_IDS
    need(
        case_id in (*SCIENCE_IDS, *CONTROL_IDS)
        and receipt["schema"] == FIXTURE
        and type(receipt["rank"]) is int
        and receipt["rank"] == 0
        and receipt["size"] == ranks
        and type(receipt["size"]) is int
        and type(receipt["adaptive"]) is bool
        and receipt["aggregate_cryptographic_binding"] is False
        and receipt["m26_pde_qualification"] is False,
        "fixture identity/scope differs",
    )
    spec = {
        "scalar-cutcell": (1, False),
        "signed-vector-permuted": (3, False),
        "partial-amr-cutcell": (3, True),
    }
    width, adaptive = spec.get(case_id, (1, False))
    need(
        receipt["width"] == width
        and type(receipt["width"]) is int
        and receipt["adaptive"] == adaptive
        and receipt["components"] == ([0] if width == 1 else [2, 0])
        and all(type(v) is int for v in receipt["components"])
        and receipt["failure"] == (None if science else case_id),
        "fixture witness declaration differs",
    )
    phases = PHASES if science else ("initial", "rejected")
    need(
        type(receipt["phases"]) is dict
        and type(receipt["checkpoints"]) is dict
        and set(receipt["phases"]) == set(receipt["checkpoints"]) == set(phases)
        and receipt["consumption_anchors"] == (ANCHORS if science else {})
        and receipt["history_source_contract"] == "pops.spatial-interaction-history-source@1",
        "phase/consumption-anchor closure differs",
    )
    need(
        type(receipt["dt"]) is float
        and receipt["dt"] == 0.01
        and type(receipt["gamma"]) is float
        and receipt["gamma"] == 0.2
        and receipt["cells"] == [8, 6]
        and all(type(v) is int for v in receipt["cells"])
        and type(receipt["workspace"]) is int
        and receipt["workspace"] == (1 if case_id == "budget" else 8 << 20),
        "declared finite witness controls differ",
    )
    need(
        type(receipt["failure_errors"]) is list and (not science or not receipt["failure_errors"]),
        "scientific receipt reports a refused attempt",
    )
    for phase in phases:
        exact(receipt["phases"][phase], ("path", "sha256", "role"), "saved phase")
        exact(receipt["checkpoints"][phase], ("path", "sha256"), "checkpoint pin")
        expected = (
            "SOURCE_SNAPSHOT_ONLY"
            if phase == "initial"
            else "REJECTED_SOURCE_SNAPSHOT"
            if phase == "rejected"
            else "ACTUAL_NATIVE_OUTPUT_AND_NEXT_SOURCE"
        )
        need(receipt["phases"][phase]["role"] == expected, "phase role differs")
    return receipt


def receipt_leaves(receipt):
    leaves = [receipt["native"], *receipt["fixture_sources"]]
    leaves += [{k: row[k] for k in ("path", "sha256")} for row in receipt["phases"].values()]
    leaves += list(receipt["checkpoints"].values())
    need(
        type(receipt["components_origins"]) is list and receipt["components_origins"],
        "component origins absent",
    )
    component_ids = set()
    for row in receipt["components_origins"]:
        program = type(row.get("component")) is str and row["component"].startswith("program-")
        exact(
            row,
            ("component", "binary", "binary_sha256", "sidecar", "sidecar_sha256")
            + (("cpp", "cpp_sha256", "ir", "ir_sha256", "program_hash") if program else ()),
            "component origin",
        )
        need(row["component"] not in component_ids, "duplicate component origin")
        component_ids.add(row["component"])
        if program:
            need(
                type(row["program_hash"]) is str and HEX.fullmatch(row["program_hash"]),
                "actual Program hash absent",
            )
        for key in ("binary", "sidecar", "cpp", "ir") if program else ("binary", "sidecar"):
            leaves.append({"path": row[key], "sha256": row[key + "_sha256"]})
    need(
        sum(name.startswith("program-") for name in component_ids) == 1,
        "fixture requires its actual single layout Program; no guessed aggregate mapping",
    )
    return leaves


def assemble(index_path):
    """Pin only explicit existing files. This command cannot approve anything."""
    index_path = canonical(index_path)
    data = strict_json(index_path.read_bytes())
    exact(
        data,
        ("schema", "capture_class", "mode", "ranks", "cases", "junit", "origins"),
        "capture index",
    )
    need(
        data["schema"] == "sol61.spatial-direct-capture-index@1"
        and data["capture_class"] in ("SOURCE_ONLY", "AUTHENTIC_NATIVE"),
        "capture schema/class differs",
    )
    need(
        type(data["ranks"]) is int
        and data["ranks"] > 0
        and data["mode"] == ("serial" if data["ranks"] == 1 else "mpi"),
        "mode/ranks differ",
    )
    need(all(type(data[k]) is list for k in ("cases", "junit", "origins")), "index list differs")
    need(len(set(data["junit"])) == len(data["junit"]), "duplicate JUnit file")
    roles = set()
    paths = [str(index_path), *data["junit"]]
    for row in data["origins"]:
        exact(row, ("role", "path"), "actual origin")
        need(
            type(row["role"]) is str and row["role"] and row["role"] not in roles,
            "empty/duplicate origin role",
        )
        roles.add(row["role"])
        paths.append(row["path"])
    ids, phase_paths = set(), set()
    for case in data["cases"]:
        exact(case, ("id", "receipt", "abi_key", "bind_identity"), "actual case")
        need(
            case["id"] in (*SCIENCE_IDS, *CONTROL_IDS) and case["id"] not in ids,
            "foreign/duplicate native case",
        )
        ids.add(case["id"])
        paths.append(case["receipt"])
        receipt = fixture_receipt(
            canonical(case["receipt"]).read_bytes(), case["id"], data["ranks"]
        )
        leaves = receipt_leaves(receipt)
        for leaf in leaves:
            exact(leaf, ("path", "sha256"), "actual fixture leaf")
            need(file_pin(leaf["path"]) == leaf, "fixture leaf changed")
            paths.append(leaf["path"])
        for row in (*receipt["phases"].values(), *receipt["checkpoints"].values()):
            need(row["path"] not in phase_paths, "phase/checkpoint path reused")
            phase_paths.add(row["path"])
    need(
        not (
            phase_paths
            & {
                str(index_path),
                *data["junit"],
                *(r["path"] for r in data["origins"]),
                *(r["receipt"] for r in data["cases"]),
            }
        ),
        "phase aliases origin/index/JUnit/receipt",
    )
    leaves = {}
    for path in paths:
        pin = file_pin(path)
        leaves.setdefault(pin["path"], pin)
        need(leaves[pin["path"]] == pin, "ambiguous file pin")
    if data["capture_class"] == "AUTHENTIC_NATIVE":
        need(
            ids == set((*SCIENCE_IDS, *CONTROL_IDS)) and len(data["junit"]) == data["ranks"],
            "native six-case/rank closure differs",
        )
    return dict(
        schema=SCHEMA,
        status="PENDING_ROOT_APPROVAL",
        capture_index=file_pin(index_path),
        files=list(leaves.values()),
        qualification=QUALIFICATION,
        source_commit=None,
        native_build_source_commit=None,
        native_real_bytes=None,
        absolute_tolerance=None,
        relative_tolerance=None,
    )


def closed_junit(index, read):
    ranks = index["ranks"]
    receipts = {row["id"]: row["receipt"] for row in index["cases"]}
    observed = []
    for path in index["junit"]:
        raw = read(path)
        need(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "JUnit entity declaration refused")
        root = ET.fromstring(raw)
        suites = list(root.iter("testsuite"))
        need(
            len(suites) == 1
            and suites[0].attrib.get("tests") == "6"
            and all(suites[0].attrib.get(key) == "0" for key in ("failures", "errors", "skipped")),
            "JUnit aggregate counters differ",
        )
        tests = list(root.iter("testcase"))
        need(
            len(tests) == 6
            and not any(
                list(t.iter(tag)) for t in tests for tag in ("failure", "error", "skipped")
            ),
            "JUnit is not a closed passing six-case batch",
        )
        ids, rank = set(), None
        for test in tests:
            name = test.attrib.get("name", "")
            science = "test_public_spatial_interaction_saved_history_and_exact_replay"
            refusal = "test_public_spatial_interaction_refuses_without_publication"
            match = re.fullmatch("(" + science + "|" + refusal + r")\[([^\]]+)\]", name)
            need(match is not None, "foreign JUnit test")
            case_id = match[2]
            need(
                case_id in (SCIENCE_IDS if match[1] == science else CONTROL_IDS)
                and case_id not in ids,
                "JUnit case identity differs",
            )
            ids.add(case_id)
            props = {}
            for p in test.iter("property"):
                key = p.attrib["name"]
                need(key not in props, "duplicate JUnit property")
                props[key] = p.attrib.get("value", "")
            actual = props.get("rank")
            need(
                actual is not None
                and re.fullmatch("0|[1-9][0-9]*", actual)
                and int(actual) < ranks
                and props.get("size") == str(ranks)
                and props.get("fixture_schema") == FIXTURE
                and props.get("evidence_path") == str(Path(receipts[case_id]).parent),
                "JUnit rank/evidence provenance differs",
            )
            rank = int(actual) if rank is None else rank
            need(int(actual) == rank, "mixed JUnit ranks")
            if case_id in SCIENCE_IDS:
                expected_width = "1" if case_id == SCIENCE_IDS[0] else "3"
                need(
                    props.get("dimension") == "2"
                    and props.get("width") == expected_width
                    and props.get("adaptive") == ("True" if case_id == SCIENCE_IDS[2] else "False"),
                    "JUnit physical declaration differs",
                )
            else:
                need(props.get("failure") == case_id, "JUnit refusal identity differs")
        need(ids == set((*SCIENCE_IDS, *CONTROL_IDS)), "JUnit six-case identity closure differs")
        observed.append(rank)
    need(sorted(observed) == list(range(ranks)), "JUnit rank closure differs")


def interaction_observations(ir, failure=None):
    """Select physical maps through their actual history-store inputs, not node order."""
    need(type(ir) is dict and type(ir.get("nodes")) is list, "actual Program IR absent")
    nodes = {}
    pending = list(ir["nodes"])
    while pending:
        node = pending.pop()
        need(type(node.get("id")) is int and node["id"] not in nodes, "duplicate/invalid IR node")
        nodes[node["id"]] = node
        for key in (
            "true_block",
            "false_block",
            "body_block",
            "cond_block",
            "apply_block",
            "residual_block",
        ):
            pending.extend(node.get("attrs", {}).get(key, ()))
    result = {}
    for name in ("I_accepted", "I_history"):
        stores = [
            n
            for n in nodes.values()
            if n.get("op") == "store_history" and n.get("attrs", {}).get("history") == name
        ]
        need(
            len(stores) == 1 and len(stores[0]["inputs"]) == 1,
            "unique original map observation absent",
        )
        node_id = stores[0]["inputs"][0]
        result[name] = interaction(ir, node_id)
    need(
        result["I_accepted"][0]["id"] != result["I_history"][0]["id"], "map observations collapsed"
    )
    a, h = result["I_accepted"][2], result["I_history"][2]
    need(
        a["contract"] == "pops.spatial-interaction@1"
        and a["source_scope"] == ("issued" if failure == "nonfinite" else "accepted")
        and h["contract"] == "pops.spatial-interaction@2"
        and h["source_scope"] == "issued",
        "accepted/history authority differs",
    )
    need(
        all(
            a[k] == h[k]
            for k in (
                "kernel",
                "measure",
                "quadrature",
                "realization",
                "max_workspace_bytes",
                "components",
                "ncomp",
            )
        ),
        "physical law/measure differs between accepted and history maps",
    )
    return result


def checkpoint_pair(a, b):
    """All physical/diagnostic/history bits strict; two envelope identities are separately sealed."""
    skip = {"pops_checkpoint_manifest", "pops_restart_identity"}
    need(set(a) == set(b), "restart checkpoint inventory differs")
    same_arrays(
        {k: v for k, v in a.items() if k not in skip},
        {k: v for k, v in b.items() if k not in skip},
        "restart checkpoint physical state",
    )
    left, right = (strict_json(str(cp["pops_checkpoint_manifest"].item())) for cp in (a, b))
    need(
        {k: v for k, v in left.items() if k not in ("run_identity", "restart_identity")}
        == {k: v for k, v in right.items() if k not in ("run_identity", "restart_identity")},
        "restart checkpoint non-continuation manifest differs",
    )


def receive(owner_path, owner_sha, approval_path, approval_sha):
    need(
        HEX.fullmatch(owner_sha or "") and HEX.fullmatch(approval_sha or ""),
        "two external ROOT digests required",
    )
    owner = strict_json(pinned(dict(path=str(canonical(owner_path)), sha256=owner_sha)))
    approval = strict_json(pinned(dict(path=str(canonical(approval_path)), sha256=approval_sha)))
    exact(approval, ("schema", "status", "owner_sha256", "qualification"), "ROOT approval")
    need(
        approval
        == dict(
            schema=SCHEMA,
            status="ROOT_APPROVED",
            owner_sha256=owner_sha,
            qualification=QUALIFICATION,
        ),
        "ROOT approval differs",
    )
    exact(
        owner,
        (
            "schema",
            "status",
            "capture_index",
            "files",
            "qualification",
            "source_commit",
            "native_build_source_commit",
            "native_real_bytes",
            "absolute_tolerance",
            "relative_tolerance",
        ),
        "ROOT owner",
    )
    need(
        owner["schema"] == SCHEMA
        and owner["status"] == "ROOT_APPROVED"
        and owner["qualification"] == QUALIFICATION,
        "owner is not approved",
    )
    for key in ("source_commit", "native_build_source_commit"):
        need(
            type(owner[key]) is str and re.fullmatch("[0-9a-f]{40}", owner[key]),
            "source/build commit absent",
        )
    need(
        type(owner["native_real_bytes"]) is int and owner["native_real_bytes"] == 8,
        "this capture profile requires ROOT-attested binary64 native coordinates",
    )
    files = {}
    for leaf in owner["files"]:
        exact(leaf, ("path", "sha256"), "file pin")
        need(
            leaf["path"] not in files and file_pin(leaf["path"]) == leaf,
            "duplicate/changed closed file inventory",
        )
        files[leaf["path"]] = leaf
    index = strict_json(pinned(owner["capture_index"]))
    pending = assemble(owner["capture_index"]["path"])
    need(owner["files"] == pending["files"], "closed actual file inventory differs")
    need(index["capture_class"] == "AUTHENTIC_NATIVE", "SOURCE_ONLY cannot become native evidence")

    def read(path):
        need(path in files, "unsealed evidence file")
        return pinned(files[path])

    closed_junit(index, read)
    roles = {r["role"]: r["path"] for r in index["origins"]}
    need(
        {
            "fixture_source",
            "physical_helper",
            "kernel_emitter_source",
            "consumer_header",
            "history_header",
            "python_package",
            "sdk_manifest",
            "native_library",
        }
        <= set(roles),
        "actual source/package/SDK/native origin missing",
    )
    tolerance = []
    for key in ("absolute_tolerance", "relative_tolerance"):
        need(type(owner[key]) is str, "explicit tolerance absent")
        v = float.fromhex(owner[key])
        need(
            math.isfinite(v) and 0 <= v <= 5e-12 and v.hex() == owner[key],
            "tolerance exceeds declared fixture guard",
        )
        tolerance.append(v)
    atol, rtol = tolerance
    results = []
    for case in index["cases"]:
        receipt = fixture_receipt(read(case["receipt"]), case["id"], index["ranks"])
        need(receipt["native"] == files[roles["native_library"]], "selected Native origin differs")
        need(
            {(r["path"], r["sha256"]) for r in receipt["fixture_sources"]}
            == {
                (p, files[p]["sha256"]) for p in (roles["fixture_source"], roles["physical_helper"])
            },
            "actual fixture/helper source differs",
        )
        program = next(
            r for r in receipt["components_origins"] if r["component"].startswith("program-")
        )
        ir = strict_json(read(program["ir"]))
        maps = interaction_observations(ir, receipt["failure"])
        for _, _, attrs in maps.values():
            need(
                attrs["max_workspace_bytes"]["uint64_hex"] == format(receipt["workspace"], "016x"),
                "IR workspace differs",
            )
            expected_tree = (
                ["div", ["constant", (1.0).hex()], ["sub", ["x", 0], ["y", 0]]]
                if receipt["failure"] == "pole"
                else [
                    "sub",
                    ["add", ["constant", (1.0).hex()], ["mul", ["x", 0], ["y", 1]]],
                    ["mul", ["constant", (2.0).hex()], ["y", 0]],
                ]
            )
            need(
                attrs["kernel"]["tree"] == expected_tree,
                "original declared physical kernel differs",
            )
        observations = maps if case["id"] in SCIENCE_IDS else None
        if observations:
            for _, source, attrs in observations.values():
                need(
                    attrs["components"] == receipt["components"]
                    and attrs["ncomp"] == len(receipt["components"])
                    and len(source["space"]["components"]) == receipt["width"],
                    "IR component selection differs",
                )
        phases = PHASES if observations else ("initial", "rejected")
        images, checkpoint_images, snapshots, meta_by_phase, levels_by_phase = {}, {}, {}, {}, {}
        for phase in phases:
            images[phase] = wire.protocol.archive(read(receipt["phases"][phase]["path"]))
            image = images[phase]
            snapshot, levels, meta = fixture_snapshot(image, receipt, index["ranks"])
            snapshots[phase], levels_by_phase[phase], meta_by_phase[phase] = snapshot, levels, meta
            expected_step = {
                "initial": 0,
                "accepted": 1,
                "continuous": 2,
                "reloaded": 1,
                "replay": 2,
                "rejected": 0,
            }[phase]
            need(
                int(image["step"]) == expected_step
                and float(image["time"]) == expected_step * receipt["dt"],
                "saved native clock differs",
            )
            cp, manifest = wire.envelope(
                read(receipt["checkpoints"][phase]["path"]),
                "initial" if expected_step == 0 else phase,
                case["abi_key"],
                receipt["artifact_identity"],
                case["bind_identity"],
            )
            need(
                manifest["clock"]
                == {"time": float(image["time"]).hex(), "macro_step": expected_step},
                "checkpoint/snapshot clock differs",
            )
            checkpoint_images[phase] = cp
            history_checkpoint(image, cp, meta, levels, index["ranks"], receipt["adaptive"])
            owners = quotient(
                snapshot,
                "source",
                levels,
                index["ranks"],
                receipt["width"],
                "<f8",
                finite_components=receipt["components"],
            )
            if expected_step == 0:
                need(
                    all(not record for record in meta.values()),
                    "initial/refusal invents an evaluated history",
                )
            else:
                for (history_level, _), records in meta.items():
                    for name in ("rho", "I_accepted", "I_history"):
                        row = records[name]
                        samples = history_samples(
                            row["sample_hex"],
                            row["native_name"],
                            history_level if receipt["adaptive"] else -1,
                            2,
                        )
                        need(
                            row["initialized"] is True
                            and row["fill_count"] == min(expected_step, 2),
                            "accepted history maturity differs",
                        )
                        wanted = (
                            (0.0, receipt["dt"])
                            if expected_step == 1
                            else (receipt["dt"], receipt["dt"])
                        )
                        need(
                            samples[1] == (2, *wanted, 1),
                            "latest history publication window differs",
                        )
                        need(
                            samples[0] == (2, 0.0, receipt["dt"], 1),
                            "older history publication window differs",
                        )
            # Physical carrier buffers are finite only on active source cells.
            need(owners, "no authentic source owner")
        if not observations:
            same_arrays(images["initial"], images["rejected"], "refusal accepted image")
            checkpoint_pair(checkpoint_images["initial"], checkpoint_images["rejected"])
            errors = receipt["failure_errors"]
            need(
                type(errors) is list
                and len(errors) == index["ranks"]
                and all(
                    type(e) is list and len(e) == 2 and all(type(v) is str and v for v in e)
                    for e in errors
                ),
                "refusal collective provenance absent",
            )
            results.append(
                dict(
                    id=case["id"], scope="collective refusal; no evaluated I", ranks=index["ranks"]
                )
            )
            continue
        same_arrays(images["accepted"], images["reloaded"], "restart accepted observation")
        same_arrays(images["continuous"], images["replay"], "continuous/replay observation")
        checkpoint_pair(checkpoint_images["accepted"], checkpoint_images["reloaded"])
        checkpoint_pair(checkpoint_images["continuous"], checkpoint_images["replay"])
        max_error = 0.0
        for phase in PHASES[1:]:
            anchor = receipt["consumption_anchors"][phase]
            need(
                levels_by_phase[phase] == levels_by_phase[anchor],
                "topology changed across consumption anchor",
            )
            need(
                images[phase]["topology_epoch"].tobytes()
                == images[anchor]["topology_epoch"].tobytes(),
                "topology epoch changed across consumer",
            )
            current, previous = snapshots[phase], snapshots[anchor]
            need(
                np.array_equal(current["source_index"], previous["source_index"])
                and np.array_equal(current["source_level"], previous["source_level"])
                and np.array_equal(current["source_resident"], previous["source_resident"]),
                "pre/post carrier row authority differs",
            )
            active = (
                (current["source_active"] == 1)
                & (current["source_coverage"] == 1)
                & (current["source_kappa"] > 0)
            )
            reference_rho = (1 - receipt["gamma"] * receipt["dt"]) * previous["source_values"][
                active
            ]
            need(
                np.all(
                    np.abs(current["source_values"][active] - reference_rho)
                    <= atol + rtol * np.abs(reference_rho)
                ),
                "declared zero-flux decay carrier differs",
            )
            for (level, rank), record in meta_by_phase[phase].items():
                need(
                    record["rho_name"] == observations["I_history"][2]["history_source"]["history"],
                    "native keeper name differs from authenticated IR history seed",
                )
                shape = tuple(int(v) for v in images[phase]["level_%d_native_cell_shape" % level])
                keeper = images[phase]["level_%d_rank_%d_rho_slot_1" % (level, rank)]
                seed = snapshots[anchor]
                for row in range(len(seed["source_level"])):
                    if (
                        int(seed["source_level"][row]) == level
                        and int(seed["source_resident"][row]) == rank
                    ):
                        ix, iy = seed["source_index"][row]
                        need(
                            keeper[:, iy, ix].tobytes() == seed["source_values"][row].tobytes(),
                            "retained keeper does not contain the actual pre-commit State.n seed",
                        )
                need(
                    keeper.shape == (receipt["width"], *shape),
                    "retained physical seed width differs",
                )
                for name in ("rho", "I_accepted", "I_history"):
                    older = images[phase]["level_%d_rank_%d_%s_slot_0" % (level, rank, name)]
                    previous = images[phase if anchor == "initial" else anchor][
                        "level_%d_rank_%d_%s_slot_1" % (level, rank, name)
                    ]
                    need(
                        older.dtype == previous.dtype
                        and older.shape == previous.shape
                        and older.tobytes() == previous.tobytes(),
                        "history cold-fill/rotation bits differ",
                    )
            for name, (_, _, attrs) in observations.items():
                source = {k: v.copy() for k, v in snapshots[anchor].items()}
                if name == "I_history" and anchor != "initial":
                    # Warm lag1 belongs to BEFORE rotation. Initial is true cold seed.
                    for row in range(len(source["source_values"])):
                        level = int(source["source_level"][row])
                        rank = int(source["source_resident"][row])
                        ix, iy = source["source_index"][row]
                        source["source_values"][row] = images[anchor][
                            "level_%d_rank_%d_rho_slot_1" % (level, rank)
                        ][:, iy, ix]
                targets = snapshots[phase]
                source.update({k: v for k, v in targets.items() if k.startswith("target_")})
                levelset = levels_by_phase[phase]
                sourcerows = quotient(
                    source,
                    "source",
                    levelset,
                    index["ranks"],
                    receipt["width"],
                    "<f8",
                    finite_components=attrs["components"],
                )
                # Output lives in its own scalar-field ring; never reinterpret rho as I.
                output = np.stack(
                    [
                        images[phase][
                            "level_%d_rank_%d_%s_slot_1"
                            % (
                                int(source["target_level"][r]),
                                int(source["target_resident"][r]),
                                name,
                            )
                        ][:, source["target_index"][r, 1], source["target_index"][r, 0]]
                        for r in range(len(source["target_level"]))
                    ]
                )
                source["target_values"] = output
                targetrows = quotient(
                    source, "target", levelset, index["ranks"], attrs["ncomp"], "<f8"
                )
                reference = convolution(
                    source, attrs["kernel"]["tree"], attrs["components"], 2, sourcerows, targetrows
                )
                actual = output[targetrows]
                active = np.array(
                    [
                        source["target_active"][r] == 1
                        and source["target_coverage"][r] == 1
                        and source["target_kappa"][r] > 0
                        for r in targetrows
                    ]
                )
                error = np.abs(actual[active] - reference[active])
                need(
                    np.all(error <= atol + rtol * np.abs(reference[active])),
                    "original finite convolution differs",
                )
                max_error = max(max_error, float(error.max(initial=0)))
        results.append(
            dict(
                id=case["id"],
                max_absolute_error=max_error,
                ranks=index["ranks"],
                levels=len(levels_by_phase["initial"]),
                observations=["I_accepted", "I_history"],
            )
        )
    return dict(
        schema=SCHEMA,
        qualification=QUALIFICATION,
        scope="finite saved convolution and checkpoint-associated histories",
        cases=results,
        owner_sha256=owner_sha,
        approval_sha256=approval_sha,
        gaps=[
            "Component CPP/IR/DSO association is explicitly ROOT-owner-attested; no aggregate cryptographic link inferred",
            "Origin source/build commits are external ROOT attestations, not inferred from a DSO",
            "No continuum convergence, M26 PDE, fast method, moving geometry or AMR candidate scratch barrier qualification",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("index")
    r = sub.add_parser("receive")
    r.add_argument("owner")
    r.add_argument("--owner-sha256", required=True)
    r.add_argument("--approval", required=True)
    r.add_argument("--approval-sha256", required=True)
    args = parser.parse_args()
    result = (
        assemble(args.index)
        if args.command == "assemble"
        else receive(args.owner, args.owner_sha256, args.approval, args.approval_sha256)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
