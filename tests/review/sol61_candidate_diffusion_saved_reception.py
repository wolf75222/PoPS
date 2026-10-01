"""Candidate-D@1 offline science. No PoPS import, native execution or self-approval."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Identity/archive primitives only. Neither old captured-D science nor its @2
# reception/approval contract is imported or promoted to candidate-D evidence.
wire = load("candidate_d_identity_wire", "sol61_m19_saved_reception.py")
components = load("candidate_d_component_identity", "sol61_physical_global_feedback_offline.py")
need, exact, digest, protocol = wire.need, wire.exact, wire.digest, wire.protocol
SCHEMA = "sol61.candidate-d-owner-pins@1"
QUALIFICATION = "saved-candidate-d-original-residual@1"
ASSOCIATION = "ROOT-attested actual execution/component association; aggregate payload not retained"
CASES = {"scalar1": (1, [0]), "coupled3-201": (3, [2, 0, 1])}
PHASES = ("accepted", "continuous", "reloaded", "replay")
CLOCKS = {
    "accepted": (0.01, 1),
    "continuous": (0.02, 2),
    "reloaded": (0.01, 1),
    "replay": (0.02, 2),
}
DT, TOL = 0.01, 3e-8
POLICY = "pops.field.coefficients.per-candidate@1"
CRITERION = "pops.field.linear.true-correction-residual@1"
CONTROLS = dict(
    tolerance=1e-10,
    max_iterations=20,
    linear_tolerance=1e-8,
    linear_max_iterations=240,
    restart=60,
    armijo=1e-4,
    minimum_step=1 / 1024,
)
SOURCE = "39b2db311148257099e47ec0b305a7d6d2d59e2d"
SOURCE_FINGERPRINTS = {
    "fixture": (
        "tests/python/integration/runtime/test_public_captured_diffusion.py",
        "374b799cbd68e33f9de4d93ee5e0e8eb247c99e47e5e285c4a6d2b2933e97623",
    ),
    "physical_helper": (
        "tests/python/support/captured_diffusion_mms.py",
        "0226d9889de555b20c4a4a97d3ce29ff055fb719c1d6829cc4935240e8ca6f34",
    ),
    "emitter": (
        "python/pops/codegen/program_emit_nonlinear_field.py",
        "729f161c996c9712341241e1f01cb02e3fc479cf4fe447c0062587bcc07decc3",
    ),
    "request_contract": (
        "python/pops/fields/_program_nonlinear_problem.py",
        "cf60a119549e2a095ac4c01dfd29b0d2ee2ee5371e22a73096d6f916dfe84f9f",
    ),
}
MAX_BYTES = 1024**3


def strict_json(raw):
    value = wire.strict_json(raw)

    def finite(node):
        if type(node) is float:
            need(math.isfinite(node), "nonfinite JSON number")
        elif type(node) in (dict, list):
            for item in node.values() if type(node) is dict else node:
                finite(item)

    finite(value)
    return value


def read(path, budget=protocol.MAX_FILE_BYTES):
    path = wire.canonical(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode), "reception input is not a regular file")
        need(0 <= before.st_size <= budget, "file exceeds offline reception budget")
        chunks, position = [], 0
        while position < before.st_size:
            part = os.pread(fd, min(1024**2, before.st_size - position), position)
            need(bool(part), "file truncated during bounded read")
            chunks.append(part)
            position += len(part)
        after = os.fstat(fd)
        attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        need(
            all(getattr(before, a) == getattr(after, a) for a in attributes),
            "file changed during read",
        )
        return path, b"".join(chunks)
    finally:
        os.close(fd)


def leaf(path, budget=protocol.MAX_FILE_BYTES):
    path, raw = read(path, budget)
    return dict(path=str(path), sha256=digest(raw))


def pinned(row, roots, budget=protocol.MAX_FILE_BYTES):
    exact(row, ("path", "sha256"), "file pin")
    need(
        type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
        "invalid SHA pin",
    )
    path = wire.canonical(row["path"])
    need(
        any(path.is_relative_to(wire.canonical(root)) for root in roots),
        "file outside approved roots",
    )
    path, raw = read(path, budget)
    need(digest(raw) == row["sha256"], "external file digest differs")
    return path, raw


def matrices(width):
    need(type(width) is int and width in (1, 3), "foreign witness width")
    rows = (
        ([[".012"]], [["1.1"]])
        if width == 1
        else (
            [[".012", ".002", "0"], ["-.001", ".014", "0"], [".001", "0", "0"]],
            [["1.1", ".03", "-.01"], ["-.02", "1.3", ".02"], [".01", "-.03", "1.5"]],
        )
    )
    return tuple(tuple(tuple(Fraction(v) for v in row) for row in matrix) for matrix in rows)


def original(q, alpha, *, rational=False):
    """One visit per periodic +x/+y face, equal/opposite contributions.

    D_rc at each endpoint depends on that endpoint's COLUMN unknown q_c.
    No roll, emitted action, solver, SPD approximation or matrix symmetrization.
    rational=True evaluates the complete action on exact binary64 inputs using
    rational decimal coefficients; it is an independent accumulation check.
    """
    width, ny, nx = q.shape
    need(alpha.shape == (ny, nx) and ny > 1 and nx > 1, "material/field geometry differs")
    diffusion, reaction = matrices(width)
    cast = Fraction.from_float if rational else float
    values = [
        [[cast(float(q[c, y, x])) for x in range(nx)] for y in range(ny)] for c in range(width)
    ]
    material = [[cast(float(alpha[y, x])) for x in range(nx)] for y in range(ny)]
    zero = Fraction(0) if rational else 0.0
    spatial = [[[zero for _ in range(nx)] for _ in range(ny)] for _ in range(width)]
    for y in range(ny):
        for x in range(nx):
            for yy, xx, cells in (((y + 1) % ny, x, ny), (y, (x + 1) % nx, nx)):
                for row in range(width):
                    flux = zero
                    for column in range(width):
                        left, right = values[column][y][x], values[column][yy][xx]
                        d = diffusion[row][column] if rational else float(diffusion[row][column])
                        dl = d * (1 + material[y][x]) * (1 + 3 * left * left)
                        dr = d * (1 + material[yy][xx]) * (1 + 3 * right * right)
                        flux += ((dl + dr) / 2) * (right - left) * cells * cells
                    spatial[row][y][x] -= flux
                    spatial[row][yy][xx] += flux
    action = []
    for row in range(width):
        plane = []
        for y in range(ny):
            line = []
            for x in range(nx):
                local = sum(
                    (reaction[row][c] if rational else float(reaction[row][c])) * values[c][y][x]
                    for c in range(width)
                )
                cubic = (Fraction(1, 5) if rational else 0.2) * values[row][y][x] ** 3
                line.append(float(spatial[row][y][x] + local + cubic))
            plane.append(line)
        action.append(plane)
    if rational:
        need(
            all(sum((v for line in plane for v in line), Fraction(0)) == 0 for plane in spatial),
            "unique-face rational conservation failed",
        )
    return np.array(action, dtype=np.float64), np.array(spatial, dtype=np.float64)


def prescribed(n, width):
    q, alpha = np.empty((width, n, n)), np.empty((n, n))
    for y in range(n):
        for x in range(n):
            px, py = 2 * math.pi * (x + 0.5) / n, 2 * math.pi * (y + 0.5) / n
            alpha[y, x] = 0.25 * math.sin(px) + 0.15 * math.cos(py)
            for c in range(width):
                q[c, y, x] = 0.15 + 0.025 * math.cos((c + 1) * px) + 0.02 * math.sin(py)
    return q, alpha


def same(a, b, label):
    need(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
        label + " differs in bytes",
    )


def array(value, shape, label):
    need(
        type(value) is np.ndarray
        and value.dtype == np.dtype("float64")
        and value.shape == shape
        and np.isfinite(value).all(),
        "invalid finite binary64 array: " + label,
    )


def norm(a):
    return math.sqrt(math.fsum(float(v) * float(v) for v in a.flat))


def science(initial, states, width, n=16):
    exact(initial, ("response", "forcing", "material", "target"), "initial NPZ")
    exact(states, PHASES, "phase inventory")
    target, alpha = prescribed(n, width)
    shape = (width, n, n)
    for key in ("response", "forcing", "target"):
        array(initial[key], shape, "initial " + key)
    array(initial["material"], (1, n, n), "initial material")
    need(
        np.max(np.abs(initial["target"] - target)) < 2e-14
        and np.max(np.abs(initial["material"][0] - alpha)) < 2e-14,
        "declared initial centre samples differ",
    )
    need(
        initial["response"].tobytes() == np.zeros(shape).tobytes(),
        "initial response is not positive zero",
    )
    forcing, _ = original(initial["target"], initial["material"][0], rational=True)
    need(
        norm(forcing - initial["forcing"]) / norm(forcing) < 2e-13,
        "original candidate-D manufactured load differs",
    )
    need(
        np.ptp(initial["material"]) > 0.5 and all(np.ptp(v) > 0.07 for v in initial["target"]),
        "nonconstant witness was replaced",
    )
    reports = {}
    for phase, saved in states.items():
        exact(saved, ("response", "forcing", "material", "solution", "time", "step"), "phase NPZ")
        time, step = CLOCKS[phase]
        need(
            protocol.scalar(saved["time"], "real").hex() == time.hex()
            and protocol.scalar(saved["step"], "int") == step,
            "exact saved phase clock differs",
        )
        for key in ("response", "forcing", "solution"):
            array(saved[key], shape, phase + " " + key)
        array(saved["material"], (1, n, n), phase + " material")
        same(saved["material"], initial["material"], "readonly alpha")
        same(saved["forcing"], initial["forcing"], "readonly forcing")
        action, spatial = original(saved["solution"], saved["material"][0])
        rational, _ = original(saved["solution"], saved["material"][0], rational=True)
        need(np.max(np.abs(action - rational)) < 2e-13, "float/Fraction action disagreement")
        residual = norm(rational - saved["forcing"]) / norm(saved["forcing"])
        error = float(np.max(np.abs(saved["solution"] - initial["target"])))
        consumer = float(np.max(np.abs(saved["response"] / time - saved["solution"])))
        need(
            error < TOL and residual < TOL and consumer < TOL,
            "candidate-D original science guard failed",
        )
        need(np.max(np.abs(spatial)) > 0.01, "diffusion action absent")
        reports[phase] = dict(
            solution_linf=error,
            original_residual_relative_l2=residual,
            consumer_linf=consumer,
            spatial_linf=float(np.max(np.abs(spatial))),
        )
    for left, right in (("accepted", "reloaded"), ("continuous", "replay")):
        for key in states[left]:
            same(states[left][key], states[right][key], left + "/" + right + " " + key)
    return reports


def diagnostic_images(arrays, ranks):
    need(
        {"program_diagnostics_state", "program_diagnostics_offsets"} <= set(arrays),
        "Native diagnostic image absent",
    )
    raw, offsets = arrays["program_diagnostics_state"], arrays["program_diagnostics_offsets"]
    need(
        raw.dtype == np.uint8
        and raw.ndim == 1
        and offsets.dtype == np.int64
        and offsets.shape == (ranks + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(raw)
        and np.all(offsets[1:] > offsets[:-1]),
        "diagnostic offsets/rank count differs",
    )
    tables = []
    for rank in range(ranks):
        image = raw[int(offsets[rank]) : int(offsets[rank + 1])].tobytes()
        need(len(image) >= 40 and image[:8] == b"POPSDIA1", "diagnostic header differs")
        width, actual_rank, actual_ranks, count = struct.unpack_from("<QQQQ", image, 8)
        need(
            (width, actual_rank, actual_ranks) == (64, rank, ranks)
            and count <= (len(image) - 40) // 16,
            "diagnostic width/rank/count differs",
        )
        table, cursor, previous = {}, 40, None
        for _ in range(count):
            need(cursor + 8 <= len(image), "truncated diagnostic length")
            length = int.from_bytes(image[cursor : cursor + 8], "little")
            cursor += 8
            need(length <= len(image) - cursor - 8, "diagnostic name exceeds actual bytes")
            name = image[cursor : cursor + length]
            cursor += length
            bits = image[cursor : cursor + 8]
            cursor += 8
            need(
                (previous is None or previous < name) and not name.startswith(b"pops.balance-term"),
                "duplicate/unordered/reserved diagnostic name",
            )
            table[name] = bits
            previous = name
        need(cursor == len(image), "diagnostic trailing bytes")
        tables.append(table)
    return tables


def diagnostic_science(tables, names):
    suffixes = (
        "residual_norm",
        "reference_residual_norm",
        "rel_residual",
        "full_residual_evaluations",
        "finite_difference_jvps",
    )
    need(set(names) == set(suffixes), "original solve diagnostic inventory differs")
    for table in tables:
        need(all(names[k].encode() in table for k in suffixes), "original solve diagnostic absent")
        values = {key: struct.unpack("<d", table[names[key].encode()])[0] for key in suffixes}
        need(
            all(math.isfinite(v) and v >= 0 for v in values.values())
            and values["reference_residual_norm"] > 0
            and values["rel_residual"] <= CONTROLS["tolerance"],
            "Native original solve diagnostic guard differs",
        )
        ratio = values["residual_norm"] / values["reference_residual_norm"]
        need(
            abs(ratio - values["rel_residual"])
            <= 8 * np.finfo(float).eps * max(ratio, values["rel_residual"], 1e-300),
            "Native residual diagnostic ratio differs",
        )
        evaluations, jvps = values["full_residual_evaluations"], values["finite_difference_jvps"]
        need(
            evaluations.is_integer() and jvps.is_integer() and evaluations > jvps > 0,
            "Native evaluation diagnostics differ",
        )


def history_identity(raw, name, step):
    header = b"POPSHID1" + struct.pack("<Q", len(name)) + name.encode() + struct.pack("<qQ", -1, 2)
    expected = header + b"".join(
        struct.pack(
            "<QQQQ",
            2,
            int.from_bytes(struct.pack("<d", start), "little"),
            int.from_bytes(struct.pack("<d", DT), "little"),
            1,
        )
        for start in (0.0, (step - 1) * DT)
    )
    need(raw == expected, "history exact issued duration/publication point differs")


def checkpoint(raw, phase, saved, first, artifact, abi, ranks, names):
    arrays, manifest = wire.envelope(raw, "accepted", abi, artifact=artifact)
    need(
        manifest["runtime_kind"] == "uniform"
        and protocol.scalar(arrays["pops_checkpoint_version"], "int") == 8,
        "checkpoint version/runtime differs",
    )
    time, step = CLOCKS[phase]
    need(
        protocol.scalar(arrays["t"], "real").hex() == time.hex()
        and protocol.scalar(arrays["macro_step"], "int") == step,
        "checkpoint phase clock differs",
    )
    geometry = strict_json(str(arrays["pops_spatial_contract"].item()))
    exact(
        geometry,
        (
            "schema_version",
            "dimension",
            "shape",
            "lower",
            "upper",
            "periodicity",
            "refinement_ratios",
            "native_layout_identity",
            "identity",
        ),
        "spatial contract",
    )
    need(
        type(geometry["schema_version"]) is int
        and geometry["schema_version"] == 1
        and type(geometry["dimension"]) is int
        and geometry["dimension"] == 2
        and geometry["shape"] == [16, 16]
        and all(type(v) is int for v in geometry["shape"])
        and geometry["lower"] == [(0.0).hex()] * 2
        and geometry["upper"] == [(1.0).hex()] * 2
        and geometry["periodicity"] == [True, True]
        and all(type(v) is bool for v in geometry["periodicity"])
        and geometry["refinement_ratios"] == [],
        "checkpoint physical geometry differs",
    )
    need(
        re.fullmatch(
            r"pops\.native-spatial-layout\.v1:sha256:[0-9a-f]{64}",
            geometry["native_layout_identity"],
        ),
        "native layout identity absent",
    )
    need(
        geometry["identity"]
        == "pops.checkpoint-spatial-layout.v1:sha256:"
        + wire.identity_hash(
            "checkpoint-spatial-layout", {k: v for k, v in geometry.items() if k != "identity"}
        ),
        "spatial identity differs",
    )
    width = saved["solution"].shape[0]
    need(
        list(arrays["blocks"]) == ["response", "forcing", "material"],
        "checkpoint block order differs",
    )
    for block, labels in (
        ("response", [f"u{i}" for i in range(width)]),
        ("forcing", [f"f{i}" for i in range(width)]),
        ("material", ["alpha"]),
    ):
        need(
            list(arrays["names_" + block]) == labels
            and protocol.scalar(arrays["ncomp_" + block], "int") == len(labels),
            "component ordering differs",
        )
        value = arrays["state_" + block]
        need(
            value.dtype == np.float64 and value.size == len(labels) * 256,
            "checkpoint physical shape differs",
        )
        same(value.reshape(len(labels), 16, 16), saved[block], "checkpoint physical " + block)
    need(
        list(arrays["history_names"]) == [f"q{i}" for i in range(width)], "history registry differs"
    )
    for i in range(width):
        name = f"q{i}"
        need(
            protocol.scalar(arrays["history_depth_" + name], "int") == 2
            and protocol.scalar(arrays["history_ncomp_" + name], "int") == 1
            and arrays["history_init_" + name].item() is True
            and protocol.scalar(arrays["history_fill_count_" + name], "int") == min(step, 2)
            and arrays["history_stored_slots_" + name].dtype == np.int64
            and arrays["history_stored_slots_" + name].tolist() == [0, 1],
            "history accepted storage differs",
        )
        same(arrays["history_slot_dt_" + name], np.array([DT, DT]), "history durations")
        sample = arrays["history_sample_identity_" + name]
        need(sample.dtype == np.uint8 and sample.ndim == 1, "history identity storage differs")
        history_identity(sample.tobytes(), name, step)
        for slot, expected in enumerate((first["solution"][i], saved["solution"][i])):
            value = arrays[f"history_{name}_{slot}"]
            need(value.dtype == np.float64 and value.size == 256, "history geometry differs")
            same(value.reshape(16, 16), expected, "history solution")
    diagnostic_science(diagnostic_images(arrays, ranks), names)
    auxiliary = arrays["auxiliary_checkpoint"]
    need(
        auxiliary.dtype == np.uint8
        and auxiliary.ndim == 1
        and auxiliary.size >= 8
        and auxiliary[:8].tobytes() == b"POPSAUX2",
        "exact accepted auxiliary image absent",
    )
    exchanges, offsets = arrays["program_exchange_state"], arrays["program_exchange_offsets"]
    need(
        exchanges.dtype == np.uint8
        and exchanges.ndim == 1
        and offsets.dtype == np.int64
        and offsets.shape == (ranks + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(exchanges)
        and np.all(offsets[1:] > offsets[:-1]),
        "exchange rank offsets differ",
    )
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        image = exchanges[int(lo) : int(hi)].tobytes()
        need(
            image in (b"POPSEX01" + bytes(8), b"POPSEX02" + bytes(24)),
            "unexpected accepted physical exchange/consumption",
        )
    temporal = strict_json(str(arrays["temporal_restart_state"].item()))
    need(
        temporal["clock"] == dict(time=time.hex(), macro_step=step)
        and temporal["status"] == "accepted"
        and temporal["synchronized"] is True
        and temporal["controller_state"]["last_accepted_dt"] == DT.hex(),
        "temporal accepted window differs",
    )
    for key in (
        "clock_cursors",
        "schedule_cursors",
        "synchronization_cursors",
        "history_cursors",
        "cache_cursors",
    ):
        need(type(temporal[key]) is dict, "temporal cursor map absent")
        for cursor in temporal[key].values():
            need(
                type(cursor) is dict and cursor.get("phase") == "accepted",
                "unaccepted temporal cursor",
            )
            if "time" in cursor:
                need(cursor["time"] == time.hex(), "temporal cursor clock differs")
            if "macro_step" in cursor:
                need(cursor["macro_step"] == step, "temporal cursor step differs")
    return arrays


def scalar(value):
    if type(value) in (int, float):
        need(type(value) is int or math.isfinite(value), "nonfinite canonical control")
        return value
    need(type(value) is dict, "control is not a canonical scalar")
    if set(value) == {"scalar"}:
        return scalar(value["scalar"])
    exact(value, ("kind", "value"), "canonical control")
    text = value["value"]
    need(type(text) is str, "canonical scalar payload is not text")
    if value["kind"] == "integer":
        need(re.fullmatch("-?(0|[1-9][0-9]*)", text), "noncanonical integer control")
        return int(text)
    need(value["kind"] == "binary64", "unsupported canonical control kind")
    result = float.fromhex(text)
    need(math.isfinite(result) and result.hex() == text, "noncanonical binary64 control")
    return result


def polynomial_contract(source, capture_nodes, width, order):
    """Normalize the closed documentary AST to exact rational polynomials.

    This authenticates the stated equation, not a C++-to-DSO compiled graph.
    Variables are alpha, physical q0.., then physical forcing f0...
    """
    dimension = 1 + 2 * width
    zero = (0,) * dimension

    def constant(value):
        return {} if value == 0 else {zero: value}

    def variable(index):
        powers = list(zero)
        powers[index] = 1
        return {tuple(powers): Fraction(1)}

    def add(a, b):
        result = a.copy()
        for powers, value in b.items():
            result[powers] = result.get(powers, Fraction(0)) + value
            if result[powers] == 0:
                del result[powers]
        return result

    def multiply(a, b):
        result = {}
        for p, x in a.items():
            for q, y in b.items():
                exponent = tuple(i + j for i, j in zip(p, q, strict=True))
                result = add(result, {exponent: x * y})
        return result

    def negate(a):
        return {k: -v for k, v in a.items()}

    unknowns = source["unknown_components"]
    need(
        len(unknowns) == width and [v["local_id"] for v in unknowns] == ["q%d" % i for i in order],
        "original unknown permutation differs",
    )
    roles = [node["state"]["local_id"] for node in capture_nodes]
    need(
        set(roles) == {"forcing", "material"} and len(roles) == 2,
        "physical readonly capture roles differ",
    )

    def evaluate(node):
        need(type(node) is list and bool(node), "expression is not a closed scalar AST")
        op = node[0]
        if op == "literal" and len(node) == 2:
            data = node[1]
            need(type(data) is dict, "invalid field literal")
            kind = data["kind"]
            if kind == "rational":
                value = Fraction(int(data["numerator"]), int(data["denominator"]))
            elif kind == "decimal":
                value = Fraction(data["value"])
            else:
                value = Fraction(scalar(data))
            return constant(value)
        if op == "unknown" and len(node) == 3:
            i = node[1]
            need(
                type(i) is int and 0 <= i < width and node[2] == unknowns[i],
                "unknown expression authority differs",
            )
            return variable(1 + order[i])
        if op == "input" and len(node) == 4:
            index, c = node[1:3]
            need(
                type(index) is int
                and 0 <= index < 2
                and type(c) is int
                and node[3] == capture_nodes[index]["state"],
                "input expression authority differs",
            )
            need(
                (roles[index] == "material" and c == 0)
                or (roles[index] == "forcing" and 0 <= c < width),
                "captured component differs",
            )
            return variable(0 if roles[index] == "material" else 1 + width + c)
        if op == "neg" and len(node) == 2:
            return negate(evaluate(node[1]))
        if op in ("add", "sub", "mul", "pow") and len(node) == 3:
            a, b = evaluate(node[1]), evaluate(node[2])
            if op == "add":
                return add(a, b)
            if op == "sub":
                return add(a, negate(b))
            if op == "mul":
                return multiply(a, b)
            need(set(b) <= {zero}, "nonconstant field power")
            exponent = b.get(zero, Fraction(0))
            need(exponent.denominator == 1 and 0 <= exponent <= 3, "unsupported field power")
            result = constant(Fraction(1))
            for _ in range(int(exponent)):
                result = multiply(result, a)
            return result
        raise ValueError("foreign operation in original documentary equation")

    diffusion, reaction = matrices(width)
    alpha = add(constant(Fraction(1)), variable(0))
    for row, r in enumerate(order):
        for column, c in enumerate(order):
            q = variable(1 + c)
            coefficient = multiply(
                constant(Fraction.from_float(float(diffusion[r][c]))),
                multiply(
                    alpha,
                    add(constant(Fraction(1)), multiply(constant(Fraction(3)), multiply(q, q))),
                ),
            )
            need(
                evaluate(source["diffusion"][row * width + column]) == coefficient,
                "original candidate-D documentary coefficient differs",
            )
        expected = negate(variable(1 + width + r))
        for c in range(width):
            expected = add(
                expected,
                multiply(constant(Fraction.from_float(float(reaction[r][c]))), variable(1 + c)),
            )
        q = variable(1 + r)
        expected = add(
            expected, multiply(constant(Fraction.from_float(0.2)), multiply(q, multiply(q, q)))
        )
        need(
            evaluate(source["local_expressions"][row]) == expected,
            "original reaction/load documentary equation differs",
        )


def cpp_contract(raw, width, ir_raw=None, *, require_ir=True):
    code = raw.decode("utf-8")
    if ir_raw is not None:
        ir = strict_json(ir_raw)
    elif components.MARKER in code:
        ir, code = components.ir_from_cpp(code)
    else:
        need(not require_ir, "exact compiled Program IR was not retained")
        return cpp_route(code, width), None
    ir_contract(ir, width)
    return cpp_route(code, width), ir


def ir_contract(ir, width):
    need(
        type(ir["version"]) is int
        and ir["version"] == 11
        and ir["name"] == "captured-diffusion-accepted-response",
        "candidate Program IR version/name differs",
    )
    solves = [node for node in ir["nodes"] if node["op"] == "solve_spatial_field"]
    need(len(solves) == 1, "original field solve occurrence differs")
    attrs = solves[0]["attrs"]
    source = attrs["source_contract"]
    need(
        {"newton_controls", "finite_difference_step", "capture_count", "seed_index", "problem_kind"}
        <= set(attrs),
        "original solver/capture metadata absent",
    )
    exact(attrs["newton_controls"], CONTROLS, "IR seven controls")
    controls = {k: scalar(v) for k, v in attrs["newton_controls"].items()}
    need(
        controls == CONTROLS
        and all(type(controls[k]) is type(v) for k, v in CONTROLS.items())
        and scalar(attrs["finite_difference_step"]) == 1e-6
        and attrs["seed_index"] is None
        and "right_preconditioner" not in attrs
        and attrs["problem_kind"] == "original_field_equations",
        "original solver/FD/initialization realization differs",
    )
    need(
        attrs["solver_identity"]
        == components.identity("prepared-spatial-newton", attrs["newton_controls"]),
        "prepared solver identity differs",
    )
    need(
        type(attrs["capture_count"]) is int and attrs["capture_count"] == 2,
        "readonly capture inventory differs",
    )
    by_id = {node["id"]: node for node in ir["nodes"]}
    need(len(by_id) == len(ir["nodes"]), "duplicate IR node identity")
    point = dict(
        schema_version=1, clock=ir["clock"], step=0, offset=dict(kind="integer", value="0")
    )
    captures = solves[0]["inputs"][2:4]
    need(
        len(captures) == 2
        and all(by_id[i]["op"] == "state" and by_id[i]["point"] == point for i in captures),
        "captures are not readonly State at point n",
    )
    stage = solves[0]["point"]
    need(
        stage["name"] == "frozen-original-coefficients"
        and type(stage["partitions"]) is dict
        and bool(stage["partitions"])
        and all(p == point for p in stage["partitions"].values()),
        "candidate field stage point differs",
    )
    order = [0] if width == 1 else [2, 0, 1]
    polynomial_contract(source, [by_id[i] for i in captures], width, order)
    need(
        attrs["contract"] == "pops.spatial-field-residual@3"
        and attrs["ncomp"] == width
        and attrs["coefficient_evaluation"] == source["coefficient_evaluation"] == POLICY
        and attrs["linear_residual_verification"]
        == source["linear_residual_verification"]
        == CRITERION
        and source["coefficient_face_policy"] == "pops.field.face-mean.arithmetic@1",
        "candidate original request authority differs",
    )
    need(
        len(source["diffusion"]) == width**2
        and len(source["local_expressions"]) == width
        and "temporal_tau" not in source
        and "evolved_stage" not in source,
        "foreign equation/stage scope",
    )


def cpp_route(code, width):
    controls = re.findall(r"pops::FieldNewtonOptions\{([^}]+)\}", code)
    need(bool(controls), "seven original Newton controls absent from retained CPP")
    for group in controls:
        rows = re.findall(r"\.([a-z_]+)\s*=\s*([^,]+)(?:,|$)", group)
        need(len(rows) == 7 and len(dict(rows)) == 7, "CPP controls duplicate/incomplete")
        parsed = {
            key: int(value.strip())
            if key in ("max_iterations", "linear_max_iterations", "restart")
            else float(value.strip())
            for key, value in rows
        }
        need(parsed == CONTROLS, "retained CPP seven controls differ")
    fd = re.findall(
        r"pops::FieldNewtonOptions\{[^}]+\},\s*([^,]+),\s*true,\s*&ctx.prepared_execution_lane\(\),\s*true",
        code,
    )
    need(
        len(fd) == len(controls) and all(float(v.strip()) == 1e-6 for v in fd),
        "retained CPP FD/correction authority differs",
    )
    signature = rf"prepare_general_field_coefficients\s*<\s*pops::kNativeDimension\s*,\s*{width}\s*,\s*{width**2}\s*,\s*false\s*,\s*true\s*>"
    need(
        re.search(signature, code)
        and "candidate coefficient point/attempt/lane authority changed" in code
        and "candidate diffusion local evaluation" in code
        and "nonfinite_original_field_residual" in code,
        "candidate coefficient evaluation/recheck route absent",
    )
    options = re.findall(
        r"PreparedSpatialResidual<pops::kNativeDimension>>\([^;]+,\s*true,\s*&ctx.prepared_execution_lane\(\),\s*true\);",
        code,
    )
    need(len(options) == 1, "true correction residual option absent/duplicate")
    records = re.findall(r'ctx\.record_scalar\("(field_residual_[0-9]+)\.([a-z_]+)"', code)
    need(
        len(records) == 5
        and len({stem for stem, _ in records}) == 1
        and len({suffix for _, suffix in records}) == 5,
        "original diagnostic CPP emission differs",
    )
    return tuple(stem + "." + suffix for stem, suffix in records)


def case_inventory(directory):
    directory = wire.canonical(directory)
    receipt = strict_json(read(directory / "receipt.json")[1])
    files = {directory / "receipt.json"}
    initial = leaf(receipt["initial_npz"])
    need(initial["sha256"] == receipt["initial_sha256"], "initial receipt hash differs")
    files.add(Path(initial["path"]))
    exact(receipt["phases"], PHASES, "phase receipt")
    phases = {}
    for phase, row in receipt["phases"].items():
        exact(row, ("npz", "sha256", "checks"), "phase receipt row")
        phases[phase] = leaf(row["npz"])
        need(phases[phase]["sha256"] == row["sha256"], "phase receipt hash differs")
        files.add(Path(phases[phase]["path"]))
    exact(receipt["checkpoints"], ("accepted", "continuous", "replay"), "checkpoint receipt")
    checkpoints = {phase: leaf(row["path"]) for phase, row in receipt["checkpoints"].items()}
    need(checkpoints == receipt["checkpoints"], "checkpoint receipt hash differs")
    files.update(Path(row["path"]) for row in checkpoints.values())
    need(
        type(receipt["sources"]) is list and len(receipt["sources"]) == 1,
        "actual retained CPP missing",
    )
    source = receipt["sources"][0]
    exact(source, ("component", "path", "sha256"), "retained source receipt")
    cpp = leaf(source["path"])
    need(cpp["sha256"] == source["sha256"], "retained source hash differs")
    files.add(Path(cpp["path"]))
    need(
        all(checkpoints[p] == phases[p] for p in checkpoints),
        "historical overwritten checkpoint aliases differ",
    )
    ir = None
    need(
        len(files) == 7
        and all(p.parent == directory for p in files)
        and set(directory.iterdir()) == files,
        "closed candidate case inventory differs",
    )
    return dict(
        directory=str(directory),
        artifact=receipt["artifact"],
        receipt=leaf(directory / "receipt.json"),
        initial=initial,
        phases=phases,
        checkpoints=checkpoints,
        cpp=cpp,
        ir=ir,
    )


def receipt_contract(receipt, width, order, ranks, owner):
    exact(
        receipt,
        (
            "kind",
            "fixture_schema",
            "artifact",
            "dimension",
            "rank",
            "size",
            "cells",
            "width",
            "order",
            "face_policy",
            "newton",
            "fd_step",
            "solution_tolerance",
            "residual_tolerance",
            "native",
            "platform",
            "binaries",
            "sources",
            "initial_npz",
            "initial_sha256",
            "phases",
            "checkpoints",
            "exact_restart_and_replay",
            "coefficient_evaluation",
            "linear_residual_verification",
            "candidate_beta",
        ),
        "candidate receipt",
    )
    need(
        receipt["fixture_schema"] == "pops.candidate-diffusion-native-fixture@1"
        and receipt["kind"] == "actual-native-candidate-D-original-MMS",
        "frozen-D receipt cannot qualify candidate D",
    )
    need(
        all(type(receipt[k]) is int for k in ("dimension", "rank", "size", "cells", "width"))
        and (
            receipt["dimension"],
            receipt["rank"],
            receipt["size"],
            receipt["cells"],
            receipt["width"],
        )
        == (2, 0, ranks, 16, width),
        "candidate case dimensions/ranks differ",
    )
    need(
        type(receipt["order"]) is list
        and all(type(v) is int for v in receipt["order"])
        and receipt["order"] == order,
        "unknown permutation differs",
    )
    exact(receipt["newton"], CONTROLS, "seven Newton controls")
    need(
        receipt["newton"] == CONTROLS
        and all(type(receipt["newton"][k]) is type(v) for k, v in CONTROLS.items()),
        "Newton controls changed",
    )
    need(
        receipt["face_policy"] == "pops.field.face-mean.arithmetic@1"
        and receipt["coefficient_evaluation"] == POLICY
        and receipt["linear_residual_verification"] == CRITERION
        and type(receipt["candidate_beta"]) is float
        and receipt["candidate_beta"] == 3.0
        and receipt["fd_step"] == 1e-6
        and receipt["solution_tolerance"] == receipt["residual_tolerance"] == TOL
        and receipt["exact_restart_and_replay"] is True,
        "candidate method/guards differ",
    )
    need(receipt["native"] == owner["native"], "selected native owner differs")


def origins(owner, roots):
    exact(
        owner,
        (
            "schema",
            "source_commit",
            "native_build_source_commit",
            "abi_key",
            "python_package",
            "sdk",
            "native",
            "sources",
            "execution_association",
        ),
        "execution owner",
    )
    need(owner["schema"] == "sol61.candidate-d-execution-owner@1", "owner schema differs")
    need(
        all(
            type(owner[k]) is str and re.fullmatch("[0-9a-f]{40}", owner[k])
            for k in ("source_commit", "native_build_source_commit")
        ),
        "source/build source identity absent",
    )
    need(type(owner["abi_key"]) is str and owner["abi_key"], "ABI absent")
    for key in ("python_package", "sdk", "native"):
        pinned(owner[key], roots, MAX_BYTES if key == "native" else protocol.MAX_FILE_BYTES)
    exact(owner["sources"], SOURCE_FINGERPRINTS, "four source identities")
    for key, (_, sha) in SOURCE_FINGERPRINTS.items():
        need(
            digest(pinned(owner["sources"][key], roots)[1]) == sha,
            "reviewed source fingerprint differs: " + key,
        )
    exact(owner["execution_association"], CASES, "ROOT execution association")


def linkage(rows, receipt, cpp, roots):
    need(type(rows) is dict, "execution association absent")
    exact(rows, ("authority", "aggregate_artifact", "components"), "case association")
    need(
        rows["authority"] == ASSOCIATION and rows["aggregate_artifact"] == receipt["artifact"],
        "unattested aggregate execution association",
    )
    binaries = receipt["binaries"]
    need(
        type(binaries) is list
        and len(binaries) == 4
        and len({r["component"] for r in binaries}) == 4,
        "actual binary component inventory differs",
    )
    by_name = {row["component"]: row for row in binaries}
    program = receipt["sources"][0]["component"]
    need(
        program.startswith("program-")
        and set(by_name) == {"block-response", "block-forcing", "block-material", program},
        "physical component owners differ",
    )
    exact(rows["components"], by_name, "component associations")
    result = {}
    for name, row in rows["components"].items():
        exact(row, ("so", "sidecar", "cpp"), "component association")
        binary = by_name[name]
        exact(binary, ("component", "path", "sha256", "compile_command"), "binary receipt")
        need(
            row["so"] == {k: binary[k] for k in ("path", "sha256")}, "association actual SO differs"
        )
        so = pinned(row["so"], roots, MAX_BYTES)[1]
        sidecar = pinned(row["sidecar"], roots)[1]
        result[name] = components.component(so, sidecar)
        if name == program:
            need(row["cpp"] == cpp, "actual Program CPP association differs")
            source = pinned(cpp, roots)[1].decode("utf-8")
            need(
                re.findall(
                    r'extern "C" const char\* pops_program_hash\(\) \{ return "([0-9a-f]{64})"; \}',
                    source,
                )
                == [result[name]["semantic_identity"].rsplit(":", 1)[1]],
                "actual CPP/Program semantic identity differs",
            )
        else:
            need(row["cpp"] is None, "unretained block source cannot be fabricated")
    return result


def junit(raw, rank, ranks, cases):
    root = ET.fromstring(raw)
    tests = root.findall(".//testcase")
    for suite in root.iter("testsuite"):
        children = suite.findall("testcase")
        need(suite.get("tests") == str(len(children)), "JUnit count differs")
        for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            need(
                suite.get(key, "0") == str(sum(bool(t.findall(tag)) for t in children)),
                "JUnit summary differs",
            )
    prefix = "test_public_candidate_diffusion_nonconstant_saved_and_exact_replay["
    selected = [t for t in tests if t.get("name", "").startswith(prefix)]
    need(len(selected) == 2, "two selected candidate cases/rank required")
    selected_ids = {id(t) for t in selected}
    for t in tests:
        if id(t) not in selected_ids:
            need(
                not any(
                    p.get("value") in {c["receipt"]["path"] for c in cases.values()}
                    for p in t.findall("./properties/property")
                ),
                "foreign JUnit test claims candidate receipt",
            )
    tests = selected
    need(
        all(not t.findall(tag) for t in tests for tag in ("failure", "error", "skipped")),
        "selected candidate Native failure/error/skip",
    )
    seen = set()
    for test in tests:
        need(
            test.get("classname", "").endswith("test_public_captured_diffusion"),
            "foreign Native fixture",
        )
        props = test.findall("./properties/property")
        need(len({p.get("name") for p in props}) == len(props), "duplicate JUnit property")
        props = {p.get("name"): p.get("value") for p in props}
        matches = [
            key
            for key, case in cases.items()
            if props.get("captured_diffusion_receipt") == case["receipt"]["path"]
        ]
        need(
            len(matches) == 1 and matches[0] not in seen, "JUnit actual receipt association differs"
        )
        key = matches[0]
        suffix = "1-order0" if key == "scalar1" else "3-order1"
        need(
            test.get("name")
            == "test_public_candidate_diffusion_nonconstant_saved_and_exact_replay[" + suffix + "]"
            and props.get("dimension") == "2"
            and props.get("rank") == str(rank)
            and props.get("size") == str(ranks)
            and props.get("artifact_identity") == cases[key]["artifact"]
            and props.get("evidence_path") == cases[key]["directory"],
            "JUnit parameters/provenance differ",
        )
        seen.add(key)
    need(seen == set(CASES), "incomplete Native case coverage")
    return dict(
        total=len(root.findall(".//testcase")),
        selected=2,
        failures=len(root.findall(".//failure")),
        errors=len(root.findall(".//error")),
        skipped=len(root.findall(".//skipped")),
    )


def assemble(root, directories, junits, owner, roots):
    qualification = QUALIFICATION
    root = wire.canonical(root)
    need(
        type(roots) is list and str(root) in roots and len(roots) == len(set(roots)),
        "file roots differ",
    )
    need(len(directories) == 2 and len(junits) in (1, 2), "two cases with Serial/MPI2 required")
    origins(owner, roots)
    cases = {}
    for directory in directories:
        case = case_inventory(directory)
        need(Path(case["directory"]).is_relative_to(root), "case escapes archive")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        key = next(
            (k for k, (w, o) in CASES.items() if (receipt["width"], receipt["order"]) == (w, o)),
            None,
        )
        need(key is not None and key not in cases, "foreign/duplicate case")
        receipt_contract(receipt, *CASES[key], len(junits), owner)
        linkage(owner["execution_association"][key], receipt, case["cpp"], roots)
        cases[key] = case
    exact(cases, CASES, "case inventory")
    junit_pins = [leaf(path) for path in junits]
    for rank, pin in enumerate(junit_pins):
        junit(pinned(pin, roots)[1], rank, len(junits), cases)
    return dict(
        schema=SCHEMA,
        qualification=qualification,
        archive_root=str(root),
        file_roots=roots,
        mode="serial" if len(junits) == 1 else "mpi2",
        ranks=len(junits),
        owner=owner,
        junit=junit_pins,
        cases=cases,
    )


def receive(pins_path, pins_sha, approval_path, approval_sha):
    need(
        all(type(v) is str and re.fullmatch("[0-9a-f]{64}", v) for v in (pins_sha, approval_sha))
        and pins_path is not None
        and approval_path is not None,
        "two external seals and ROOT approval are required before reception",
    )
    raw, approved = read(pins_path)[1], read(approval_path)[1]
    need(digest(raw) == pins_sha and digest(approved) == approval_sha, "external seal differs")
    approval = strict_json(approved)
    qualification = approval.get("qualification")
    need(qualification == QUALIFICATION, "foreign qualification")
    need(
        approval
        == dict(
            schema="sol61.candidate-d-root-approval@1",
            approved_by="ROOT",
            pins_sha256=pins_sha,
            qualification=qualification,
        ),
        "external ROOT approval scope differs",
    )
    pins = strict_json(raw)
    exact(
        pins,
        (
            "schema",
            "qualification",
            "archive_root",
            "file_roots",
            "mode",
            "ranks",
            "owner",
            "junit",
            "cases",
        ),
        "owner pins",
    )
    need(
        pins["schema"] == SCHEMA
        and pins["qualification"] == qualification
        and type(pins["ranks"]) is int
        and pins["ranks"] in (1, 2)
        and pins["mode"] == ("serial" if pins["ranks"] == 1 else "mpi2"),
        "candidate owner scope differs",
    )
    roots = pins["file_roots"]
    need(
        type(roots) is list and pins["archive_root"] in roots and len(roots) == len(set(roots)),
        "approved roots differ",
    )
    origins(pins["owner"], roots)
    exact(pins["cases"], CASES, "sealed cases")
    need(
        type(pins["junit"]) is list
        and len(pins["junit"]) == pins["ranks"]
        and len({p["path"] for p in pins["junit"]}) == pins["ranks"],
        "all-rank JUnit inventory differs",
    )
    batches = [
        junit(pinned(row, roots)[1], rank, pins["ranks"], pins["cases"])
        for rank, row in enumerate(pins["junit"])
    ]
    reports = {}
    for key, case in pins["cases"].items():
        need(
            Path(case["directory"]).is_relative_to(wire.canonical(pins["archive_root"])),
            "case outside archive",
        )
        need(case_inventory(case["directory"]) == case, "sealed case inventory differs")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        width, order = CASES[key]
        receipt_contract(receipt, width, order, pins["ranks"], pins["owner"])
        binary_evidence = linkage(
            pins["owner"]["execution_association"][key], receipt, case["cpp"], roots
        )
        need(
            case["ir"] is None
            and all(case["checkpoints"][p] == case["phases"][p] for p in case["checkpoints"]),
            "states-only historical inventory differs",
        )
        cpp_contract(pinned(case["cpp"], roots)[1], width, require_ir=False)
        initial = protocol.archive(pinned(case["initial"], roots)[1])
        states = {
            phase: protocol.archive(pinned(row, roots)[1]) for phase, row in case["phases"].items()
        }
        reports[key] = dict(
            science=science(initial, states, width), component_bytes=binary_evidence
        )
    return dict(
        schema="sol61.candidate-d-scientific-reception@1",
        qualification=qualification,
        checkpoint_reception=False,
        documentary_ir_reception=False,
        junit_batches=batches,
        status="received",
        reviewed_source=SOURCE,
        source_commit=pins["owner"]["source_commit"],
        native_build_source_commit=pins["owner"]["native_build_source_commit"],
        abi_key=pins["owner"]["abi_key"],
        native_sha256=pins["owner"]["native"]["sha256"],
        sdk_sha256=pins["owner"]["sdk"]["sha256"],
        ranks=pins["ranks"],
        cases=reports,
        component_binary_binding_recomputed=True,
        cryptographic_aggregate_binding=False,
        execution_association=ASSOCIATION,
        gaps=[
            "states-only qualification has no conserved checkpoint, history, diagnostics or IR proof; full scope requires distinct files",
            "independent reloaded checkpoint not saved; NPZ exactness checked and same accepted CP anchored",
            "block compiler CPP and original aggregate payload not retained",
            "private candidate/capture lease image not saved",
            "no AMR/GPU/convergence/arbitrary-D/SPD/partition qualification",
        ],
        evidence="external ROOT-approved originals; checker performs no Native execution",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble", help="pending inventory only; never creates ROOT approval")
    for name in ("archive-root", "owner", "output"):
        a.add_argument("--" + name, required=True)
    for name in ("case", "junit", "file-root"):
        a.add_argument("--" + name, action="append", required=True)
    r = sub.add_parser("receive")
    for name in ("pins", "pins-sha256", "approval", "approval-sha256"):
        r.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        result = assemble(
            args.archive_root,
            args.case,
            args.junit,
            strict_json(read(args.owner)[1]),
            args.file_root,
        )
        target = Path(args.output)
        need(not target.exists(), "pending inventory output already exists")
        target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("pending external ROOT approval; no Native/scientific reception claimed")
    else:
        print(
            json.dumps(
                receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
