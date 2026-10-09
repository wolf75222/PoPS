"""Independent composite oracle, restricted to complete finest coverage.
Covered coarse cells carry zero measure. CP12 establishes reduction to the
periodic finest stencil; partial coverage is rejected. No PoPS imports.
"""

import io
import json
import hashlib
from pathlib import Path
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

DT = 1 / 64
NAMES = ("reservoir_z", "zeta_laminated_material", "reservoir_a")
WEIGHTS = ((1 / 8, 1 / 8, 1 / 4, 1 / 2), (-1 / 4, 0, 1 / 4, 0))
READER_CONTRACT = "sol61.m19-amr-consumed-offline@2"


def _carrier_owner_map(child, level, patch_count, rank_count):
    """Read CP12 ownership without inventing owners for replicated levels."""
    mode_value = np.asarray(child["distribution_mode_%d" % level])
    owner_map = np.asarray(child["dmap_%d" % level])
    if (
        mode_value.ndim != 0
        or mode_value.dtype.kind not in "US"
        or owner_map.dtype != np.dtype("int64")
        or owner_map.ndim != 1
        or type(patch_count) is not int
        or patch_count < 1
        or type(rank_count) is not int
        or rank_count < 1
    ):
        raise ValueError("carrier distribution authority")
    mode = str(mode_value.item())
    if mode == "replicated":
        if owner_map.size:
            raise ValueError("carrier distribution authority")
        return (-1,) * patch_count
    if (
        mode != "partitioned"
        or len(owner_map) != patch_count
        or np.any(owner_map < 0)
        or np.any(owner_map >= rank_count)
    ):
        raise ValueError("carrier distribution authority")
    return tuple(int(owner) for owner in owner_map)


def potential(load):
    if (
        load.dtype != np.dtype("float64")
        or load.ndim != 3
        or load.shape[0] != 2
        or not np.isfinite(load).all()
    ):
        raise ValueError("invalid saved source State")
    rhs = load[1]
    ny, nx = rhs.shape
    matrix = np.eye(nx * ny)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for yy, xx, coefficient in (
                (y, (x - 1) % nx, 0.125 * nx * nx),
                (y, (x + 1) % nx, 0.125 * nx * nx),
                ((y - 1) % ny, x, 0.125 * ny * ny),
                ((y + 1) % ny, x, 0.125 * ny * ny),
            ):
                matrix[i, i] += coefficient
                matrix[i, yy * nx + xx] -= coefficient
    phi = np.linalg.solve(matrix, rhs.ravel()).reshape(ny, nx)
    if np.max(abs(matrix @ phi.ravel() - rhs.ravel())) > 1e-12:
        raise ValueError("independent Helmholtz residual")
    return phi


def overlap(phi, n):
    old = len(phi)
    out = np.zeros(n)
    for j in range(n):
        for i, value in enumerate(phi):
            out[j] += (
                float(value) * max(0, min((i + 1) / old, (j + 1) / n) - max(i / old, j / n)) * n
            )
    return out


def remap(value, ny, nx):
    rows = np.stack([overlap(value[:, x], ny) for x in range(value.shape[1])], axis=1)
    return np.stack([overlap(row, nx) for row in rows])


def mapped(phi, index, n):
    return np.asarray(WEIGHTS[index]) @ remap(phi, 4, n)


def verify_step(before, after):
    if set(before) != set(NAMES) or set(after) != set(NAMES):
        raise ValueError("foreign block partition")
    phi = potential(before[NAMES[1]])
    for i, name in enumerate((NAMES[0], NAMES[2])):
        old = before[name]
        new = after[name]
        if (
            old.dtype != np.dtype("float64")
            or new.dtype != old.dtype
            or old.ndim != 3
            or new.ndim != 3
            or old.shape[0] != 3
            or new.shape[0] != 3
            or old.shape[2] != 1
            or new.shape[2] != 1
            or not np.isfinite(new).all()
        ):
            raise ValueError("target storage")
        expected = np.stack([remap(old[c], *new.shape[1:]) for c in range(3)])
        expected[1, :, 0] += (i + 1) * DT * mapped(phi, i, new.shape[1])
        if np.max(abs(new - expected)) > 2e-11:
            raise ValueError("mapped Field/RHS differs from independent stencil and overlap")
    source = after[NAMES[1]]
    expected = np.stack([remap(before[NAMES[1]][c], *source.shape[1:]) for c in range(2)])
    expected[1] += 32 * DT
    if (
        source.dtype != np.dtype("float64")
        or source.shape[0] != 2
        or not np.isfinite(source).all()
        or np.max(abs(source - expected)) > 2e-11
    ):
        raise ValueError("source evolution/injection")
    return {
        "phi_span": float(np.ptp(phi)),
        "mapped_span": float(np.ptp(mapped(phi, 0, after[NAMES[0]].shape[1]))),
        "signed_map_max": float(np.max(abs(mapped(phi, 1, after[NAMES[2]].shape[1])))),
        "field_rhs_tolerance": 2e-11,
        "source_shape": list(source.shape),
        "native_received": False,
    }


def checkpoint_children(path):
    with np.load(path, allow_pickle=False) as outer:
        ids = tuple(str(x) for x in outer["layout_ids"])
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("composite layout partition")
        result = {}
        for i, name in enumerate(ids):
            image = outer["layout_checkpoint_%d" % i]
            if image.dtype != np.uint8 or image.ndim != 1:
                raise ValueError("child checkpoint bytes")
            with np.load(io.BytesIO(image.tobytes()), allow_pickle=False) as child:
                result[name] = {key: child[key].copy() for key in child.files}
        if any(
            float(row["t"]) != float(outer["t"])
            or int(row["macro_step"]) != int(outer["macro_step"])
            for row in result.values()
        ):
            raise ValueError("layouts differ from composite common time")
        return result


def checkpoint_states(path):
    """Reconstruct valid finest bits and establish complete active coverage."""
    result = {}
    authority = {}
    for layout, child in checkpoint_children(path).items():
        if int(child["pops_amr_checkpoint_version"]) != 12:
            raise ValueError("CP12 required")
        archive = decode(child["state_carriers_checkpoint"])
        if (
            archive["dim"] != 2
            or archive["real"] != 64
            or archive["shard"] != -1
            or archive["levels"] != int(child["n_levels"])
            or archive["ranks"] != int(child["n_ranks"])
            or archive["blocks"] != list(child["blocks"])
        ):
            raise ValueError("CP12 full carrier authority")
        spatial = json.loads(str(child["pops_spatial_contract"].item()))
        nx, ny = spatial["shape"]
        source = NAMES[1] in archive["blocks"]
        if set(archive["blocks"]) != ({NAMES[1]} if source else {NAMES[0], NAMES[2]}):
            raise ValueError("checkpoint layout block partition")
        if (
            spatial["dimension"] != 2
            or spatial["shape"] != ([3, 4] if source else [1, 3])
            or spatial["periodicity"] != [True, True]
            or [float.fromhex(x) for x in spatial["lower"]] != [0.0, 0.0]
            or [float.fromhex(x) for x in spatial["upper"]] != [1.0, 1.0]
            or spatial["refinement_ratios"] != ([[2, 2]] if source else [[1, 2]])
            or archive["levels"] not in (1, 2)
        ):
            raise ValueError("oracle spatial scope")
        level = archive["levels"] - 1
        for ratio in spatial["refinement_ratios"][:level]:
            nx *= ratio[0]
            ny *= ratio[1]
        for block, name in enumerate(archive["blocks"]):
            components = 2 if name == NAMES[1] else 3
            bits = np.zeros((components, ny, nx), dtype=np.uint64)
            coverage = np.zeros((ny, nx), dtype=np.uint8)
            selected = [p for p in archive["patches"] if p["key"][:2] == (block, level)]
            owners = _carrier_owner_map(child, level, len(selected), archive["ranks"])
            native_boxes = child["patch_boxes"][child["patch_boxes"][:, 0] == level]
            for patch in selected:
                (xlo, xhi, gxlo, gxhi), (ylo, yhi, gylo, gyhi) = patch["axes"]
                index = patch["key"][2]
                if index >= len(selected) or patch["owner"] != owners[index]:
                    raise ValueError("carrier rank owner")
                if level and (
                    len(native_boxes) != len(selected)
                    or tuple(native_boxes[index, 1:]) != (xlo, ylo, xhi, yhi)
                ):
                    raise ValueError("carrier/native refinement geometry")
                if patch["components"] != components or not (
                    0 <= xlo <= xhi < nx and 0 <= ylo <= yhi < ny
                ):
                    raise ValueError("carrier valid geometry")
                full = np.asarray(patch["bits"], dtype=np.uint64).reshape(
                    components, gyhi - gylo + 1, gxhi - gxlo + 1
                )
                bits[:, ylo : yhi + 1, xlo : xhi + 1] = full[
                    :, ylo - gylo : yhi - gylo + 1, xlo - gxlo : xhi - gxlo + 1
                ]
                coverage[ylo : yhi + 1, xlo : xhi + 1] += 1
            if not np.all(coverage == 1):
                raise ValueError("oracle requires complete finest coverage")
            if name in result or name not in NAMES:
                raise ValueError("checkpoint block partition")
            result[name] = bits.view(np.float64)
            durable = child["state_%s_%d" % (name, level)]
            if (
                durable.dtype != np.float64
                or durable.size != bits.size
                or durable.tobytes() != bits.tobytes()
                or not np.isfinite(result[name]).all()
            ):
                raise ValueError("carrier/CP12 valid State bits")
            authority[name] = {
                "layout": layout,
                "level": level,
                "topology_epoch": int(child["topology_epoch"]),
                "time": float(child["t"]),
                "macro_step": int(child["macro_step"]),
            }
    if set(result) != set(NAMES):
        raise ValueError("checkpoint block partition")
    return result, authority


def receive_phase(path, values, clock):
    carried, authority = checkpoint_states(path)
    for name in NAMES:
        x, y = carried[name], values[name]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            raise ValueError("NPY differs from CP12 finest valid State")
        if (authority[name]["time"], authority[name]["macro_step"]) != tuple(clock):
            raise ValueError("NPY phase common time")
    return authority


def assert_rollback(before, after):
    a, b = checkpoint_children(before), checkpoint_children(after)
    if set(a) != set(b):
        raise ValueError("checkpoint layout authority drift")
    required = (
        "state_carriers_checkpoint",
        "t",
        "macro_step",
        "topology_epoch",
        "patch_boxes",
        "amr_accepted_contract",
        "temporal_restart_state",
        "regrid_count",
        "pops_spatial_contract",
    )
    for name in a:
        for key in required:
            if key not in a[name] or key not in b[name]:
                raise ValueError("missing rollback authority " + key)
            x, y = a[name][key], b[name][key]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise ValueError("rollback changed " + key)
        for key in a[name]:
            if key.startswith(
                ("phi_", "field_provider_phi_", "program_accepted_state", "auxiliary_checkpoint_")
            ):
                if (
                    key not in b[name]
                    or a[name][key].dtype != b[name][key].dtype
                    or a[name][key].shape != b[name][key].shape
                    or a[name][key].tobytes() != b[name][key].tobytes()
                ):
                    raise ValueError("rollback changed " + key)
    with np.load(before, allow_pickle=False) as x, np.load(after, allow_pickle=False) as y:
        if x["mapping_evaluations"].tobytes() != y["mapping_evaluations"].tobytes():
            raise ValueError("rollback changed mapping evaluation authority")
    return {
        "fullgrown_state_restored": True,
        "common_time_and_epoch_restored": True,
        "Field_payload_received": False,
        "history_received": False,
    }


def audit(directory):
    """Read real saved phases without re-running PoPS. Hashes are integrity only."""
    directory = Path(directory).resolve()
    phases = {}
    proofs = {}

    def pinned(row):
        path = Path(row["file"]).resolve()
        if (
            not path.is_relative_to(directory)
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("phase payload pin differs")
        return path

    for label, step in (
        ("initial", 0),
        ("accepted1", 1),
        ("accepted2", 2),
        ("before-failure", 2),
        ("after-failure", 2),
    ):
        receipts = sorted(directory.glob(label + "-rank*.json"))
        if not receipts:
            raise ValueError("missing saved phase " + label)
        rows = []
        for receipt in receipts:
            row = json.loads(receipt.read_text())
            if (
                row["schema"] != "sol61.amr-consumed-phase@1"
                or row["phase"] != label
                or row["capture_complete"] is not True
                or row["clock"] != [step * DT, step]
            ):
                raise ValueError("phase receipt clock/completion")
            values = {
                name: np.load(pinned(pin), allow_pickle=False) for name, pin in row["files"].items()
            }
            checkpoint = pinned(row["checkpoint"])
            authority = receive_phase(checkpoint, values, row["clock"])
            if row["authority"] != authority:
                raise ValueError("phase epoch/level receipt differs")
            rows.append((row, values, checkpoint))
        count = rows[0][0]["ranks"]
        if (
            type(count) is not int
            or len(rows) != count
            or {x[0]["rank"] for x in rows} != set(range(count))
            or any(x[0]["ranks"] != count for x in rows)
        ):
            raise ValueError("phase MPI rank registry")
        for _row, values, _cp in rows[1:]:
            if any(values[name].tobytes() != rows[0][1][name].tobytes() for name in NAMES):
                raise ValueError("global phase State differs by rank")
        phases[label] = rows
    for step in (1, 2):
        before = phases["initial" if step == 1 else "accepted1"][0][1]
        proofs["step%d" % step] = verify_step(before, phases["accepted%d" % step][0][1])
        if (
            proofs["step%d" % step]["mapped_span"] <= 1e-3
            or proofs["step%d" % step]["signed_map_max"] <= 1e-4
        ):
            raise ValueError("spatial/signed witness is trivial")
    start = checkpoint_children(phases["initial"][0][2])
    end = checkpoint_children(phases["accepted2"][0][2])
    if (
        set(start) != set(end)
        or any(int(x["n_levels"]) != 2 for x in end.values())
        or not any(
            int(end[key]["topology_epoch"]) > int(start[key]["topology_epoch"]) for key in start
        )
    ):
        raise ValueError("two-level epoch refresh witness missing")
    for before, after in zip(phases["before-failure"], phases["after-failure"], strict=True):
        if (
            before[0]["rank"] != after[0]["rank"]
            or before[0]["consumer_cursors"] != after[0]["consumer_cursors"]
        ):
            raise ValueError("rollback consumer cursor changed")
        proofs["rollback-rank%d" % before[0]["rank"]] = assert_rollback(before[2], after[2])
    ranks = len(phases["after-failure"])
    targets = [
        json.loads((directory / ("target-rank%d.json" % rank)).read_text()) for rank in range(ranks)
    ]
    observed = {item for rows in targets[0]["observed"] for item in rows}
    if not observed or any(type(item) is not int or not 0 <= item < ranks for item in observed):
        raise ValueError("actual integral rank election is absent")
    target = max(observed)
    for rank, row in enumerate(targets):
        if (
            row["target"] != target
            or row["observed"] != targets[0]["observed"]
            or row["agreement"] != [target] * ranks
        ):
            raise ValueError("failure target vote differs by rank")
        failure = json.loads((directory / ("failure-rank%d.json" % rank)).read_text())
        if (
            failure["target"] != target
            or len(failure["failures"]) != ranks
            or any(
                not x
                or x[2] is not True
                or "nonfinite" not in x[1].lower().replace("-", "").replace(" ", "")
                for x in failure["failures"]
            )
        ):
            raise ValueError("finite-guard failure vote is absent")
        trace = directory / ("integral-rank%d.log" % rank)
        lines = trace.read_text().splitlines() if trace.exists() else []
        injections = [line for line in lines if line.startswith("injected:")]
        if (
            (rank == target and not injections)
            or any(line != "injected:" + str(target) for line in injections)
            or (rank != target and injections)
        ):
            raise ValueError("NaN injection target trace differs")
    proofs["failure"] = {"target": target, "collective_finite_guard": True}
    return {
        "reader_contract": READER_CONTRACT,
        "scope": "offline saved-data numerical and rollback audit",
        "proofs": proofs,
        "Native_received": False,
        "ROOT_approved": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(
        json.dumps(audit(parser.parse_args().directory), sort_keys=True, allow_nan=False, indent=2)
    )
